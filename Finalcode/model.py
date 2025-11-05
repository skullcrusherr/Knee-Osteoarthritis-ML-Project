# model.py — maximized accuracy: MixUp + cosine LR + deeper FT + better aug + MLflow AUC/QWK
import os, json, random
from math import ceil
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.applications import EfficientNetV2B0
import tensorflow.keras.applications.efficientnet_v2 as effv2
import cv2

# plotting & metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, cohen_kappa_score

# ============ Optional MLflow ============
USE_MLFLOW = True
mlflow = None
try:
    import mlflow
    import mlflow.keras
    if os.getenv("MLFLOW_TRACKING_URI") is None:
        mlflow.set_tracking_uri("file:" + os.path.abspath("mlruns"))
    mlflow.set_experiment("knee_oa_efficientnetv2")
except Exception as e:
    print("[Info] MLflow not available -> continuing without experiment tracking:", e)
    USE_MLFLOW = False

# =========================
# Reproducibility & GPU QoL
# =========================
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
try:
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

# Let TF grow GPU memory instead of grabbing all
try:
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

# =========================
# Paths / Config
# =========================
DATA_ROOT = "dataset"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR   = os.path.join(DATA_ROOT, "val")
TEST_DIR  = os.path.join(DATA_ROOT, "test")

# Toggle: 3-class merge improves robustness vs class imbalance
MERGE_TO_THREE = True   # True => Healthy/Mild/Severe ; False => original 5 classes 0..4

IMG_SIZE = (320, 320)
BATCH = 16
WARMUP_EPOCHS = 4
FINETUNE_EPOCHS = 24
DROPOUT = 0.5
LABEL_SMOOTHING = 0.05
BASE_LR = 2e-3           # head training
FINETUNE_LR = 1e-4       # initial LR for cosine schedule
N_UNFREEZE = 240

ARTIF_DIR = "artifacts"
CHECKPOINT_PATH = os.path.join(ARTIF_DIR, "best_model.keras")
FINAL_MODEL_PATH = "model.keras"

os.makedirs(ARTIF_DIR, exist_ok=True)

# =========================
# Label mapping (5 -> 3)
# =========================
FIVE_TO_THREE = {
    "0": "Healthy",
    "1": "Healthy",
    "2": "Mild",
    "3": "Severe",
    "4": "Severe"
}
with open("label_mapping.json", "w") as f:
    json.dump({"merge_to_three": MERGE_TO_THREE, "five_to_three": FIVE_TO_THREE}, f, indent=2)

# =========================
# CLAHE + preprocess (train vs eval)
# =========================
def preprocess_xray_clahe(img_uint8_rgb: np.ndarray) -> np.ndarray:
    """CLAHE on grayscale → back to RGB (uint8)."""
    gray = cv2.cvtColor(img_uint8_rgb, cv2.COLOR_RGB2GRAY)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)

def train_preprocess_function(x: np.ndarray) -> np.ndarray:
    """
    Train-only preprocessing:
    - CLAHE
    - Light gamma jitter (0.9–1.1)
    - Occasional small Gaussian blur
    - EfficientNetV2 normalization
    """
    x_uint8 = np.clip(x, 0, 255).astype(np.uint8)
    x = preprocess_xray_clahe(x_uint8)

    # gamma jitter
    if np.random.rand() < 0.7:
        gamma = np.random.uniform(0.9, 1.1)
        x = np.clip(255.0 * ((x / 255.0) ** gamma), 0, 255).astype(np.uint8)

    # slight blur (simulate defocus)
    if np.random.rand() < 0.25:
        x = cv2.GaussianBlur(x, (3, 3), sigmaX=0.6)

    x = x.astype(np.float32)
    return effv2.preprocess_input(x)

def eval_preprocess_function(x: np.ndarray) -> np.ndarray:
    """Validation/Test preprocessing: CLAHE + EfficientNetV2 normalization."""
    x_uint8 = np.clip(x, 0, 255).astype(np.uint8)
    x = preprocess_xray_clahe(x_uint8).astype(np.float32)
    return effv2.preprocess_input(x)

# =========================
# Base 5-class iterators
# =========================
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    shear_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=train_preprocess_function,
)
eval_datagen = ImageDataGenerator(
    preprocessing_function=eval_preprocess_function
)

train_iter_5 = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode="categorical", shuffle=True, color_mode='rgb'
)
val_iter_5 = eval_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode="categorical", shuffle=False, color_mode='rgb'
)
test_iter_5 = eval_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode="categorical", shuffle=False, color_mode='rgb'
)

# =========================
# Merge 5->3 on the fly (if enabled)
# =========================
if not MERGE_TO_THREE:
    train_gen = train_iter_5
    val_gen   = val_iter_5
    test_gen  = test_iter_5
    NUM_CLASSES = train_iter_5.num_classes
    class_indices_to_save = train_iter_5.class_indices
    steps_train = steps_val = steps_test = None
else:
    inv5 = {v: k for k, v in train_iter_5.class_indices.items()}           # idx -> "0","1","2","3","4"
    names3 = sorted({FIVE_TO_THREE[name5] for name5 in inv5.values()})     # ["Healthy","Mild","Severe"]
    to3 = {name3: i for i, name3 in enumerate(names3)}                     # "Healthy"->0 etc.

    def merge_batch(y_onehot: np.ndarray) -> np.ndarray:
        idx5 = np.argmax(y_onehot, axis=1)
        idx3 = np.array([to3[FIVE_TO_THREE[inv5[i]]] for i in idx5], dtype=np.int32)
        y3 = tf.keras.utils.to_categorical(idx3, num_classes=len(names3))
        return y3

    def merged_generator(base_iter):
        while True:
            x, y5 = next(base_iter)
            y3 = merge_batch(y5)
            # label smoothing
            if LABEL_SMOOTHING > 0:
                eps = LABEL_SMOOTHING
                y3 = (1 - eps) * y3 + eps / y3.shape[1]
            yield x, y3

    train_gen = merged_generator(train_iter_5)
    val_gen   = merged_generator(val_iter_5)
    test_gen  = merged_generator(test_iter_5)

    steps_train = ceil(train_iter_5.samples / BATCH)
    steps_val   = ceil(val_iter_5.samples   / BATCH)
    steps_test  = ceil(test_iter_5.samples  / BATCH)

    NUM_CLASSES = len(names3)
    class_indices_to_save = {name: i for i, name in enumerate(names3)}

# Persist class names for your Streamlit app
with open("class_indices.json", "w") as f:
    json.dump(class_indices_to_save, f, indent=2)

# =========================
# Class Weights (info only; not applied with Python generators)
# =========================
if not MERGE_TO_THREE:
    counts5 = Counter(train_iter_5.classes)
    maxc = max(counts5.values())
    class_weight = {int(k): float(maxc/v) for k, v in counts5.items()}
else:
    counts5 = Counter(train_iter_5.classes)
    counts3 = Counter()
    for k5_idx, cnt in counts5.items():
        name5 = inv5[k5_idx]
        name3 = FIVE_TO_THREE[name5]
        counts3[to3[name3]] += cnt
    maxc = max(counts3.values())
    class_weight = {int(k): float(maxc/v) for k, v in counts3.items()}
print("[Info] Class weights (not applied with generator inputs):", class_weight)

# =========================
# MixUp wrapper (train only)
# =========================
def mixup_batch(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    idx = np.random.permutation(x.shape[0])
    x_mix = lam * x + (1 - lam) * x[idx]
    y_mix = lam * y + (1 - lam) * y[idx]
    return x_mix, y_mix

def mixup_generator(gen, prob=0.5, alpha=0.3):
    while True:
        x, y = next(gen)
        if np.random.rand() < prob:
            x, y = mixup_batch(x, y, alpha=alpha)
        yield x, y

# Wrap ONLY the training generator with MixUp
if MERGE_TO_THREE:
    train_gen = mixup_generator(train_gen, prob=0.5, alpha=0.3)
else:
    train_gen = mixup_generator(train_iter_5, prob=0.5, alpha=0.3)

# =========================
# Build EfficientNetV2-B0 model
# =========================
base = EfficientNetV2B0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Phase 1: freeze backbone
for layer in base.layers:
    layer.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(DROPOUT)(x)
out = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base.input, outputs=out)
try:
    opt_head = AdamW(learning_rate=BASE_LR, weight_decay=1e-4)
except Exception:
    opt_head = Adam(learning_rate=BASE_LR)

model.compile(optimizer=opt_head, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# =========================
# Callbacks
# =========================
ckpt = ModelCheckpoint(
    CHECKPOINT_PATH, monitor="val_loss", mode="min",
    save_best_only=True, verbose=1
)
early = EarlyStopping(
    monitor="val_loss", patience=8, min_delta=1e-3,
    restore_best_weights=True, verbose=1
)
# keep ReduceLROnPlateau for warm-up only (we'll switch to cosine in finetune)
reduce = ReduceLROnPlateau(
    monitor="val_loss", factor=0.3, patience=3,
    min_lr=1e-7, verbose=1
)

# =========================
# Helpers for metrics/plots
# =========================
def _get_true_labels_from_iterator(it5, merge_to_three, mapping_5_to_name3, name3_to_idx):
    y5 = it5.classes.astype(int)
    if not merge_to_three:
        return y5
    inv5_local = {v: k for k, v in it5.class_indices.items()}
    y3 = []
    for idx in y5:
        name5 = inv5_local[idx]
        name3 = mapping_5_to_name3[name5]
        y3.append(name3_to_idx[name3])
    return np.array(y3, dtype=int)

def _predict_probs(model, iterator_or_gen, steps=None):
    return model.predict(iterator_or_gen, steps=steps, verbose=1)

def _plot_cm_roc(split, y_true, y_prob, class_names, out_prefix):
    os.makedirs(ARTIF_DIR, exist_ok=True)
    n_classes = len(class_names)
    y_pred = np.argmax(y_prob, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    fig_cm, ax = plt.subplots(figsize=(4.2, 3.6), dpi=150)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion Matrix — {split}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="w" if cm[i, j] > cm.max()/2 else "black", fontsize=8)
    fig_cm.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cm_path = os.path.join(ARTIF_DIR, f"{out_prefix}_{split}_cm.png")
    fig_cm.tight_layout(); fig_cm.savefig(cm_path); plt.close(fig_cm)

    # ROC curves (+ micro/macro)
    fig_roc, ax2 = plt.subplots(figsize=(4.6, 3.6), dpi=150)
    y_true_bin = np.eye(n_classes)[y_true]
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_prob[:, c])
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, lw=1.2, label=f"{class_names[c]} (AUC={roc_auc:.3f})")
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    ax2.plot(fpr_micro, tpr_micro, linestyle="--", lw=1.2, label=f"micro (AUC={auc_micro:.3f})")
    all_fpr = np.unique(np.concatenate([roc_curve(y_true_bin[:, c], y_prob[:, c])[0] for c in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_prob[:, c])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= n_classes
    auc_macro = auc(all_fpr, mean_tpr)
    ax2.plot(all_fpr, mean_tpr, linestyle=":", lw=1.2, label=f"macro (AUC={auc_macro:.3f})")
    ax2.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax2.set_xlim([0.0, 1.0]); ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
    ax2.set_title(f"ROC — {split}"); ax2.legend(fontsize=7, loc="lower right")
    roc_path = os.path.join(ARTIF_DIR, f"{out_prefix}_{split}_roc.png")
    fig_roc.tight_layout(); fig_roc.savefig(roc_path); plt.close(fig_roc)

    return cm_path, roc_path

# =========================
# Training (with optional MLflow)
# =========================
def run_training():
    # track basic params
    params_to_log = {
        "merge_to_three": MERGE_TO_THREE,
        "img_w": IMG_SIZE[0], "img_h": IMG_SIZE[1],
        "batch": BATCH,
        "warmup_epochs": WARMUP_EPOCHS,
        "finetune_epochs": FINETUNE_EPOCHS,
        "dropout": DROPOUT,
        "label_smoothing": LABEL_SMOOTHING,
        "base_lr": BASE_LR,
        "finetune_lr_init": FINETUNE_LR,
        "n_unfreeze": N_UNFREEZE,
        "num_classes": len(class_indices_to_save),
        "mixup": True,
        "cosine_decay_restarts": True,
    }

    if USE_MLFLOW:
        mlflow.start_run()
        for k,v in params_to_log.items():
            mlflow.log_param(k, v)
        mlflow.log_artifact("class_indices.json", artifact_path="artifacts")
        mlflow.log_artifact("label_mapping.json", artifact_path="artifacts")

    # -------- Phase 1: Warm-up (frozen backbone, ReduceLROnPlateau) --------
    print("\n[Phase 1] Warm-up (frozen backbone)")
    if MERGE_TO_THREE:
        hist1 = model.fit(
            train_gen,
            epochs=WARMUP_EPOCHS,
            steps_per_epoch=steps_train,
            validation_data=val_gen,
            validation_steps=steps_val,
            callbacks=[ckpt, early, reduce],
            verbose=1
        )
    else:
        hist1 = model.fit(
            train_gen,
            epochs=WARMUP_EPOCHS,
            validation_data=val_gen,
            callbacks=[ckpt, early, reduce],
            verbose=1
        )

    # -------- Phase 2: Fine-tune with cosine decay restarts --------
    for layer in base.layers[-N_UNFREEZE:]:
        layer.trainable = True

    steps_per_epoch = steps_train if MERGE_TO_THREE else (train_iter_5.samples // BATCH)
    schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=FINETUNE_LR,
        first_decay_steps=steps_per_epoch * 5,  # ~5 epochs per restart
        t_mul=2.0,
        m_mul=0.9
    )
    try:
        opt_ft = AdamW(learning_rate=schedule, weight_decay=2e-5)
    except Exception:
        opt_ft = Adam(learning_rate=schedule)

    model.compile(optimizer=opt_ft, loss="categorical_crossentropy", metrics=["accuracy"])
    print("\n[Phase 2] Fine-tuning last blocks (cosine LR + restarts)")

    if MERGE_TO_THREE:
        hist2 = model.fit(
            train_gen,
            epochs=FINETUNE_EPOCHS,
            steps_per_epoch=steps_train,
            validation_data=val_gen,
            validation_steps=steps_val,
            callbacks=[ckpt, early],  # no ReduceLROnPlateau here
            verbose=1
        )
    else:
        hist2 = model.fit(
            train_gen,
            epochs=FINETUNE_EPOCHS,
            validation_data=val_gen,
            callbacks=[ckpt, early],
            verbose=1
        )

    # -------- Load best & save final --------
    if os.path.exists(CHECKPOINT_PATH):
        print(f"[Info] Loading best checkpoint: {CHECKPOINT_PATH}")
        best_model = tf.keras.models.load_model(CHECKPOINT_PATH)
    else:
        best_model = model

    best_model.save(FINAL_MODEL_PATH)
    print(f"[Info] Saved final model to {FINAL_MODEL_PATH}")
    if USE_MLFLOW:
        mlflow.log_artifact(CHECKPOINT_PATH, artifact_path="artifacts")
        mlflow.log_artifact(FINAL_MODEL_PATH, artifact_path="artifacts")

    # -------- Evaluate & log CM/ROC + macro AUC + QWK --------
    class_names = [None] * len(class_indices_to_save)
    for name, idx in class_indices_to_save.items():
        class_names[idx] = name

    # Validation
    print("\n[Eval] Validation:")
    y_true_val = _get_true_labels_from_iterator(val_iter_5, MERGE_TO_THREE, FIVE_TO_THREE, {k:v for k,v in class_indices_to_save.items()})
    y_prob_val = _predict_probs(best_model, val_gen if MERGE_TO_THREE else val_iter_5,
                                steps=steps_val if MERGE_TO_THREE else None)
    val_metrics = best_model.evaluate(val_gen if MERGE_TO_THREE else val_iter_5,
                                      steps=steps_val if MERGE_TO_THREE else None, verbose=1)
    print(dict(zip(best_model.metrics_names, val_metrics)))
    cm_v_path, roc_v_path = _plot_cm_roc("val", y_true_val, y_prob_val, class_names, out_prefix="effnetv2")

    # Additional MLflow metrics: macro AUC & QWK
    y_pred_val = np.argmax(y_prob_val, axis=1)
    y_true_val_bin = np.eye(len(class_names))[y_true_val]
    try:
        macro_auc_val = roc_auc_score(y_true_val_bin, y_prob_val, average="macro", multi_class="ovr")
    except Exception:
        macro_auc_val = float("nan")
    qwk_val = cohen_kappa_score(y_true_val, y_pred_val, weights="quadratic")

    # Test
    print("\n[Eval] Test:")
    y_true_test = _get_true_labels_from_iterator(test_iter_5, MERGE_TO_THREE, FIVE_TO_THREE, {k:v for k,v in class_indices_to_save.items()})
    y_prob_test = _predict_probs(best_model, test_gen if MERGE_TO_THREE else test_iter_5,
                                 steps=steps_test if MERGE_TO_THREE else None)
    test_metrics = best_model.evaluate(test_gen if MERGE_TO_THREE else test_iter_5,
                                       steps=steps_test if MERGE_TO_THREE else None, verbose=1)
    print(dict(zip(best_model.metrics_names, test_metrics)))
    cm_t_path, roc_t_path = _plot_cm_roc("test", y_true_test, y_prob_test, class_names, out_prefix="effnetv2")

    y_pred_test = np.argmax(y_prob_test, axis=1)
    y_true_test_bin = np.eye(len(class_names))[y_true_test]
    try:
        macro_auc_test = roc_auc_score(y_true_test_bin, y_prob_test, average="macro", multi_class="ovr")
    except Exception:
        macro_auc_test = float("nan")
    qwk_test = cohen_kappa_score(y_true_test, y_pred_test, weights="quadratic")

    print({"val_macro_auc": macro_auc_val, "val_qwk": qwk_val,
           "test_macro_auc": macro_auc_test, "test_qwk": qwk_test})

    if USE_MLFLOW:
        try:
            for name, val in zip(best_model.metrics_names, val_metrics):
                mlflow.log_metric(f"val_{name}", float(val))
            for name, val in zip(best_model.metrics_names, test_metrics):
                mlflow.log_metric(f"test_{name}", float(val))
            mlflow.log_metric("val_macro_auc", float(macro_auc_val))
            mlflow.log_metric("test_macro_auc", float(macro_auc_test))
            mlflow.log_metric("val_qwk", float(qwk_val))
            mlflow.log_metric("test_qwk", float(qwk_test))
        except Exception:
            pass
        mlflow.log_artifact(cm_v_path, artifact_path="artifacts/plots")
        mlflow.log_artifact(roc_v_path, artifact_path="artifacts/plots")
        mlflow.log_artifact(cm_t_path, artifact_path="artifacts/plots")
        mlflow.log_artifact(roc_t_path, artifact_path="artifacts/plots")
        mlflow.end_run()

if __name__ == "__main__":
    run_training()
