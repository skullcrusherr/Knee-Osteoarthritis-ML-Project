import warnings, os, pathlib, sys
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ------------------- CONFIG -------------------
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 1       # increase for better training
SEED = 42

DATA_DIR = "dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")
TEST_DIR  = os.path.join(DATA_DIR, "test")

# Five exemplar images (one per grade) for Grad-CAM:
GRADE_SAMPLES = {
    "Grade 0": "dataset/test/0/9003175L.png",
    "Grade 1": "dataset/test/1/9001400L.png",
    "Grade 2": "dataset/test/2/9003316R.png",
    "Grade 3": "dataset/test/3/9011053L.png",
    "Grade 4": "dataset/test/4/9012867R.png",
}

# Saved-model filenames (auto-loaded if present)
MODEL_FILES = {
    "DenseNet121":         "densenet121_model.keras",
    "DenseNet161_as169":   "densenet169_as_161_model.keras",
    "ResNet34_as50":       "resnet50_as_34_model.keras",
    "VGG19":               "vgg19_model.keras",
}
# ------------------------------------------------

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import (
    DenseNet121, DenseNet169,
    ResNet50, VGG19
)
from tensorflow.keras.applications.densenet import preprocess_input as densenet_pre
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_pre

np.random.seed(SEED)
tf.random.set_seed(SEED)

# --------- Data loaders ----------
def get_ds(split_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True):
    return image_dataset_from_directory(
        split_dir,
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=SEED
    )

def cache_prefetch(ds):
    return ds.cache().prefetch(tf.data.AUTOTUNE)

# --------- Model builders (transfer learning heads) ----------
def make_head(num_classes):
    return models.Sequential([
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

def build_densenet121(num_classes):
    base = DenseNet121(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    x = base.output
    head = make_head(num_classes)(x)
    model = models.Model(base.input, head, name="DenseNet121")
    return model, densenet_pre

def build_densenet169_as_161(num_classes):
    base = DenseNet169(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    x = base.output
    head = make_head(num_classes)(x)
    model = models.Model(base.input, head, name="DenseNet161_as169")
    return model, densenet_pre

def build_resnet50_as_34(num_classes):
    base = ResNet50(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    x = base.output
    head = make_head(num_classes)(x)
    model = models.Model(base.input, head, name="ResNet34_as50")
    return model, resnet_pre

def build_vgg19(num_classes):
    base = VGG19(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    x = base.output
    head = make_head(num_classes)(x)
    model = models.Model(base.input, head, name="VGG19")
    return model, vgg_pre

BUILDERS = {
    "DenseNet121":         build_densenet121,
    "DenseNet161_as169":   build_densenet169_as_161,
    "ResNet34_as50":       build_resnet50_as_34,
    "VGG19":               build_vgg19,
}

# -------------- Training/Eval helpers --------------
def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

def apply_preprocess(ds, preprocess_fn):
    def map_fn(x, y):
        return preprocess_fn(tf.cast(x, tf.float32)), y
    return ds.map(map_fn)

def predict_dataset(model, ds):
    y_true = []
    y_score = []
    for batch, labels in ds:
        preds = model.predict(batch, verbose=0)
        y_score.append(preds)
        y_true.append(labels.numpy())
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    y_pred = y_score.argmax(axis=1)
    return y_true, y_score, y_pred

# -------------- Plotting: Confusion & ROC --------------
def make_confusion_matrix_panel(model_results, normalize=False, cmap="magma", figsize=(12, 12), title="Confusion Matrices"):
    n_models = len(model_results)
    ncols = 2 if n_models > 1 else 1
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for idx, (name, (y_true, y_pred, labels)) in enumerate(model_results.items()):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)
        im = ax.imshow(cm, cmap=cmap)
        ax.set_title(f"Confusion Matrix: {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(len(labels))); ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for k in range(n_models, nrows*ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return fig

def make_roc_panel(model_results, labels, figsize=(12, 12), title="ROC-AUC Curves (One-vs-Rest)"):
    n_models = len(model_results)
    ncols = 2 if n_models > 1 else 1
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    n_classes = len(labels)
    for idx, (name, (y_true, y_score)) in enumerate(model_results.items()):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            ax.plot(fpr[i], tpr[i], lw=1, label=f"{labels[i]} (AUC={roc_auc[i]:.3f})")
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        ax.plot(fpr["micro"], tpr["micro"], lw=2, linestyle="--", label=f"micro (AUC={roc_auc['micro']:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="black")
        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC: {name}")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return fig

# -------------- Grad-CAM (Keras, auto last conv) --------------
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    raise ValueError("No Conv2D-like layer found.")

def grad_cam_keras(model, img_array, layer_name):
    grad_model = tf.keras.models.Model([model.inputs],
                                       [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)[0]       # (H, W, C)
    conv_np  = conv_outputs[0].numpy()                 # (H, W, C)
    grads_np = grads.numpy()                           # (H, W, C)
    weights = grads_np.mean(axis=(0, 1))               # (C,)
    cam = np.tensordot(conv_np, weights, axes=([2],[0]))  # (H,W)
    cam = np.maximum(cam, 0)
    cam -= cam.min(); cam /= (cam.max() + 1e-8)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return cam

def make_gradcam_panel(models_and_pre, sample_paths, title="Grad-CAM – all models & grades"):
    """
    Rows = Original + each model. Cols = Grade 0..4 (order of sample_paths).
    models_and_pre: dict name -> (model, preprocess_fn)
    sample_paths: dict grade -> path
    """
    grades = list(sample_paths.keys())
    n_cols = len(grades)
    n_rows = 1 + len(models_and_pre)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0*n_cols, 3.6*n_rows))

    # Top row: originals
    for j, g in enumerate(grades):
        img = tf.keras.utils.load_img(sample_paths[g], target_size=(IMG_SIZE, IMG_SIZE))
        axes[0, j].imshow(img)
        axes[0, j].set_title(g); axes[0, j].axis("off")

    # Each model row
    for i, (name, (model, preprocess_fn)) in enumerate(models_and_pre.items(), start=1):
        try:
            last_conv = find_last_conv_layer(model)
        except Exception:
            last_conv = None
        axes[i, 0].set_ylabel(name, rotation=90, va="center", ha="right")
        for j, g in enumerate(grades):
            img = tf.keras.utils.load_img(sample_paths[g], target_size=(IMG_SIZE, IMG_SIZE))
            arr = tf.keras.utils.img_to_array(img)[None, ...]
            if preprocess_fn is not None:
                arr = preprocess_fn(arr.copy())
            else:
                arr = arr / 255.0
            if last_conv is not None:
                cam = grad_cam_keras(model, arr, last_conv)
                heat = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
                heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
                overlay = np.uint8(0.55 * heat + arr[0])
                axes[i, j].imshow(overlay.astype(np.uint8))
            else:
                axes[i, j].imshow(img)  # fallback
            axes[i, j].axis("off")

    fig.suptitle(title, fontsize=14, y=0.99)
    fig.tight_layout()
    return fig

# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    # 1) Data
    for p in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if not pathlib.Path(p).exists():
            print(f"❌ Missing folder: {p}  (please create dataset/train|val|test/0..4)")
            sys.exit(1)

    print("🔹 Loading datasets ...")
    train_ds = cache_prefetch(get_ds(TRAIN_DIR))
    val_ds   = cache_prefetch(get_ds(VAL_DIR, shuffle=False))
    test_ds  = cache_prefetch(get_ds(TEST_DIR, shuffle=False))

    class_names = sorted([d.name for d in pathlib.Path(TRAIN_DIR).iterdir() if d.is_dir()])
    labels_txt = [f"Grade {c}" for c in range(len(class_names))]
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    # 2) Build or load models
    models_ready = {}          # name -> (model, preprocess_fn)
    ytrue_yscore = {}          # name -> (y_true, y_score, y_pred)

    for name, builder in BUILDERS.items():
        fname = MODEL_FILES[name]
        if pathlib.Path(fname).exists():
            print(f"✅ Loading saved model: {fname}")
            model = load_model(fname)
            if "DenseNet" in name:
                preprocess_fn = densenet_pre
            elif "ResNet" in name:
                preprocess_fn = resnet_pre
            elif "VGG" in name:
                preprocess_fn = vgg_pre
            else:
                preprocess_fn = None
        else:
            print(f"🛠️ Training {name} (no saved model found: {fname})")
            model, preprocess_fn = builder(num_classes)
            compile_model(model)
            train_pp = apply_preprocess(train_ds, preprocess_fn)
            val_pp   = apply_preprocess(val_ds, preprocess_fn)
            model.fit(train_pp, validation_data=val_pp, epochs=EPOCHS, verbose=1)
            model.save(fname)

        models_ready[name] = (model, preprocess_fn)

        # Evaluate on test set
        test_pp = apply_preprocess(test_ds, models_ready[name][1])
        y_true, y_score, y_pred = predict_dataset(models_ready[name][0], test_pp)
        ytrue_yscore[name] = (y_true, y_score, y_pred)

    # 3) Ensemble (soft vote)
    print("🤝 Building Ensemble (soft-vote average of probabilities)...")
    model_names = list(models_ready.keys())
    y_true_ref = ytrue_yscore[model_names[0]][0]
    y_scores_stack = np.stack([ytrue_yscore[m][1] for m in model_names], axis=0)  # (M, N, C)
    y_score_ens = y_scores_stack.mean(axis=0)                                      # (N, C)
    y_pred_ens = y_score_ens.argmax(axis=1)
    ytrue_yscore["Ensemble"] = (y_true_ref, y_score_ens, y_pred_ens)

    # 4) Plots: Confusion & ROC
    print("📊 Saving confusion matrices and ROC curves ...")
    cm_input = {name: (vals[0], vals[2], labels_txt) for name, vals in ytrue_yscore.items()}
    fig_cm = make_confusion_matrix_panel(cm_input, normalize=False, title="Confusion Matrices (Test)")
    fig_cm.savefig("confusion_matrices.png", dpi=300, bbox_inches="tight")

    roc_input = {name: (vals[0], vals[1]) for name, vals in ytrue_yscore.items()}
    fig_roc = make_roc_panel(roc_input, labels_txt, title="ROC-AUC Curves (Test)")
    fig_roc.savefig("roc_curves.png", dpi=300, bbox_inches="tight")

    # 5) Grad-CAM: rows = models + Ensemble (CAM from first model)
    print("🔥 Saving Grad-CAM grid for all models ...")
    models_for_cam = {}
    for name in model_names:
        models_for_cam[name] = models_ready[name]
    if model_names:
        models_for_cam["Ensemble_CAM_from_" + model_names[0]] = models_ready[model_names[0]]

    # Verify sample image paths exist
    for g, p in GRADE_SAMPLES.items():
        if not pathlib.Path(p).exists():
            print(f"⚠️ Missing sample for {g}: {p} (CAM will fail). Please update GRADE_SAMPLES.")
    fig_cam = make_gradcam_panel(models_for_cam, GRADE_SAMPLES, title="Grad-CAM – All Models x Grades")
    fig_cam.savefig("gradcam_all_models.png", dpi=300, bbox_inches="tight")

    print("✅ All PNGs saved: confusion_matrices.png, roc_curves.png, gradcam_all_models.png")
