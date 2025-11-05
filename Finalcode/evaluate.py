# evaluate.py

import json
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, matthews_corrcoef,
    roc_auc_score, average_precision_score, brier_score_loss
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input

def expected_calibration_error(probs, y_true, n_bins=15):
    # probs: (N, C), y_true: int labels (N,)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if np.any(mask):
            gap = abs(accuracies[mask].mean() - confidences[mask].mean())
            ece += gap * (mask.sum() / len(confidences))
    return ece

model = tf.keras.models.load_model('model.h5')
BATCH=32; IMG_SIZE=(299,299)
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = datagen.flow_from_directory('dataset/test', target_size=IMG_SIZE,
                                       batch_size=BATCH, class_mode='categorical', shuffle=False)

probs = model.predict(test_gen, verbose=1)
y_true = test_gen.classes
y_onehot = label_binarize(y_true, classes=list(range(test_gen.num_classes)))

preds = probs.argmax(axis=1)

# Metrics
mcc = matthews_corrcoef(y_true, preds)
balanced_acc = (confusion_matrix(y_true, preds, normalize='true').diagonal().mean())
macro_auroc = roc_auc_score(y_onehot, probs, average='macro', multi_class='ovr')
macro_auprc = average_precision_score(y_onehot, probs, average='macro')
brier = brier_score_loss(y_onehot.ravel(), probs.ravel())
ece = expected_calibration_error(probs, y_true, n_bins=15)

print(f"MCC: {mcc:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"Macro-AUROC: {macro_auroc:.4f}")
print(f"Macro-AUPRC: {macro_auprc:.4f}")
print(f"Brier Score (lower better): {brier:.4f}")
print(f"ECE (lower better): {ece:.4f}")
