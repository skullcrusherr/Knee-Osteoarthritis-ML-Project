from knee_eval_plots import (
    ModelEval,
    make_confusion_matrix_panel,
    make_roc_panel,
    make_eigencam_grid
)

# 1) Build inputs per model
results = {
    "DenseNet-121": ModelEval(y_true, y_proba_121, labels=["Grade 0","Grade 1","Grade 2","Grade 3","Grade 4"]),
    "DenseNet-161": ModelEval(y_true, y_proba_161, labels=...),
    "ResNet-34":   ModelEval(y_true, y_proba_res34, labels=...),
    "VGG-19":      ModelEval(y_true, y_proba_vgg19, labels=...),
    "Ensemble":    ModelEval(y_true, y_proba_ens, labels=...),
}

fig_cm = make_confusion_matrix_panel(results, normalize=False, title="Confusion Matrices")
fig_cm.savefig("confusion_matrices.png", dpi=300, bbox_inches="tight")

fig_roc = make_roc_panel(results, title="ROC-AUC Curves")
fig_roc.savefig("roc_curves.png", dpi=300, bbox_inches="tight")

# 3) EigenCAM grid (PyTorch):
# map each model to its last conv target_layer
model_and_layers = {
    "DenseNet-121": (d121_model, d121_model.features[-1]),
    "DenseNet-161": (d161_model, d161_model.features[-1]),
    "ResNet-34":    (res34_model, res34_model.layer4[-1].conv2),
    "VGG-19":       (vgg19_model, vgg19_model.features[-1]),
    "Ensemble":     (ens_model,   ens_model.backbone.layer4[-1].conv2)  # example
}
grade_images = {
    "Grade 0": img0, "Grade 1": img1, "Grade 2": img2, "Grade 3": img3, "Grade 4": img4
}
fig_cam = make_eigencam_grid(model_and_layers, grade_images, input_size=(224,224))
fig_cam.savefig("eigencam_grid.png", dpi=300, bbox_inches="tight")
