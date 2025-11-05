# explain.py
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import Model

# ---------- Preproc (same as training) ----------
def preprocess_xray(img_uint8_rgb):
    gray = cv2.cvtColor(img_uint8_rgb, cv2.COLOR_RGB2GRAY)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    rgb = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
    return rgb

# ---------- Grad-CAM ----------
# --- replace the two functions below in explain.py ---

def find_last_conv_layer(model):
    """
    Return the last layer *object* in the model graph that yields a 4D tensor
    (H, W, C). Handles nested models and layers with multiple outputs.
    """
    # 1) direct scan (reverse order)
    for layer in reversed(model.layers):
        try:
            out = layer.output  # could be a Tensor or list/tuple
            # normalize to a list
            outs = out if isinstance(out, (list, tuple)) else [out]
            for t in outs:
                if hasattr(t, "shape") and len(t.shape) == 4:
                    return layer, t  # return layer and the specific tensor
        except Exception:
            continue

    # 2) try diving into nested submodels
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            try:
                sub_layer, sub_tensor = find_last_conv_layer(layer)
                if sub_layer is not None:
                    return sub_layer, sub_tensor
            except Exception:
                continue

    raise ValueError("No 4D conv layer found for Grad-CAM.")

def get_gradcam_heatmap(img_array, model, class_index=None, layer_name=None):
    """
    img_array: (1,H,W,3) preprocessed input
    class_index: int or None (defaults to top-1)
    layer_name: optional string to force a specific layer by name
    returns: heatmap (H, W) in [0,1]
    """
    # Find target conv tensor
    if layer_name is not None:
        layer = model.get_layer(layer_name)
        # pick any 4D output from this layer
        outs = layer.output if isinstance(layer.output, (list, tuple)) else [layer.output]
        target_tensor = None
        for t in outs:
            if len(t.shape) == 4:
                target_tensor = t; break
        if target_tensor is None:
            raise ValueError(f"Layer '{layer_name}' has no 4D output for Grad-CAM.")
    else:
        layer, target_tensor = find_last_conv_layer(model)

    # Build a model that outputs (target feature map, predictions)
    grad_model = Model(inputs=model.inputs, outputs=[target_tensor, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    # Global-average-pool the gradients over spatial dims (H,W)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    # Weighted sum of channels
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap)  # keep only positive influence
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    # Convert to numpy if needed
    heatmap = heatmap.numpy() if hasattr(heatmap, "numpy") else heatmap
    return heatmap

# ---------- Integrated Gradients ----------
def integrated_gradients(img_array, model, class_index=None, m_steps=50, baseline=None):
    """
    img_array: (1,H,W,3) preprocessed input (float32)
    baseline:  (1,H,W,3) preprocessed baseline (defaults to zeros)
    returns IG attribution map (H,W) normalized to [0,1]
    """
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
    if baseline is None:
        baseline = tf.zeros_like(img_array)

    if class_index is None:
        preds = model(img_array, training=False)
        class_index = tf.argmax(preds[0])

    # interpolate between baseline and input
    alphas = tf.linspace(0.0, 1.0, m_steps + 1)
    interpolated = baseline + alphas[:, None, None, None] * (img_array - baseline)

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        preds = model(interpolated, training=False)
        target = preds[:, class_index]

    grads = tape.gradient(target, interpolated)  # (m+1,H,W,3)
    # path integral approximation: average grads, then multiply by (input - baseline)
    avg_grads = tf.reduce_mean(grads, axis=0)  # (H,W,3)
    ig = (img_array[0] - baseline[0]) * avg_grads  # (H,W,3)

    # aggregate channels and normalize to [0,1]
    ig = tf.reduce_sum(tf.abs(ig), axis=-1)
    ig = ig / (tf.reduce_max(ig) + 1e-8)
    ig_np = ig.numpy() if hasattr(ig, "numpy") else ig
    return ig_np  # (H,W)

# ---------- Overlay ----------
def overlay_heatmap(heatmap, image_rgb_uint8, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image_rgb_uint8.shape[1], image_rgb_uint8.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_rgb_uint8, 1 - alpha, heatmap_color, alpha, 0)
    return overlay
