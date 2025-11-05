# app.py

import os, json, glob, re, base64
import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image
from tensorflow.keras.applications import efficientnet_v2 as effv2

# --------- PAGE CONFIG ---------
st.set_page_config(page_title="Knee OA Severity", layout="wide")

# --------- LIGHT CSS POLISH ---------
APP_CSS = """
<style>
/* slim page */
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1200px; }

/* metric spacing */
.css-1xarl3l, .stMetric { margin-top: 0.5rem; }

/* chat bubbles */
.chat-bubble {
  border-radius: 16px; padding: 12px 16px; margin: 6px 0; display: inline-block;
  max-width: 100%;
  box-shadow: 0 2px 10px rgba(0,0,0,.05);
}
.chat-user { background: #e8f0fe; color:#111; }
.chat-bot  { background: #f6f8fa; color:#111; }

/* citations as chips */
.citation-chip {
  display:inline-block; padding:4px 8px; margin:0 6px 6px 0; border-radius:12px;
  background:#eef2ff; font-size:0.85rem; border:1px solid #dbeafe;
}

/* section headers tighter */
h2, h3, h4 { margin-top: 0.6rem; }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

# --------- TITLE ---------
st.title("🦴 Knee Osteoarthritis – Severity Grading")

# --------- LABELS (match training) ---------
def load_labels():
    with open("class_indices.json", "r") as f:
        idx_map = json.load(f)  # {"Healthy":0,...} or {"0":0,...,"4":4}
    names = [None] * len(idx_map)
    for name, i in idx_map.items():
        names[i] = name
    merge_info = {}
    if os.path.exists("label_mapping.json"):
        with open("label_mapping.json", "r") as f:
            merge_info = json.load(f)
    return names, merge_info

CLASS_NAMES, MERGE_INFO = load_labels()

# --------- MODEL ---------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")
model = load_model()

# --------- IMAGE PREPROCESS (match training) ---------
IMG_SIZE = (256, 256)

def preprocess_xray_numpy(img_uint8_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_uint8_rgb, cv2.COLOR_RGB2GRAY)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    rgb = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
    return rgb

def import_and_predict(pil_image: Image.Image):
    img = pil_image.convert("RGB").resize(IMG_SIZE, Image.Resampling.LANCZOS)
    arr = np.asarray(img).astype(np.uint8)
    arr = preprocess_xray_numpy(arr).astype(np.float32)
    arr = effv2.preprocess_input(arr)
    probs = model.predict(arr[None, ...], verbose=0)[0]
    return probs

# --------- TABS ---------
tab_pred, tab_expl, tab_chat, tab_about = st.tabs(
    ["🔮 Predict", "🧠 Explain", "💬 Chatbot", "ℹ️ About"]
)

# =======================
# PREDICT TAB
# =======================
with tab_pred:
    st.subheader("Upload an X-ray")
    file = st.file_uploader("Upload a knee X-ray (jpg/png)", type=["jpg","png"])
    if file:
        image = Image.open(file)
        colA, colB = st.columns([3,2])
        with colA:
            st.image(image, caption="Input", use_container_width=True)
        with colB:
            probs = import_and_predict(image)
            pred_idx = int(np.argmax(probs))
            pred_label = CLASS_NAMES[pred_idx]
            st.metric("Predicted Severity", pred_label)
            st.write("Confidence by class:")
            for name, p in zip(CLASS_NAMES, probs):
                st.progress(int(p * 100), text=f"{name}: {p:.2f}")
        st.caption(f"Max confidence: {probs[pred_idx]:.2f} • Classes: {', '.join(CLASS_NAMES)}")

# =======================
# EXPLAIN TAB
# =======================
from explain import preprocess_xray, get_gradcam_heatmap, integrated_gradients, overlay_heatmap

with tab_expl:
    st.subheader("Model Explanation")

    method = st.radio("Explanation method", ["Grad-CAM", "Integrated Gradients"], horizontal=True)
    alpha = st.slider("Heatmap opacity", 0.0, 1.0, 0.4, 0.05)
    class_choice = st.selectbox("Class to explain (optional; defaults to top-1)", ["(auto: top-1)"] + CLASS_NAMES)

    file2 = st.file_uploader("Upload an X-ray to explain", type=["jpg","png"], key="exp")
    if file2:
        img_pil = Image.open(file2).convert("RGB").resize(IMG_SIZE, Image.Resampling.LANCZOS)
        img_uint8 = np.asarray(img_pil).astype(np.uint8)
        img_clahe = preprocess_xray(img_uint8).astype(np.float32)
        img_model = effv2.preprocess_input(img_clahe.copy())
        x = img_model[None, ...]

        probs = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        class_index = CLASS_NAMES.index(class_choice) if class_choice != "(auto: top-1)" else pred_idx

        st.write(
            f"Predicted: **{CLASS_NAMES[pred_idx]}** (p={probs[pred_idx]:.2f}) • "
            f"Explaining: **{CLASS_NAMES[class_index]}**"
        )

        if method == "Grad-CAM":
            heatmap = get_gradcam_heatmap(x, model, class_index=class_index)
        else:
            heatmap = integrated_gradients(x, model, class_index=class_index, m_steps=50)

        overlay = overlay_heatmap(heatmap, img_uint8, alpha=alpha)
        c1, c2 = st.columns(2)
        with c1: st.image(img_pil, caption="Original", use_container_width=True)
        with c2: st.image(overlay,  caption=f"{method} Heatmap", use_container_width=True)

        with st.expander("Class probabilities"):
            for name, p in zip(CLASS_NAMES, probs):
                st.progress(int(p * 100), text=f"{name}: {p:.2f}")

# =======================
# CHATBOT TAB (RAG over local markdown)
# =======================
from chatbot import KBIndex, answer_query

def _kb_fingerprint(knowledge_dir="knowledge"):
    files = sorted(glob.glob(os.path.join(knowledge_dir, "**", "*.md"), recursive=True))
    stats = [(f, os.path.getmtime(f), os.path.getsize(f)) for f in files if os.path.isfile(f)]
    return str(stats)

@st.cache_resource
def load_kb(_fp: str):
    kb = KBIndex(knowledge_dir="knowledge", chunk_size=800, overlap=120)
    kb.build()
    return kb

kb = load_kb(_kb_fingerprint())

# Pretty extraction: if the chunk contains multiple Q/A, extract the best matching pair
def extract_best_qa(text: str, query: str) -> str:
    """
    Looks for Q:/A: pairs in the retrieved chunk and returns the most relevant A:
    Falls back to cleaned chunk.
    """
    qa_pairs = re.findall(r"Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)", text, flags=re.IGNORECASE | re.DOTALL)
    if not qa_pairs:
        # Try a simple A: extraction
        m = re.search(r"A:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            ans = m.group(1).strip()
            return re.sub(r"\s+", " ", ans)
        return re.sub(r"\s+", " ", text.strip())

    # Pick the Q that best matches the query using a tiny similarity heuristic
    q_lower = query.lower().strip()
    best = None
    best_score = -1
    for q_str, a_str in qa_pairs:
        # word overlap score
        q_words = set(re.findall(r"[a-z0-9]+", q_str.lower()))
        query_words = set(re.findall(r"[a-z0-9]+", q_lower))
        overlap = len(q_words & query_words)
        score = overlap / (len(query_words) + 1e-6)
        if score > best_score:
            best_score = score
            best = (q_str.strip(), a_str.strip())

    if best is None:
        # fallback: first pair
        _, a = qa_pairs[0]
        return re.sub(r"\s+", " ", a.strip())
    else:
        _, a = best
        return re.sub(r"\s+", " ", a.strip())

# Chat UI state
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of dicts: {"role":"user"/"bot","text":..., "cites":[(file,section),...]}

with tab_chat:
    st.subheader("Ask about Osteoarthritis & this model (local, cited answers)")

    top_row = st.columns([1,1,3,1])
    with top_row[0]:
        if st.button("🧹 Clear chat"):
            st.session_state.chat = []
    with top_row[1]:
        if st.button("🔄 Rebuild index"):
            load_kb.clear()
            kb = load_kb(_kb_fingerprint())
            st.success("Knowledge index rebuilt.")

    # chat history
    if st.session_state.chat:
        for msg in st.session_state.chat:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-bubble chat-user"><b>You:</b> {msg["text"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble chat-bot"><b>Bot:</b> {msg["text"]}</div>', unsafe_allow_html=True)
                if msg.get("cites"):
                    st.caption("Sources:")
                    for f, sec in msg["cites"]:
                        st.markdown(f'<span class="citation-chip">{f} — {sec}</span>', unsafe_allow_html=True)

    # input row
    q = st.text_input("Your question", key="chat_q", placeholder="e.g., What is osteoarthritis?")
    ask = st.button("Ask")

    if ask and q.strip():
        # display user bubble immediately
        st.session_state.chat.append({"role":"user", "text": q.strip(), "cites":[]})

        # run retrieval
        ret = answer_query(kb, q.strip())

        # make the answer pretty (extract the A: line if present)
        pretty = extract_best_qa(ret.answer, q.strip())

        st.session_state.chat.append({"role":"bot", "text": pretty, "cites": ret.citations})
        st.rerun()  # refresh to show bubbles in order

# =======================
# ABOUT TAB
# =======================

import platform

with tab_about:
    st.header("About this app")

    # --------- Quick facts from runtime ---------
    num_classes = len(CLASS_NAMES)
    params = getattr(model, "count_params", lambda: None)()
    total_params = f"{params:,}" if params is not None else "—"
    merged_note = "Yes (KL 0–1 → Healthy, 2 → Mild, 3–4 → Severe)" if MERGE_INFO.get("merge_to_three") else "No (KL 0–4)"

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Base model", "EfficientNetV2-B0")
    with c2: st.metric("Input size", f"{IMG_SIZE[0]}×{IMG_SIZE[1]}")
    with c3: st.metric("Classes", num_classes)
    with c4: st.metric("Model params", total_params)

    st.markdown("---")

    # --------- What this app does ---------
    st.subheader("What this app does")
    st.markdown(
        """
- **Analyzes knee X-ray images** and predicts **osteoarthritis (OA) severity**.
- Offers **explanations** (Grad-CAM / Integrated Gradients) to visualize model attention.
- Includes a **local, cited chatbot** that answers from your `knowledge/` Markdown files (offline).
- Built for **research / education** — **not** a clinical device. Always involve clinicians.
        """
    )

    # --------- Brief OA intro ---------
    st.subheader("A quick intro to osteoarthritis (OA)")
    st.markdown(
        f"""
- OA is a **degenerative joint disease** with cartilage loss, osteophytes, joint-space narrowing, and subchondral changes.
- **Kellgren–Lawrence (KL) grading**: 0 (none) → 4 (severe).  
- App can run in **5-class mode** or **3-class merged mode** (Healthy / Mild / Severe).  
**Merged classes enabled:** **{merged_note}**
        """
    )

    # --------- Pipeline ---------
    st.subheader("How the model works (end-to-end pipeline)")
    st.markdown(
        f"""
**1) Preprocessing**
- Grayscale → **CLAHE** contrast enhancement → back to RGB  
- Resize to **{IMG_SIZE[0]}×{IMG_SIZE[1]}**, then **EfficientNetV2 preprocessing**

**2) Backbone & head**
- **EfficientNetV2-B0** (transfer learning)  
- Global pooling → Dropout → Dense **softmax** for **{num_classes}** classes

**3) Training recipe**
- Loss: **Categorical Cross-Entropy**  
- Optimizer: **Adam** (+ **ReduceLROnPlateau**)  
- Regularization: **Augmentation** (flip/rotate/zoom/shift), **Dropout**, **EarlyStopping**  
- **Checkpoint** best validation score

**4) Outputs**
- Per-class **probabilities**, **top-1 label**, and **XAI heatmaps** on demand
        """
    )

    # --------- XAI ---------
    st.subheader("Explainability (XAI) used here")
    st.markdown(
        """
- **Grad-CAM**: gradients w.r.t. last conv feature map → **coarse localization** heatmap.  
- **Integrated Gradients (IG)**: integrates pixel gradients from a baseline → **fine-grained saliency**.  
- Heatmaps are overlaid on the X-ray with adjustable **opacity**.
        """
    )

    # --------- Dataset & labels ---------
    st.subheader("Dataset & labels")
    st.markdown(
        """
- Trained on **Knee Osteoarthritis Dataset with Severity** (Kaggle).  
- Labels follow **KL grades 0–4**. Optional **3-class merge**:  
  - Healthy = KL 0–1  
  - Mild = KL 2  
  - Severe = KL 3–4
        """
    )

    # --------- Chatbot / RAG ---------
    st.subheader("Chatbot (local RAG)")
    st.markdown(
        """
- Uses a **lightweight TF-IDF retriever** over `knowledge/*.md`.  
- Returns **verbatim snippets** with **source citations** (filename & section).  
- Low-confidence queries yield **“I don’t know”** to avoid hallucinations.  
- Use the **Rebuild index** button after updating Markdown files.
        """
    )

    # --------- Libraries ---------
    st.subheader("Key libraries & tools")
    st.markdown(
        """
- **TensorFlow / Keras** — model training & inference  
- **OpenCV** — CLAHE & image operations  
- **NumPy** — numerical arrays  
- **Streamlit** — UI for prediction, explanation & chatbot  
- **scikit-learn** — TF-IDF retriever for the chatbot  
- *(optional during training)* **Matplotlib/Seaborn** (plots), **MLflow** (experiment tracking)
        """
    )

    # --------- Safety / limitations ---------
    with st.expander("Safety & limitations"):
        st.markdown(
            """
- Not a medical device; **do not use for diagnosis or treatment** without clinician oversight.
- Performance depends on **image quality** and **dataset distribution**.
- **Domain shift** (different scanners/sites/populations) can reduce accuracy — consider **local fine-tuning** and **external validation**.
- XAI heatmaps are **supportive**, not proof; always interpret in clinical context.
            """
        )

    # --------- Artifacts / versions ---------
    with st.expander("Artifacts & versions"):
        st.markdown(
            f"""
- **Model file**: `model.keras`  
- **Labels**: `class_indices.json` (UI order), optional `label_mapping.json` (3-class map)  
- **Input size**: `{IMG_SIZE[0]}×{IMG_SIZE[1]}`  
- **Classes**: `{", ".join(CLASS_NAMES)}`  
- **Parameters**: `{total_params}`  
- **Explain methods**: Grad-CAM, Integrated Gradients  
- **Knowledge base**: Markdown files in `knowledge/` (use **Rebuild index** after edits)
            """
        )

    # --------- System info ---------
    with st.expander("System info (runtime)"):
        try:
            tf_build = getattr(tf.sysconfig, "get_build_info", lambda: {})()
        except Exception:
            tf_build = {}
        cuda_ver  = tf_build.get("cuda_version", "unknown")
        cudnn_ver = tf_build.get("cudnn_version", "unknown")

        # GPU info (friendly)
        gpus = tf.config.list_physical_devices('GPU')
        gpu_lines = []
        for g in gpus:
            # best-effort friendly name
            name = None
            try:
                details = tf.config.experimental.get_device_details(g)  # may not exist on some TF builds
                name = details.get("device_name") or details.get("compute_capability")
            except Exception:
                pass
            gpu_lines.append(f"- {getattr(g, 'name', 'GPU')} " + (f"({name})" if name else ""))

        # library versions
        try:
            import mlflow  # optional
            mlflow_ver = mlflow.__version__
        except Exception:
            mlflow_ver = "not installed"

        st.markdown(
            f"""
**Python**: `{platform.python_version()}`  
**OS**: `{platform.system()} {platform.release()} ({platform.machine()})`  

**TensorFlow**: `{tf.__version__}`  
**CUDA**: `{cuda_ver}` • **cuDNN**: `{cudnn_ver}`  
**NumPy**: `{np.__version__}` • **OpenCV**: `{cv2.__version__}` • **Streamlit**: `{st.__version__}` • **MLflow**: `{mlflow_ver}`  

**CPU cores**: `{os.cpu_count()}`  
**GPUs detected**: `{len(gpus)}`  
{("\n".join(gpu_lines) if gpu_lines else "No GPU devices reported by TensorFlow.")}
            """
        )

    # --------- Download "About" as Markdown ---------
    about_md = f"""# Knee OA App — About

**Base model**: EfficientNetV2-B0  
**Input size**: {IMG_SIZE[0]}×{IMG_SIZE[1]}  
**Classes**: {", ".join(CLASS_NAMES)}  
**Merged classes enabled**: {merged_note}  
**Parameters**: {total_params}

## Pipeline
1. Preprocess: Grayscale → CLAHE → RGB → Resize {IMG_SIZE[0]}×{IMG_SIZE[1]} → EffNetV2 preprocess  
2. Backbone: EfficientNetV2-B0, pooling + dropout + softmax  
3. Training: Cross-entropy, Adam, ReduceLROnPlateau, Augmentation, EarlyStopping, Checkpoint  
4. Outputs: Probabilities, top-1 label, XAI (Grad-CAM / IG)

## Explainability
- Grad-CAM (coarse localization)  
- Integrated Gradients (pixel-level saliency)

## Dataset & labels
- Knee OA Dataset with Severity (Kaggle)  
- KL 0–4; optional 3-class merge (0–1 Healthy, 2 Mild, 3–4 Severe)

## Chatbot (local RAG)
- TF-IDF over local Markdown (`knowledge/`), cited, low-confidence refusal

## Libraries
- TensorFlow/Keras, OpenCV, NumPy, Streamlit, scikit-learn

## System
- Python: {platform.python_version()}
- OS: {platform.system()} {platform.release()} ({platform.machine()})
- TensorFlow: {tf.__version__} | CUDA: {cuda_ver} | cuDNN: {cudnn_ver}
- NumPy: {np.__version__} | OpenCV: {cv2.__version__} | Streamlit: {st.__version__}
- MLflow: {mlflow_ver}
- CPU cores: {os.cpu_count()}
- GPUs: {len(gpus)}
{("\n".join(gpu_lines) if gpu_lines else "No GPU devices reported by TensorFlow.")}
"""
    st.download_button(
        label="⬇️ Download this About page (Markdown)",
        data=about_md.encode("utf-8"),
        file_name="about_knee_oa_app.md",
        mime="text/markdown"
    )

    st.caption(
        "This app is intended for research and education. Always consult radiology/orthopedic experts for clinical decisions."
    )
