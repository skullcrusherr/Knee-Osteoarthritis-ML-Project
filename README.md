# 🦴 Knee Osteoarthritis (OA) Severity Classification  
**Deep Learning · Explainable AI · Streamlit App · MLflow · Medical Imaging**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![MLflow](https://img.shields.io/badge/MLflow-enabled-green)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow)

> ⚠️ **Disclaimer:** This project is a *research and educational tool*, **not a medical diagnostic system**.  
> Always seek a licensed radiologist/doctor for diagnosis and treatment.

---

## 📌 Project Summary
This project builds a full ML pipeline to detect **Knee Osteoarthritis severity from X-ray images** using **EfficientNetV2 + Explainable AI**.  
It includes **model training, evaluation, visualization, deployment UI, and a retrieval-based medical chatbot.**

✅ Classifies OA severity (3 or 5 grades)  
✅ Live Grad-CAM & Integrated Gradients visualization  
✅ MLflow experiment tracking & artifact logging  
✅ Streamlit-based clinical-style UI  
✅ Local citation-based OA chatbot (no hallucinations)  
✅ Supports GPU (NVIDIA RTX / CUDA / cuDNN)  
✅ Fully reproducible + MLOps-ready

---

## 🖼️ Demo — App Interface

| Predict Tab | Explainability Tab |
|-------------|-------------------|
| *(add screenshot)* | *(add screenshot)* |

---

## 🏗️ System Architecture
```
Dataset  →  Preprocessing  →  EfficientNetV2 Training  →  Metrics & Plots  →  Streamlit Deployment
   │               │                   │                        │                     │
   │               ├─ CLAHE + resize   │                        │                     │
   │               ├─ Augmentation     ├─ MLflow experiment     │                     │
   │               └─ 5 → 3 class map  │                        └─ GradCAM / IG       │
   │                                                                                  │
   └──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure
```
├── model.py                 # Training pipeline (EffNetV2 + MLflow + CM/ROC plots)
├── app.py                   # Streamlit UI (Predict, Explain, Chatbot, About)
├── explain.py               # Grad-CAM + Integrated Gradients utilities
├── chatbot.py               # Local retrieval-based medical QA system
├── knowledge/               # Markdown-based OA knowledge base for chatbot
├── artifacts/               # Saved plots, metrics, checkpoints (auto-created)
├── class_indices.json       # Exported label mapping
├── label_mapping.json       # 5 → 3 class merge rule
├── requirements.txt         # Python dependencies
└── dataset/ https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity  # train/val/test folders
```

---

## 🧠 Model Details
| Feature | Description |
|---------|-------------|
| Base Model | **EfficientNetV2-B0 (ImageNet pretrained)** |
| Image Preprocessing | CLAHE + normalization |
| Input Size | 256×256 RGB |
| Class Setup | 5-grade or merged 3-grade mapping |
| Optimizers | AdamW (head), AdamW (fine-tune) |
| Training Strategy | Warmup (frozen) → Finetune (unfrozen top N layers) |
| Regularization | Label smoothing, dropout, LR scheduler |
| Explainability | Grad-CAM + Integrated Gradients |
| Metrics Logged | Accuracy, Loss, ROC-AUC, Confusion Matrix |

---

## 🧪 Results (Sample)
| Metric | 3-Class Model |
|--------|--------------|
| Accuracy (Test) | ~92% |
| Macro ROC-AUC | ~0.97 |
| F1 (avg) | ~0.91 |

Confusion matrices & ROC curves are auto-saved under `artifacts/`.

---

## 🚀 Quickstart

### 1️⃣ Create Environment
```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Train Model
```bash
python model.py
```

✔ Saves `model.keras`  
✔ Logs training metrics  
✔ Generates plots inside `artifacts/`  
✔ (Optional) Logs to MLflow UI

### 3️⃣ Run Web App
```bash
streamlit run app.py
```

---

## 🧬 Explainable AI (XAI)
| Method | Purpose |
|--------|---------|
| Grad-CAM | Shows discriminative regions on X-ray |
| Integrated Gradients | Pixel-level explanation via gradients |

---

## 💬 Local Chatbot (Zero Hallucination)
🔹 Uses retrieval (no LLM guesses)  
🔹 Answers only from `knowledge/*.md`  
🔹 Shows source citation  
🔹 Rejects medical advice questions safely  
🔹 Works **offline**  

---

## 📈 MLOps Features
| Feature | Status |
|---------|--------|
| MLflow experiment tracking | ✅ |
| Auto saving CM + ROC plots | ✅ |
| Run reproducibility (seed + config) | ✅ |
| GPU support | ✅ |
| Future add-ons | Docker, ONNX, CI/CD |

---

## 🛠️ Tech Stack
| Category | Tools |
|----------|-------|
| Deep Learning | TensorFlow / Keras |
| Model | EfficientNetV2-B0 |
| Deployment | Streamlit |
| Explainability | Grad-CAM, Integrated Gradients |
| MLOps | MLflow |
| Metrics | sklearn, matplotlib |
| Preprocessing | OpenCV (CLAHE) |

---
