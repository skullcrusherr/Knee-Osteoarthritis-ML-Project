# AI Model & App FAQ for Knee OA Prediction

Q: What does this AI model do?  
A: It analyzes knee X-ray images to predict the severity of osteoarthritis based on visual patterns similar to a radiologist’s grading.

Q: What grading system does the model follow?  
A: The model approximates the Kellgren–Lawrence (KL) grading system from 0 to 4.

Q: What do grades 0 to 4 mean?  
A: 0 – Normal, 1 – Doubtful, 2 – Mild, 3 – Moderate, 4 – Severe osteoarthritis.

Q: What dataset was the model trained on?  
A: The publicly available **Knee Osteoarthritis Dataset with Severity** from Kaggle, containing labeled knee X-rays.

Q: Does the model detect other diseases?  
A: No, it’s trained only for osteoarthritis grading, not for fractures, tumors, or infections.

Q: Can the AI replace a doctor?  
A: No. It’s a decision-support tool meant to assist, not replace, clinical judgment.

Q: How accurate is the model?  
A: Depending on architecture and preprocessing, validation accuracy is typically between 80 % and 90 %.

Q: What architectures work best?  
A: Transfer-learning backbones such as EfficientNetB3 or InceptionV3 generally perform well.

Q: What is transfer learning?  
A: It’s using a model pre-trained on a large dataset (like ImageNet) and fine-tuning it for a specific medical task.

Q: How were the images preprocessed?  
A: They were resized, contrast-enhanced using CLAHE, normalized, and sometimes augmented for rotation, zoom, and brightness.

Q: What is CLAHE?  
A: Contrast Limited Adaptive Histogram Equalization — a technique to improve visibility in X-ray images.

Q: What kind of input does the model need?  
A: A clear anterior–posterior or lateral knee X-ray in JPG or PNG format.

Q: How does the model output predictions?  
A: It returns probabilities for each grade (0–4); the highest probability indicates the predicted class.

Q: Why is the prediction sometimes uncertain?  
A: The image may be unclear, underexposed, or the case may be borderline between grades.

Q: What does the confidence percentage mean?  
A: It’s the model’s internal probability that an image belongs to a certain class — not a guarantee.

Q: How is the AI model evaluated?  
A: Using metrics like accuracy, precision, recall, F1-score, and confusion matrix.

Q: What is a confusion matrix?  
A: A table comparing predicted versus actual labels to show where the model performs well or poorly.

Q: How does Grad-CAM help explain predictions?  
A: It visualizes which regions of the X-ray most influenced the AI’s decision.

Q: What does Integrated Gradients show?  
A: It highlights pixel-level contributions to the model’s output, showing subtle decision areas.

Q: Why are explainability tools important?  
A: They build user trust and help verify that the model focuses on clinically relevant features.

Q: Does the model learn patient identity?  
A: No, it learns only patterns from image pixels; personal data isn’t used.

Q: How is patient privacy protected?  
A: The dataset is anonymized and contains no personal identifiers.

Q: What causes model bias?  
A: Unequal representation of age groups, genders, or imaging devices in the training set.

Q: Can model bias affect predictions?  
A: Yes — imbalance may lead to over- or under-prediction for certain patient groups.

Q: How can bias be reduced?  
A: By using balanced datasets, data augmentation, and fairness evaluation.

Q: What optimizer is used in training?  
A: Typically Adam with learning-rate scheduling (e.g., ReduceLROnPlateau).

Q: What is the loss function used?  
A: Categorical cross-entropy, since the task involves multi-class classification.

Q: How many classes are there?  
A: Five (KL 0–4) or sometimes three merged classes (Healthy, Mild, Severe) for simplicity.

Q: Why merge classes?  
A: To simplify prediction and improve balance when extreme classes are under-represented.

Q: What does dropout do in a neural network?  
A: It randomly disables neurons during training to prevent overfitting.

Q: What is data augmentation?  
A: Randomly altering training images (rotating, flipping, zooming) to make the model generalize better.

Q: Why does the app use EfficientNet or Inception?  
A: They are well-optimized CNNs that achieve strong accuracy with fewer parameters.

Q: What is fine-tuning?  
A: Unfreezing some of the pre-trained layers and retraining them on your dataset.

Q: What hardware is required for training?  
A: Ideally an NVIDIA GPU with CUDA and cuDNN installed; CPU training is slower.

Q: Why might training stop early?  
A: EarlyStopping callback halts when validation accuracy no longer improves to prevent overfitting.

Q: What is ReduceLROnPlateau?  
A: A learning-rate scheduler that lowers the learning rate when progress stalls.

Q: What file format is the model saved in?  
A: The native Keras `.keras` or legacy `.h5` format.

Q: What library is used for deployment?  
A: TensorFlow/Keras integrated into a Streamlit web interface.

Q: Can the app run offline?  
A: Yes, once dependencies and model files are installed locally.

Q: What is Streamlit?  
A: A Python framework that turns scripts into interactive web apps easily.

Q: What does the Explain tab do?  
A: It generates Grad-CAM or Integrated-Gradient heatmaps to visualize AI reasoning.

Q: Why might Grad-CAM fail?  
A: If the model’s last convolution layer name isn’t found or uses non-CNN layers.

Q: Can you re-train the model on hospital data?  
A: Yes, with consent and anonymization, fine-tuning on local data improves performance.

Q: How much data is needed to re-train?  
A: Typically several thousand labeled images per class for reliable generalization.

Q: What if the model predicts 0 for everything?  
A: It likely overfits or is mis-calibrated; re-check training balance and normalization.

Q: Why are confidence scores low?  
A: The image may differ from training distribution (lighting, resolution, view).

Q: What are model calibration metrics?  
A: Expected Calibration Error (ECE) measures how well predicted probabilities match reality.

Q: What is explainable AI (XAI)?  
A: AI systems designed to make their decision processes understandable to humans.

Q: What are limitations of AI diagnosis?  
A: It can misclassify rare cases, artifacts, or diseases outside its training scope.

Q: Can AI help triage patients?  
A: Yes, it can prioritize cases for review, improving efficiency.

Q: How does the chatbot work?  
A: It retrieves curated Q&A pairs from markdown knowledge files and returns relevant answers.

Q: Where are these knowledge files stored?  
A: In the `knowledge/` folder, each markdown file grouped by topic.

Q: How does retrieval differ from generation?  
A: Retrieval pulls pre-written answers; generation creates new text (risking errors).

Q: Why use retrieval over generation here?  
A: It ensures factual, consistent, and safe responses.

Q: Can the chatbot explain medical images?  
A: It can describe AI outputs but not interpret new medical findings.

Q: What’s the purpose of model explainability in healthcare?  
A: To ensure transparency, accountability, and clinician trust.

Q: What does “softmax” mean?  
A: A mathematical function that converts raw model scores into probabilities.

Q: Why normalize images before prediction?  
A: To match the training data’s scale and improve consistency.

Q: What happens if an image is overexposed?  
A: Contrast adjustments like CLAHE can improve prediction quality.

Q: What’s the advantage of using Keras over PyTorch here?  
A: Keras integrates smoothly with Streamlit and TensorFlow Serving for deployment.

Q: How do callbacks help training?  
A: They monitor metrics and adjust learning parameters automatically.

Q: What is model checkpointing?  
A: Saving the best version of the model during training to avoid performance loss.

Q: Can we combine AI predictions from multiple models?  
A: Yes, ensemble methods often improve reliability.

Q: What is an ensemble model?  
A: A combination of several models whose outputs are averaged or voted for better performance.

Q: What is class imbalance?  
A: When some severity grades have many more images than others.

Q: How is imbalance handled?  
A: Through data augmentation, class weighting, or resampling.

Q: What’s the danger of overfitting?  
A: The model performs well on training data but poorly on unseen data.

Q: How do we detect overfitting?  
A: Validation accuracy stops improving while training accuracy keeps rising.

Q: What is cross-validation?  
A: Splitting data into multiple folds to test model stability.

Q: Why visualize training curves?  
A: To check learning stability and convergence behavior.

Q: What does “model precision” mean?  
A: The proportion of positive predictions that are correct.

Q: What is recall?  
A: The proportion of actual positives correctly identified.

Q: What is the F1-score?  
A: The harmonic mean of precision and recall, balancing both.

Q: Can you integrate this model with hospital PACS systems?  
A: Yes, via DICOM API wrappers and local deployment, with compliance checks.

Q: What legal or ethical issues exist?  
A: Data privacy, informed consent, fairness, and accountability.

Q: How often should the model be re-validated?  
A: At least annually, or when new imaging protocols are introduced.

Q: What’s the future of AI in radiology?  
A: Collaborative AI that assists clinicians, not replaces them, for faster and safer care.

Q: Can this app export reports?  
A: Yes, the prediction results can be exported as PDFs or logs for records.

Q: Can AI help teach medical students?  
A: Definitely — visualization tools and AI-assisted grading improve learning.

Q: What happens if I upload a non-knee image?  
A: The model’s output will be unreliable since it wasn’t trained on other anatomy.

Q: Why does the model use 299 × 299 or 224 × 224 input size?  
A: Those are the required input dimensions for Inception and EfficientNet models.

Q: What libraries are essential for running the app?  
A: TensorFlow, Pillow, Streamlit, NumPy, and OpenCV.

Q: How can results be improved further?  
A: Collect more diverse data, use transfer learning from medical datasets, and fine-tune longer.

Q: What does “explain tab” in the app show?  
A: Heatmaps (Grad-CAM / Integrated Gradients) showing which areas the AI looked at.

Q: How should users interpret heatmaps?  
A: Bright red/yellow areas show regions most influencing the prediction.

Q: What is transparency in AI?  
A: The ability for users to understand why a model gave a certain result.

Q: Can AI make mistakes on high-quality images?  
A: Yes — even perfect images can be misclassified if features overlap across classes.

Q: Why is clinical validation needed?  
A: To confirm the model performs well on real-world hospital data, not just public datasets.

Q: Who should review AI results before diagnosis?  
A: Always a qualified radiologist or orthopedic specialist.

Q: Can the chatbot learn new questions automatically?  
A: Not yet; new Q&A must be added manually to the markdown files.

Q: What is the long-term goal of this project?  
A: To make knee-OA screening faster, more accessible, and explainable for clinicians and students.
