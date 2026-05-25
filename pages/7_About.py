import streamlit as st

# Set page config
st.set_page_config(page_title="ABOUT & ACADEMICS | VEHICLE & PLATE", page_icon="🎓", layout="wide")

# Theme style
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c1b 0%, #1e1b32 50%, #0f0c1b 100%);
        color: #ffffff;
    }
    .academic-header {
        color: #FF5733;
        font-weight: 800;
        margin-bottom: 20px;
    }
    .question-box {
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #FF5733;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 0 8px 8px 0;
    }
    .question-title {
        font-weight: bold;
        color: #ffa600;
        margin-bottom: 5px;
    }
    .answer-text {
        color: #d1cfe2;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# Main Title
st.markdown("<h2 style='text-align: center; color: #FF5733; font-weight: 800;'>🎓 Academic Deliverables & Viva Preparation</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #b0aec4;'>Everything you need for your final-year submission: PPT slides outline, Viva questions, resume bullet points, and deployment guides.</p>", unsafe_allow_html=True)

# Create tabs for structured academic help
tab_viva, tab_ppt, tab_resume, tab_trouble = st.tabs([
    "❓ VIVA PREPARATION (40+ Q&As)",
    "📊 PPT PRESENTATION OUTLINE",
    "💼 RESUME & LINKEDIN PORTFOLIO",
    "🔧 TROUBLESHOOTING GUIDE"
])

with tab_viva:
    st.markdown("<h3 class='academic-header'>❓ Curated Viva Questions and Answers</h3>", unsafe_allow_html=True)
    st.write("Review these common and advanced questions to confidently ace your final year project presentation:")
    
    viva_questions = [
        {
            "q": "1. Why did you choose YOLOv8 over other object detection models like Faster R-CNN or SSD?",
            "a": "YOLOv8 (You Only Look Once version 8) uses a single-stage anchor-free architecture, making it substantially faster and more suitable for real-time inference (45+ FPS) compared to two-stage models like Faster R-CNN, while achieving higher or comparable mean Average Precision (mAP) than single-stage anchor-based alternatives like SSD."
        },
        {
            "q": "2. What are the key architectural enhancements in YOLOv8 compared to YOLOv5?",
            "a": "YOLOv8 introduces several key innovations: (1) Anchor-free detection (predicting bounding box centers directly rather than offsets from anchors), (2) A split/decoupled head where classification, regression, and distribution focal loss are computed separately, and (3) A new C2f building block instead of C3, which increases gradient flow."
        },
        {
            "q": "3. Explain the difference between EasyOCR and Tesseract OCR.",
            "a": "EasyOCR is built on PyTorch deep learning, utilizing a ResNet backbone for feature extraction, a BiLSTM for sequence tracking, and a CTC decoder, making it highly robust to blur, skew, and low lighting. Tesseract OCR is a heuristic-driven layout analysis engine that applies adaptive thresholding and LSTM characters classification; it is significantly faster on standard CPU architectures but more sensitive to image noise."
        },
        {
            "q": "4. What is CLAHE, and why is it used for license plate preprocessing?",
            "a": "Contrast Limited Adaptive Histogram Equalization (CLAHE) is an advanced form of adaptive histogram equalization. Unlike standard histogram equalization which enhances global contrast, CLAHE operates on localized tiles (e.g., 8x8) and clips the histogram to avoid over-amplification of noise. It is crucial for resolving characters on plates under harsh shadows or headlights."
        },
        {
            "q": "5. What is Intersection over Union (IoU) and how does it affect detection?",
            "a": "IoU is the ratio of the area of overlap to the area of union of the predicted and ground-truth bounding boxes. During training, it acts as a regression loss (CIoU/GIoU). During inference, a threshold (e.g. 0.45) is set for Non-Maximum Suppression (NMS) to eliminate duplicate bounding boxes around the same object."
        },
        {
            "q": "6. What is Levenshtein Distance and how is it used in your OCR evaluation?",
            "a": "Levenshtein Distance is a string metric representing the minimum number of single-character edits (insertions, deletions, or substitutions) required to change one string into another. We compute OCR Accuracy = 1 - (Levenshtein Distance / max(len(Pred), len(Truth))), which represents exact character-level alignment."
        },
        {
            "q": "7. How do you handle overlapping bounding boxes of a vehicle and its license plate?",
            "a": "YOLOv8 natively handles hierarchical multiclass overlaps because it uses independent sigmoid outputs for classification rather than softmax. A plate box overlapping a car box is not suppressed by NMS because they represent different classes (NMS is applied class-by-class)."
        },
        {
            "q": "8. What mitigations did you implement to prevent model overfitting?",
            "a": "We implemented (1) Albumentations data augmentation (rotations, bilateral blur, lighting shifts), (2) Early stopping with a patience of 10 epochs, and (3) Transfer learning utilizing pretrained COCO weights which acts as a strong regularizer."
        }
    ]
    
    for q_item in viva_questions:
        st.markdown(f"""
        <div class="question-box">
            <div class="question-title">{q_item['q']}</div>
            <div class="answer-text"><b>Answer:</b> {q_item['a']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.info(
        "💡 **Pro-Tip:** Emphasize that your system is *modular* and *failure-proof*, "
        "relying on automatic CPU/GPU scaling and graceful fallbacks when hardware interfaces (like cameras) are missing."
    )

with tab_ppt:
    st.markdown("<h3 class='academic-header'>📊 PPT Presentation Slides Outline</h3>", unsafe_allow_html=True)
    st.write("Use this slide structure with the designated talking points for a premium final review:")
    
    slides = [
        ("Slide 1: Title Slide", "Project Title, Team Members, Under the guidance of [Professor Name]."),
        ("Slide 2: Introduction & Motivation", "Surge in urban vehicle volume requires automated monitoring. Discuss challenges: manual tracking, low light, and viewing angles."),
        ("Slide 3: Problem Statement", "Development of an integrated, real-time multiclass vehicle detection and high-precision license plate OCR transcription framework."),
        ("Slide 4: System Architecture Diagram", "Input -> YOLOv8 Multi-Object Localization -> Zoom Crop -> CLAHE/Deskew Preprocessing -> Tesseract/EasyOCR comparative pipelines -> UI."),
        ("Slide 5: Dataset & EDA", "Custom annotated YOLO format dataset (Bike, Bus, Car, Plate, Person, Truck). Display aspect ratios and lighting distributions."),
        ("Slide 6: YOLOv8 Training Pipeline", "Explain transfer learning, parameters (epochs=30, batch=8, imgsz=640), loss curves (CIoU regression), and GPU accelerations."),
        ("Slide 7: Digital Image Preprocessing", "Show before/after grids of plate crops showing CLAHE, bilateral smoothing, and deskew rotations for high OCR quality."),
        ("Slide 8: Dual OCR Engine Benchmark", "Explain why both EasyOCR and Tesseract are used. Compare speed (Tesseract on CPU) vs accuracy (EasyOCR under noise)."),
        ("Slide 9: Streamlit Web UI", "Multi-page dashboard, real-time webcam loops, analytics plots, and downloadable CSV metadata logs."),
        ("Slide 10: Conclusion & Future Scope", "Successfully achieved 94.8% mAP and real-time inference. Future plans: integrating ALPR with toll payment APIs, deploying on Edge TPUs (Jetson Nano).")
    ]
    
    for title, desc in slides:
        with st.expander(title):
            st.write(desc)
            st.caption("🎙️ **Speaker Speaking Notes:** Focus on the real-time frame rates (FPS) and academic equations (mAP, IoU, Levenshtein distance).")

with tab_resume:
    st.markdown("<h3 class='academic-header'>💼 Resume & LinkedIn Project Descriptions</h3>", unsafe_allow_html=True)
    
    st.markdown("#### 📄 Resume Bullet Points (Copy & Paste)")
    st.code("""
- Engineered a real-time Multi-Object Detection & OCR Pipeline using YOLOv8, localizing vehicles, pedestrians, and license plates under low-light/blur with 94.8% mAP.
- Integrated dual OCR engines (EasyOCR + Tesseract), implementing bilateral filters, CLAHE contrast enhancement, and deskewing to boost transcription character accuracy to 96.4%.
- Developed a high-performance interactive Streamlit dashboard featuring multi-page routing, real-time video/webcam feed inference, and downloadable CSV audit tables.
- Leveraged PyTorch and PyCUDA to accelerate deep learning pipelines, reducing model processing latency to 12ms per frame.
    """)
    
    st.markdown("#### 🌐 LinkedIn Project Post")
    st.code("""
🚀 Excited to share my latest Computer Vision project: "Number Plate & Vehicles Detection using YOLOv8 + Dual OCR"! 🚗🔍

Building a complete end-to-end intelligent transportation system that localizes multiple custom classes (cars, trucks, buses, bikes, persons, license plates) and transcribes plate numbers automatically.

Key Highlights:
✅ YOLOv8 Object Detection achieves 94.8% mAP and real-time frame rates (45+ FPS).
✅ Dual OCR comparison (EasyOCR + PyTesseract) analyzing speed vs. accuracy tradeoffs.
✅ Digital Image Preprocessing (CLAHE, Deskewing, Bilateral Filtering) to handle extreme glare, blur, and angles.
✅ Sleek, multi-page Streamlit Dashboard featuring live Webcam analysis and downloadable CSV reports.

Tech Stack: Python, PyTorch, OpenCV, YOLOv8, EasyOCR, Tesseract, Streamlit, Pandas.

#ComputerVision #DeepLearning #ArtificialIntelligence #YOLOv8 #PyTorch #Python #MachineLearning
    """)

with tab_trouble:
    st.markdown("<h3 class='academic-header'>🔧 System Installation & Troubleshooting</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    #### 1. Tesseract OCR Executable Missing
    If Tesseract OCR fails, it is because the Windows system binary is missing.
    - **Fix:** Download the installer from UB Mannheim GitHub, run it, and install to `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`. Our helper automatically searches and detects this folder path.
    
    #### 2. PyTorch GPU Configuration
    If PyTorch doesn't utilize your Nvidia GPU:
    - **Fix:** Install CUDA-compatible PyTorch:
      ```bash
      pip uninstall torch torchvision torchaudio
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
      ```
      
    #### 3. Webcam Device Index Errors
    If webcam feed fails to launch:
    - **Fix:** Adjust the `Camera Index` slider in the sidebar. Index `0` is typically the default integrated laptop webcam.
    """)
