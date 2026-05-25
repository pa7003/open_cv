import streamlit as st
import os
import cv2
import time
import pandas as pd
from PIL import Image

# Import custom helpers
from detect import load_yolo_model
from utils.preprocessing import preprocess_for_ocr, convert_to_grayscale, apply_clahe, sharpen_image, deskew_plate
from utils.ocr_helpers import perform_easyocr, perform_tesseract
from utils.metrics import calculate_ocr_accuracy

# Set page config
st.set_page_config(page_title="OCR COMPARISON | VEHICLE & PLATE", page_icon="🔠", layout="wide")

# Styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c1b 0%, #1e1b32 50%, #0f0c1b 100%);
        color: #ffffff;
    }
    .engine-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        text-align: center;
    }
    .engine-title {
        font-size: 1.4rem;
        font-weight: bold;
        color: #33FFF3;
        margin-bottom: 15px;
    }
    .engine-text {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ffa600, #ff5733);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .engine-stat {
        font-size: 1rem;
        color: #b0aec4;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Main Title
st.markdown("<h2 style='text-align: center; color: #33FFF3; font-weight: 800;'>🔠 EasyOCR vs Tesseract OCR Deep-Dive Comparison</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #b0aec4;'>Upload a vehicle or license plate crop to analyze digital image processing improvements and OCR engine benchmarks side-by-side.</p>", unsafe_allow_html=True)

# Load model
weights_path = 'weights/best.pt'
model = load_yolo_model(weights_path)

# File input
uploaded_file = st.file_uploader("Upload image containing a license plate...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Save temp image
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_image_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    img = cv2.imread(temp_image_path)
    
    # 1. Run YOLO to find number plate boxes
    st.info("Localizing license plate regions via YOLOv8...")
    results = model.predict(img, conf=0.25, verbose=False)[0]
    
    boxes = results.boxes.xyxy.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    model_classes = results.names
    
    # Isolate number plates
    plate_crops = []
    
    for i in range(len(boxes)):
        box = boxes[i].astype(int)
        cls_id = int(class_ids[i])
        
        # Check class
        if len(model_classes) == 6:
            cls_name = model_classes[cls_id]
        else:
            coco_classes = {0: "Person", 2: "Car", 3: "Bike", 5: "Bus", 7: "Truck"}
            cls_name = coco_classes.get(cls_id, "Other")
            
        if cls_name == "Number plate" or cls_name in ["Car", "Truck", "Bus"]:
            crop = img[box[1]:box[3], box[0]:box[2]]
            if crop.size > 0:
                plate_crops.append((crop, cls_name))
                
    # If no plates found, use entire image as crop
    if not plate_crops:
        st.warning("⚠️ YOLOv8 did not localize standard license plate bounding boxes. Using entire uploaded image for OCR analysis.")
        plate_crops.append((img, "Full Image"))
        
    # Select which crop to analyze if multiple found
    crop_idx = 0
    if len(plate_crops) > 1:
        crop_idx = st.selectbox(
            f"Multiple localized objects found ({len(plate_crops)}). Select one for OCR:",
            options=range(len(plate_crops)),
            format_func=lambda x: f"Crop #{x+1} — Label: {plate_crops[x][1]}"
        )
        
    selected_crop, label_type = plate_crops[crop_idx]
    
    # Grid: Pre-processing Visualizer
    st.markdown("### 🛠️ Step 1: Digital Image Preprocessing Stages")
    
    # Perform pre-processing steps
    c_gray = convert_to_grayscale(selected_crop)
    c_deskew = deskew_plate(selected_crop)
    c_clahe = apply_clahe(c_gray, clip_limit=3.0)
    c_preprocess = preprocess_for_ocr(selected_crop)
    
    pv1, pv2, pv3, pv4 = st.columns(4)
    with pv1:
        st.image(cv2.cvtColor(selected_crop, cv2.COLOR_BGR2RGB), caption="1. Raw Localized Crop", use_container_width=True)
    with pv2:
        st.image(c_deskew, caption="2. Rotational Deskewing", use_container_width=True)
    with pv3:
        st.image(c_clahe, caption="3. Grayscale CLAHE Contrast", use_container_width=True)
    with pv4:
        st.image(c_preprocess, caption="4. Final Bilateral Denoise & Sharpen", use_container_width=True)
        
    # Step 2: Running OCR Comparisons
    st.markdown("### 🔠 Step 2: Side-by-Side OCR Engine Benchmark")
    
    # Run OCR Engines
    e_text, e_conf, e_lat = perform_easyocr(c_preprocess)
    t_text, t_conf, t_lat = perform_tesseract(c_preprocess)
    
    # Columns for engines
    col_e, col_t = st.columns(2)
    
    with col_e:
        st.markdown(f"""
        <div class="engine-card">
            <div class="engine-title">🔍 EasyOCR Engine (PyTorch Deep Learning)</div>
            <div class="engine-text">"{e_text or 'EMPTY'}"</div>
            <div class="engine-stat">⏱️ <b>Processing Speed:</b> {e_lat} ms</div>
            <div class="engine-stat">📊 <b>Confidence Level:</b> {e_conf*100:.2f}%</div>
            <div class="engine-stat">🧬 <b>Architecture:</b> ResNet + BiLSTM + CTC</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_t:
        st.markdown(f"""
        <div class="engine-card">
            <div class="engine-title">⚙️ Tesseract OCR Engine (Heuristic Layout Model)</div>
            <div class="engine-text">"{t_text or 'EMPTY'}"</div>
            <div class="engine-stat">⏱️ <b>Processing Speed:</b> {t_lat} ms</div>
            <div class="engine-stat">📊 <b>Confidence Level:</b> {t_conf*100:.2f}%</div>
            <div class="engine-stat">🧬 <b>Architecture:</b> LSTM + Page Layout Heuristics</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Step 3: Match Matrix & Academic Calculations
    st.markdown("### 📊 Step 3: Engine Similarity & Character Analysis")
    
    # Calculate Character Accuracy
    ocr_acc = calculate_ocr_accuracy(e_text, t_text)
    
    sm1, sm2, sm3 = st.columns(3)
    with sm1:
        st.metric("Text Match status", "MATCHING" if e_text == t_text and e_text else "MISMATCH", delta="Exact Match" if e_text == t_text else "Character Difference")
    with sm2:
        st.metric("Alphanumeric Mutual Similarity", f"{ocr_acc*100:.1f}%", help="1 - Levenshtein Distance / max(len1, len2)")
    with sm3:
        fastest = "Tesseract" if t_lat < e_lat else "EasyOCR"
        diff = abs(e_lat - t_lat)
        st.metric("Speed Benchmark Winner", fastest, delta=f"-{diff} ms faster")
        
    # Ground truth validation entry
    st.markdown("### 📝 Ground Truth Validation")
    ground_truth = st.text_input("Enter the actual license plate text manually (for exact accuracy log):").upper().strip()
    
    if ground_truth:
        e_accuracy = calculate_ocr_accuracy(e_text, ground_truth)
        t_accuracy = calculate_ocr_accuracy(t_text, ground_truth)
        
        st.success("Accuracy calculated using Levenshtein distance against Ground Truth!")
        v1, v2 = st.columns(2)
        with v1:
            st.metric("EasyOCR Accuracy", f"{e_accuracy*100:.1f}%")
        with v2:
            st.metric("Tesseract OCR Accuracy", f"{t_accuracy*100:.1f}%")
            
    # Clean up temp image
    try:
        os.remove(temp_image_path)
    except Exception:
        pass
