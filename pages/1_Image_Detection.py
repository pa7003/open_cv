import streamlit as st
import os
import cv2
import pandas as pd
from datetime import datetime
import numpy as np
from PIL import Image

# Import custom helpers
from detect import load_yolo_model, run_inference_on_image
from utils.ocr_helpers import compare_ocr_methods
from utils.preprocessing import preprocess_for_ocr

# Set page config
st.set_page_config(page_title="IMAGE DETECTION | VEHICLE & PLATE", page_icon="🖼️", layout="wide")

# Dark Theme Custom Styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c1b 0%, #1e1b32 50%, #0f0c1b 100%);
        color: #ffffff;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #FF5733;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #b0aec4;
    }
</style>
""", unsafe_allow_html=True)

# Main Title
st.markdown("<h2 style='text-align: center; color: #FF5733; font-weight: 800;'>🖼️ Multi-Object Image Detection & OCR</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #b0aec4;'>Upload an image to detect vehicles, license plates, and extract alphanumeric numbers using EasyOCR & Tesseract OCR side-by-side.</p>", unsafe_allow_html=True)

# Sidebar settings
st.sidebar.header("🔧 Inference Configurations")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.35, 0.05)
iou_threshold = st.sidebar.slider("IoU Overlap Threshold", 0.1, 1.0, 0.45, 0.05)

# Load model
weights_path = 'weights/best.pt'
model = load_yolo_model(weights_path)

# Initialize Session State for Logs
if "detection_logs" not in st.session_state:
    st.session_state["detection_logs"] = []

# File Uploader
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_image_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.success("File uploaded successfully! Starting detection pipeline...")
    
    # Run core YOLOv8 + OCR inference
    annotated_img, detections, metrics = run_inference_on_image(
        temp_image_path, model, 
        conf_threshold=conf_threshold, 
        iou_threshold=iou_threshold
    )
    
    # Columns layout for side-by-side visual output
    col_vis1, col_vis2 = st.columns(2)
    
    with col_vis1:
        st.markdown("### 📤 Original Input Image")
        st.image(uploaded_file, use_container_width=True)
        
    with col_vis2:
        st.markdown("### 🎯 Detection & Bounding Boxes Output")
        if annotated_img is not None:
            # Convert BGR (OpenCV) to RGB for Streamlit image rendering
            rgb_annotated = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            st.image(rgb_annotated, use_container_width=True)
            
    # Metrics breakdown columns
    st.markdown("### 📊 Inference & Detection Analysis")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics.get('inference_time_ms', 0)} ms</div>
            <div class="metric-label">⏱️ YOLO Inference Speed</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics.get('vehicles_count', 0)}</div>
            <div class="metric-label">🚗 Vehicles Detected</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics.get('plates_count', 0)}</div>
            <div class="metric-label">💳 License Plates Detected</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics.get('persons_count', 0)}</div>
            <div class="metric-label">🚶 Pedestrians Detected</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Table of Detections & OCR Results
    if detections:
        st.markdown("### 📑 Detailed Class Metadata & OCR Comparison Table")
        
        # Format detection objects for display
        display_records = []
        for d in detections:
            # Crop image logic to display isolated plate crops in Streamlit
            cropped_plate_html = ""
            box = d["box"]
            
            record = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Class Label": d["class"],
                "Confidence Score": f"{d['confidence']*100:.2f}%",
                "Bounding Box [xmin, ymin, xmax, ymax]": str(box),
                "EasyOCR Text": d.get("easyocr_text", "N/A"),
                "EasyOCR Conf": f"{d.get('easyocr_conf', 0.0)*100:.1f}%" if d.get("easyocr_text") != "N/A" else "N/A",
                "EasyOCR Latency (ms)": f"{d.get('easyocr_lat', 0)} ms" if d.get("easyocr_text") != "N/A" else "N/A",
                "Tesseract OCR Text": d.get("tesseract_text", "N/A"),
                "Tesseract Conf": f"{d.get('tesseract_conf', 0.0)*100:.1f}%" if d.get("tesseract_text") != "N/A" else "N/A",
                "Tesseract Latency (ms)": f"{d.get('tesseract_lat', 0)} ms" if d.get("tesseract_text") != "N/A" else "N/A"
            }
            display_records.append(record)
            
            # Save logs to Session State
            st.session_state["detection_logs"].append(record)
            
        df = pd.DataFrame(display_records)
        st.dataframe(df, use_container_width=True)
        
        # Download log button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Session Detection Logs as CSV",
            data=csv,
            file_name=f"yolo_detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Render isolated Crops if license plates detected
        plate_detections = [d for d in detections if d["class"] in ["Number plate", "Car", "Truck"] and d.get("easyocr_text") != "N/A"]
        
        if plate_detections:
            st.markdown("### 🔠 License Plate Regions & OCR Zoom")
            crop_cols = st.columns(min(len(plate_detections), 4))
            
            raw_cv_img = cv2.imread(temp_image_path)
            for idx, p_det in enumerate(plate_detections):
                col_idx = idx % 4
                with crop_cols[col_idx]:
                    box = p_det["box"]
                    # Calculate bounding box zoom crop
                    crop = raw_cv_img[box[1]:box[3], box[0]:box[2]]
                    if crop.size > 0:
                        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        st.image(rgb_crop, caption=f"Crop #{idx+1} ({p_det['class']})", use_container_width=True)
                        
                        # Apply pre-processing and show
                        preprocessed = preprocess_for_ocr(crop)
                        st.image(preprocessed, caption=f"Preprocessed OCR Input", use_container_width=True)
                        
                        st.markdown(f"""
                        **EasyOCR Text:** `{p_det['easyocr_text']}`  
                        **Tesseract OCR:** `{p_det['tesseract_text']}`
                        """)
    else:
        st.info("No vehicles, persons, or license plates detected at this confidence threshold.")
        
    # Clean up temp image
    try:
        os.remove(temp_image_path)
    except Exception:
        pass
