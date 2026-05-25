import streamlit as st
import os
import cv2
import time
import pandas as pd
from datetime import datetime
from PIL import Image

# Import custom helpers
from detect import load_yolo_model
from utils.preprocessing import preprocess_for_ocr
from utils.ocr_helpers import perform_easyocr

# Set page config
st.set_page_config(page_title="WEBCAM DETECTION | VEHICLE & PLATE", page_icon="📹", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c1b 0%, #1e1b32 50%, #0f0c1b 100%);
        color: #ffffff;
    }
    .webcam-status {
        background: rgba(51, 255, 87, 0.1);
        border: 1px solid rgba(51, 255, 87, 0.3);
        border-radius: 8px;
        padding: 10px;
        color: #33FF57;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Main Title
st.markdown("<h2 style='text-align: center; color: #FF5733; font-weight: 800;'>📹 Real-time Webcam Inference Engine</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #b0aec4;'>Trigger your local system camera to run real-time YOLOv8 object localization and license plate OCR recognition.</p>", unsafe_allow_html=True)

# Load model
weights_path = 'weights/best.pt'
model = load_yolo_model(weights_path)

# Sidebar configurations
st.sidebar.header("🔧 Camera Settings")
camera_idx = st.sidebar.number_input("Select Camera Index", min_value=0, max_value=5, value=0, step=1)
conf_threshold = st.sidebar.slider("Webcam Confidence", 0.1, 1.0, 0.35, 0.05)

# Control button
st.markdown("### ⏯️ Camera Controls")
run_cam = st.checkbox("💻 START CAMERA FEED")

if run_cam:
    st.markdown("<div class='webcam-status'>🔴 SYSTEM CAMERA ACTIVE — INFERENCE RUNNING</div>", unsafe_allow_html=True)
    
    # Placeholders
    col_feed, col_log = st.columns([2, 1])
    with col_feed:
        frame_placeholder = st.empty()
    with col_log:
        st.markdown("##### 📝 Session Detections")
        log_placeholder = st.empty()
        
    # Open camera stream
    cap = cv2.VideoCapture(camera_idx)
    
    # Fallback simulation flag if camera won't open
    simulation_mode = False
    if not cap.isOpened():
        st.warning("⚠️ Local camera hardware not detected. Switching to System Simulation Mode...")
        simulation_mode = True
        
        # Gather images from dataset
        val_dir = os.path.join("dataset", "images", "val")
        sim_images = []
        if os.path.exists(val_dir):
            sim_images = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
        if not sim_images:
            st.error("No sample images found in validation split to simulate. Make sure to generate dataset first!")
            run_cam = False
            
    # Active frame loop
    frame_count = 0
    fps_start = time.time()
    session_logs = []
    CLASSES = ["Bike", "Bus", "Car", "Number plate", "Person", "Truck"]
    
    try:
        while run_cam:
            if simulation_mode:
                # Loop through mock images to simulate camera
                img_path = sim_images[frame_count % len(sim_images)]
                frame = cv2.imread(img_path)
                time.sleep(0.8) # Simulate processing delay
            else:
                ret, frame = cap.read()
                if not ret:
                    st.error("Error reading from webcam device.")
                    break
            
            # YOLO predict
            start_t = time.time()
            results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
            latency = int((time.time() - start_t) * 1000)
            
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            model_classes = results.names
            
            annotated_frame = frame.copy()
            frame_logs = []
            
            for i in range(len(boxes)):
                box = boxes[i].astype(int)
                conf = float(confs[i])
                cls_id = int(class_ids[i])
                
                if len(model_classes) == 6:
                    cls_name = CLASSES[cls_id]
                else:
                    coco_classes = {0: "Person", 2: "Car", 3: "Bike", 5: "Bus", 7: "Truck"}
                    cls_name = coco_classes.get(cls_id, "Other")
                    if cls_name == "Other":
                        continue
                        
                # Draw Box
                cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (51, 255, 87), 2)
                
                # Run OCR on plates
                ocr_txt = ""
                if cls_name == "Number plate":
                    plate_crop = frame[box[1]:box[3], box[0]:box[2]]
                    if plate_crop.size > 0:
                        prep = preprocess_for_ocr(plate_crop)
                        ocr_txt, _, _ = perform_easyocr(prep)
                
                label = f"{cls_name} ({conf:.2f})"
                if ocr_txt:
                    label += f" | {ocr_txt}"
                    
                cv2.putText(
                    annotated_frame, label, (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 87, 51), 1
                )
                
                log_entry = {
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Class": cls_name,
                    "Confidence": f"{conf*100:.1f}%",
                    "OCR": ocr_txt if ocr_txt else "N/A"
                }
                session_logs.append(log_entry)
                frame_logs.append(log_entry)
                
            # Compute real FPS
            fps = 1.0 / (time.time() - start_t)
            cv2.putText(
                annotated_frame, f"FPS: {fps:.1f} | Latency: {latency} ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (33, 255, 243), 2
            )
            
            # Display frame
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            
            # Display logs
            if frame_logs:
                log_placeholder.dataframe(pd.DataFrame(frame_logs), use_container_width=True)
                
            frame_count += 1
            
            # Check standard streamlit toggle change to end loop
            # Streamlit rerun mechanism handles updates when user untoggles start camera
            
    except Exception as e:
        logger.error(f"Webcam execution error: {e}")
    finally:
        if not simulation_mode:
            cap.release()
else:
    st.info("💡 Check the checkbox above to activate your system camera stream!")
