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
st.set_page_config(page_title="VIDEO DETECTION | VEHICLE & PLATE", page_icon="🎥", layout="wide")

# Custom dark theme styles
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c1b 0%, #1e1b32 50%, #0f0c1b 100%);
        color: #ffffff;
    }
    .video-stat {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        margin-bottom: 10px;
    }
    .video-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #FF5733;
    }
</style>
""", unsafe_allow_html=True)

# Main Title
st.markdown("<h2 style='text-align: center; color: #FF5733; font-weight: 800;'>🎥 Video Object Detection & OCR Pipeline</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #b0aec4;'>Upload a traffic video file to localize vehicles, track pedestrians, and run real-time number plate OCR recognition.</p>", unsafe_allow_html=True)

# Configurations
st.sidebar.header("🔧 Settings")
conf_threshold = st.sidebar.slider("YOLO Confidence", 0.1, 1.0, 0.35, 0.05)
frame_skip = st.sidebar.slider("Frame Skip Rate (1=Every frame, 3=Faster)", 1, 10, 2, 1)

# Load model
weights_path = 'weights/best.pt'
model = load_yolo_model(weights_path)

# File Uploader
uploaded_video = st.file_uploader("Upload video file (MP4, AVI, MOV)", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None:
    # Save video locally
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_video_path = os.path.join(temp_dir, uploaded_video.name)
    
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())
        
    st.success("Video uploaded! Opening frame reader...")
    
    # Open Video
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        st.error("Could not open video file.")
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Display Stats
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f"<div class='video-stat'><div class='video-val'>{width}x{height}</div><div class='metric-label'>Resolution</div></div>", unsafe_allow_html=True)
        with s2:
            st.markdown(f"<div class='video-stat'><div class='video-val'>{total_frames}</div><div class='metric-label'>Total Frames</div></div>", unsafe_allow_html=True)
        with s3:
            st.markdown(f"<div class='video-stat'><div class='video-val'>{fps} FPS</div><div class='metric-label'>Original Frame Rate</div></div>", unsafe_allow_html=True)
        with s4:
            st.markdown(f"<div class='video-stat'><div class='video-val'>{total_frames/fps:.1f}s</div><div class='metric-label'>Duration</div></div>", unsafe_allow_html=True)
            
        # Video Player Area
        col_play, col_meta = st.columns([2, 1])
        
        with col_play:
            st.markdown("### 🎬 Real-time Inference Playback")
            image_placeholder = st.empty() # Placeholder for real-time frame display
            progress_bar = st.progress(0)
            
        with col_meta:
            st.markdown("### 📑 Live Traffic Logs")
            logs_placeholder = st.empty()
            
        # Processing Output Config
        output_name = "annotated_" + uploaded_video.name
        output_path = os.path.join("outputs", "detections", output_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Set up VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'VP80') # VP80 works better in standard web browsers
        # Fallback path change if it's .mp4
        web_friendly_path = output_path.replace(os.path.splitext(output_path)[-1], '.webm')
        out_writer = cv2.VideoWriter(web_friendly_path, fourcc, fps / frame_skip, (width, height))
        
        frame_idx = 0
        all_detections = []
        CLASSES = ["Bike", "Bus", "Car", "Number plate", "Person", "Truck"]
        
        # Stream frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
                
            # YOLO inference
            start_t = time.time()
            results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
            lat = int((time.time() - start_t) * 1000)
            
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
                
                # Model class checking
                if len(model_classes) == 6:
                    cls_name = CLASSES[cls_id]
                else:
                    coco_classes = {0: "Person", 2: "Car", 3: "Bike", 5: "Bus", 7: "Truck"}
                    cls_name = coco_classes.get(cls_id, "Other")
                    if cls_name == "Other":
                        continue
                        
                # Draw boxes
                cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (255, 87, 51), 2)
                
                # License Plate OCR on frame
                ocr_txt = ""
                if cls_name == "Number plate":
                    plate_crop = frame[box[1]:box[3], box[0]:box[2]]
                    if plate_crop.size > 0:
                        prep = preprocess_for_ocr(plate_crop)
                        ocr_txt, _, _ = perform_easyocr(prep)
                        
                label = f"{cls_name} ({conf:.2f})"
                if ocr_txt:
                    label += f" | OCR: {ocr_txt}"
                    
                cv2.putText(
                    annotated_frame, label, (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (51, 255, 87), 1
                )
                
                # Add to logs
                log_entry = {
                    "Timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    "Frame": frame_idx,
                    "Class": cls_name,
                    "Confidence": f"{conf*100:.1f}%",
                    "OCR_Plate": ocr_txt if ocr_txt else "N/A",
                    "Inference_ms": lat
                }
                all_detections.append(log_entry)
                frame_logs.append(log_entry)
                
            # Write annotated frame
            out_writer.write(annotated_frame)
            
            # Update UI frame display
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            image_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            
            # Update Live Logs
            if frame_logs:
                logs_placeholder.dataframe(pd.DataFrame(frame_logs).drop(columns=["Frame"]), use_container_width=True)
                
            # Update progress
            progress_val = min(1.0, frame_idx / total_frames)
            progress_bar.progress(progress_val)
            
            frame_idx += 1
            
        cap.release()
        out_writer.release()
        
        progress_bar.progress(1.0)
        st.success("Video processing completed!")
        
        # Save CSV Logs
        if all_detections:
            df = pd.DataFrame(all_detections)
            csv_log_path = web_friendly_path.replace('.webm', '_log.csv')
            df.to_csv(csv_log_path, index=False)
            
            st.markdown("### 📥 Download Video Detection Outputs")
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                csv_bytes = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Detection Logs (CSV)",
                    data=csv_bytes,
                    file_name=f"video_detections_{uploaded_video.name}.csv",
                    mime="text/csv"
                )
            with d_col2:
                if os.path.exists(web_friendly_path):
                    with open(web_friendly_path, "rb") as vf:
                        v_bytes = vf.read()
                    st.download_button(
                        "📥 Download Annotated WebM Video",
                        data=v_bytes,
                        file_name=f"annotated_{uploaded_video.name.replace('.mp4', '.webm')}",
                        mime="video/webm"
                    )
                    
        # Clean up uploaded temp video
        try:
            os.remove(temp_video_path)
        except Exception:
            pass
