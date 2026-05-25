import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import local helpers
from utils.eda_helpers import get_annotations_dataframe, analyze_image_properties, plot_class_distribution, plot_bbox_size_distribution, plot_image_lighting_and_blur

# Set page config
st.set_page_config(page_title="ANALYTICS DASHBOARD | VEHICLE & PLATE", page_icon="📊", layout="wide")

# Theme style
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c1b 0%, #1e1b32 50%, #0f0c1b 100%);
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Main Title
st.markdown("<h2 style='text-align: center; color: #FFC300; font-weight: 800;'>📊 Traffic Analytics & Dataset Insights</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #b0aec4;'>Detailed statistical analysis of custom classes distributions, bounding box aspects, image lighting levels, and blur metrics.</p>", unsafe_allow_html=True)

# Dataset path
dataset_dir = "dataset"
dataset_exists = os.path.exists(dataset_dir) and os.path.exists(os.path.join(dataset_dir, "labels", "train"))

# Control
st.sidebar.header("🔧 Settings")
split_choice = st.sidebar.selectbox("Select Split to Analyze", ["train", "val"])

# Let's check session logs for live logs
live_logs_exist = "detection_logs" in st.session_state and len(st.session_state["detection_logs"]) > 0

if dataset_exists:
    st.sidebar.success("📊 Loading real dataset metrics...")
    df_annotations = get_annotations_dataframe(dataset_dir, split=split_choice)
    df_images = analyze_image_properties(dataset_dir, split=split_choice)
else:
    st.sidebar.warning("⚠️ Using high-fidelity synthetic/simulation database for preview.")
    
    # Generate mock annotations database
    np.random.seed(42)
    mock_records = []
    classes = ["Bike", "Bus", "Car", "Number plate", "Person", "Truck"]
    
    for i in range(150):
        cls_id = np.random.choice(range(len(classes)), p=[0.1, 0.05, 0.45, 0.20, 0.12, 0.08])
        w = np.random.uniform(0.1, 0.6)
        h = np.random.uniform(0.1, 0.5) if classes[cls_id] != "Number plate" else np.random.uniform(0.05, 0.15)
        
        mock_records.append({
            "filename": f"sim_{i:04d}.jpg",
            "class_id": cls_id,
            "class_name": classes[cls_id],
            "width": w,
            "height": h,
            "area": w * h,
            "aspect_ratio": w / (h + 1e-6)
        })
    df_annotations = pd.DataFrame(mock_records)
    
    # Generate mock image properties
    mock_imgs = []
    for i in range(50):
        brightness = np.random.normal(120, 35)
        blur = np.random.exponential(150) + 15
        mock_imgs.append({
            "filename": f"sim_{i:04d}.jpg",
            "width": 640,
            "height": 480,
            "brightness": np.clip(brightness, 10, 255),
            "blur_score": blur,
            "lighting_type": "Low Light" if brightness < 80 else ("Overexposed" if brightness > 200 else "Standard")
        })
    df_images = pd.DataFrame(mock_imgs)

# --- Visual Render grid ---
grid_col1, grid_col2 = st.columns(2)

with grid_col1:
    st.markdown("### 📊 Class Distribution (Balance Analysis)")
    fig_class = plot_class_distribution(df_annotations)
    if fig_class:
        st.pyplot(fig_class)
        
    st.markdown("### 💡 Object Density Insights")
    # Quick insights
    if not df_annotations.empty:
        counts = df_annotations["class_name"].value_counts()
        most_common = counts.index[0]
        st.info(
            f"📈 **Statistical Peak:** The most common class detected is **{most_common}** "
            f"with {counts[most_common]} individual labels in this database split. "
            f"Number plates occupy {counts.get('Number plate', 0)} total instances."
        )

with grid_col2:
    st.markdown("### 📐 Bounding Box Aspect Ratio (Anchor Boxes)")
    fig_bbox = plot_bbox_size_distribution(df_annotations)
    if fig_bbox:
        st.pyplot(fig_bbox)
        
    st.markdown("### 📐 Aspect Ratio Analysis")
    if not df_annotations.empty:
        avg_ar_plate = df_annotations[df_annotations["class_name"] == "Number plate"]["aspect_ratio"].mean()
        st.info(
            f"📐 **License Plate Aspect Ratio:** Localized license plates show an average "
            f"aspect ratio of **{avg_ar_plate:.2f} : 1.0** (typically horizontal layout), "
            f"confirming optimal anchor box ratios for YOLO scale adjustments."
        )

# Image Quality Grid (Lighting & Blur)
st.markdown("---")
st.markdown("### 🌤️ Image Quality & Environment Analysis")
fig_quality = plot_image_lighting_and_blur(df_images)
if fig_quality:
    st.pyplot(fig_quality)
    
# Render details about blur/low light stats
iq_col1, iq_col2 = st.columns(2)
with iq_col1:
    low_light_pct = (df_images["brightness"] < 80).mean() * 100
    over_pct = (df_images["brightness"] > 200).mean() * 100
    st.metric("Low Light Frame Rate", f"{low_light_pct:.1f}%", help="Percentage of images with mean brightness < 80")
    st.write("📊 Grayscale average values demonstrate the operational resilience of the model in varying ambient light conditions.")
with iq_col2:
    blur_pct = (df_images["blur_score"] < 100).mean() * 100
    st.metric("Blurry Frame Rate", f"{blur_pct:.1f}%", help="Percentage of images with Laplacian variance < 100")
    st.write("📐 Sharpness scores indicate camera stability and movement speeds. Higher blur values prompt bilateral filter smoothing prior to OCR.")

# Session Live statistics if they exist
if live_logs_exist:
    st.markdown("---")
    st.markdown("### 💻 Live Session Inference Timings")
    df_live = pd.DataFrame(st.session_state["detection_logs"])
    
    # Calculate averages
    latencies = pd.to_numeric(df_live["EasyOCR Latency (ms)"].str.replace(" ms", "", regex=False), errors="coerce")
    avg_inf = latencies.mean() if not latencies.dropna().empty else 0.0
    st.write(df_live.head(10))
