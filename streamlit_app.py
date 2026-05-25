import streamlit as st
import os

# Set page configuration
st.set_page_config(
    page_title="VEHICLE & PLATE DETECTOR | HOME",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Glassmorphic dark mode styling)
st.markdown("""
<style>
    /* Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c1b 0%, #1e1b32 50%, #0f0c1b 100%);
        color: #ffffff;
    }
    
    /* Premium Title Header */
    .header-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 30px;
    }
    .header-box h1 {
        background: linear-gradient(90deg, #FF5733, #FFC300, #33FF57, #33FFF3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 3rem !important;
        margin-bottom: 10px;
    }
    .header-box p {
        color: #b0aec4;
        font-size: 1.2rem;
        margin: 0;
    }
    
    /* Feature Cards */
    .card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        transition: transform 0.3s ease, border 0.3s ease;
        height: 100%;
    }
    .card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(255, 87, 51, 0.4);
        box-shadow: 0 4px 20px 0 rgba(255, 87, 51, 0.15);
    }
    .card-title {
        color: #FF5733;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-text {
        color: #d1cfe2;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Stat Badge */
    .stat-badge {
        background: rgba(51, 255, 87, 0.1);
        border: 1px solid rgba(51, 255, 87, 0.3);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: bold;
        color: #33FF57;
        width: fit-content;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("""
<div class="header-box">
    <h1>Number Plate & Vehicles Detection</h1>
    <p>Multiple Custom Objects Detection Engine powered by YOLOv8 + Dual OCR (EasyOCR & Tesseract)</p>
</div>
""", unsafe_allow_html=True)

# Layout: Two columns (Main info and System Stats/Metadata)
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🌟 Project Abstract")
    st.write(
        "This project implements an end-to-end intelligent transportation and vehicle surveillance system. "
        "Leveraging the state-of-the-art **YOLOv8** Deep Learning model, the system detects and classifies "
        "multiple custom classes: **Cars, Trucks, Buses, Bikes, Persons, and License Plates** in real-time. "
        "Once a license plate is localized, the system crops the region of interest, applies custom digital "
        "image processing filters (CLAHE, deskewing, binarization), and extracts the alphanumeric text "
        "using a comparative integration of **EasyOCR** and **Tesseract OCR**."
    )
    
    st.markdown("### 🛠️ Core Capabilities")
    
    # Feature grid using standard columns
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="card">
            <div class="card-title">🚗 Multiple Custom Object Detection</div>
            <div class="card-text">
                Simultaneous detection and localization of cars, trucks, buses, bikes, pedestrians, and license plates using YOLOv8 transfer learning weights.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <div class="card-title">🔍 Dual OCR Engine Comparison</div>
            <div class="card-text">
                Compares character-level accuracy and processing latency between EasyOCR (deep-learning-based) and Tesseract OCR (heuristic engine).
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div class="card">
            <div class="card-title">🎞️ Multi-Source Processing</div>
            <div class="card-text">
                Full pipeline capability running inference on single uploaded images, pre-recorded video files, and real-time live webcam feeds.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <div class="card-title">📊 Analytics & Performance Metrics</div>
            <div class="card-text">
                Detailed performance breakdowns, including mAP calculations, inference speed charts, confusion matrices, and exportable CSV metadata tables.
            </div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### 📊 System Status & Setup")
    
    # Setup directories automatically
    project_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(project_dir, 'dataset')
    weights_path = os.path.join(project_dir, 'weights', 'best.pt')
    
    # Check dataset
    dataset_exists = os.path.exists(dataset_dir) and len(os.listdir(os.path.join(dataset_dir, 'images', 'train'))) > 0 if os.path.exists(os.path.join(dataset_dir, 'images', 'train')) else False
    model_exists = os.path.exists(weights_path)
    
    # Render badges
    if dataset_exists:
        st.markdown('<div class="stat-badge">✅ CUSTOM DATASET READY</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="stat-badge" style="background: rgba(255, 87, 51, 0.1); border: 1px solid rgba(255, 87, 51, 0.3); color: #FF5733;">⚠️ MOCK DATA FALLBACK RUNNING</div>', unsafe_allow_html=True)
        
    if model_exists:
        st.markdown('<div class="stat-badge">✅ CUSTOM MODEL WEIGHTS READY</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="stat-badge" style="background: rgba(255, 87, 51, 0.1); border: 1px solid rgba(255, 87, 51, 0.3); color: #FF5733;">⚠️ COCO MODEL FALLBACK RUNNING</div>', unsafe_allow_html=True)
    
    st.info(
        "💡 **Developer Recommendation:** Set up your project directory by copying your local dataset folder "
        "into the root as `dataset/` or trigger `train.py` which will automatically handle it!"
    )
    
    # Architecture list
    st.markdown("### 🧬 Pipeline Architecture")
    st.markdown("""
    1. **Data Ingestion**: Image / Video Frame
    2. **Vehicle/Plate Detection**: YOLOv8 model inference
    3. **Localization**: Isolate bounding boxes for "Number plate"
    4. **Image Filtering**: Upscaling + Deskewing + CLAHE
    5. **OCR Recognition**: Simultaneous Tesseract & EasyOCR inference
    6. **Post-Processing**: Alphanumeric cleansing + Regex validation
    7. **UI Dashboard**: Render boxes + OCR results table + analytics graphs
    """)

# Custom Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #b0aec4;'>Number Plate & Vehicles Detection Final Year Project — Created with ❤️ using Streamlit & YOLOv8</p>", unsafe_allow_html=True)
