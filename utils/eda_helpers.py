import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CLASSES = ["Bike", "Bus", "Car", "Number plate", "Person", "Truck"]

def analyze_dataset_structure(dataset_dir):
    """
    Counts the number of images and annotations in the train and val splits.
    """
    stats = {}
    splits = ['train', 'val']
    for split in splits:
        img_dir = os.path.join(dataset_dir, "images", split)
        lbl_dir = os.path.join(dataset_dir, "labels", split)
        
        imgs = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))] if os.path.exists(img_dir) else []
        lbls = [f for f in os.listdir(lbl_dir) if os.path.isfile(os.path.join(lbl_dir, f))] if os.path.exists(lbl_dir) else []
        
        stats[f"{split}_images"] = len(imgs)
        stats[f"{split}_labels"] = len(lbls)
    
    return stats

def get_annotations_dataframe(dataset_dir, split='train'):
    """
    Parses all YOLO label files and builds a pandas DataFrame with:
    - filename
    - class_id
    - class_name
    - x_center, y_center, width, height
    - bbox_area (width * height)
    - aspect_ratio (width / height)
    """
    records = []
    lbl_dir = os.path.join(dataset_dir, "labels", split)
    img_dir = os.path.join(dataset_dir, "images", split)
    
    if not os.path.exists(lbl_dir):
        return pd.DataFrame()
        
    for lbl_file in os.listdir(lbl_dir):
        if not lbl_file.endswith('.txt'):
            continue
            
        lbl_path = os.path.join(lbl_dir, lbl_file)
        img_name = lbl_file.replace('.txt', '.jpg')
        # Check standard extensions if .jpg not found
        if not os.path.exists(os.path.join(img_dir, img_name)):
            img_name = lbl_file.replace('.txt', '.png')
        if not os.path.exists(os.path.join(img_dir, img_name)):
            img_name = lbl_file.replace('.txt', '.jpeg')
            
        try:
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_c, y_c, w, h = [float(x) for x in parts[1:]]
                    
                    records.append({
                        "filename": img_name,
                        "class_id": class_id,
                        "class_name": CLASSES[class_id] if class_id < len(CLASSES) else f"Unknown ({class_id})",
                        "x_center": x_c,
                        "y_center": y_c,
                        "width": w,
                        "height": h,
                        "area": w * h,
                        "aspect_ratio": w / (h + 1e-6)
                    })
        except Exception:
            continue
            
    return pd.DataFrame(records)

def analyze_image_properties(dataset_dir, split='train'):
    """
    Computes brightness and blur metrics for the images.
    - Brightness: mean of grayscale pixels
    - Blur: variance of Laplacian
    """
    records = []
    img_dir = os.path.join(dataset_dir, "images", split)
    
    if not os.path.exists(img_dir):
        return pd.DataFrame()
        
    for img_file in os.listdir(img_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
            
        img_path = os.path.join(img_dir, img_file)
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            h, w, c = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Brightness
            brightness = float(np.mean(gray))
            
            # Blur (Laplacian variance)
            blur_metric = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            
            records.append({
                "filename": img_file,
                "width": w,
                "height": h,
                "channels": c,
                "brightness": brightness,
                "blur_score": blur_metric,
                "lighting_type": "Low Light" if brightness < 80 else ("Overexposed" if brightness > 200 else "Standard")
            })
        except Exception:
            continue
            
    return pd.DataFrame(records)

# Plotting Functions for the EDA dashboard
def plot_class_distribution(df, save_path=None):
    """Generates class frequency plot."""
    if df.empty:
        return None
    plt.figure(figsize=(10, 5))
    sns.set_theme(style="darkgrid")
    
    # Custom vibrant palette
    colors = ["#FF5733", "#33FF57", "#3357FF", "#F3FF33", "#FF33F3", "#33FFF3"]
    sns.countplot(data=df, x="class_name", order=CLASSES, palette=colors[:len(CLASSES)])
    
    plt.title("Class Frequency Distribution", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Object Class", fontsize=12)
    plt.ylabel("Occurrences Count", fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    return plt.gcf()

def plot_bbox_size_distribution(df, save_path=None):
    """Generates scatter plot of width vs height to show aspect ratios."""
    if df.empty:
        return None
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    
    sns.scatterplot(
        data=df, x="width", y="height", hue="class_name",
        alpha=0.7, palette="Set2", style="class_name"
    )
    
    # Draw reference aspect ratio line (1:1)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', alpha=0.5, label='1:1 Aspect')
    
    plt.title("Bounding Box Dimensions (Normalized Width vs Height)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Normalized Width", fontsize=12)
    plt.ylabel("Normalized Height", fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    return plt.gcf()

def plot_image_lighting_and_blur(df, save_path=None):
    """Generates lighting/brightness & blur analysis plot."""
    if df.empty:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.set_theme(style="darkgrid")
    
    # Left: Brightness Distribution
    sns.histplot(data=df, x="brightness", kde=True, ax=axes[0], color="#ffa600")
    axes[0].axvline(80, color='red', linestyle='--', alpha=0.7, label='Low Light Threshold')
    axes[0].axvline(200, color='blue', linestyle='--', alpha=0.7, label='Overexposed Threshold')
    axes[0].set_title("Image Average Brightness Analysis", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Mean Grayscale Value (0-255)")
    axes[0].legend()
    
    # Right: Blur Score Distribution
    sns.histplot(data=df, x="blur_score", kde=True, ax=axes[1], color="#bc5090")
    axes[1].axvline(100, color='red', linestyle='--', alpha=0.7, label='Blurry Threshold')
    axes[1].set_title("Image Sharpness (Laplacian Variance)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Blur Score (Higher = Sharper)")
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    return fig
