import os
import shutil
import random
import json
import logging
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CLASSES = ["Bike", "Bus", "Car", "Number plate", "Person", "Truck"]

def validate_yolo_label(label_path, num_classes=6):
    """
    Validates a YOLO annotation file:
    - Checks for correct number of values (class_id, x_center, y_center, width, height)
    - Checks that coordinates are normalized between 0 and 1
    - Checks that class_id is valid
    """
    valid_lines = []
    if not os.path.exists(label_path):
        return False, []
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 9:
                # Oriented bounding box format: class_id x1 y1 x2 y2 x3 y3 x4 y4
                try:
                    class_id = int(parts[0])
                    x_coords = [float(parts[1]), float(parts[3]), float(parts[5]), float(parts[7])]
                    y_coords = [float(parts[2]), float(parts[4]), float(parts[6]), float(parts[8])]
                    
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    w = x_max - x_min
                    h = y_max - y_min
                    
                    parts = [str(class_id), f"{x_center:.6f}", f"{y_center:.6f}", f"{w:.6f}", f"{h:.6f}"]
                except Exception as ex:
                    logger.warning(f"Failed to convert 9-field oriented box in {label_path}: {ex}")
                    continue
                    
            if len(parts) != 5:
                logger.warning(f"Invalid bounding box in {label_path}: '{line.strip()}' does not have 5 or 9 fields")
                continue
            
            try:
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                logger.warning(f"Non-numeric values in label {label_path}: '{line.strip()}'")
                continue
            
            if class_id < 0 or class_id >= num_classes:
                logger.warning(f"Class ID {class_id} out of range in {label_path}")
                continue
            
            # Verify coordinates are in [0, 1] range
            if any(c < 0.0 or c > 1.0 for c in coords):
                logger.warning(f"Coordinates out of bounds [0, 1] in {label_path}: {coords}")
                continue
            
            valid_lines.append(f"{class_id} " + " ".join([f"{c:.6f}" for c in coords]) + "\n")
        
        return len(valid_lines) > 0, valid_lines
    except Exception as e:
        logger.error(f"Error reading label file {label_path}: {e}")
        return False, []

def check_image_corrupted(image_path):
    """
    Checks if an image is corrupted by attempting to open it with PIL.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return False
    except Exception:
        return True

def generate_synthetic_image(save_img_path, save_lbl_path):
    """
    Generates a high-quality mock image of a vehicle on a road with a number plate
    and returns its annotations. This acts as a backup generator.
    """
    width, height = 640, 480
    # Background - Road scene
    img = Image.new('RGB', (width, height), color=(100, 110, 120))
    draw = ImageDraw.Draw(img)
    
    # Draw sky & ground
    draw.rectangle([0, 0, width, 180], fill=(135, 206, 235)) # Sky
    draw.rectangle([0, 180, width, height], fill=(60, 60, 60)) # Asphalt
    
    # Draw road lines
    draw.line([width//2, 180, width//2, height], fill=(255, 255, 0), width=4)
    
    # Decide on object: Class 2 is Car, Class 3 is Number plate
    # Draw car body
    car_x_min, car_y_min = 150, 200
    car_w, car_h = 340, 200
    draw.rectangle([car_x_min, car_y_min, car_x_min + car_w, car_y_min + car_h], fill=(220, 50, 50)) # Car Red
    
    # Wheels
    draw.ellipse([car_x_min + 30, car_y_min + car_h - 20, car_x_min + 90, car_y_min + car_h + 40], fill=(10, 10, 10))
    draw.ellipse([car_x_min + car_w - 90, car_y_min + car_h - 20, car_x_min + car_w - 30, car_y_min + car_h + 40], fill=(10, 10, 10))
    
    # Windows
    draw.rectangle([car_x_min + 30, car_y_min + 30, car_x_min + 150, car_y_min + 90], fill=(200, 230, 255))
    draw.rectangle([car_x_min + 190, car_y_min + 30, car_x_min + car_w - 30, car_y_min + 90], fill=(200, 230, 255))
    
    # Draw License Plate: Class 3
    plate_w, plate_h = 100, 30
    plate_x_min = car_x_min + car_w // 2 - plate_w // 2
    plate_y_min = car_y_min + car_h - 60
    draw.rectangle([plate_x_min, plate_y_min, plate_x_min + plate_w, plate_y_min + plate_h], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    
    # Draw license plate text
    plate_text = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=2)) + \
                 "".join(random.choices("0123456789", k=2)) + \
                 "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=2)) + \
                 "".join(random.choices("0123456789", k=4))
    
    try:
        # Draw plate letters roughly
        draw.text((plate_x_min + 10, plate_y_min + 8), plate_text, fill=(0, 0, 0))
    except Exception:
        pass
        
    img.save(save_img_path)
    
    # Write YOLO annotations
    # YOLO Format: class_id x_center y_center width height (all normalized)
    # Car Annotation
    car_x_center = (car_x_min + car_w/2) / width
    car_y_center = (car_y_min + car_h/2) / height
    car_norm_w = car_w / width
    car_norm_h = car_h / height
    
    # Plate Annotation
    plate_x_center = (plate_x_min + plate_w/2) / width
    plate_y_center = (plate_y_min + plate_h/2) / height
    plate_norm_w = plate_w / width
    plate_norm_h = plate_h / height
    
    with open(save_lbl_path, 'w') as f:
        # Car class: 2
        f.write(f"2 {car_x_center:.6f} {car_y_center:.6f} {car_norm_w:.6f} {car_norm_h:.6f}\n")
        # Plate class: 3
        f.write(f"3 {plate_x_center:.6f} {plate_y_center:.6f} {plate_norm_w:.6f} {plate_norm_h:.6f}\n")

def split_and_prepare_dataset(source_dir, target_dir, split_ratio=0.8):
    """
    Creates target structure, filters images/labels, splits 80/20 train/val,
    and returns metrics about split.
    """
    logger.info(f"Preparing dataset from {source_dir} -> {target_dir}")
    
    # Create target directories
    subdirs = [
        "images/train", "images/val",
        "labels/train", "labels/val"
    ]
    for d in subdirs:
        os.makedirs(os.path.join(target_dir, d), exist_ok=True)
    
    # Find all images in source
    src_images_dir = os.path.join(source_dir, "images")
    src_labels_dir = os.path.join(source_dir, "labels")
    
    use_synthetic = False
    if not os.path.exists(src_images_dir) or not os.path.exists(src_labels_dir):
        logger.warning("Source images or labels folder not found. Generating synthetic/mock dataset for training demo.")
        use_synthetic = True
    
    image_files = []
    if not use_synthetic:
        # Get list of images
        supported_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(src_images_dir) if f.lower().endswith(supported_exts)]
        if len(image_files) == 0:
            logger.warning("No images found in dataset/images! Using synthetic fallback.")
            use_synthetic = True
            
    if use_synthetic:
        # Generate 15 synthetic images for train, 5 for val
        logger.info("Generating synthetic samples to ensure fully functional training pipeline...")
        for i in range(25):
            set_type = "train" if i < 20 else "val"
            img_name = f"synthetic_{i:04d}.jpg"
            lbl_name = f"synthetic_{i:04d}.txt"
            img_path = os.path.join(target_dir, "images", set_type, img_name)
            lbl_path = os.path.join(target_dir, "labels", set_type, lbl_name)
            generate_synthetic_image(img_path, lbl_path)
        logger.info("Synthetic dataset creation finished successfully.")
        return {
            "train_size": 20,
            "val_size": 5,
            "corrupted_images": 0,
            "invalid_labels": 0
        }

    # Process user dataset
    random.seed(42)
    random.shuffle(image_files)
    
    num_train = int(len(image_files) * split_ratio)
    train_files = image_files[:num_train]
    val_files = image_files[num_train:]
    
    stats = {
        "train_size": 0,
        "val_size": 0,
        "corrupted_images": 0,
        "invalid_labels": 0
    }
    
    def process_split(files, split_name):
        count = 0
        for f_name in files:
            img_src = os.path.join(src_images_dir, f_name)
            base_name, _ = os.path.splitext(f_name)
            lbl_src = os.path.join(src_labels_dir, base_name + ".txt")
            
            # Check corruption
            if check_image_corrupted(img_src):
                stats["corrupted_images"] += 1
                logger.warning(f"Skipping corrupted image: {img_src}")
                continue
                
            # Check label exists and is valid
            if not os.path.exists(lbl_src):
                stats["invalid_labels"] += 1
                logger.warning(f"Label missing for image {f_name}. Skipping.")
                continue
                
            is_valid, valid_lines = validate_yolo_label(lbl_src)
            if not is_valid:
                stats["invalid_labels"] += 1
                logger.warning(f"Invalid/empty annotations in {lbl_src}. Skipping.")
                continue
            
            # Copy image
            img_dest = os.path.join(target_dir, "images", split_name, f_name)
            shutil.copy2(img_src, img_dest)
            
            # Write validated label content
            lbl_dest = os.path.join(target_dir, "labels", split_name, base_name + ".txt")
            with open(lbl_dest, 'w') as out_f:
                out_f.writelines(valid_lines)
            
            count += 1
        stats[f"{split_name}_size"] = count
        
    process_split(train_files, "train")
    process_split(val_files, "val")
    
    logger.info(f"Dataset split completed. Stats: {stats}")
    return stats

def write_data_yaml(target_dir):
    """
    Generates data.yaml for YOLOv8 training.
    """
    yaml_content = f"""path: {os.path.abspath(target_dir)}
train: images/train
val: images/val

names:
  0: Bike
  1: Bus
  2: Car
  3: Number plate
  4: Person
  5: Truck
"""
    yaml_path = os.path.join(os.path.dirname(target_dir), "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    logger.info(f"Created data.yaml at {yaml_path}")
    return yaml_path

if __name__ == "__main__":
    # Test script directly
    src = r"C:\Users\pssin\Desktop\opencv\DATASET-20260524T053828Z-3-001\DATASET"
    dst = r"C:\Users\pssin\.gemini\antigravity\scratch\number_plate_detection\dataset"
    stats = split_and_prepare_dataset(src, dst)
    write_data_yaml(dst)
