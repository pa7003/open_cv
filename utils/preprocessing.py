import cv2
import numpy as np
from PIL import Image, ImageEnhance

def convert_to_grayscale(image):
    """Converts image to grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Applies Contrast Limited Adaptive Histogram Equalization."""
    gray = convert_to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)

def denoise_image(image, method='bilateral'):
    """Applies bilateral or Gaussian denoising to keep edges sharp."""
    if method == 'bilateral':
        # bilateralFilter preserves edges well which is useful for OCR
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    elif method == 'gaussian':
        return cv2.GaussianBlur(image, (5, 5), 0)
    return image

def sharpen_image(image):
    """Sharpens the image using an unsharp mask kernel."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def apply_thresholding(image, block_size=11, c=2):
    """Applies adaptive thresholding for high contrast binarization."""
    gray = convert_to_grayscale(image)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, c
    )

def deskew_plate(image):
    """
    Deskews an image (corrects rotation/tilt) of a license plate
    to improve OCR accuracy.
    """
    gray = convert_to_grayscale(image)
    # Binarize
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Grab coordinates of non-zero pixels
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return image
        
    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust the angle for minAreaRect output
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # Rotate if tilt is significant
    if abs(angle) > 1.0 and abs(angle) < 45.0:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    return image

def preprocess_for_ocr(image):
    """
    Combines preprocessing steps to optimize license plate crops for EasyOCR / Tesseract.
    """
    # 1. Resize/Upscale (OCR performs better on higher resolution text)
    h, w = image.shape[:2]
    upscaled = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    
    # 2. Deskew
    deskewed = deskew_plate(upscaled)
    
    # 3. Grayscale
    gray = convert_to_grayscale(deskewed)
    
    # 4. Enhance contrast using CLAHE
    contrast = apply_clahe(gray, clip_limit=3.0)
    
    # 5. Denoise and sharpen
    denoised = denoise_image(contrast, method='bilateral')
    sharpened = sharpen_image(denoised)
    
    return sharpened

# Augmentations for Training/Simulation
def augment_brightness(image, factor=1.2):
    """Adjusts the brightness of the image."""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(pil_img)
    enhanced = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)

def augment_blur(image, ksize=(5, 5)):
    """Applies simple Gaussian blur to the image."""
    return cv2.GaussianBlur(image, ksize, 0)

def augment_noise(image):
    """Injects Gaussian random noise into the image."""
    row, col, ch = image.shape
    mean = 0
    var = 100
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def augment_flip(image, flip_code=1):
    """Flips image horizontally (1), vertically (0), or both (-1)."""
    return cv2.flip(image, flip_code)
