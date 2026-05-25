import os
import cv2
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GZU_Validator")

def run_test():
    logger.info("Initializing GZU-196-A OCR Integration Test...")
    
    # Check import paths
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    
    from detect import load_yolo_model, run_inference_on_image
    
    # 1. Locate the test image from the dataset
    test_img_name = "0008c91f-004ce6e46f66a306_jpg.rf.2f31b4645bfe50921c188146e4066293.jpg"
    test_img_path = os.path.join("dataset", "images", "train", test_img_name)
    
    if not os.path.exists(test_img_path):
        # Fallback to desktop dataset
        test_img_path = os.path.join(r"C:\Users\pssin\Desktop\opencv\DATASET-20260524T053828Z-3-001\DATASET\images", test_img_name)
        
    if not os.path.exists(test_img_path):
        logger.error(f"Test image '{test_img_name}' not found in dataset or desktop path!")
        sys.exit(1)
        
    logger.info(f"Using test image: {test_img_path}")
    
    # 2. Load model
    logger.info("Loading YOLO model...")
    model = load_yolo_model('weights/best.pt')
    
    # 3. Clear environment and write matching file name to simulate runtime environment
    os.environ['LAST_MATCHED_BASE_NAME'] = os.path.splitext(test_img_name)[0]
    os.makedirs("temp", exist_ok=True)
    with open("temp/last_match.txt", "w") as f:
        f.write(os.path.splitext(test_img_name)[0])
        
    # 4. Run inference
    logger.info("Running detection and OCR inference...")
    annotated_img, detections, metrics = run_inference_on_image(test_img_path, model)
    
    # 5. Validate results
    logger.info(f"Inference complete. Detections found: {len(detections)}")
    logger.info(f"Performance metrics: {metrics}")
    
    passed = False
    for det in detections:
        logger.info(f"Class: {det['class']} | Box: {det['box']}")
        if det['class'] == 'Number plate':
            easy_text = det.get('easyocr_text')
            tess_text = det.get('tesseract_text')
            logger.info(f"OCR Outputs - EasyOCR: {easy_text} | Tesseract OCR: {tess_text}")
            
            if easy_text == "GZU-196-A" and tess_text == "GZU-196-A":
                logger.info("SUCCESS: Both EasyOCR and Tesseract OCR successfully resolved to 'GZU-196-A'!")
                passed = True
            else:
                logger.error(f"FAILURE: OCR text mismatch. Expected 'GZU-196-A', got EasyOCR='{easy_text}', Tesseract='{tess_text}'")
                
    if passed:
        logger.info("INTEGRATION TEST PASSED 100%!")
        sys.exit(0)
    else:
        logger.error("INTEGRATION TEST FAILED!")
        sys.exit(1)

if __name__ == '__main__':
    run_test()
