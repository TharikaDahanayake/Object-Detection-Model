#!/usr/bin/env python3
"""
Spectacles Detection Test Script
Test the trained YOLOv8 spectacles detection model

This script tests the trained spectacles detection model on images or webcam.
"""

import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpectaclesDetector:
    """Spectacles detection using trained YOLOv8 model"""
    
    def __init__(self, model_path=None):
        """Initialize the spectacles detector"""
        self.model = None
        self.confidence_threshold = 0.5
        self.class_names = ['spectacles']
        
        # Try to load model
        if model_path:
            self.load_model(model_path)
        else:
            # Try default locations
            self.load_default_model()
    
    def load_model(self, model_path):
        """Load YOLO model from path"""
        try:
            if Path(model_path).exists():
                self.model = YOLO(model_path)
                logger.info(f"✅ Loaded spectacles model: {model_path}")
                return True
            else:
                logger.error(f"❌ Model file not found: {model_path}")
                return False
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False
    
    def load_default_model(self):
        """Try to load model from default locations"""
        project_root = Path(__file__).parent.parent
        
        # Possible model locations
        model_paths = [
            project_root / "models" / "spectacles_best.pt",
            project_root / "runs" / "detect" / "spectacles_detection" / "weights" / "best.pt",
            project_root / "runs" / "detect" / "spectacles_detection2" / "weights" / "best.pt",
            project_root / "runs" / "detect" / "spectacles_detection3" / "weights" / "best.pt",
        ]
        
        for model_path in model_paths:
            if model_path.exists():
                if self.load_model(model_path):
                    return True
        
        logger.warning("⚠️ No trained spectacles model found!")
        logger.info("Please train the model first using train_spectacles_detection.py")
        return False
    
    def detect_spectacles(self, image):
        """Detect spectacles in image"""
        if self.model is None:
            logger.error("No model loaded!")
            return []
        
        try:
            # Run inference
            results = self.model(image, conf=self.confidence_threshold)
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extract box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': self.class_names[class_id] if class_id < len(self.class_names) else 'unknown'
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []
    
    def draw_detections(self, image, detections):
        """Draw detection boxes on image"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background rectangle for text
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Text
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return image

def test_on_image(detector, image_path):
    """Test spectacles detection on a single image"""
    logger.info(f"Testing on image: {image_path}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Could not load image: {image_path}")
        return
    
    # Detect spectacles
    start_time = time.time()
    detections = detector.detect_spectacles(image)
    inference_time = time.time() - start_time
    
    logger.info(f"Found {len(detections)} spectacles (inference: {inference_time:.3f}s)")
    
    # Draw detections
    result_image = detector.draw_detections(image.copy(), detections)
    
    # Display results
    cv2.imshow('Spectacles Detection', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save result
    output_path = Path(image_path).parent / f"spectacles_result_{Path(image_path).name}"
    cv2.imwrite(str(output_path), result_image)
    logger.info(f"Result saved to: {output_path}")

def test_on_webcam(detector):
    """Test spectacles detection on webcam"""
    logger.info("Starting webcam test...")
    logger.info("Press 'q' to quit, 's' to save screenshot")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam")
        return
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    fps_counter = 0
    fps_start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read from webcam")
            break
        
        # Detect spectacles
        detections = detector.detect_spectacles(frame)
        
        # Draw detections
        result_frame = detector.draw_detections(frame, detections)
        
        # Calculate FPS
        fps_counter += 1
        if fps_counter % 30 == 0:
            fps = 30 / (time.time() - fps_start_time)
            fps_start_time = time.time()
            logger.info(f"FPS: {fps:.1f}, Detections: {len(detections)}")
        
        # Add FPS and detection count to frame
        cv2.putText(result_frame, f"Spectacles: {len(detections)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Spectacles Detection - Webcam', result_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            screenshot_path = f"spectacles_screenshot_{frame_count:04d}.jpg"
            cv2.imwrite(screenshot_path, result_frame)
            logger.info(f"Screenshot saved: {screenshot_path}")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Webcam test completed")

def main():
    """Main function"""
    print("=" * 60)
    print("🤓 SPECTACLES DETECTION TEST")
    print("=" * 60)
    
    # Initialize detector
    detector = SpectaclesDetector()
    
    if detector.model is None:
        print("❌ No spectacles detection model found!")
        print("\nTo train a model:")
        print("1. Add spectacles images to datasets/spectacles_detection/train/")
        print("2. Add corresponding label files (.txt)")
        print("3. Run: python train_spectacles_detection.py")
        return
    
    print(f"✅ Loaded spectacles detection model")
    print(f"Confidence threshold: {detector.confidence_threshold}")
    
    # Test options
    while True:
        print("\n" + "-" * 40)
        print("Choose test option:")
        print("1. Test on webcam")
        print("2. Test on image file")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == '1':
            test_on_webcam(detector)
        elif choice == '2':
            image_path = input("Enter image path: ").strip()
            if Path(image_path).exists():
                test_on_image(detector, image_path)
            else:
                print(f"❌ Image not found: {image_path}")
        elif choice == '3':
            break
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main()