#!/usr/bin/env python3
"""
Chess Pieces Detection Test Script
Test the trained YOLOv8 chess pieces detection model

This script tests the trained chess pieces detection model on images or webcam.
It can identify all 12 types of chess pieces with confidence scores.
"""

import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import logging
import time
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChessDetector:
    """Chess pieces detection using trained YOLOv8 model"""
    
    def __init__(self, model_path=None):
        """Initialize the chess detector"""
        self.model = None
        self.chess_classes = [
            'white-king', 'white-queen', 'white-bishop', 'white-knight', 'white-rook', 'white-pawn',
            'black-king', 'black-queen', 'black-bishop', 'black-knight', 'black-rook', 'black-pawn'
        ]
        self.piece_symbols = {
            'white-king': '♔', 'white-queen': '♕', 'white-bishop': '♗',
            'white-knight': '♘', 'white-rook': '♖', 'white-pawn': '♙',
            'black-king': '♚', 'black-queen': '♛', 'black-bishop': '♝', 
            'black-knight': '♞', 'black-rook': '♜', 'black-pawn': '♟'
        }
        
        if model_path:
            self.load_model(model_path)
        else:
            self.load_default_model()
    
    def load_model(self, model_path):
        """Load YOLO model from path"""
        try:
            self.model = YOLO(model_path)
            logger.info(f"✅ Chess model loaded from: {model_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model from {model_path}: {e}")
            return False
    
    def load_default_model(self):
        """Try to load model from default locations"""
        default_paths = [
            "runs/detect/chess_detection/weights/best.pt",
            "../runs/detect/chess_detection/weights/best.pt",
            "../../runs/detect/chess_detection/weights/best.pt",
            "C:/runs/detect/chess_detection/weights/best.pt"
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                if self.load_model(path):
                    return True
        
        logger.error("❌ No trained chess model found!")
        logger.info("💡 Train a model first using train_chess_detection.py")
        return False
    
    def detect_chess_pieces(self, image):
        """Detect chess pieces in image"""
        if self.model is None:
            return []
        
        try:
            # Run inference
            results = self.model(image, conf=0.3, verbose=False)
            
            detections = []
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy()
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = box.astype(int)
                    class_name = self.chess_classes[int(cls_id)]
                    symbol = self.piece_symbols.get(class_name, '♟')
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(conf),
                        'class_name': class_name,
                        'symbol': symbol,
                        'class_id': int(cls_id)
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def draw_detections(self, image, detections):
        """Draw detection boxes on image"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            symbol = detection['symbol']
            
            # Choose color based on piece color
            if 'white' in class_name:
                color = (255, 255, 255)  # White
                text_color = (0, 0, 0)   # Black text
            else:
                color = (0, 0, 0)        # Black
                text_color = (255, 255, 255)  # White text
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with symbol
            label = f"{symbol} {class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background rectangle for text
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Text
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        return image

def test_on_image(detector, image_path):
    """Test chess detection on a single image"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return
        
        print(f"🔍 Analyzing image: {image_path}")
        
        # Detect chess pieces
        detections = detector.detect_chess_pieces(image)
        
        if detections:
            print(f"♟️ Found {len(detections)} chess pieces:")
            for i, det in enumerate(detections, 1):
                print(f"  {i}. {det['symbol']} {det['class_name']} (confidence: {det['confidence']:.3f})")
            
            # Draw detections
            result_image = detector.draw_detections(image.copy(), detections)
            
            # Show result
            cv2.imshow('Chess Pieces Detection', result_image)
            print("Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("❌ No chess pieces detected")
            
    except Exception as e:
        print(f"❌ Error testing image: {e}")

def test_on_webcam(detector):
    """Test chess detection on webcam"""
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open webcam")
            return
        
        print("📹 Webcam opened successfully!")
        print("♟️ Point your camera at a chess board")
        print("📋 Press 'q' to quit")
        
        frame_count = 0
        fps_start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect chess pieces
            detections = detector.detect_chess_pieces(frame)
            
            # Draw detections
            result_frame = detector.draw_detections(frame, detections)
            
            # Add info overlay
            cv2.putText(result_frame, f"Chess Pieces: {len(detections)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Calculate and display FPS
            if frame_count % 30 == 0:  # Update FPS every 30 frames
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
            else:
                fps = 0
            
            if fps > 0:
                cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(result_frame, "Press 'q' to quit", (10, result_frame.shape[0]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Chess Pieces Detection', result_frame)
            
            # Print detections to console (less frequent)
            if frame_count % 60 == 0 and detections:  # Every 2 seconds at 30fps
                print(f"♟️ Frame {frame_count}: Found {len(detections)} pieces")
                for det in detections:
                    print(f"   {det['symbol']} {det['class_name']} ({det['confidence']:.2f})")
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("📹 Webcam test completed!")
        
    except Exception as e:
        print(f"❌ Webcam test error: {e}")

def main():
    """Main function"""
    print("♟️ Chess Pieces Detection Test")
    print("=" * 40)
    
    # Create detector
    detector = ChessDetector()
    
    if detector.model is None:
        print("❌ No model loaded. Please train a chess detection model first.")
        return
    
    print("✅ Chess detection model loaded successfully!")
    
    # Test options
    while True:
        print("\n🎯 Choose test option:")
        print("1. 📹 Test with webcam")
        print("2. 🖼️  Test with image file")
        print("3. 🏁 Exit")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == '1':
            test_on_webcam(detector)
        elif choice == '2':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                test_on_image(detector, image_path)
            else:
                print("❌ Image file not found!")
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main()