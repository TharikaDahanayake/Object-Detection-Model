#!/usr/bin/env python3
"""
Interactive Book Detection Tuning
Allows you to adjust confidence threshold in real-time to find optimal settings
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def load_book_model():
    """Load YOLOv8 book detection model"""
    model_paths = [
        "C:/runs/detect/book_detection3/weights/best.pt",
        "C:/runs/detect/book_detection2/weights/best.pt",
        "C:/runs/detect/book_detection/weights/best.pt"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = YOLO(path)
                print(f"✅ YOLOv8 book model loaded: {path}")
                return model
            except Exception as e:
                print(f"❌ Error loading {path}: {e}")
                continue
    
    print("❌ No book detection model found!")
    return None

def detect_books_with_validation(frame, book_model, confidence_threshold):
    """Detect books with size and aspect ratio validation"""
    if book_model is None:
        return []
    
    try:
        results = book_model(frame, conf=confidence_threshold, verbose=False)
        
        detections = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = box.astype(int)
                
                # Size validation
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = width / height if height > 0 else 0
                
                # Validation criteria
                valid_aspect = 0.3 <= aspect_ratio <= 3.0
                valid_size = (width > 30 and height > 30 and 
                             width < frame.shape[1] * 0.8 and 
                             height < frame.shape[0] * 0.8)
                
                status = "✅" if (valid_aspect and valid_size) else "❌"
                
                detections.append({
                    'class': 'book',
                    'confidence': conf,
                    'box': (x1, y1, x2, y2),
                    'color': (0, 255, 0) if (valid_aspect and valid_size) else (0, 0, 255),
                    'valid': valid_aspect and valid_size,
                    'width': width,
                    'height': height,
                    'aspect_ratio': aspect_ratio,
                    'status': status
                })
        
        return detections
    except Exception as e:
        print(f"Error in book detection: {e}")
        return []

def draw_enhanced_detections(frame, detections, confidence_threshold):
    """Draw detections with detailed information"""
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        conf = detection['confidence']
        color = detection['color']
        status = detection['status']
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare detailed label
        label = f"{status} book: {conf:.2f}"
        details = f"AR: {detection['aspect_ratio']:.2f}, {detection['width']}x{detection['height']}"
        
        # Draw labels
        y_offset = y1 - 35 if y1 - 35 > 35 else y1 + 50
        cv2.rectangle(frame, (x1, y_offset - 25), (x1 + 250, y_offset + 5), color, -1)
        cv2.putText(frame, label, (x1 + 5, y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, details, (x1 + 5, y_offset + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def main():
    """Main tuning interface"""
    print("🔧 Book Detection Tuning Tool")
    print("📚 Green boxes = Valid book detections")
    print("🔴 Red boxes = Invalid detections (filtered out)")
    print("🎛️ Controls:")
    print("   'q' - Quit")
    print("   '+' - Increase confidence threshold")
    print("   '-' - Decrease confidence threshold")
    print("   's' - Save current threshold")
    print("=" * 50)
    
    # Load model
    book_model = load_book_model()
    if book_model is None:
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    confidence_threshold = 0.4
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect books
        detections = detect_books_with_validation(frame, book_model, confidence_threshold)
        
        # Filter valid detections
        valid_detections = [d for d in detections if d['valid']]
        invalid_detections = [d for d in detections if not d['valid']]
        
        # Draw all detections
        draw_enhanced_detections(frame, detections, confidence_threshold)
        
        # Add info overlay
        info_text = [
            f"Confidence Threshold: {confidence_threshold:.2f}",
            f"Total Detections: {len(detections)}",
            f"Valid Books: {len(valid_detections)}",
            f"Invalid (Filtered): {len(invalid_detections)}",
            f"Frame: {frame_count}"
        ]
        
        for i, text in enumerate(info_text):
            y_pos = 30 + i * 25
            cv2.rectangle(frame, (10, y_pos - 20), (400, y_pos + 5), (0, 0, 0), -1)
            cv2.putText(frame, text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Book Detection Tuning', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            confidence_threshold = min(0.95, confidence_threshold + 0.05)
            print(f"🔼 Confidence threshold increased to {confidence_threshold:.2f}")
        elif key == ord('-'):
            confidence_threshold = max(0.1, confidence_threshold - 0.05)
            print(f"🔽 Confidence threshold decreased to {confidence_threshold:.2f}")
        elif key == ord('s'):
            print(f"💾 Current optimal threshold: {confidence_threshold:.2f}")
            print(f"   Valid detections: {len(valid_detections)}")
            print(f"   Invalid detections: {len(invalid_detections)}")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"🎯 Final recommended confidence threshold: {confidence_threshold:.2f}")

if __name__ == "__main__":
    main()