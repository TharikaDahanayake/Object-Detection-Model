#!/usr/bin/env python3
"""
Simple Chess Detection Test
Quick test for chess pieces detection model

This script provides a simple way to test chess pieces detection
without the full enhanced detection system.
"""

import cv2
from ultralytics import YOLO
import os

def test_chess_detection():
    print("♟️ Simple Chess Detection Test")
    print("=" * 40)
    
    # Try to find the trained chess model
    model_paths = [
        "C:/Users/thari/Documents/Nethmi/KDU/6th Semester/AI/runs/detect/exp/weights/best.pt",
        "../../runs/detect/exp/weights/best.pt",
        "runs/detect/chess_detection/weights/best.pt",
        "../runs/detect/chess_detection/weights/best.pt", 
        "../../runs/detect/chess_detection/weights/best.pt",
        "C:/runs/detect/chess_detection/weights/best.pt"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ No trained chess model found!")
        print("💡 Train a model first:")
        print("   cd training_scripts")
        print("   python train_chess_detection.py")
        return
    
    print(f"✅ Loading chess model: {model_path}")
    
    try:
        model = YOLO(model_path)
        print("✅ Chess model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    print("📹 Webcam opened successfully!")
    print("♟️ Point your camera at a chess board")
    print("📋 Press 'q' to quit")
    
    # Chess piece symbols for display
    piece_symbols = {
        'white-king': '♔', 'white-queen': '♕', 'white-bishop': '♗',
        'white-knight': '♘', 'white-rook': '♖', 'white-pawn': '♙',
        'black-king': '♚', 'black-queen': '♛', 'black-bishop': '♝', 
        'black-knight': '♞', 'black-rook': '♜', 'black-pawn': '♟'
    }
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run chess detection
        try:
            results = model(frame, conf=0.3, verbose=False)
            
            chess_count = 0
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy()
                
                # Chess piece classes (should match training order)
                chess_classes = [
                    'white-king', 'white-queen', 'white-bishop', 'white-knight', 'white-rook', 'white-pawn',
                    'black-king', 'black-queen', 'black-bishop', 'black-knight', 'black-rook', 'black-pawn'
                ]
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    if int(cls_id) < len(chess_classes):
                        class_name = chess_classes[int(cls_id)]
                        symbol = piece_symbols.get(class_name, '♟')
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Choose color based on piece color
                        if 'white' in class_name:
                            color = (255, 255, 255)  # White
                            text_color = (0, 0, 0)   # Black text
                        else:
                            color = (64, 64, 64)     # Dark gray
                            text_color = (255, 255, 255)  # White text
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label with symbol
                        label = f"{symbol} {class_name}: {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                        
                        chess_count += 1
                        
                        # Print detection (less frequently)
                        if frame_count % 60 == 0:  # Every 2 seconds
                            print(f"♟️ Detected: {symbol} {class_name} (confidence: {conf:.2f})")
            
            # Add info overlay
            cv2.putText(frame, f"Chess Pieces: {chess_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            cv2.putText(frame, "Detection Error", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show frame
        cv2.imshow('Simple Chess Detection', frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Chess detection test completed!")

if __name__ == "__main__":
    test_chess_detection()