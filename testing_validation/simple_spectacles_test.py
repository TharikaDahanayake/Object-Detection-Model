#!/usr/bin/env python3
"""
Simple spectacles detection test
"""

import cv2
from ultralytics import YOLO
import os

def test_spectacles():
    print("🔄 Testing spectacles detection...")
    
    # Load spectacles model
    model_path = "../models/spectacles_quick.pt"
    if os.path.exists(model_path):
        model = YOLO(model_path)
        print(f"✅ Spectacles model loaded: {model_path}")
    else:
        print(f"❌ Model not found: {model_path}")
        return
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    print("📹 Webcam started! Press 'q' to quit")
    print("👓 Try wearing glasses/spectacles in front of the camera")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run spectacles detection
        try:
            results = model(frame, conf=0.3, verbose=False)
            
            # Process results
            detections = 0
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Draw detection
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    label = f"Spectacles: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    detections += 1
            
            # Show detection count
            if detections > 0:
                print(f"👓 Frame {frame_count}: Detected {detections} spectacles!")
            
            # Add info to frame
            cv2.putText(frame, f"Spectacles: {detections}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        except Exception as e:
            print(f"Error in detection: {e}")
        
        # Show frame
        cv2.imshow('Spectacles Detection Test', frame)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Spectacles detection test completed!")

if __name__ == "__main__":
    test_spectacles()