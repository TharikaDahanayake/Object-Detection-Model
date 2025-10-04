#!/usr/bin/env python3
"""
Enhanced Real-Time Object Detection
All 4 models: MobileNet + Books + Spectacles + Cellphones
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

# MobileNet classes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def enhanced_detection():
    print("� Enhanced Detection - All 4 Models")
    print("📱 MobileNet + 📚 Books + 👓 Spectacles + 📱 Cellphones")
    
    # Load all models
    print("�🔄 Loading models...")
    
    # Load MobileNet
    net = None
    try:
        if os.path.exists("../models/MobileNetSSD_deploy.prototxt.txt"):
            net = cv2.dnn.readNetFromCaffe("../models/MobileNetSSD_deploy.prototxt.txt", 
                                         "../models/MobileNetSSD_deploy.caffemodel")
            print("✅ MobileNet loaded")
    except:
        print("⚠️ MobileNet not found")
    
    # Load book model
    book_model = None
    if os.path.exists("C:/runs/detect/book_detection3/weights/best.pt"):
        book_model = YOLO("C:/runs/detect/book_detection3/weights/best.pt")
        print("✅ Book model loaded")
    
    # Load spectacles model
    spectacles_model = None
    model_path = "../models/spectacles_quick.pt"
    if os.path.exists(model_path):
        spectacles_model = YOLO(model_path)
        print(f"✅ Spectacles model loaded: {model_path}")
    else:
        print(f"❌ Spectacles model not found: {model_path}")
        return
    
    # Load general YOLO for cellphones
    general_model = None
    gen_paths = ["../models/yolov8n.pt", "models/yolov8n.pt", "yolov8n.pt"]
    for path in gen_paths:
        if os.path.exists(path):
            general_model = YOLO(path)
            print(f"✅ General YOLO loaded: {path}")
            break
    
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    print("📹 Webcam started! Press 'q' to quit")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        all_detections = []
        
        # MobileNet detection
        if net is not None:
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.2:
                    idx = int(detections[0, 0, i, 1])
                    if idx < len(CLASSES):
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x1, y1, x2, y2) = box.astype("int")
                        all_detections.append({
                            'class': CLASSES[idx],
                            'confidence': confidence,
                            'box': (x1, y1, x2, y2),
                            'color': COLORS[idx]
                        })
        
        # Book detection
        if book_model is not None:
            try:
                results = book_model(frame, conf=0.3, verbose=False)
                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    for box, conf in zip(boxes, confidences):
                        x1, y1, x2, y2 = box.astype(int)
                        all_detections.append({
                            'class': 'book',
                            'confidence': conf,
                            'box': (x1, y1, x2, y2),
                            'color': (0, 255, 0)
                        })
                        print(f"📚 Book detected!")
            except:
                pass
        
        # Spectacles detection
        if spectacles_model is not None:
            try:
                results = spectacles_model(frame, conf=0.3, verbose=False)
                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    for box, conf in zip(boxes, confidences):
                        x1, y1, x2, y2 = box.astype(int)
                        all_detections.append({
                            'class': 'spectacles',
                            'confidence': conf,
                            'box': (x1, y1, x2, y2),
                            'color': (255, 0, 255)
                        })
                        print(f"👓 Spectacles detected!")
            except:
                pass
        
        # Cellphone detection
        if general_model is not None:
            try:
                results = general_model(frame, conf=0.3, verbose=False)
                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy()
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        class_name = results[0].names[int(cls_id)]
                        if class_name == 'cell phone':
                            x1, y1, x2, y2 = box.astype(int)
                            all_detections.append({
                                'class': 'cellphone',
                                'confidence': conf,
                                'box': (x1, y1, x2, y2),
                                'color': (0, 255, 255)
                            })
                            print(f"� Cellphone detected!")
            except:
                pass
        
        # Draw all detections
        for det in all_detections:
            x1, y1, x2, y2 = det['box']
            label = f"{det['class']}: {det['confidence']:.2f}"
            color = det['color']
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add info
        cv2.putText(frame, f"Objects: {len(all_detections)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0]-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Enhanced Detection', frame)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Enhanced detection completed!")

if __name__ == "__main__":
    enhanced_detection()