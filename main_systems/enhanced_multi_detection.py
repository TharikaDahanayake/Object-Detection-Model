#!/usr/bin/env python3
"""
Enhanced Multi-Model Object Detection System
5 AI Models: MobileNet + Books + Spectacles + General Objects + Chess Pieces
Features: Smart deduplication, high accuracy, real-time detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from collections import defaultdict

# MobileNet classes (21 classes)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    # Calculate intersection
    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def smart_deduplicate(detections):
    """Remove overlapping detections with smart priority system"""
    if not detections:
        return []
    
    # Priority mapping for conflicting classes
    class_priority = {
        'cell phone': 10,      # Highest priority
        'cellphone': 10,
        'person': 9,
        'book': 8,
        'chair': 7,
        'cat': 6,
        'bottle': 3,           # Lower priority (often misclassified)
        'pottedplant': 2,      # Lower priority (often misclassified)
        'tvmonitor': 5
    }
    
    # First pass: Remove obvious conflicts (cell phone vs bottle/pottedplant)
    filtered_detections = []
    cell_phone_boxes = []
    
    # Find all cell phone detections first
    for det in detections:
        if det['class'] in ['cell phone', 'cellphone']:
            cell_phone_boxes.append(det['box'])
            filtered_detections.append(det)
    
    # Filter out bottle/pottedplant that overlap with cell phones
    for det in detections:
        if det['class'] in ['bottle', 'pottedplant']:
            # Check if this overlaps with any cell phone
            overlaps_with_phone = False
            for phone_box in cell_phone_boxes:
                iou = calculate_iou(det['box'], phone_box)
                if iou > 0.3:  # 30% overlap
                    overlaps_with_phone = True
                    break
            
            if not overlaps_with_phone:
                filtered_detections.append(det)
        elif det['class'] not in ['cell phone', 'cellphone']:
            filtered_detections.append(det)
    
    # Second pass: Standard deduplication
    final_detections = []
    filtered_detections.sort(key=lambda x: class_priority.get(x['class'], 1), reverse=True)
    
    for current_det in filtered_detections:
        # Check if this detection overlaps significantly with any kept detection
        overlap_found = False
        for kept_det in final_detections:
            iou = calculate_iou(current_det['box'], kept_det['box'])
            if iou > 0.4:  # 40% overlap threshold
                overlap_found = True
                break
        
        if not overlap_found:
            final_detections.append(current_det)
    
    return final_detections

def enhanced_multi_detection():
    print("🎯 Enhanced Multi-Model Detection System")
    print("🤖 5 AI Models: MobileNet + Books + Spectacles + General Objects + Chess Pieces")
    print("✨ Features: Smart Deduplication + High Accuracy + Real-time Processing")
    
    # Load all models
    print("🔄 Loading models...")
    
    # Load MobileNet
    net = None
    try:
        mobilenet_paths = [
            "../../models/MobileNetSSD_deploy.prototxt.txt",
            "../models/MobileNetSSD_deploy.prototxt.txt",
            "../../models/MobileNetSSD_deploy.prototxt.txt"
        ]
        
        for base_path in mobilenet_paths:
            prototxt_path = base_path
            model_path = base_path.replace("prototxt.txt", "caffemodel")
            if os.path.exists(prototxt_path) and os.path.exists(model_path):
                net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                print("✅ MobileNet loaded")
                break
        
        if net is None:
            print("⚠️ MobileNet files not found")
    except Exception as e:
        print(f"⚠️ MobileNet loading failed: {e}")
    
    # Load book model
    book_model = None
    if os.path.exists("C:/runs/detect/book_detection3/weights/best.pt"):
        book_model = YOLO("C:/runs/detect/book_detection3/weights/best.pt")
        print("✅ Book model loaded")
    
    # Load spectacles model
    spectacles_model = None
    spectacles_paths = [
        "../../models/spectacles_quick.pt",
        "../models/spectacles_quick.pt",
        "../../models/spectacles_quick.pt"
    ]
    for path in spectacles_paths:
        if os.path.exists(path):
            spectacles_model = YOLO(path)
            print(f"✅ Spectacles model loaded: {path}")
            break
    
    # Load general YOLO model
    general_model = None
    yolo_paths = [
        "../../models/yolov8n.pt",
        "../models/yolov8n.pt",
        "models/yolov8n.pt"
    ]
    for path in yolo_paths:
        if os.path.exists(path):
            general_model = YOLO(path)
            print(f"✅ General YOLO loaded: {path}")
            break
    
    if general_model is None:
        print("🔄 Downloading YOLOv8...")
        general_model = YOLO('yolov8n.pt')  # Auto-download
        print("✅ General YOLO downloaded and loaded")
    
    # Load chess model
    chess_model = None
    chess_path = "C:/Users/thari/Documents/Nethmi/KDU/6th Semester/AI/runs/detect/exp/weights/best.pt"
    if os.path.exists(chess_path):
        chess_model = YOLO(chess_path)
        print(f"✅ Chess model loaded: {chess_path}")
    else:
        print("⚠️ Chess model not found")
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    print("📹 Webcam started! Press 'q' to quit")
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            frame_count += 1
            all_detections = []
            
            # MobileNet detection
            if net is not None:
                try:
                    (h, w) = frame.shape[:2]
                    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
                    net.setInput(blob)
                    detections = net.forward()
                    
                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > 0.6:  # Higher threshold for better accuracy
                            idx = int(detections[0, 0, i, 1])
                            if idx < len(CLASSES):
                                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                                (x1, y1, x2, y2) = box.astype("int")
                                class_name = CLASSES[idx]
                                
                                # Filter out commonly misclassified objects with low confidence
                                if class_name in ['bottle', 'pottedplant'] and confidence < 0.8:
                                    continue  # Skip these if not very confident
                                
                                all_detections.append({
                                    'class': class_name,
                                    'confidence': confidence,
                                    'box': (x1, y1, x2, y2),
                                    'color': tuple(map(int, COLORS[idx])),
                                    'source': 'MobileNet'
                                })
                except Exception as e:
                    pass
            
            # Book detection
            if book_model is not None:
                try:
                    results = book_model(frame, conf=0.4, verbose=False)
                    if len(results) > 0 and len(results[0].boxes) > 0:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        confidences = results[0].boxes.conf.cpu().numpy()
                        for box, conf in zip(boxes, confidences):
                            x1, y1, x2, y2 = box.astype(int)
                            all_detections.append({
                                'class': 'book',
                                'confidence': conf,
                                'box': (x1, y1, x2, y2),
                                'color': (0, 255, 0),
                                'source': 'Book_Model'
                            })
                except:
                    pass
            
            # Spectacles detection
            if spectacles_model is not None:
                try:
                    results = spectacles_model(frame, conf=0.4, verbose=False)
                    if len(results) > 0 and len(results[0].boxes) > 0:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        confidences = results[0].boxes.conf.cpu().numpy()
                        for box, conf in zip(boxes, confidences):
                            x1, y1, x2, y2 = box.astype(int)
                            all_detections.append({
                                'class': 'spectacles',
                                'confidence': conf,
                                'box': (x1, y1, x2, y2),
                                'color': (255, 0, 255),
                                'source': 'Spectacles_Model'
                            })
                except:
                    pass
            
            # General YOLO detection
            if general_model is not None:
                try:
                    results = general_model(frame, conf=0.4, verbose=False)
                    if len(results) > 0 and len(results[0].boxes) > 0:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        confidences = results[0].boxes.conf.cpu().numpy()
                        class_ids = results[0].boxes.cls.cpu().numpy()
                        for box, conf, cls_id in zip(boxes, confidences, class_ids):
                            class_name = results[0].names[int(cls_id)]
                            x1, y1, x2, y2 = box.astype(int)
                            
                            # Normalize class names
                            if class_name in ['cell phone']:
                                display_name = 'cellphone'
                                color = (0, 255, 255)
                            else:
                                display_name = class_name
                                color = (255, 0, 0) if class_name == 'person' else (0, 255, 0)
                            
                            all_detections.append({
                                'class': display_name,
                                'confidence': conf,
                                'box': (x1, y1, x2, y2),
                                'color': color,
                                'source': 'YOLO'
                            })
                except Exception as e:
                    pass
            
            # Chess pieces detection
            if chess_model is not None:
                try:
                    results = chess_model(frame, conf=0.4, verbose=False)
                    if len(results) > 0 and len(results[0].boxes) > 0:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        confidences = results[0].boxes.conf.cpu().numpy()
                        class_ids = results[0].boxes.cls.cpu().numpy()
                        
                        chess_classes = [
                            'white-king', 'white-queen', 'white-bishop', 'white-knight', 'white-rook', 'white-pawn',
                            'black-king', 'black-queen', 'black-bishop', 'black-knight', 'black-rook', 'black-pawn'
                        ]
                        
                        for box, conf, cls_id in zip(boxes, confidences, class_ids):
                            if int(cls_id) < len(chess_classes):
                                class_name = chess_classes[int(cls_id)]
                                x1, y1, x2, y2 = box.astype(int)
                                
                                # Better colors for chess pieces
                                if 'white' in class_name:
                                    color = (255, 255, 0)    # Bright cyan for white pieces
                                else:
                                    color = (255, 0, 255)    # Bright magenta for black pieces
                                
                                all_detections.append({
                                    'class': class_name,
                                    'confidence': conf,
                                    'box': (x1, y1, x2, y2),
                                    'color': color,
                                    'source': 'Chess_Model'
                                })
                except:
                    pass
            
            # Apply smart deduplication
            original_count = len(all_detections)
            all_detections = smart_deduplicate(all_detections)
            final_count = len(all_detections)
            
            # Real-time terminal output - show detections every frame
            if final_count > 0:
                unique_objects = list(set([det['class'] for det in all_detections]))
                for obj in sorted(unique_objects):
                    print(f"{obj} detected")
            else:
                print("No objects detected")
            
            # Draw all detections with source info
            for det in all_detections:
                x1, y1, x2, y2 = det['box']
                label = f"{det['class']}: {det['confidence']:.2f}"
                color = det['color']
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add minimal info overlay
            cv2.putText(frame, f"Objects: {final_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Object Detection', frame)
            
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n⚠️ Detection interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Enhanced Multi-Model Detection completed!")
        print("🎯 Thanks for using the 5-Model AI Detection System!")

if __name__ == "__main__":
    enhanced_multi_detection()