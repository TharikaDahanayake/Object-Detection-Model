#!/usr/bin/env python3
"""
Test Book Detection Model
Tests the trained YOLOv8 model for book detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def test_trained_model():
    """Test the trained book detection model"""
    print("🧪 Testing Trained Book Detection Model")
    print("=" * 50)
    
    # Find the latest trained model
    model_paths = [
        "C:/runs/detect/book_detection3/weights/best.pt",
        "C:/runs/detect/book_detection2/weights/best.pt",
        "C:/runs/detect/book_detection/weights/best.pt",
        "C:/runs/detect/book_detection4/weights/best.pt"
    ]
    
    model_path = None
    for path in reversed(model_paths):  # Check latest first
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ No trained model found!")
        print("Expected locations:", model_paths)
        return
    
    print(f"✅ Found trained model: {model_path}")
    
    # Load the trained model
    try:
        model = YOLO(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test on validation images
    test_images_dir = "datasets/book_detection/images/val"
    if not os.path.exists(test_images_dir):
        print(f"❌ Test images directory not found: {test_images_dir}")
        return
    
    test_images = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not test_images:
        print("❌ No test images found!")
        return
    
    print(f"📸 Found {len(test_images)} test images")
    
    # Test on a few images
    for i, img_name in enumerate(test_images[:5]):  # Test first 5 images
        img_path = os.path.join(test_images_dir, img_name)
        print(f"\n🔍 Testing on: {img_name}")
        
        try:
            # Run inference
            results = model(img_path, conf=0.25)  # 25% confidence threshold
            
            # Print results
            if len(results[0].boxes) > 0:
                detections = len(results[0].boxes)
                confidences = results[0].boxes.conf.cpu().numpy()
                avg_confidence = np.mean(confidences)
                print(f"  📚 Found {detections} book(s)")
                print(f"  🎯 Average confidence: {avg_confidence:.2f}")
                
                # Save result image
                result_img = results[0].plot()
                output_path = f"test_results_{img_name}"
                cv2.imwrite(output_path, result_img)
                print(f"  💾 Result saved: {output_path}")
            else:
                print(f"  ❌ No books detected")
                
        except Exception as e:
            print(f"  ❌ Error processing {img_name}: {e}")
    
    print("\n🎉 Testing completed!")
    return model

def test_webcam_detection():
    """Test book detection with webcam"""
    print("\n📹 Testing Webcam Book Detection")
    print("=" * 40)
    
    # Find the trained model
    model_paths = [
        "C:/runs/detect/book_detection3/weights/best.pt",
        "C:/runs/detect/book_detection2/weights/best.pt", 
        "C:/runs/detect/book_detection/weights/best.pt"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ No trained model found!")
        return
    
    print(f"✅ Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    print("📹 Webcam opened successfully!")
    print("📚 Point your camera at a book")
    print("🔄 Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run book detection
        results = model(frame, conf=0.3)  # 30% confidence
        
        # Draw results
        annotated_frame = results[0].plot()
        
        # Add text overlay
        cv2.putText(annotated_frame, "Book Detection - Press 'q' to quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Book Detection', annotated_frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("📹 Webcam test completed!")

if __name__ == "__main__":
    # Test on validation images first
    model = test_trained_model()
    
    if model:
        # Ask user if they want to test with webcam
        response = input("\n🎥 Do you want to test with webcam? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            test_webcam_detection()