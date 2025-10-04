# Real-time object detection with YOLOv8 (more flexible for custom training)
# Install required packages: pip install ultralytics opencv-python

import cv2
from ultralytics import YOLO
import numpy as np

class YOLOObjectDetection:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """
        Initialize YOLO object detection
        
        Args:
            model_path: Path to YOLO model (.pt file)
                       'yolov8n.pt' - nano (fastest)
                       'yolov8s.pt' - small
                       'yolov8m.pt' - medium
                       'yolov8l.pt' - large
                       'yolov8x.pt' - extra large (most accurate)
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Generate random colors for each class
        self.colors = np.random.uniform(0, 255, size=(len(self.model.names), 3))
    
    def detect_objects(self, frame):
        """
        Detect objects in a frame
        
        Args:
            frame: Input image/frame
            
        Returns:
            frame: Frame with bounding boxes and labels
        """
        # Run inference
        results = self.model(frame, conf=self.confidence_threshold)
        
        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Get confidence and class
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    # Draw bounding box and label
                    color = self.colors[class_id]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Create label with class name and confidence
                    label = f"{class_name}: {confidence:.2f}"
                    
                    # Calculate label size and position
                    (label_width, label_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    
                    # Draw label background
                    cv2.rectangle(
                        frame, 
                        (x1, y1 - label_height - 10), 
                        (x1 + label_width, y1), 
                        color, 
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        frame, 
                        label, 
                        (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 255, 255), 
                        1
                    )
                    
                    print(f"Detected: {label}")
        
        return frame
    
    def run_real_time_detection(self, camera_index=0):
        """
        Run real-time object detection using webcam
        
        Args:
            camera_index: Camera index (usually 0 for default camera)
        """
        # Initialize video capture
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Starting real-time object detection. Press 'q' to quit.")
        
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Detect objects in the frame
            frame = self.detect_objects(frame)
            
            # Display the frame
            cv2.imshow('YOLOv8 Object Detection', frame)
            
            # Check for 'q' key press to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped.")

def main():
    # Create detector instance
    # You can download different model sizes:
    # - yolov8n.pt (smallest, fastest)
    # - yolov8s.pt
    # - yolov8m.pt  
    # - yolov8l.pt
    # - yolov8x.pt (largest, most accurate)
    
    detector = YOLOObjectDetection(
        model_path='yolov8n.pt',  # Will auto-download if not present
        confidence_threshold=0.5
    )
    
    # Start real-time detection
    detector.run_real_time_detection(camera_index=0)

if __name__ == "__main__":
    main()