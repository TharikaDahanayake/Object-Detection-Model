# Custom Object Training with YOLOv8
# This script helps you train YOLOv8 to detect custom objects

import os
import yaml
from ultralytics import YOLO
import shutil
from pathlib import Path

class CustomObjectTrainer:
    def __init__(self, project_name="custom_detection"):
        self.project_name = project_name
        self.dataset_path = f"datasets/{project_name}"
        
    def create_dataset_structure(self):
        """
        Create the required dataset structure for YOLO training
        
        Dataset structure:
        datasets/
        └── custom_detection/
            ├── images/
            │   ├── train/
            │   ├── val/
            │   └── test/
            └── labels/
                ├── train/
                ├── val/
                └── test/
        """
        print("Creating dataset structure...")
        
        # Create directories
        dirs = [
            f"{self.dataset_path}/images/train",
            f"{self.dataset_path}/images/val", 
            f"{self.dataset_path}/images/test",
            f"{self.dataset_path}/labels/train",
            f"{self.dataset_path}/labels/val",
            f"{self.dataset_path}/labels/test"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created: {dir_path}")
    
    def create_yaml_config(self, class_names):
        """
        Create YAML configuration file for training
        
        Args:
            class_names: List of class names to detect (e.g., ['apple', 'banana', 'orange'])
        """
        config = {
            'path': os.path.abspath(self.dataset_path),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(class_names),  # number of classes
            'names': class_names     # class names
        }
        
        config_path = f"{self.dataset_path}/config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Created config file: {config_path}")
        return config_path
    
    def train_model(self, config_path, epochs=100, img_size=640, batch_size=16):
        """
        Train YOLOv8 model on custom dataset
        
        Args:
            config_path: Path to YAML config file
            epochs: Number of training epochs
            img_size: Image size for training
            batch_size: Batch size for training
        """
        print("Starting training...")
        
        # Load a pre-trained model (recommended for transfer learning)
        model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt, etc.
        
        # Train the model
        results = model.train(
            data=config_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name=self.project_name,
            save=True,
            plots=True
        )
        
        print("Training completed!")
        return results
    
    def test_custom_model(self, model_path, test_image_path=None):
        """
        Test the trained custom model
        
        Args:
            model_path: Path to trained model weights
            test_image_path: Path to test image (optional)
        """
        print("Testing custom model...")
        
        # Load the trained model
        model = YOLO(model_path)
        
        if test_image_path and os.path.exists(test_image_path):
            # Test on specific image
            results = model(test_image_path)
            results[0].show()
        else:
            # Test with webcam
            print("Testing with webcam...")
            import cv2
            
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = model(frame)
                annotated_frame = results[0].plot()
                
                cv2.imshow('Custom Model Detection', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()

def create_annotation_guide():
    """
    Create a guide for data annotation
    """
    guide = """
# Data Annotation Guide for Custom Object Training

## Step 1: Collect Images
- Gather 100-1000+ images of your target objects
- Use diverse backgrounds, lighting conditions, and angles
- Include images with multiple objects and occlusions

## Step 2: Annotate Images
You can use these tools for annotation:

### Option A: LabelImg (Recommended for beginners)
1. Install: pip install labelimg
2. Run: labelimg
3. Select image folder and save folder
4. Draw bounding boxes around objects
5. Save as YOLO format

### Option B: Roboflow (Online tool)
1. Go to roboflow.com
2. Create account and upload images
3. Annotate objects online
4. Export in YOLOv8 format

### Option C: CVAT (Computer Vision Annotation Tool)
1. Set up CVAT locally or use cloud version
2. Create annotation project
3. Annotate objects
4. Export in YOLO format

## Step 3: Organize Data
Place your annotated data in this structure:

datasets/
└── custom_detection/
    ├── images/
    │   ├── train/        # 70-80% of your images
    │   ├── val/          # 10-15% of your images
    │   └── test/         # 10-15% of your images
    └── labels/
        ├── train/        # Corresponding label files
        ├── val/          # Corresponding label files
        └── test/         # Corresponding label files

## YOLO Label Format
Each label file should contain one line per object:
class_id center_x center_y width height

Where:
- class_id: 0, 1, 2, ... (starting from 0)
- center_x, center_y: Center coordinates (normalized 0-1)
- width, height: Bounding box dimensions (normalized 0-1)

Example label file content:
0 0.5 0.3 0.2 0.4
1 0.7 0.6 0.15 0.25

## Tips for Better Training:
1. Balance your dataset (similar number of images per class)
2. Include difficult cases (occlusion, poor lighting, etc.)
3. Augment your data if you have limited images
4. Use transfer learning with pre-trained weights
5. Start with a smaller model (yolov8n) for faster iteration
"""
    
    with open("annotation_guide.md", "w") as f:
        f.write(guide)
    
    print("Created annotation_guide.md - Please read for data preparation steps!")

def main():
    """
    Example usage of the custom trainer
    """
    print("Custom Object Training Setup")
    print("=" * 40)
    
    # Example: Training to detect custom objects
    # Replace with your own class names
    class_names = ['apple', 'banana', 'orange']  # Example classes
    
    # Create trainer
    trainer = CustomObjectTrainer("fruit_detection")
    
    # Create dataset structure
    trainer.create_dataset_structure()
    
    # Create config file
    config_path = trainer.create_yaml_config(class_names)
    
    # Create annotation guide
    create_annotation_guide()
    
    print("\nSetup completed! Next steps:")
    print("1. Read annotation_guide.md for data preparation")
    print("2. Collect and annotate your images")
    print("3. Place images and labels in the created folders")
    print("4. Run training with: trainer.train_model(config_path)")
    
    # Uncomment these lines when your data is ready:
    # trainer.train_model(config_path, epochs=100)
    # trainer.test_custom_model('runs/detect/fruit_detection/weights/best.pt')

if __name__ == "__main__":
    main()