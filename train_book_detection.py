# Book Detection Training Script
# This script will help you train YOLOv8 to detect books

import os
import yaml
from ultralytics import YOLO

def create_book_dataset_structure():
    """Create dataset structure for book detection"""
    print("Creating book detection dataset structure...")
    
    # Create directories
    base_dir = "datasets/book_detection"
    dirs = [
        f"{base_dir}/images/train",
        f"{base_dir}/images/val", 
        f"{base_dir}/images/test",
        f"{base_dir}/labels/train",
        f"{base_dir}/labels/val",
        f"{base_dir}/labels/test"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created: {dir_path}")
    
    return base_dir

def create_book_config():
    """Create YAML config file for book detection"""
    base_dir = create_book_dataset_structure()
    
    # Use absolute paths to avoid path issues
    config = {
        'path': os.path.abspath("Real-Time-Object-Detection-With-OpenCV/datasets/book_detection"),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,  # number of classes (just 'book')
        'names': ['book']  # class names
    }
    
    config_path = "Real-Time-Object-Detection-With-OpenCV/datasets/book_detection/config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✓ Created config file: {config_path}")
    return config_path

def train_book_detection():
    """Train YOLOv8 model to detect books"""
    print("Starting book detection training...")
    
    # Create config
    config_path = create_book_config()
    
    # Load pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')  # Start with nano version for speed
    
    print("🚀 Starting book detection training!")
    
    # Change to the project directory before training
    original_dir = os.getcwd()
    os.chdir("Real-Time-Object-Detection-With-OpenCV")
    
    try:
        # Train the model with absolute path to config
        results = model.train(
            data=os.path.abspath("datasets/book_detection/config.yaml"),
            epochs=50,  # Reduced for faster training
            imgsz=640,
            batch=8,   # Reduced batch size for better compatibility
            name='book_detection',
            plots=True,
            save=True
        )
        
        print("🎉 Training completed!")
        print(f"📁 Model saved to: runs/detect/book_detection/weights/best.pt")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Config exists: {os.path.exists('datasets/book_detection/config.yaml')}")
        
    finally:
        # Return to original directory
        os.chdir(original_dir)
    
    return results

def test_book_detection():
    """Test the trained book detection model"""
    model_path = 'runs/detect/book_detection/weights/best.pt'
    
    if os.path.exists(model_path):
        print("Testing trained book detection model...")
        model = YOLO(model_path)
        
        # Test with webcam
        print("Starting webcam test... Press 'q' to quit")
        results = model.predict(source=0, show=True, conf=0.5)
    else:
        print(f"❌ Trained model not found at {model_path}")
        print("Please train the model first!")

if __name__ == "__main__":
    print("📚 Book Detection Training Setup")
    print("=" * 50)
    
    # Create project structure
    config_path = train_book_detection()
    
    print("\n" + "=" * 50)
    print("SETUP COMPLETE! 🎉")
    print("=" * 50)
    
    print(f"\nDataset structure created for book detection!")
    print(f"Config file: {config_path}")
    
    print("\nNEXT STEPS:")
    print("1. 📸 Collect 100-500+ images of books")
    print("2. 🏷️  Annotate your images (mark books with bounding boxes)")
    print("3. 📁 Organize your data in the created folders")
    print("4. 🚀 Start training")
    print("5. 🧪 Test your trained model")
    
    print("\n📋 DETAILED INSTRUCTIONS:")
    print("1. Take photos of books in different:")
    print("   - Positions (open, closed, stacked)")
    print("   - Lighting conditions")
    print("   - Backgrounds")
    print("   - Angles and distances")
    
    print("\n2. Use annotation tools:")
    print("   - LabelImg (offline): pip install labelimg")
    print("   - Roboflow (online): https://roboflow.com")
    
    print("\n3. Save annotations in YOLO format")
    print("   Each image should have a corresponding .txt file")
    print("   Format: class_id center_x center_y width height")
    print("   For books: 0 0.5 0.5 0.3 0.4 (example)")
    
    choice = input("\nWould you like to start the annotation tool LabelImg? (y/n): ")
    if choice.lower() == 'y':
        try:
            os.system("labelimg")
        except:
            print("LabelImg not found. Install with: pip install labelimg")