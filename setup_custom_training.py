#!/usr/bin/env python3
"""
Quick Setup Script for Custom Object Detection Training
This script helps you get started with training custom objects quickly.
"""

import os
import sys
import subprocess

def check_and_install_packages():
    """Check and install required packages"""
    required_packages = [
        'ultralytics',
        'opencv-python', 
        'PyYAML',
        'numpy',
        'matplotlib'
    ]
    
    print("Checking required packages...")
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")

def setup_project():
    """Setup project structure for custom training"""
    print("\nSetting up project structure...")
    
    # Get project details from user
    project_name = input("Enter your project name (e.g., 'fruit_detection'): ").strip()
    if not project_name:
        project_name = "custom_detection"
    
    classes_input = input("Enter object classes separated by commas (e.g., 'apple,banana,orange'): ").strip()
    if not classes_input:
        classes = ['object1', 'object2']
        print("Using default classes: object1, object2")
    else:
        classes = [cls.strip() for cls in classes_input.split(',')]
    
    # Create directory structure
    base_dir = f"datasets/{project_name}"
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
        print(f"Created: {dir_path}")
    
    # Create config file
    config_content = f"""path: {os.path.abspath(base_dir)}
train: images/train
val: images/val  
test: images/test
nc: {len(classes)}
names: {classes}
"""
    
    config_path = f"{base_dir}/config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Created config file: {config_path}")
    
    # Create a simple training script
    training_script = f"""# Training script for {project_name}
from ultralytics import YOLO

def train_model():
    # Load pre-trained model
    model = YOLO('yolov8n.pt')  # or yolov8s.pt for better accuracy
    
    # Train the model
    results = model.train(
        data='{config_path}',
        epochs=100,
        imgsz=640,
        batch=16,
        name='{project_name}',
        plots=True
    )
    
    print("Training completed!")
    print(f"Best model saved at: runs/detect/{project_name}/weights/best.pt")
    
    return results

def test_model():
    # Load trained model
    model = YOLO('runs/detect/{project_name}/weights/best.pt')
    
    # Test with webcam
    results = model.predict(source=0, show=True, conf=0.5)

if __name__ == "__main__":
    print("1. Make sure you have prepared your dataset")
    print("2. Place images in {base_dir}/images/train, val, test")
    print("3. Place corresponding labels in {base_dir}/labels/train, val, test")
    print("4. Run train_model() to start training")
    
    # Uncomment to start training
    # train_model()
    
    # Uncomment to test trained model
    # test_model()
"""
    
    script_path = f"train_{project_name}.py"
    with open(script_path, 'w') as f:
        f.write(training_script)
    
    print(f"Created training script: {script_path}")
    
    return project_name, classes, base_dir

def show_next_steps(project_name, classes, base_dir):
    """Show user what to do next"""
    print("\n" + "="*50)
    print("SETUP COMPLETE! 🎉")
    print("="*50)
    
    print(f"\nProject: {project_name}")
    print(f"Classes: {', '.join(classes)}")
    print(f"Dataset folder: {base_dir}")
    
    print("\nNEXT STEPS:")
    print("1. 📸 Collect images of your objects")
    print("2. 🏷️  Annotate your images using:")
    print("   - LabelImg: pip install labelimg")
    print("   - Roboflow: https://roboflow.com")
    print("   - CVAT: https://cvat.org")
    
    print(f"\n3. 📁 Organize your data:")
    print(f"   Place training images in: {base_dir}/images/train/")
    print(f"   Place training labels in: {base_dir}/labels/train/")
    print(f"   Place validation images in: {base_dir}/images/val/")
    print(f"   Place validation labels in: {base_dir}/labels/val/")
    
    print(f"\n4. 🚀 Start training:")
    print(f"   python train_{project_name}.py")
    
    print("\n5. 📖 Read TRAINING_GUIDE.md for detailed instructions")
    
    print("\nTIPS:")
    print("- Collect 100-1000+ images per class")
    print("- Use diverse backgrounds and lighting")
    print("- Include objects at different angles and sizes")
    print("- Balance your dataset (similar number of images per class)")

def main():
    """Main setup function"""
    print("🤖 Custom Object Detection Training Setup")
    print("="*50)
    
    # Check and install packages
    check_and_install_packages()
    
    # Setup project
    project_name, classes, base_dir = setup_project()
    
    # Show next steps
    show_next_steps(project_name, classes, base_dir)
    
    # Ask if user wants to test YOLOv8 immediately
    test_now = input("\nWould you like to test YOLOv8 with pre-trained model now? (y/n): ").strip().lower()
    
    if test_now == 'y':
        print("Starting YOLOv8 with webcam...")
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')  # Will download automatically
            model.predict(source=0, show=True, conf=0.5)
        except Exception as e:
            print(f"Error testing model: {e}")
            print("Please check your camera connection and try again.")

if __name__ == "__main__":
    main()