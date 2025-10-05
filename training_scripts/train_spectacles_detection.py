#!/usr/bin/env python3
"""
Spectacles Detection Training Script
YOLOv8 Custom Object Detection Training

This script trains a YOLOv8 model to detect spectacles/eyeglasses.
"""

import os
import sys
import torch
from ultralytics import YOLO
from pathlib import Path
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dataset_structure(dataset_path):
    """Check if dataset has proper YOLO structure"""
    required_dirs = ['train', 'valid', 'test']  # test is optional
    required_files = ['data.yaml']
    
    logger.info(f"Checking dataset structure at: {dataset_path}")
    
    for req_dir in required_dirs[:2]:  # train and valid are required
        dir_path = dataset_path / req_dir
        if not dir_path.exists():
            logger.error(f"Required directory missing: {req_dir}")
            return False
        logger.info(f"✓ Found directory: {req_dir}")
    
    for req_file in required_files:
        file_path = dataset_path / req_file
        if not file_path.exists():
            logger.error(f"Required file missing: {req_file}")
            return False
        logger.info(f"✓ Found file: {req_file}")
    
    return True

def count_dataset_files(dataset_path):
    """Count images and labels in dataset"""
    counts = {}
    
    for split in ['train', 'valid', 'test']:
        split_path = dataset_path / split
        if split_path.exists():
            # Count images - check both direct and images subdirectory
            image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
            images = []
            
            # Check direct directory
            for ext in image_exts:
                images.extend(list(split_path.glob(f'*{ext}')))
                images.extend(list(split_path.glob(f'*{ext.upper()}')))
            
            # Check images subdirectory
            images_subdir = split_path / 'images'
            if images_subdir.exists():
                for ext in image_exts:
                    images.extend(list(images_subdir.glob(f'*{ext}')))
                    images.extend(list(images_subdir.glob(f'*{ext.upper()}')))
            
            # Count labels - check both direct and labels subdirectory
            labels = list(split_path.glob('*.txt'))
            labels_subdir = split_path / 'labels'
            if labels_subdir.exists():
                labels.extend(list(labels_subdir.glob('*.txt')))
            
            counts[split] = {
                'images': len(images),
                'labels': len(labels)
            }
            
            logger.info(f"{split.capitalize()} set: {len(images)} images, {len(labels)} labels")
        else:
            counts[split] = {'images': 0, 'labels': 0}
    
    return counts

def train_spectacles_model():
    """Train YOLOv8 model for spectacles detection"""
    
    # Paths
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "datasets" / "spectacles_detection"
    models_path = project_root / "models"
    
    logger.info("=== Spectacles Detection Training ===")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Dataset path: {dataset_path}")
    
    # Check if dataset exists and is properly structured
    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_path}")
        logger.info("Please create the dataset structure first:")
        logger.info("1. Add images to datasets/spectacles_detection/train/")
        logger.info("2. Add images to datasets/spectacles_detection/valid/")
        logger.info("3. Add corresponding .txt label files")
        return False
    
    # Verify dataset structure
    if not check_dataset_structure(dataset_path):
        logger.error("Dataset structure check failed!")
        return False
    
    # Count dataset files
    counts = count_dataset_files(dataset_path)
    
    # Check if we have training data
    if counts['train']['images'] == 0:
        logger.error("No training images found!")
        logger.info("Please add training images to: datasets/spectacles_detection/train/")
        return False
    
    if counts['valid']['images'] == 0:
        logger.warning("No validation images found!")
        logger.info("Consider adding validation images to: datasets/spectacles_detection/valid/")
    
    # Load data.yaml configuration
    data_yaml_path = dataset_path / "data.yaml"
    try:
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        logger.info("✓ Loaded data.yaml configuration")
    except Exception as e:
        logger.error(f"Error loading data.yaml: {e}")
        return False
    
    # Initialize YOLO model
    try:
        # Start with pre-trained YOLOv8 nano model for faster training
        model = YOLO('yolov8n.pt')  # nano version for faster training
        logger.info("✓ Loaded YOLOv8n pre-trained model")
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        return False
    
    # Training parameters
    training_params = {
        'data': str(data_yaml_path),  # Path to data.yaml
        'epochs': 100,                # Number of epochs
        'imgsz': 640,                # Image size
        'batch': 16,                 # Batch size (adjust based on your GPU memory)
        'name': 'spectacles_detection',  # Project name
        'project': str(project_root / 'runs' / 'detect'),  # Save location
        'save': True,                # Save checkpoints
        'save_period': 10,           # Save every N epochs
        'device': 'cpu',             # Use CPU (since CUDA not available)
        'workers': 4,                # Number of dataloader workers
        'patience': 50,              # Early stopping patience
        'optimizer': 'AdamW',        # Optimizer
        'lr0': 0.01,                # Initial learning rate
        'weight_decay': 0.0005,      # Weight decay
        'warmup_epochs': 3,          # Warmup epochs
        'box': 7.5,                  # Box loss gain
        'cls': 0.5,                  # Class loss gain
        'dfl': 1.5,                  # DFL loss gain
        'plots': True,               # Save training plots
        'verbose': True,             # Verbose output
    }
    
    logger.info("Starting training with parameters:")
    for key, value in training_params.items():
        logger.info(f"  {key}: {value}")
    
    try:
        # Start training
        logger.info("🚀 Starting spectacles detection training...")
        results = model.train(**training_params)
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"Results saved to: {results.save_dir}")
        
        # Model validation
        logger.info("📊 Running validation...")
        metrics = model.val()
        
        logger.info("📈 Training Summary:")
        logger.info(f"  mAP50: {metrics.box.map50:.4f}")
        logger.info(f"  mAP50-95: {metrics.box.map:.4f}")
        logger.info(f"  Precision: {metrics.box.mp:.4f}")
        logger.info(f"  Recall: {metrics.box.mr:.4f}")
        
        # Save the trained model to models directory
        models_path.mkdir(exist_ok=True)
        best_model_path = results.save_dir / "weights" / "best.pt"
        if best_model_path.exists():
            import shutil
            destination = models_path / "spectacles_best.pt"
            shutil.copy2(best_model_path, destination)
            logger.info(f"✅ Best model saved to: {destination}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("🤓 SPECTACLES DETECTION TRAINING")
    print("=" * 60)
    
    # Check PyTorch and CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    else:
        print("CUDA not available - using CPU")
    
    print("-" * 60)
    
    # Start training
    success = train_spectacles_model()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Check training results in runs/detect/spectacles_detection/")
        print("2. Test the model with test_spectacles_detection.py")
        print("3. Integrate with enhanced_detection.py")
    else:
        print("\n" + "=" * 60)
        print("❌ TRAINING FAILED!")
        print("=" * 60)
        print("\nPlease check the error messages above and:")
        print("1. Ensure you have training images in datasets/spectacles_detection/train/")
        print("2. Ensure you have corresponding label files (.txt)")
        print("3. Check the dataset structure")

if __name__ == "__main__":
    main()