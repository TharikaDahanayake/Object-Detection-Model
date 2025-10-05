#!/usr/bin/env python3
"""
Train spectacles detection with 500 images subset for faster training
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training_500.log')
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Main training function for 500-image spectacles subset"""
    
    print("=" * 60)
    print("🤓 SPECTACLES DETECTION TRAINING - 500 Images")
    print("=" * 60)
    
    logger = setup_logging()
    
    try:
        # Import after logging setup
        import torch
        from ultralytics import YOLO
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
        else:
            print("Using CPU for training")
        print("-" * 60)
        
        # Setup paths
        project_root = Path(__file__).parent.parent
        dataset_path = project_root / "datasets" / "spectacles_detection_500"
        data_yaml = dataset_path / "data.yaml"
        
        logger.info("=== Spectacles Detection Training (500 Images) ===")
        logger.info(f"Project root: {project_root}")
        logger.info(f"Dataset path: {dataset_path}")
        
        # Check if dataset exists
        if not dataset_path.exists():
            logger.error(f"Dataset not found at: {dataset_path}")
            logger.error("Please run create_subset_spectacles.py first!")
            return False
            
        if not data_yaml.exists():
            logger.error(f"data.yaml not found at: {data_yaml}")
            return False
            
        logger.info(f"✓ Found dataset at: {dataset_path}")
        logger.info(f"✓ Found data.yaml at: {data_yaml}")
        
        # Count images
        train_images = list((dataset_path / "train" / "images").glob("*.jpg"))
        valid_images = list((dataset_path / "valid" / "images").glob("*.jpg"))
        
        logger.info(f"Training images: {len(train_images)}")
        logger.info(f"Validation images: {len(valid_images)}")
        
        if len(train_images) == 0:
            logger.error("No training images found!")
            return False
            
        # Load pre-trained YOLOv8 model
        logger.info("Loading YOLOv8n pre-trained model...")
        model = YOLO('yolov8n.pt')
        logger.info("✓ Loaded YOLOv8n pre-trained model")
        
        # Training parameters optimized for 500 images
        training_params = {
            'data': str(data_yaml),
            'epochs': 50,              # Reduced epochs for faster training
            'imgsz': 640,
            'batch': 8,               # Smaller batch size for CPU
            'name': 'spectacles_500',
            'project': str(project_root / "runs" / "detect"),
            'save': True,
            'save_period': 10,
            'device': 'cpu',          # Use CPU
            'workers': 2,             # Reduced workers for CPU
            'patience': 20,           # Early stopping patience
            'optimizer': 'AdamW',
            'lr0': 0.01,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'plots': True,
            'verbose': True
        }
        
        logger.info("Starting training with parameters:")
        for key, value in training_params.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("🚀 Starting spectacles detection training (500 images)...")
        
        # Start training
        results = model.train(**training_params)
        
        # Training completed successfully
        logger.info("🎉 Training completed successfully!")
        
        # Get the best model path
        run_dir = Path(training_params['project']) / training_params['name']
        best_model = run_dir / "weights" / "best.pt"
        last_model = run_dir / "weights" / "last.pt"
        
        if best_model.exists():
            logger.info(f"✓ Best model saved at: {best_model}")
        if last_model.exists():
            logger.info(f"✓ Last model saved at: {last_model}")
            
        # Copy best model to models directory for easy access
        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)
        
        if best_model.exists():
            import shutil
            dest_model = models_dir / "spectacles_yolov8_500.pt"
            shutil.copy2(str(best_model), str(dest_model))
            logger.info(f"✓ Copied best model to: {dest_model}")
        
        print("\n" + "=" * 60)
        print("🎉 SPECTACLES TRAINING COMPLETED!")
        print("=" * 60)
        print(f"Best model: {best_model}")
        print(f"Results saved in: {run_dir}")
        print(f"Model ready for integration: {dest_model}")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please install required packages:")
        logger.error("pip install ultralytics torch")
        return False
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n" + "=" * 60)
        print("❌ TRAINING FAILED!")
        print("=" * 60)
        print("\nPlease check the error messages above and:")
        print("1. Ensure the 500-image dataset exists (run create_subset_spectacles.py)")
        print("2. Check that you have the required packages installed")
        print("3. Verify the dataset structure")
        sys.exit(1)
    else:
        print("\nTraining completed successfully! ✨")