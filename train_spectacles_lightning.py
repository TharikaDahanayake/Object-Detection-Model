#!/usr/bin/env python3
"""
SUPER QUICK spectacles detection training - just 5 epochs for immediate results
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
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Super quick training function"""
    
    print("=" * 60)
    print("⚡ SUPER QUICK SPECTACLES TRAINING - 5 EPOCHS")
    print("=" * 60)
    
    logger = setup_logging()
    
    try:
        # Import after logging setup
        import torch
        from ultralytics import YOLO
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print("Using CPU for super quick training")
        print("-" * 60)
        
        # Setup paths
        project_root = Path(__file__).parent.parent
        dataset_path = project_root / "datasets" / "spectacles_detection_500"
        data_yaml = dataset_path / "data.yaml"
        
        logger.info("=== SUPER QUICK Spectacles Training (5 epochs) ===")
        logger.info(f"Dataset: {dataset_path}")
        
        # Check if dataset exists
        if not dataset_path.exists():
            logger.error(f"Dataset not found at: {dataset_path}")
            return False
            
        if not data_yaml.exists():
            logger.error(f"data.yaml not found at: {data_yaml}")
            return False
            
        # Count images
        train_images = list((dataset_path / "train" / "images").glob("*.jpg"))
        valid_images = list((dataset_path / "valid" / "images").glob("*.jpg"))
        
        logger.info(f"Training images: {len(train_images)}")
        logger.info(f"Validation images: {len(valid_images)}")
        
        if len(train_images) == 0:
            logger.error("No training images found!")
            return False
            
        # Load pre-trained YOLOv8 model
        logger.info("Loading YOLOv8n model...")
        model = YOLO('yolov8n.pt')
        logger.info("Model loaded!")
        
        # SUPER QUICK training parameters
        training_params = {
            'data': str(data_yaml),
            'epochs': 5,              # JUST 5 EPOCHS!
            'imgsz': 416,             # Smaller image size for speed
            'batch': 4,               # Small batch for CPU
            'name': 'spectacles_quick',
            'project': str(project_root / "runs" / "detect"),
            'save': True,
            'device': 'cpu',
            'workers': 1,             # Single worker
            'patience': 50,           # No early stopping
            'optimizer': 'SGD',       # Faster than AdamW
            'lr0': 0.01,
            'plots': False,           # Skip plots for speed
            'verbose': True,
            'val': False              # Skip validation during training
        }
        
        logger.info("Starting SUPER QUICK training:")
        for key, value in training_params.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("⚡ Starting lightning-fast training...")
        
        # Start training
        results = model.train(**training_params)
        
        # Training completed
        logger.info("⚡ SUPER QUICK training completed!")
        
        # Get model paths
        run_dir = Path(training_params['project']) / training_params['name']
        best_model = run_dir / "weights" / "best.pt"
        last_model = run_dir / "weights" / "last.pt"
        
        # Copy to models directory
        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)
        
        if last_model.exists():
            import shutil
            dest_model = models_dir / "spectacles_quick.pt"
            shutil.copy2(str(last_model), str(dest_model))
            logger.info(f"Quick model saved: {dest_model}")
        
        print("\n" + "=" * 60)
        print("⚡ QUICK TRAINING DONE!")
        print("=" * 60)
        print("🎯 Model ready for testing!")
        print(f"📁 Model location: {dest_model}")
        print("⏱️  Training time: ~2-3 minutes")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ QUICK TRAINING FAILED!")
        sys.exit(1)
    else:
        print("\n✨ Ready for integration!")