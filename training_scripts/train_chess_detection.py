#!/usr/bin/env python3
"""
Chess Pieces Detection Training Script
Train YOLOv8 to detect various chess pieces with high accuracy

This script sets up and trains a chess pieces detection model using YOLOv8.
It can detect all 12 types of chess pieces: white and black pieces.
"""

import os
import sys
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO
import requests
import zipfile

def download_chess_dataset():
    """Download and extract chess pieces dataset from Roboflow"""
    print("🔄 Setting up chess pieces dataset...")
    
    # Create dataset directory
    dataset_dir = "../../datasets/chess_detection"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Dataset URL from Roboflow (Chess Pieces v24 dataset)
    dataset_url = "https://universe.roboflow.com/ds/chess-pieces-new"
    
    print("📦 Chess pieces dataset setup completed!")
    print("⚠️  Note: You'll need to download the dataset manually from:")
    print("   https://universe.roboflow.com/joseph-nelson/chess-pieces-new")
    print("   Choose 'YOLOv8' format and download to datasets/chess_detection/")
    
    return dataset_dir

def create_chess_config():
    """Create YAML config file for chess pieces detection"""
    dataset_dir = "../../datasets/chess_detection"
    
    # Chess piece classes (12 total: 6 white + 6 black)
    chess_classes = [
        'white-king', 'white-queen', 'white-bishop', 'white-knight', 'white-rook', 'white-pawn',
        'black-king', 'black-queen', 'black-bishop', 'black-knight', 'black-rook', 'black-pawn'
    ]
    
    config = {
        'path': os.path.abspath(dataset_dir),
        'train': 'train/images',
        'val': 'valid/images', 
        'test': 'test/images',
        'nc': len(chess_classes),  # number of classes
        'names': chess_classes
    }
    
    config_path = os.path.join(dataset_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Created chess config file: {config_path}")
    return config_path

def prepare_chess_dataset():
    """Prepare chess dataset structure"""
    dataset_dir = "../../datasets/chess_detection"
    
    # Create directory structure
    directories = [
        "train/images", "train/labels",
        "valid/images", "valid/labels", 
        "test/images", "test/labels"
    ]
    
    for dir_path in directories:
        full_path = os.path.join(dataset_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
    
    print("✅ Chess dataset structure created!")
    return dataset_dir

def train_chess_detection():
    """Train YOLOv8 model to detect chess pieces"""
    print("🚀 Starting chess pieces detection training...")
    
    # Setup dataset
    dataset_dir = prepare_chess_dataset()
    config_path = create_chess_config()
    
    # Check if dataset exists
    train_images = os.path.join(dataset_dir, "train/images")
    if not os.path.exists(train_images) or len(os.listdir(train_images)) == 0:
        print("❌ No training images found!")
        print("📥 Please download the chess pieces dataset:")
        print("   1. Go to: https://universe.roboflow.com/joseph-nelson/chess-pieces-new")
        print("   2. Choose 'YOLOv8' format")
        print("   3. Download and extract to:", os.path.abspath(dataset_dir))
        return None
    
    # Load pre-trained YOLOv8 model
    model = YOLO('yolov8s.pt')  # Using small model for good balance of speed/accuracy
    
    print("♟️ Starting chess pieces detection training!")
    
    try:
        # Train the model
        results = model.train(
            data=config_path,
            epochs=100,     # More epochs for better chess piece recognition
            imgsz=416,      # Standard chess board image size
            batch=16,       # Batch size
            name='chess_detection',
            plots=True,
            save=True,
            patience=15,    # Early stopping patience
            lr0=0.01,       # Learning rate
            augment=True    # Data augmentation
        )
        
        print("🎉 Chess detection training completed!")
        print(f"📁 Model saved to: runs/detect/chess_detection/weights/best.pt")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def test_chess_detection():
    """Test the trained chess detection model"""
    model_path = 'runs/detect/chess_detection/weights/best.pt'
    
    if os.path.exists(model_path):
        print("🧪 Testing trained chess detection model...")
        model = YOLO(model_path)
        
        # Test with webcam
        print("📹 Starting webcam test... Press 'q' to quit")
        print("♟️ Point camera at a chess board to detect pieces")
        results = model.predict(source=0, show=True, conf=0.4, save=False)
    else:
        print(f"❌ Trained model not found at {model_path}")
        print("Please train the model first!")

def show_chess_classes():
    """Display all chess piece classes that can be detected"""
    chess_classes = [
        '♔ white-king', '♕ white-queen', '♗ white-bishop', 
        '♘ white-knight', '♖ white-rook', '♙ white-pawn',
        '♚ black-king', '♛ black-queen', '♝ black-bishop', 
        '♞ black-knight', '♜ black-rook', '♟ black-pawn'
    ]
    
    print("♟️ Chess Pieces Detection Classes:")
    print("=" * 40)
    for i, piece in enumerate(chess_classes, 1):
        print(f"{i:2d}. {piece}")
    print("=" * 40)

if __name__ == "__main__":
    print("♟️ Chess Pieces Detection Training Setup")
    print("=" * 50)
    
    # Show available classes
    show_chess_classes()
    
    # Setup dataset
    download_chess_dataset()
    
    print("\n📋 NEXT STEPS:")
    print("1. 📥 Download chess dataset from Roboflow:")
    print("   https://universe.roboflow.com/joseph-nelson/chess-pieces-new")
    print("2. 📁 Extract to: datasets/chess_detection/")
    print("3. 🚀 Run training: python train_chess_detection.py")
    print("4. 🧪 Test model: Use test function after training")
    
    # Ask if user wants to start training (if dataset exists)
    choice = input("\n🚀 Start training now? (y/n): ").strip().lower()
    if choice == 'y':
        train_chess_detection()
    
    # Ask if user wants to test (if model exists)
    if os.path.exists('runs/detect/chess_detection/weights/best.pt'):
        test_choice = input("\n🧪 Test trained model? (y/n): ").strip().lower()
        if test_choice == 'y':
            test_chess_detection()