#!/usr/bin/env python3
"""
Chess Dataset Downloader
Downloads and prepares the Roboflow chess pieces dataset for training

This script automates the download and setup of the chess pieces dataset
from Roboflow Universe for training YOLOv8 models.
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
import shutil
import yaml

def download_roboflow_dataset():
    """Download chess pieces dataset from Roboflow"""
    print("♟️ Downloading Chess Pieces Dataset from Roboflow...")
    print("=" * 60)
    
    # Dataset information
    dataset_info = {
        'name': 'Chess Pieces Detection',
        'classes': 12,
        'total_images': 693,
        'source': 'Roboflow Universe'
    }
    
    print(f"📊 Dataset: {dataset_info['name']}")
    print(f"🎯 Classes: {dataset_info['classes']} (6 white + 6 black pieces)")
    print(f"📸 Images: {dataset_info['total_images']}")
    print(f"🌐 Source: {dataset_info['source']}")
    
    # Create dataset directory
    dataset_dir = "../../datasets/chess_detection"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # For now, provide manual download instructions
    # (Roboflow requires API key for automated downloads)
    print("\n📥 MANUAL DOWNLOAD REQUIRED:")
    print("1. Go to: https://universe.roboflow.com/joseph-nelson/chess-pieces-new")
    print("2. Click 'Download Dataset'")
    print("3. Choose format: 'YOLOv8'")
    print("4. Choose split: 'Train/Valid/Test'")
    print("5. Download the ZIP file")
    print("6. Extract to:", os.path.abspath(dataset_dir))
    
    return dataset_dir

def setup_chess_dataset_structure():
    """Setup proper YOLOv8 dataset structure"""
    dataset_dir = "../../datasets/chess_detection"
    
    # Expected structure after Roboflow download
    roboflow_structure = {
        'train': ['images', 'labels'],
        'valid': ['images', 'labels'],
        'test': ['images', 'labels']
    }
    
    print("\n📁 Setting up dataset structure...")
    
    # Create directories if they don't exist
    for split in roboflow_structure:
        for folder in roboflow_structure[split]:
            dir_path = os.path.join(dataset_dir, split, folder)
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ Created: {dir_path}")
    
    # Create data.yaml file
    create_chess_config(dataset_dir)
    
    return dataset_dir

def create_chess_config(dataset_dir):
    """Create YOLOv8 configuration file for chess pieces"""
    
    # Chess piece classes (12 total)
    chess_classes = [
        'white-king', 'white-queen', 'white-bishop', 'white-knight', 'white-rook', 'white-pawn',
        'black-king', 'black-queen', 'black-bishop', 'black-knight', 'black-rook', 'black-pawn'
    ]
    
    config = {
        'path': os.path.abspath(dataset_dir),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(chess_classes),
        'names': chess_classes
    }
    
    # Save as data.yaml (YOLOv8 standard)
    config_path = os.path.join(dataset_dir, "data.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Created config file: {config_path}")
    
    # Also save as config.yaml for backwards compatibility
    config_path_alt = os.path.join(dataset_dir, "config.yaml")
    with open(config_path_alt, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return config_path

def verify_dataset(dataset_dir):
    """Verify the dataset is properly downloaded and structured"""
    print("\n🔍 Verifying dataset...")
    
    required_files = [
        "data.yaml",
        "train/images",
        "train/labels", 
        "valid/images",
        "valid/labels"
    ]
    
    all_good = True
    for file_path in required_files:
        full_path = os.path.join(dataset_dir, file_path)
        if os.path.exists(full_path):
            if os.path.isdir(full_path):
                count = len(os.listdir(full_path))
                print(f"✅ {file_path}: {count} files")
                if count == 0:
                    all_good = False
                    print(f"⚠️  {file_path} is empty!")
            else:
                print(f"✅ {file_path}: exists")
        else:
            print(f"❌ {file_path}: missing")
            all_good = False
    
    if all_good:
        print("🎉 Dataset verification passed!")
        print("🚀 Ready for training!")
    else:
        print("⚠️  Dataset incomplete. Please download manually.")
    
    return all_good

def show_chess_piece_info():
    """Display information about chess pieces that can be detected"""
    pieces_info = {
        'White Pieces': {
            'white-king': '♔ King - Most important piece',
            'white-queen': '♕ Queen - Most powerful piece', 
            'white-bishop': '♗ Bishop - Diagonal movement',
            'white-knight': '♘ Knight - L-shaped movement',
            'white-rook': '♖ Rook - Horizontal/vertical movement',
            'white-pawn': '♙ Pawn - Forward movement only'
        },
        'Black Pieces': {
            'black-king': '♚ King - Most important piece',
            'black-queen': '♛ Queen - Most powerful piece',
            'black-bishop': '♝ Bishop - Diagonal movement', 
            'black-knight': '♞ Knight - L-shaped movement',
            'black-rook': '♜ Rook - Horizontal/vertical movement',
            'black-pawn': '♟ Pawn - Forward movement only'
        }
    }
    
    print("\n♟️ CHESS PIECES DETECTION CLASSES")
    print("=" * 50)
    
    for color, pieces in pieces_info.items():
        print(f"\n{color}:")
        for piece_name, description in pieces.items():
            print(f"  {description}")
    
    print("=" * 50)
    print(f"Total classes: {sum(len(pieces) for pieces in pieces_info.values())}")

def main():
    """Main function"""
    print("♟️ Chess Pieces Dataset Setup")
    print("=" * 50)
    
    # Show chess piece information
    show_chess_piece_info()
    
    # Download dataset (manual instructions)
    dataset_dir = download_roboflow_dataset()
    
    # Setup structure
    setup_chess_dataset_structure()
    
    # Wait for user to download manually
    input("\n⏸️  Press Enter after you've downloaded and extracted the dataset...")
    
    # Verify dataset
    if verify_dataset(dataset_dir):
        print("\n✅ SETUP COMPLETE!")
        print("\n🎯 NEXT STEPS:")
        print("1. 🚀 Train the model:")
        print("   cd training_scripts")
        print("   python train_chess_detection.py")
        print("\n2. 🧪 Test the model:")
        print("   cd testing_validation") 
        print("   python test_chess_detection.py")
        print("\n3. 🌟 Use in enhanced detection:")
        print("   cd main_systems")
        print("   python enhanced_detection.py")
    else:
        print("\n❌ Setup incomplete. Please follow the manual download steps.")

if __name__ == "__main__":
    main()