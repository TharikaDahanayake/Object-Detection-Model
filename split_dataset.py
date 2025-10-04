#!/usr/bin/env python3
"""
Split Dataset for Book Detection Training
Splits the 100 images from train folder into train/val splits
"""

import os
import shutil
import glob
import random

def split_dataset():
    """Split dataset into train/val (80/20 split)"""
    print("📂 Splitting dataset into train/val...")
    
    # Paths
    train_images = "datasets/book_detection/images/train"
    val_images = "datasets/book_detection/images/val"
    train_labels = "datasets/book_detection/labels/train"
    val_labels = "datasets/book_detection/labels/val"
    
    # Get all image files in train folder
    image_files = glob.glob(f"{train_images}/*.jpg") + glob.glob(f"{train_images}/*.png")
    print(f"📸 Found {len(image_files)} images in train folder")
    
    # Shuffle and split (80% train, 20% val)
    random.seed(42)  # For reproducible results
    random.shuffle(image_files)
    
    split_index = int(0.8 * len(image_files))
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]
    
    print(f"📊 Split: {len(train_files)} train, {len(val_files)} validation")
    
    # Move validation files
    for img_path in val_files:
        # Move image
        img_name = os.path.basename(img_path)
        shutil.move(img_path, os.path.join(val_images, img_name))
        
        # Move corresponding label
        label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        label_src = os.path.join(train_labels, label_name)
        label_dst = os.path.join(val_labels, label_name)
        
        if os.path.exists(label_src):
            shutil.move(label_src, label_dst)
        else:
            print(f"⚠️ Warning: No label found for {img_name}")
    
    print("✅ Dataset split completed!")
    print(f"📁 Train: {len(os.listdir(train_images))} images")
    print(f"📁 Val: {len(os.listdir(val_images))} images")

if __name__ == "__main__":
    split_dataset()