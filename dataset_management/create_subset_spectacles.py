#!/usr/bin/env python3
"""
Create a subset of spectacles dataset with 500 training images for faster training
"""

import os
import shutil
import random
from pathlib import Path

def create_spectacles_subset():
    """Create a subset of 500 images from the spectacles dataset"""
    
    # Paths
    project_root = Path(__file__).parent.parent
    original_dataset = project_root / "datasets" / "spectacles_detection"
    subset_dataset = project_root / "datasets" / "spectacles_detection_500"
    
    print("🔧 Creating 500-image spectacles subset...")
    print(f"Original dataset: {original_dataset}")
    print(f"Subset dataset: {subset_dataset}")
    
    # Create subset directories
    subset_dirs = [
        subset_dataset / "train" / "images",
        subset_dataset / "train" / "labels",
        subset_dataset / "valid" / "images", 
        subset_dataset / "valid" / "labels",
        subset_dataset / "test" / "images",
        subset_dataset / "test" / "labels"
    ]
    
    for dir_path in subset_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")
    
    # Get all training images
    train_images_dir = original_dataset / "train" / "images"
    train_labels_dir = original_dataset / "train" / "labels"
    
    all_images = list(train_images_dir.glob("*.jpg"))
    print(f"Found {len(all_images)} training images")
    
    # Randomly select 500 images
    if len(all_images) > 500:
        selected_images = random.sample(all_images, 500)
    else:
        selected_images = all_images
        
    print(f"Selected {len(selected_images)} images for training")
    
    # Copy selected training images and their labels
    copied_images = 0
    copied_labels = 0
    
    for img_path in selected_images:
        # Copy image
        dest_img = subset_dataset / "train" / "images" / img_path.name
        try:
            shutil.copy2(str(img_path), str(dest_img))
            copied_images += 1
            
            # Copy corresponding label
            label_name = img_path.stem + ".txt"
            label_path = train_labels_dir / label_name
            if label_path.exists():
                dest_label = subset_dataset / "train" / "labels" / label_name
                shutil.copy2(str(label_path), str(dest_label))
                copied_labels += 1
        except Exception as e:
            print(f"Error copying {img_path.name}: {e}")
            continue
    
    print(f"✓ Copied {copied_images} training images")
    print(f"✓ Copied {copied_labels} training labels")
    
    # Copy all validation data (it's already small)
    valid_images_dir = original_dataset / "valid" / "images"
    valid_labels_dir = original_dataset / "valid" / "labels"
    
    if valid_images_dir.exists():
        valid_images = list(valid_images_dir.glob("*.jpg"))
        for img_path in valid_images:
            # Copy validation image
            dest_img = subset_dataset / "valid" / "images" / img_path.name
            shutil.copy2(str(img_path), str(dest_img))
            
            # Copy validation label
            label_name = img_path.stem + ".txt"
            label_path = valid_labels_dir / label_name
            if label_path.exists():
                dest_label = subset_dataset / "valid" / "labels" / label_name
                shutil.copy2(str(label_path), str(dest_label))
        
        print(f"✓ Copied {len(valid_images)} validation images and labels")
    
    # Copy test data (optional, for evaluation)
    test_images_dir = original_dataset / "test" / "images"
    test_labels_dir = original_dataset / "test" / "labels"
    
    if test_images_dir.exists():
        test_images = list(test_images_dir.glob("*.jpg"))
        for img_path in test_images:
            # Copy test image
            dest_img = subset_dataset / "test" / "images" / img_path.name
            shutil.copy2(str(img_path), str(dest_img))
            
            # Copy test label
            label_name = img_path.stem + ".txt"
            label_path = test_labels_dir / label_name
            if label_path.exists():
                dest_label = subset_dataset / "test" / "labels" / label_name
                shutil.copy2(str(label_path), str(dest_label))
        
        print(f"✓ Copied {len(test_images)} test images and labels")
    
    # Create data.yaml for subset
    data_yaml_content = """train: train/images
val: valid/images
test: test/images

nc: 1
names: ['spectacles']
"""
    
    data_yaml_path = subset_dataset / "data.yaml"
    with open(data_yaml_path, 'w') as f:
        f.write(data_yaml_content)
    
    print(f"✓ Created data.yaml: {data_yaml_path}")
    
    print("\n🎉 Spectacles 500-image subset created successfully!")
    print(f"📁 Dataset location: {subset_dataset}")
    print("\nYou can now train with this smaller dataset for faster iteration.")

if __name__ == "__main__":
    random.seed(42)  # For reproducible results
    create_spectacles_subset()