#!/usr/bin/env python3
"""
Spectacles Dataset Preparation Script
Organize and split spectacles dataset for YOLO training

This script helps prepare your spectacles images and labels for training.
"""

import os
import shutil
from pathlib import Path
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_dataset(dataset_path, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
    """Split dataset into train/valid/test sets"""
    
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    
    # Create split directories
    for split in ['train', 'valid', 'test']:
        (dataset_path / split).mkdir(exist_ok=True)
    
    if not images_dir.exists() or not labels_dir.exists():
        logger.error("Images or labels directory not found!")
        logger.info("Please add your images to: datasets/spectacles_detection/images/")
        logger.info("Please add your labels to: datasets/spectacles_detection/labels/")
        return False
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(list(images_dir.glob(f'*{ext}')))
        all_images.extend(list(images_dir.glob(f'*{ext.upper()}')))
    
    if not all_images:
        logger.error("No images found in images/ directory!")
        return False
    
    logger.info(f"Found {len(all_images)} images")
    
    # Check for corresponding labels
    valid_pairs = []
    for img_path in all_images:
        label_path = labels_dir / (img_path.stem + '.txt')
        if label_path.exists():
            valid_pairs.append((img_path, label_path))
        else:
            logger.warning(f"No label found for: {img_path.name}")
    
    logger.info(f"Found {len(valid_pairs)} image-label pairs")
    
    if len(valid_pairs) == 0:
        logger.error("No valid image-label pairs found!")
        return False
    
    # Shuffle the pairs
    random.shuffle(valid_pairs)
    
    # Calculate split sizes
    total = len(valid_pairs)
    train_size = int(total * train_ratio)
    valid_size = int(total * valid_ratio)
    test_size = total - train_size - valid_size
    
    logger.info(f"Dataset split: Train={train_size}, Valid={valid_size}, Test={test_size}")
    
    # Split and copy files
    splits = {
        'train': valid_pairs[:train_size],
        'valid': valid_pairs[train_size:train_size + valid_size],
        'test': valid_pairs[train_size + valid_size:]
    }
    
    for split_name, pairs in splits.items():
        split_dir = dataset_path / split_name
        
        logger.info(f"Copying {len(pairs)} pairs to {split_name}/")
        
        for img_path, label_path in pairs:
            # Copy image
            shutil.copy2(img_path, split_dir / img_path.name)
            # Copy label
            shutil.copy2(label_path, split_dir / label_path.name)
    
    logger.info("✅ Dataset split completed successfully!")
    return True

def validate_labels(dataset_path):
    """Validate YOLO format labels"""
    dataset_path = Path(dataset_path)
    
    errors = []
    total_labels = 0
    
    for split in ['train', 'valid', 'test']:
        split_dir = dataset_path / split
        if not split_dir.exists():
            continue
            
        label_files = list(split_dir.glob('*.txt'))
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        errors.append(f"{label_file.name}:{line_num} - Invalid format (expected 5 values)")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:])
                        
                        # Check if values are normalized (0-1)
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            errors.append(f"{label_file.name}:{line_num} - Values not normalized (0-1)")
                        
                        # Check class ID (should be 0 for spectacles)
                        if class_id != 0:
                            errors.append(f"{label_file.name}:{line_num} - Invalid class ID: {class_id}")
                        
                        total_labels += 1
                        
                    except ValueError:
                        errors.append(f"{label_file.name}:{line_num} - Invalid number format")
                        
            except Exception as e:
                errors.append(f"{label_file.name} - Error reading file: {e}")
    
    logger.info(f"Validated {total_labels} labels")
    
    if errors:
        logger.error(f"Found {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10 errors
            logger.error(f"  {error}")
        if len(errors) > 10:
            logger.error(f"  ... and {len(errors) - 10} more errors")
        return False
    else:
        logger.info("✅ All labels are valid!")
        return True

def count_dataset_stats(dataset_path):
    """Count and display dataset statistics"""
    dataset_path = Path(dataset_path)
    
    stats = {}
    
    for split in ['train', 'valid', 'test']:
        split_dir = dataset_path / split
        if split_dir.exists():
            # Count images
            image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
            images = []
            for ext in image_exts:
                images.extend(list(split_dir.glob(f'*{ext}')))
                images.extend(list(split_dir.glob(f'*{ext.upper()}')))
            
            # Count labels
            labels = list(split_dir.glob('*.txt'))
            
            # Count total annotations
            total_annotations = 0
            for label_file in labels:
                try:
                    with open(label_file, 'r') as f:
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                        total_annotations += len(lines)
                except:
                    pass
            
            stats[split] = {
                'images': len(images),
                'labels': len(labels),
                'annotations': total_annotations
            }
        else:
            stats[split] = {'images': 0, 'labels': 0, 'annotations': 0}
    
    # Display stats
    print("\n" + "=" * 50)
    print("📊 DATASET STATISTICS")
    print("=" * 50)
    
    for split, data in stats.items():
        print(f"{split.upper():>8}: {data['images']:>3} images, {data['labels']:>3} labels, {data['annotations']:>3} annotations")
    
    total_images = sum(data['images'] for data in stats.values())
    total_labels = sum(data['labels'] for data in stats.values())
    total_annotations = sum(data['annotations'] for data in stats.values())
    
    print("-" * 50)
    print(f"{'TOTAL':>8}: {total_images:>3} images, {total_labels:>3} labels, {total_annotations:>3} annotations")
    print("=" * 50)
    
    return stats

def main():
    """Main function"""
    print("=" * 60)
    print("🤓 SPECTACLES DATASET PREPARATION")
    print("=" * 60)
    
    # Get dataset path
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "datasets" / "spectacles_detection"
    
    print(f"Dataset path: {dataset_path}")
    
    if not dataset_path.exists():
        logger.error("Dataset directory not found!")
        return
    
    while True:
        print("\n" + "-" * 40)
        print("Choose option:")
        print("1. Split dataset (images/ → train/valid/test/)")
        print("2. Validate labels")
        print("3. Show dataset statistics")
        print("4. Exit")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            print("\nSplitting dataset...")
            success = split_dataset(dataset_path)
            if success:
                print("✅ Dataset split completed!")
            else:
                print("❌ Dataset split failed!")
                
        elif choice == '2':
            print("\nValidating labels...")
            valid = validate_labels(dataset_path)
            if valid:
                print("✅ All labels are valid!")
            else:
                print("❌ Label validation failed!")
                
        elif choice == '3':
            count_dataset_stats(dataset_path)
            
        elif choice == '4':
            break
            
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main()