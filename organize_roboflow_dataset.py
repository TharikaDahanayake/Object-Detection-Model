# Organize Roboflow Dataset Script
# This script helps you move Roboflow files to the correct project structure

import os
import shutil
from pathlib import Path

def organize_roboflow_dataset():
    """
    Organize Roboflow dataset into the project structure
    """
    print("📁 Organizing Roboflow Book Dataset")
    print("=" * 50)
    
    # Ask user for the path to their extracted Roboflow folder
    roboflow_path = input("Enter the path to your extracted Roboflow folder: ").strip().strip('"')
    
    # Project dataset path
    project_dataset = "Real-Time-Object-Detection-With-OpenCV/datasets/book_detection"
    
    # Mapping from Roboflow structure to project structure
    mappings = [
        # (roboflow_folder, project_folder)
        ("train/images", "images/train"),
        ("train/labels", "labels/train"),
        ("valid/images", "images/val"),
        ("valid/labels", "labels/val"),
        ("test/images", "images/test"),
        ("test/labels", "labels/test"),
    ]
    
    # Alternative mappings if Roboflow uses different names
    alt_mappings = [
        ("train", "images/train"),
        ("train", "labels/train"),
        ("val/images", "images/val"),
        ("val/labels", "labels/val"),
        ("validation/images", "images/val"),
        ("validation/labels", "labels/val"),
    ]
    
    if not os.path.exists(roboflow_path):
        print(f"❌ Error: Path '{roboflow_path}' does not exist!")
        print("\n💡 Make sure you:")
        print("1. Extracted the ZIP file")
        print("2. Entered the correct path")
        print("3. Used forward slashes (/) or double backslashes (\\\\)")
        return False
    
    print(f"✅ Found Roboflow folder: {roboflow_path}")
    
    # Show contents of Roboflow folder
    print(f"\n📋 Contents of your Roboflow folder:")
    for item in os.listdir(roboflow_path):
        item_path = os.path.join(roboflow_path, item)
        if os.path.isdir(item_path):
            print(f"📁 {item}/")
            # Show subcontents
            try:
                for subitem in os.listdir(item_path):
                    print(f"   📄 {subitem}")
            except:
                pass
        else:
            print(f"📄 {item}")
    
    # Copy files
    total_images = 0
    total_labels = 0
    
    for robo_folder, proj_folder in mappings:
        robo_full_path = os.path.join(roboflow_path, robo_folder)
        proj_full_path = os.path.join(project_dataset, proj_folder)
        
        if os.path.exists(robo_full_path):
            print(f"\n📂 Copying from {robo_folder} to {proj_folder}")
            
            # Create destination directory
            os.makedirs(proj_full_path, exist_ok=True)
            
            # Copy files
            files = os.listdir(robo_full_path)
            for file in files:
                src = os.path.join(robo_full_path, file)
                dst = os.path.join(proj_full_path, file)
                
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                    if file.endswith(('.jpg', '.jpeg', '.png')):
                        total_images += 1
                    elif file.endswith('.txt'):
                        total_labels += 1
                    print(f"  ✅ Copied: {file}")
        else:
            print(f"⚠️  Not found: {robo_folder}")
    
    print(f"\n🎉 COPY COMPLETE!")
    print(f"📸 Total images copied: {total_images}")
    print(f"🏷️  Total labels copied: {total_labels}")
    
    # Update config.yaml if needed
    update_config()
    
    return total_images > 0

def update_config():
    """Update the config.yaml file with correct paths"""
    config_path = "Real-Time-Object-Detection-With-OpenCV/datasets/book_detection/config.yaml"
    
    config_content = f"""path: {os.path.abspath('Real-Time-Object-Detection-With-OpenCV/datasets/book_detection')}
train: images/train
val: images/val
test: images/test
nc: 1
names: ['book']
"""
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Updated config file: {config_path}")

def verify_dataset():
    """Verify that the dataset is properly organized"""
    print("\n🔍 VERIFYING DATASET...")
    
    base_path = "Real-Time-Object-Detection-With-OpenCV/datasets/book_detection"
    
    folders_to_check = [
        "images/train",
        "images/val", 
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test"
    ]
    
    for folder in folders_to_check:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            files = os.listdir(folder_path)
            print(f"📁 {folder}: {len(files)} files")
        else:
            print(f"❌ {folder}: MISSING")
    
    # Check if we have images and labels
    train_images = len(os.listdir(os.path.join(base_path, "images/train"))) if os.path.exists(os.path.join(base_path, "images/train")) else 0
    train_labels = len(os.listdir(os.path.join(base_path, "labels/train"))) if os.path.exists(os.path.join(base_path, "labels/train")) else 0
    
    print(f"\n📊 DATASET SUMMARY:")
    print(f"🖼️  Training images: {train_images}")
    print(f"🏷️  Training labels: {train_labels}")
    
    if train_images > 0 and train_labels > 0:
        print("✅ Dataset looks good! Ready for training.")
        return True
    else:
        print("❌ Dataset incomplete. Please check file organization.")
        return False

if __name__ == "__main__":
    success = organize_roboflow_dataset()
    
    if success:
        verify_dataset()
        print("\n🚀 NEXT STEPS:")
        print("1. ✅ Dataset organized")
        print("2. 🔄 Run training: python train_book_detection.py") 
        print("3. ⏳ Wait for training to complete")
        print("4. 🧪 Test your trained model")
    else:
        print("\n💡 TROUBLESHOOTING:")
        print("1. Check that you extracted the Roboflow ZIP file")
        print("2. Make sure the path is correct")
        print("3. Verify the folder structure matches Roboflow export")