# Training Progress Monitor
# Run this to check how your book detection training is going

import os
import time
from pathlib import Path

def check_training_progress():
    """Check the progress of book detection training"""
    print("🔍 Checking Book Detection Training Progress...")
    print("=" * 50)
    
    # Check if training has started
    runs_dir = "Real-Time-Object-Detection-With-OpenCV/runs/detect"
    
    if not os.path.exists(runs_dir):
        print("❌ Training hasn't started yet or runs folder not created")
        return
    
    # Look for book_detection training folders
    training_folders = []
    if os.path.exists(runs_dir):
        for folder in os.listdir(runs_dir):
            if "book_detection" in folder:
                training_folders.append(folder)
    
    if not training_folders:
        print("🔄 Training is starting... No progress folders found yet")
        print("💡 This is normal - training setup takes a few minutes")
        return
    
    # Get the latest training folder
    latest_folder = max(training_folders)
    training_path = os.path.join(runs_dir, latest_folder)
    
    print(f"📁 Training folder: {latest_folder}")
    
    # Check for weights folder (created when training starts)
    weights_path = os.path.join(training_path, "weights")
    if os.path.exists(weights_path):
        print("✅ Training has started!")
        
        # Check for model files
        best_model = os.path.join(weights_path, "best.pt")
        last_model = os.path.join(weights_path, "last.pt")
        
        if os.path.exists(best_model):
            print("🎉 Best model available!")
            print(f"📍 Location: {best_model}")
        
        if os.path.exists(last_model):
            print("🔄 Latest checkpoint available!")
            print(f"📍 Location: {last_model}")
    else:
        print("🔄 Training is initializing...")
    
    # Check for results and plots
    results_file = os.path.join(training_path, "results.png")
    if os.path.exists(results_file):
        print("📊 Training plots available!")
        print(f"📍 Results chart: {results_file}")
    
    # Check training log
    print("\n📋 Training Status:")
    if os.path.exists(training_path):
        files = os.listdir(training_path)
        print(f"📁 Files in training folder: {len(files)}")
        for file in files:
            if file.endswith('.yaml'):
                print(f"  ⚙️  Config: {file}")
            elif file.endswith('.png'):
                print(f"  📊 Chart: {file}")
            elif file == 'weights':
                print(f"  💾 Weights folder: ✅")
    
    print("\n💡 Tips:")
    print("- Training typically takes 15-30 minutes")
    print("- You'll see 'best.pt' when training completes")
    print("- Results.png shows training progress charts")

def wait_for_training():
    """Wait and monitor training completion"""
    print("⏳ Monitoring training progress...")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            check_training_progress()
            print("\n" + "="*50)
            print("⏳ Checking again in 30 seconds...")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")
        print("💡 Training may still be running in the background")

if __name__ == "__main__":
    choice = input("Choose option:\n1. Check progress once\n2. Monitor continuously\nEnter (1 or 2): ")
    
    if choice == "2":
        wait_for_training()
    else:
        check_training_progress()
        
        print("\n🚀 WHAT'S HAPPENING:")
        print("1. 🔄 YOLOv8 is training on your 100 book images")
        print("2. 📊 It's learning to recognize books in different scenarios")
        print("3. 💾 The best model will be saved automatically")
        print("4. 📈 Progress charts are being generated")
        
        print("\n⏱️  ESTIMATED TIME:")
        print("- Small dataset (100 images): 15-30 minutes")
        print("- The training runs 50 epochs (complete passes through data)")
        
        print("\n🎯 WHEN COMPLETE:")
        print("- You'll have a custom book detection model")
        print("- Test it with: python test_book_detection.py")
        print("- Use it in real-time detection!")