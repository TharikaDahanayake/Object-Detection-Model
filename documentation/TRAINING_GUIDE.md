# Custom Object Detection Training Guide

This guide will help you train your object detection model to recognize new objects using modern YOLOv8 framework.

## Quick Start Options

### Option 1: Extend Current Model (Easiest)
Use a pre-trained model with more object classes:

```powershell
python real_time_object_detection_coco.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
```

This uses 80+ object classes instead of the original 21.

### Option 2: Use YOLOv8 (Recommended)
Modern, flexible framework that's easier to train:

1. **Install requirements:**
```powershell
pip install -r requirements.txt
```

2. **Run with pre-trained model:**
```powershell
python yolo_object_detection.py
```

3. **Train custom objects:**
```powershell
python train_custom_objects.py
```

## Training Your Own Custom Objects

### Step 1: Setup Environment
```powershell
# Install required packages
pip install ultralytics opencv-python PyYAML numpy matplotlib

# For annotation tool
pip install labelImg
```

### Step 2: Collect Data
- Gather 100-1000+ images of your target objects
- Use diverse backgrounds, lighting, and angles
- Include various object sizes and orientations

### Step 3: Annotate Your Data
Choose one of these annotation tools:

#### Option A: LabelImg (Offline)
```powershell
pip install labelImg
labelImg
```
- Select your image folder
- Draw bounding boxes around objects
- Save in YOLO format

#### Option B: Roboflow (Online)
1. Go to [roboflow.com](https://roboflow.com)
2. Upload your images
3. Annotate online
4. Export in YOLOv8 format

### Step 4: Organize Your Dataset
```
datasets/
└── your_project/
    ├── images/
    │   ├── train/     (70-80% of images)
    │   ├── val/       (10-15% of images)
    │   └── test/      (10-15% of images)
    └── labels/
        ├── train/     (corresponding labels)
        ├── val/       (corresponding labels)
        └── test/      (corresponding labels)
```

### Step 5: Configure Training
Edit `train_custom_objects.py` and update:
```python
class_names = ['your_object1', 'your_object2', 'your_object3']
```

### Step 6: Start Training
```powershell
python train_custom_objects.py
```

### Step 7: Test Your Model
After training, test with:
```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/detect/your_project/weights/best.pt')

# Test with webcam
results = model.predict(source=0, show=True)
```

## Label Format
YOLO uses this format for labels (one file per image):
```
class_id center_x center_y width height
```

Example:
```
0 0.5 0.3 0.2 0.4
1 0.7 0.6 0.15 0.25
```

Where values are normalized (0-1).

## Training Tips

1. **Start Small**: Use YOLOv8n for faster experimentation
2. **Transfer Learning**: Always start with pre-trained weights
3. **Data Quality**: Good annotations are more important than quantity
4. **Balance Classes**: Try to have similar amounts of each object type
5. **Augmentation**: Use built-in augmentation for small datasets

## Common Training Parameters

```python
model.train(
    data='config.yaml',
    epochs=100,           # Number of training rounds
    imgsz=640,           # Image size
    batch=16,            # Batch size (reduce if out of memory)
    lr0=0.01,            # Learning rate
    patience=10,         # Early stopping patience
    save=True,           # Save checkpoints
    plots=True           # Generate training plots
)
```

## Troubleshooting

### Out of Memory Error
- Reduce batch size: `batch=8` or `batch=4`
- Use smaller model: `yolov8n.pt` instead of larger variants
- Reduce image size: `imgsz=416` instead of `imgsz=640`

### Poor Detection Results
- Collect more diverse training data
- Improve annotation quality
- Increase training epochs
- Try different model sizes (yolov8s, yolov8m)

### Training Too Slow
- Use GPU if available
- Reduce image size
- Use smaller model variant
- Reduce batch size if using CPU

## Model Variants

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| YOLOv8n | Smallest | Fastest | Good |
| YOLOv8s | Small | Fast | Better |
| YOLOv8m | Medium | Moderate | Very Good |
| YOLOv8l | Large | Slow | Excellent |
| YOLOv8x | Largest | Slowest | Best |

## Example: Training to Detect Custom Objects

```python
# 1. Setup
from ultralytics import YOLO

# 2. Create config.yaml
"""
path: /path/to/dataset
train: images/train
val: images/val
test: images/test
nc: 3
names: ['apple', 'banana', 'orange']
"""

# 3. Train
model = YOLO('yolov8n.pt')
results = model.train(
    data='config.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)

# 4. Use trained model
model = YOLO('runs/detect/train/weights/best.pt')
results = model('test_image.jpg')
```

## Next Steps
1. Follow the annotation guide in `annotation_guide.md`
2. Prepare your dataset
3. Run training
4. Integrate your trained model into real-time detection

For more advanced features, see the [Ultralytics documentation](https://docs.ultralytics.com/).