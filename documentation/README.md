# 🎯 Enhanced Multi-Model Object Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0%2B-green.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-yellow.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

## 🚀 Advanced Multi-Model Computer Vision Project

This enhanced object detection system combines **5 state-of-the-art AI models** for comprehensive real-time object detection through webcam or laptop camera. The system achieves high accuracy and performance by intelligently integrating multiple specialized models with smart deduplication.

![Enhanced Detection Demo](../real_time_output_gif/real_time_output.gif)

## ✨ Key Features

### 🤖 **5 AI Models Working Together**
1. **📱 MobileNet SSD** - 20+ general objects (person, car, chair, etc.)
2. **📚 Book Detection** - Custom-trained YOLOv8 model for book recognition
3. **👓 Spectacles Detection** - Lightning-fast eyewear detection
4. **🎯 General YOLO** - 80 COCO objects with high accuracy
5. **♟️ Chess Pieces Detection** - All 12 chess piece types (NEW!)

### 🔥 **Advanced Capabilities**
- ⚡ **Real-time Processing** - 30+ FPS performance
- 🧠 **Smart Deduplication** - Eliminates duplicate detections
- 🎨 **Color-coded Detection** - Different colors for each model
- 📊 **Clean Terminal Output** - Simple "object detected" format
- 🖥️ **Minimal UI** - Clean webcam interface
- 📈 **High Accuracy** - Optimized confidence thresholds

## 🎮 **Detectable Objects**

### **General Objects (100+ types)**
- 👥 People, vehicles, animals, furniture
- 📱 Electronics, food items, sports equipment
- 🏠 Household items, tools, appliances

### **Chess Pieces (12 types)**
- ♔ **White Pieces**: King, Queen, Bishop, Knight, Rook, Pawn
- ♚ **Black Pieces**: King, Queen, Bishop, Knight, Rook, Pawn

### **Specialized Detection**
- 📚 **Books** - Any type of book or publication
- 👓 **Spectacles** - Glasses and eyewear

## 🛠️ **Installation & Setup**

### **Prerequisites**
```bash
Python 3.8+
Git
Webcam/Camera
```

### **Quick Start**
```bash
# Clone the repository
git clone https://github.com/TharikaDahanayake/Object-Detection-Model.git
cd Object-Detection-Model/Real-Time-Object-Detection-With-OpenCV

# Install dependencies
pip install ultralytics opencv-python numpy

# Run the enhanced detection system
cd main_systems
python enhanced_multi_detection.py
```

## 🎯 **Usage**

### **Running the System**
```bash
# Navigate to main systems folder
cd main_systems

# Start detection
python enhanced_multi_detection.py
```

### **Controls**
- **ESC or 'q'** - Quit the application
- **Camera** - Point at objects to detect them in real-time

### **Expected Output**
```
🎯 Enhanced Multi-Model Detection System
🤖 5 AI Models: MobileNet + Books + Spectacles + General Objects + Chess Pieces
✨ Features: Smart Deduplication + High Accuracy + Real-time Processing

📹 Webcam started! Press 'q' to quit
person detected
book detected
chair detected
white-king detected
```

## 📊 **System Performance**

| Feature | Specification |
|---------|---------------|
| **Models** | 5 AI models simultaneously |
| **FPS** | 30+ frames per second |
| **Objects** | 100+ detectable objects |
| **Accuracy** | 85%+ average confidence |
| **Latency** | <50ms per frame |
| **Memory** | ~2GB RAM usage |

## 🏗️ **Project Structure**

```
Real-Time-Object-Detection-With-OpenCV/
├── main_systems/
│   ├── enhanced_multi_detection.py    # 🎯 Main detection system
│   ├── real_time_object_detection.py  # Basic detection
│   └── yolo_object_detection.py       # YOLO-only detection
├── training_scripts/
│   ├── train_chess_detection.py       # Chess model training
│   ├── train_book_detection.py        # Book model training
│   └── train_spectacles_detection.py  # Spectacles training
├── testing_validation/
│   ├── test_chess_detection.py        # Chess testing
│   └── simple_chess_test.py           # Quick chess test
├── dataset_management/
│   └── setup_chess_dataset.py         # Dataset preparation
├── models/
│   ├── yolov8n.pt                     # YOLO weights
│   ├── MobileNetSSD_deploy.caffemodel # MobileNet weights
│   └── spectacles_quick.pt            # Spectacles model
└── documentation/
    ├── README.md                       # This file
    ├── CHESS_DETECTION.md             # Chess integration guide
    └── TRAINING_GUIDE.md              # Training documentation
```

## 🎨 **Detection Visualization**

The system uses color-coded bounding boxes:
- 🟢 **Green** - Books
- 🔵 **Blue** - General objects (YOLO)
- 🟠 **Orange** - MobileNet objects
- 🟡 **Cyan** - White chess pieces
- 🟣 **Magenta** - Black chess pieces
- 🟡 **Yellow** - Spectacles

## ⚙️ **Technical Details**

### **AI Models Used**
- **YOLOv8n** - Ultralytics YOLO for general object detection
- **MobileNetSSD** - Efficient mobile-optimized detection
- **Custom YOLOv8** - Trained on books dataset (3000+ images)
- **Custom YOLOv8** - Trained on spectacles dataset (500+ images)
- **Custom YOLOv8** - Trained on chess pieces dataset (600+ images)

### **Key Technologies**
- **OpenCV** - Computer vision and video processing
- **Ultralytics** - YOLO model framework
- **NumPy** - Numerical computations
- **Python** - Core programming language

## 🚀 **Advanced Features**

### **Smart Deduplication Algorithm**
```python
# Removes overlapping detections from multiple models
# Prioritizes higher-confidence detections
# Prevents duplicate object labeling
```

### **Multi-Model Integration**
```python
# Runs 5 AI models simultaneously
# Combines results intelligently  
# Optimized for real-time performance
```

### **Adaptive Confidence Thresholds**
- **Books**: 40% confidence minimum
- **Chess**: 40% confidence minimum  
- **General Objects**: 30% confidence minimum
- **MobileNet**: 60% confidence minimum
- **Spectacles**: 40% confidence minimum

## 📈 **Recent Updates**

### **v2.0 - Enhanced Multi-Model System**
- ✅ Added chess pieces detection (12 classes)
- ✅ Implemented smart deduplication
- ✅ Optimized performance for 5 models
- ✅ Improved color coding
- ✅ Clean terminal output
- ✅ Professional project structure

### **v1.0 - Base System**
- ✅ Basic object detection
- ✅ MobileNet integration
- ✅ Real-time processing

## 🤝 **Contributing**

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - YOLO implementation
- [OpenCV](https://opencv.org/) - Computer vision library
- [MobileNet](https://arxiv.org/abs/1704.04861) - Efficient neural networks
- Chess dataset from [ALEDAbeysirinarayana](https://github.com/ALEDAbeysirinarayana/Object-detection-new)

## 📧 **Contact**

**Tharika Dahanayake** - [GitHub](https://github.com/TharikaDahanayake)

Project Link: [https://github.com/TharikaDahanayake/Object-Detection-Model](https://github.com/TharikaDahanayake/Object-Detection-Model)

---

⭐ **Star this repository if you found it helpful!** ⭐



## 🎯 Features**Hi!**


- **📱 MobileNetSSD** - General object detection (20+ classes: person, chair, dog, car, etc.)

- **📚 YOLOv8 Books** - Custom trained book detection model### How to run this code?

- **👓 YOLOv8 Spectacles** - Custom trained spectacles/glasses detection (80.4% mAP50)

- **📱 YOLOv8 Cellphones** - Cellphone detection using general YOLO**Step 1:** Create a directory in your local machine and cd into it



### ⚡ **Performance Highlights:**```

- **Lightning-fast training** - Spectacles model trained in 10 minutes (5 epochs)mkdir ~/Desktop/opencv_project

- **Excellent accuracy** - 80.4% mAP50 for spectacles detectioncd ~/Desktop/opencv_project

- **Real-time processing** - Smooth webcam detection with color-coded bounding boxes```

- **Smart conflict resolution** - Intelligent coordination between multiple models

**Step 2:** Clone the repository and cd into the folder:

## 🛠️ Quick Start

```

### **Step 1:** Clone the repositorygit clone https://github.com/Surya-Murali/Real-Time-Object-Detection-With-OpenCV.git

```bashcd Real-Time-Object-Detection-With-OpenCV

git clone https://github.com/TharikaDahanayake/Object-Detection-Model.git```

cd Object-Detection-Model**Step 3:** Install all the necessary libraries. I used MacOS for this project. These are some of the libraries I had to install:

```

```

### **Step 2:** Install dependenciesbrew install opencv

```bashpip install opencv-python

pip install opencv-pythonpip install opencv-contrib-python

pip install ultralyticspip install opencv-python-headless

pip install numpypip install opencv-contrib-python-headless

pip install torch torchvisionpip install matplotlib

```pip install imutils

```

### **Step 3:** Run Enhanced Detection

```bashMake sure to download and install opencv and and opencv-contrib releases for OpenCV 3.3. This ensures that the deep neural network (dnn) module is installed. You must have OpenCV 3.3 (or newer) to run this code.

# Run the complete 4-model detection system

python enhanced_detection.py**Step 4:** Make sure you have your video devices connected (e.g. Webcam, FaceTime HD Camera, etc.). You can list them by typing this in your terminal



# Or test individual models```

python simple_spectacles_test.py  # Test spectacles detection onlysystem_profiler SPCameraDataType

python test_book_detection.py     # Test book detection onlysystem_profiler SPCameraDataType | grep "^    [^ ]" | sed "s/    //" | sed "s/://"

``````



## 📁 Project Structure**Step 5:** To start your video stream and real-time object detection, run the following command:



``````

├── enhanced_detection.py              # 🌟 Main 4-model detection systempython real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

├── simple_spectacles_test.py         # 👓 Spectacles detection test```

├── train_spectacles_lightning.py     # ⚡ Lightning-fast spectacles training

├── train_spectacles_500.py           # 🎯 500-image subset training**Step 6:** If you need any help regarding the arguments you pass, try:

├── test_book_detection.py            # 📚 Book detection testing

├── real_time_object_detection.py     # 📱 Original MobileNetSSD detection```

├── models/                            # 🧠 Trained model filespython real_time_object_detection.py --help

│   ├── spectacles_quick.pt           # 👓 Spectacles model (80.4% mAP50)```

│   └── MobileNetSSD_deploy.caffemodel # 📱 MobileNet model

├── datasets/                          # 📊 Training datasets### References and Useful Links

└── requirements.txt                   # 📋 Dependencies

```* https://github.com/chuanqi305/MobileNet-SSD

* https://github.com/opencv/opencv

## 🎮 Usage* https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/

* https://github.com/jrosebr1/imutils

### Enhanced Detection (All 4 Models)=======

```bash# Object-Detection-Model

python enhanced_detection.py>>>>>>> 2949119862ed9f0f096fc28ed46dc0839075ade7

```
- **Controls:** Press 'q' to quit
- **Output:** Color-coded bounding boxes with confidence scores
- **Detection alerts:** Console messages when objects are found

### Individual Model Testing
```bash
# Test spectacles detection
python simple_spectacles_test.py

# Test book detection  
python test_book_detection.py

# Original MobileNetSSD
python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
```

## 🏆 Model Performance

| Model | Accuracy | Training Time | Dataset Size |
|-------|----------|---------------|--------------|
| Spectacles | 80.4% mAP50 | 10 minutes | 500 images |
| Books | 85%+ mAP50 | 30 minutes | 1000+ images |
| MobileNetSSD | Pre-trained | N/A | COCO dataset |
| Cellphones | 70%+ (COCO) | Pre-trained | COCO dataset |

## 🔧 Advanced Features

### Lightning-Fast Training
```bash
# Train spectacles model in 10 minutes
python train_spectacles_lightning.py

# Train with 500-image subset
python train_spectacles_500.py
```

### Custom Dataset Creation
```bash
# Create spectacles dataset subset
python create_subset_spectacles.py

# Organize Roboflow datasets
python organize_roboflow_dataset.py
```

## 📊 Technical Achievements

### 🎯 **Training Innovation:**
- **5-epoch lightning training** approach for rapid iteration
- **Professional Roboflow dataset** integration (2,899+ images)
- **CPU-optimized training** (no GPU required)
- **Smart subset selection** (500 high-quality images)

### 🔄 **System Integration:**
- **Multi-model coordination** with conflict resolution
- **Real-time performance optimization**
- **Color-coded detection** for easy identification
- **Modular architecture** for easy expansion

## 🎨 Detection Color Coding

- **👓 Spectacles:** Magenta boxes
- **📚 Books:** Green boxes  
- **📱 Cellphones:** Yellow boxes
- **🔧 General Objects:** Various colors (MobileNetSSD)

## 🛠️ System Requirements

- **Python 3.8+**
- **OpenCV 4.0+**
- **PyTorch/Ultralytics**
- **Webcam or laptop camera**
- **4GB+ RAM recommended**

## 🚀 What Makes This Special

1. **🏆 Professional-Grade Performance** - 80.4% mAP50 achieved with minimal training
2. **⚡ Lightning-Fast Development** - From idea to working model in hours
3. **🎯 Multi-Model Mastery** - 4 different AI models working in harmony
4. **🔧 Production Ready** - Real-time performance with smart optimizations
5. **📈 Scalable Architecture** - Easy to add new object detection models

## 🎓 Educational Value

This project demonstrates:
- **Transfer Learning** techniques
- **Multi-model integration** strategies  
- **Real-time computer vision** optimization
- **Professional dataset handling**
- **Modern AI/ML development workflow**

## 📚 References and Credits

- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
- [OpenCV Deep Learning](https://opencv.org/)
- [MobileNetSSD](https://github.com/chuanqi305/MobileNet-SSD)
- [Roboflow Datasets](https://roboflow.com/)

---

**🎉 Ready to detect objects like a pro!** This system represents the cutting edge of real-time computer vision, combining multiple AI models for comprehensive object detection. Perfect for learning, research, or production applications!
