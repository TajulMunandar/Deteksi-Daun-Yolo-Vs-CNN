# YOLOv11 vs CNN Leaf Detection System

An end-to-end comparison system for leaf detection and classification using YOLOv11 (Object Detection) and CNN (Image Classification). This project provides training, evaluation, testing, and a Flask REST API for both models, designed for thesis research comparing the two approaches.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [CNN Training](#cnn-training)
- [YOLOv11 Training](#yolov11-training)
- [Evaluation](#evaluation)
- [Model Comparison](#model-comparison)
- [Flask API](#flask-api)
- [Web Interface](#web-interface)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Overview

This system compares two approaches for leaf classification:

| Feature | YOLOv11 | CNN |
|---------|----------|-----|
| **Task** | Object Detection | Image Classification |
| **Output** | Bounding boxes + Class | Single Class |
| **Multi-object** | ✅ Yes | ❌ No |
| **Localization** | ✅ Yes | ❌ No |
| **Speed** | Real-time | Very Fast |
| **Accuracy** | Good | Excellent (single object) |

**Leaf Classes:**
- daun jeruk (citrus leaf)
- daun kari (curry leaf)
- daun kunyit (turmeric leaf)
- daun pandan (pandan leaf)
- daun salam (bay leaf)

---

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (optional, for faster training)
- 8GB+ RAM
- 10GB+ Disk space

---

## Dataset Structure

The dataset should be structured as follows:

```
Deteksi_Daun.v4i.yolov11/
├── train/
│   ├── images/          # Training images
│   └── labels/          # Training labels (YOLO format)
├── valid/
│   ├── images/          # Validation images
│   └── labels/          # Validation labels
├── test/
│   ├── images/          # Test images
│   └── labels/          # Test labels
└── data.yaml            # Dataset configuration
```

---

## Installation

### 1. Create a virtual environment (recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 2. Install dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# For GPU support (CUDA), install PyTorch separately:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## CNN Training

### Basic Training

```bash
python cnn_training.py
```

### Custom Training Parameters

```bash
# Train with custom CNN architecture
python cnn_training.py --model custom --epochs 50 --batch 32

# Train with EfficientNet transfer learning
python cnn_training.py --model efficient --epochs 30 --batch 16

# Train with ResNet transfer learning
python cnn_training.py --model resnet --epochs 30 --batch 16
```

### CNN Training Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | custom | Model type: custom, efficient, resnet |
| `--epochs` | 20 | Number of training epochs |
| `--batch` | 16 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--img-size` | 224 | Input image size |
| `--no-pretrained` | False | Disable pretrained weights |
| `--dropout` | 0.5 | Dropout rate |

### CNN Model Architectures

| Model | Parameters | Accuracy | Speed |
|-------|------------|----------|-------|
| custom | ~2M | Good | Fast |
| efficientnet-b0 | 5.3M | Better | Medium |
| resnet-34 | 21.8M | Best | Slower |

### CNN Training Output

Results are saved to:
- `runs/cnn/best_model.pth` - Best model weights
- `runs/cnn/training_history.png` - Training curves
- `runs/cnn/training_results.json` - Final metrics

---

## YOLOv11 Training

### Basic Training

```bash
python training.py
```

### Custom Training Parameters

```bash
python training.py --epochs 100 --batch 32
```

### YOLOv11 Model Sizes

| Size | Parameters | Speed | Accuracy |
|------|------------|-------|----------|
| n (nano) | 2.6M | Fastest | Baseline |
| s (small) | 9.1M | Fast | Good |
| m (medium) | 20.1M | Medium | Better |
| l (large) | 25.9M | Slow | High |
| x (xlarge) | 57.2M | Slowest | Highest |

Edit `training.py` to change model size:
```python
MODEL_SIZE = "n"  # Change to 's', 'm', 'l', or 'x'
```

### YOLOv11 Training Output

Results are saved to:
- `runs/detect/leaf_detection/weights/best.pt` - Best model
- `runs/detect/leaf_detection/weights/last.pt` - Last checkpoint

---

## Evaluation

### CNN Evaluation

```bash
python cnn_evaluate.py --model runs/cnn/best_model.pth
```

CNN metrics include:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Per-class Performance
- ROC-AUC Curves

### YOLOv11 Evaluation

```bash
python evaluate.py
```

YOLOv11 metrics include:
- Precision, Recall
- mAP50, mAP50-95
- Confusion Matrix

---

## Model Comparison

### Compare Both Models

```bash
python compare_models.py --show-arch
```

This script:
1. Runs both models on test data
2. Calculates accuracy metrics
3. Measures inference time
4. Generates comparison charts
5. Shows architecture differences

### Comparison Output

Results saved to:
- `runs/model_comparison_results.json` - Detailed metrics
- `runs/metrics_comparison.png` - Accuracy comparison chart
- `runs/confusion_matrices_comparison.png` - Both confusion matrices
- `runs/complexity_comparison.png` - Model size/complexity comparison

---

## Flask API

A REST API supporting both YOLOv11 and CNN models.

### Start the API Server

```bash
python app.py
```

The API will be available at `http://localhost:5000`

### API Endpoints

#### Health Check
```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "yolo_loaded": true,
  "cnn_loaded": true,
  "device": "cuda"
}
```

#### Get Classes
```bash
curl http://localhost:5000/classes
```

#### YOLO Prediction (Detection)
```bash
curl -X POST -F "image=@leaf.jpg" http://localhost:5000/predict/yolo
```

#### CNN Prediction (Classification)
```bash
curl -X POST -F "image=@leaf.jpg" http://localhost:5000/predict/cnn
```

#### Both Models (Comparison)
```bash
curl -X POST -F "image=@leaf.jpg" http://localhost:5000/predict
```

#### Model Comparison Info
```bash
curl http://localhost:5000/compare
```

### API Configuration

Set environment variables:

```bash
# YOLO model path
export YOLO_MODEL_PATH=runs/detect/leaf_detection/weights/best.pt

# CNN model path
export CNN_MODEL_PATH=runs/cnn/best_model.pth

# Confidence threshold
export YOLO_CONFIDENCE=0.3

# Server port
export PORT=5000

python app.py
```

---

## Web Interface

Open `http://localhost:5000` in your browser.

### Features:
- **Model Selection Tabs**: Switch between YOLO, CNN, or Comparison mode
- **Image Upload**: Drag & drop or click to select images
- **Visual Results**: See annotated images with detections/classifications
- **Confidence Display**: View confidence scores for each prediction
- **API Status**: Shows which models are loaded

### Using the Interface:

1. Select a model tab:
   - 🎯 **YOLOv11** - For object detection with bounding boxes
   - 🧠 **CNN** - For simple image classification
   - ⚖️ **Compare** - Run both models together

2. Click "Pilih Gambar Daun" to upload an image

3. View results:
   - YOLO: Shows detected leaves with bounding boxes
   - CNN: Shows predicted class with confidence
   - Compare: Shows both results side by side

---

## Project Structure

```
├── cnn_model.py            # CNN model architectures
├── cnn_training.py         # CNN training script
├── cnn_evaluate.py         # CNN evaluation script
├── compare_models.py       # YOLO vs CNN comparison script
├── training.py            # YOLOv11 training script
├── evaluate.py            # YOLOv11 evaluation script
├── app.py                 # Flask REST API
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── Deteksi_Daun.v4i.yolov11/   # Dataset
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
│
├── runs/
│   ├── cnn/
│   │   ├── best_model.pth
│   │   ├── training_history.png
│   │   └── training_results.json
│   │
│   └── detect/
│       └── leaf_detection/
│           └── weights/
│               ├── best.pt
│               └── last.pt
│
└── templates/
    └── index.html         # Web interface
```

---

## Configuration

### CNN Training Configuration

Edit `cnn_training.py`:

```python
# Model settings
model_type = 'custom'  # custom, efficient, resnet
epochs = 20
batch_size = 16
learning_rate = 0.001
img_size = 224
dropout_rate = 0.5
```

### YOLOv11 Training Configuration

Edit `training.py`:

```python
MODEL_SIZE = "n"  # n, s, m, l, x
IMGSZ = 640
BATCH = 16
EPOCHS = 50
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

Reduce batch size:
```bash
python cnn_training.py --batch 8
```

#### 2. Model Not Found

Ensure models are trained:
```bash
# Train CNN
python cnn_training.py

# Train YOLOv11
python training.py
```

#### 3. Flask App Won't Start

Check if port is in use:
```bash
# On Windows
netstat -ano | findstr :5000

# Kill the process
taskkill /PID <PID> /F
```

#### 4. GPU Not Detected

```python
import torch
print(torch.cuda.is_available())
```

Install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 5. Module Not Found Errors

Reinstall dependencies:
```bash
pip install -r requirements.txt
```

---

## Best Practices

1. **Preprocessing**: Ensure images are in consistent format (JPG/PNG)
2. **Confidence Threshold**: Adjust based on your use case
3. **Model Selection**: 
   - Use CNN for single leaf classification
   - Use YOLOv11 for multiple leaves with localization
4. **Transfer Learning**: Use pretrained weights for better accuracy
5. **Validation**: Always validate before testing
6. **Comparison**: Use `compare_models.py` for thesis analysis

---

## License

This project is for educational and research purposes.

---

## References

- [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/)
- [PyTorch Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train CNN
python cnn_training.py --model custom --epochs 20

# 3. Train YOLOv11
python training.py

# 4. Run API
python app.py

# 5. Open browser
# http://localhost:5000

# 6. Compare models
python compare_models.py --show-arch
```
