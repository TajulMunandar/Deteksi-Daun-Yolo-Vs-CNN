# YOLOv11 Leaf Detection System

An end-to-end object detection system for leaf detection using YOLOv11. This project provides training, evaluation, testing, and a Flask REST API for leaf detection inference.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Flask API](#flask-api)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

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
Deteksi_Daun.v2i.yolov11/
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

If you have the zip file, extract it first:
```bash
unzip "Deteksi Daun.v2i.yolov11.zip"
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

## Training

### Basic Training

```bash
python training.py
```

### Custom Training Parameters

```bash
python training.py --model yolov11n.pt --epochs 100 --batch 32
```

### Training Configuration

The default training configuration:
- **Model**: YOLOv11n (nano) - change `MODEL_SIZE` in training.py
- **Image Size**: 640
- **Batch Size**: 16 (auto-adjusted based on GPU memory)
- **Epochs**: 50
- **Optimizer**: AdamW
- **Learning Rate**: 0.01 (initial), 0.0001 (final)

### Model Sizes

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

### Training Output

Training results are saved to:
- `runs/detect/leaf_detection/weights/best.pt` - Best model
- `runs/detect/leaf_detection/weights/last.pt` - Last checkpoint
- `runs/detect/leaf_detection/` - Training plots and metrics

---

## Evaluation

Evaluate the trained model on the validation set:

### Basic Evaluation

```bash
python evaluate.py
```

### Custom Evaluation

```bash
python evaluate.py --model runs/detect/leaf_detection/weights/best.pt --conf 0.3
```

### Evaluation Metrics

The evaluation script provides:
- **Precision**: Overall precision score
- **Recall**: Overall recall score
- **mAP50**: Mean Average Precision at IoU 0.50
- **mAP50-95**: Mean Average Precision at IoU 0.50-0.95
- **Confusion Matrix**: Visual representation of predictions

### Output

Evaluation results are saved to:
- `runs/detect/confusion_matrix.png` - Confusion matrix visualization
- `runs/detect/` - Evaluation metrics

---

## Testing

Run inference on test images:

### Basic Testing

```bash
python test.py
```

### Custom Testing

```bash
python test.py \
    --model runs/detect/leaf_detection/weights/best.pt \
    --source Deteksi_Daun.v2i.yolov11/test/images \
    --output runs/detect/test \
    --conf 0.3
```

### Testing Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | auto | Path to trained model |
| `--source` | test/images | Path to test images |
| `--output` | runs/detect/test | Output directory |
| `--conf` | 0.25 | Confidence threshold |
| `--iou` | 0.45 | IoU threshold |

### Output

Test results are saved to:
- `runs/detect/test/` - Predicted images with bounding boxes
- `runs/detect/test/labels/` - YOLO format predictions (.txt)
- `runs/detect/test/predictions.json` - JSON format predictions

---

## Flask API

A REST API for leaf detection inference.

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
  "model_loaded": true,
  "device": "cuda"
}
```

#### Get Classes
```bash
curl http://localhost:5000/classes
```

#### Predict (Multipart Form)
```bash
curl -X POST \
    -F "image=@path/to/leaf.jpg" \
    http://localhost:5000/predict
```

#### Predict (Base64)
```bash
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"image": "<base64_encoded_image>"}' \
    http://localhost:5000/predict/base64
```

### Predict Response Format

```json
{
  "success": true,
  "predictions": [
    {
      "detected_class": "daun_jeruk",
      "confidence": 0.95,
      "bounding_box": {
        "x1": 100.5,
        "y1": 50.2,
        "x2": 300.8,
        "y2": 250.6
      }
    }
  ],
  "total_detections": 1,
  "annotated_image": "<base64_encoded_image_with_boxes>"
}
```

### API Configuration

Set environment variables to customize the API:

```bash
# Set model path
export MODEL_PATH=runs/detect/leaf_detection/weights/best.pt

# Set confidence threshold
export CONFIDENCE_THRESHOLD=0.3

# Set port
export PORT=5000

# Enable debug mode
export DEBUG=true

python app.py
```

---

## Project Structure

```
├── training.py           # YOLOv11 training script
├── evaluate.py           # Model evaluation script
├── test.py              # Test inference script
├── app.py               # Flask REST API
├── requirements.txt     # Python dependencies
├── README.md            # This file
│
├── Deteksi_Daun.v2i.yolov11/   # Dataset (after extraction)
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
│
└── runs/
    └── detect/
        └── leaf_detection/
            ├── weights/
            │   ├── best.pt
            │   └── last.pt
            ├── train/
            ├── val/
            └── confusion_matrix.png
```

---

## Configuration

### Training Configuration

Edit `training.py` to customize:

```python
# Model settings
MODEL_SIZE = "n"  # n, s, m, l, x
IMGSZ = 640
BATCH = 16
EPOCHS = 50
OPTIMIZER = "AdamW"
```

### API Configuration

Environment variables:
- `MODEL_PATH`: Path to trained model
- `CONFIDENCE_THRESHOLD`: Detection confidence threshold (default: 0.25)
- `IOU_THRESHOLD`: IoU threshold for NMS (default: 0.45)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 5000)
- `DEBUG`: Enable debug mode (default: false)

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

Reduce batch size:
```python
BATCH = 8  # or 4
```

#### 2. Model Not Found

Ensure the model is trained:
```bash
python training.py
```

Or specify the model path:
```bash
python evaluate.py --model runs/detect/leaf_detection/weights/best.pt
```

#### 3. No Detections

Lower the confidence threshold:
```bash
python test.py --conf 0.1
```

#### 4. Flask App Won't Start

Check if port is in use:
```bash
# Find process using port 5000
netstat -ano | findstr :5000

# Kill the process
taskkill /PID <PID> /F
```

#### 5. GPU Not Detected

Verify CUDA installation:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Best Practices

1. **Preprocessing**: Ensure images are in consistent format (JPG/PNG)
2. **Confidence Threshold**: Adjust based on your use case (lower = more detections)
3. **Model Selection**: Use larger models for better accuracy
4. **Data Augmentation**: YOLOv11 applies augmentation automatically
5. **Validation**: Always validate before testing
6. **Version Control**: Keep the `best.pt` model safe

---

## License

This project is for educational and research purposes.

---

## References

- [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/)
- [YOLOv11 GitHub Repository](https://github.com/ultralytics/ultralytics)
- [YOLO Format Annotation](https://docs.ultralytics.com/datasets/detect/)

---

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Ultralytics documentation
3. Check your dataset annotations
