"""
YOLOv11 Leaf Detection Training Script
========================================
This script trains a YOLOv11 object detection model for leaf detection.

Usage:
    python training.py

Note: YOLOv11 is Ultralytics v8.x, so pretrained weights use yolov11n.pt naming.
"""

from ultralytics import YOLO
import os
import torch

# Configuration
DATA_YAML = "Deteksi_Daun.v4i.yolov11/data.yaml"
MODEL_SIZE = "n"  # YOLOv11 nano (use 's', 'm', 'l', 'x' for larger models)
IMGSZ = 640
BATCH = 16
EPOCHS = 5
OPTIMIZER = "AdamW"
PROJECT_NAME = "runs/detect"
EXPERIMENT_NAME = "leaf_detection"


def check_device():
    """Check if CUDA is available."""
    if torch.cuda.is_available():
        device = 0  # Use first GPU
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("💻 Using CPU")
    return device


def get_model_path(model_name):
    """Get model path, downloading if necessary."""
    # Check if model exists locally in current directory
    if os.path.exists(model_name):
        return model_name
    
    # Check Ultralytics cache directory
    cache_dir = os.path.expanduser("~/.cache/ultralytics")
    cache_model_path = os.path.join(cache_dir, "weights", model_name)
    if os.path.exists(cache_model_path):
        return cache_model_path
    
    return None


def download_model(model_name):
    """Download model from Ultralytics."""
    print(f"📥 Downloading {model_name}...")
    try:
        import urllib.request
        cache_dir = os.path.expanduser("~/.cache/ultralytics")
        os.makedirs(cache_dir, exist_ok=True)
        cache_model_path = os.path.join(cache_dir, "weights", model_name)
        
        # Model download URL
        model_url = f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{model_name}"
        urllib.request.urlretrieve(model_url, cache_model_path)
        print(f"✅ Model downloaded to: {cache_model_path}")
        return cache_model_path
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return None


def train_model():
    """Train YOLOv11 model for leaf detection."""
    print("=" * 60)
    print("YOLOv11 Leaf Detection Training")
    print("=" * 60)
    
    # Check device
    device = check_device()
    
    # Model name - YOLOv11 uses yolov11n.pt naming (Ultralytics v8.x)
    model_name = f"yolov11{MODEL_SIZE}.pt"
    
    # Try to get model path (local or cached)
    model_path = get_model_path(model_name)
    
    if model_path is None:
        print(f"\n⚠️  Model {model_name} not found locally.")
        print("   Attempting automatic download...")
        model_path = model_name
    
    # Load YOLOv11 model
    print(f"\n📦 Loading YOLOv11 {MODEL_SIZE.upper()} model...")
    try:
        model = YOLO(model_path)
    except FileNotFoundError:
        print(f"\n❌ Could not load {model_name}")
        print("\n   Trying alternative approach...")
        try:
            # Try newer naming convention yolo11n.pt
            alt_name = f"yolo11{MODEL_SIZE}.pt"
            print(f"   Attempting: {alt_name}")
            model = YOLO(alt_name)
        except FileNotFoundError:
            # Fall back to training from scratch with yaml config
            yaml_config = f"yolov11{MODEL_SIZE}.yaml"
            print(f"   Training from scratch with: {yaml_config}")
            model = YOLO(yaml_config)
    
    # Training configuration
    print(f"\n🔧 Training Configuration:")
    print(f"   - Image size: {IMGSZ}")
    print(f"   - Batch size: {BATCH}")
    print(f"   - Epochs: {EPOCHS}")
    print(f"   - Optimizer: {OPTIMIZER}")
    print(f"   - Experiment: {EXPERIMENT_NAME}")
    print()
    
    results = model.train(
        data=DATA_YAML,
        imgsz=IMGSZ,
        batch=BATCH,
        epochs=EPOCHS,
        optimizer=OPTIMIZER,
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        device=device,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3.0,
        cos_lr=True,
        close_mosaic=10,
    )
    
    # Print training results summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Try to get metrics
    try:
        print(f"\n📊 Final Results:")
        if hasattr(results, 'results_dict') and results.results_dict:
            print(f"   - Box Loss: {results.results_dict.get('train/box_loss', 'N/A'):.4f}")
            print(f"   - cls Loss: {results.results_dict.get('train/cls_loss', 'N/A'):.4f}")
            print(f"   - mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
            print(f"   - mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    except Exception:
        print(f"   (Metrics summary available in runs/detect/leaf_detection/)")
    
    best_model_path = os.path.join(PROJECT_NAME, EXPERIMENT_NAME, "weights", "best.pt")
    print(f"\n📁 Model saved to: {PROJECT_NAME}/{EXPERIMENT_NAME}/weights/")
    print(f"✅ Best model: {best_model_path}")
    
    return best_model_path


if __name__ == "__main__":
    try:
        best_model = train_model()
        print(f"\n🎉 Training finished successfully!")
        print(f"Best model saved at: {best_model}")
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        raise
