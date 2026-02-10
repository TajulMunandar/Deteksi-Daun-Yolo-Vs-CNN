"""
YOLOv11 Leaf Detection Evaluation Script
==========================================
This script evaluates a trained YOLOv11 model on the validation set.

Usage:
    python evaluate.py --model path/to/best.pt
    python evaluate.py  # Uses default model path
"""

from ultralytics import YOLO
import argparse
import os
import torch
from pathlib import Path

# Default model paths
DEFAULT_MODEL = "runs/detect/leaf_detection/weights/best.pt"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate YOLOv11 Leaf Detection Model")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Path to trained model (default: runs/detect/leaf_detection/weights/best.pt)")
    parser.add_argument("--data", type=str, 
                        default="Deteksi_Daun.v4i.yolov11/data.yaml",
                        help="Path to data.yaml file")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save evaluation results")
    parser.add_argument("--plots", action="store_true", default=True,
                        help="Generate and save confusion matrix")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for evaluation")
    parser.add_argument("--iou", type=float, default=0.6,
                        help="IoU threshold for evaluation")
    return parser.parse_args()

def check_device():
    """Check if CUDA is available."""
    if torch.cuda.is_available():
        device = 0
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("💻 Using CPU")
    return device

def evaluate_model(model_path, data_yaml, save=True, plots=True, conf=0.25, iou=0.6):
    """Evaluate YOLOv11 model on validation set."""
    print("=" * 60)
    print("YOLOv11 Leaf Detection Model Evaluation")
    print("=" * 60)
    
    # Check device
    device = check_device()
    
    # Validate model path
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first or provide a valid model path.")
        return None
    
    print(f"\n📦 Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Run evaluation
    print("\n🔍 Running evaluation on validation set...")
    results = model.val(
        data=data_yaml,
        imgsz=640,
        batch=16,
        device=device,
        save_json=save,
        save_txt=save,
        save=save,
        plots=plots,
        conf=conf,
        iou=iou,
        verbose=True
    )
    
    # Extract metrics
    metrics = results.results_dict
    confusion_matrix = results.confusion_matrix
    
    # Helper function to safely format metrics
    def fmt_metric(value, decimals=4):
        if value is None or value == 'N/A':
            return 'N/A'
        try:
            return f"{float(value):.{decimals}f}"
        except (ValueError, TypeError):
            return str(value)
    
    # Print evaluation results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"\n📊 Precision:      {fmt_metric(metrics.get('metrics/precision(B)'))}")
    print(f"📊 Recall:         {fmt_metric(metrics.get('metrics/recall(B)'))}")
    print(f"📊 mAP50:          {fmt_metric(metrics.get('metrics/mAP50(B)'))}")
    print(f"📊 mAP50-95:       {fmt_metric(metrics.get('metrics/mAP50-95(B)'))}")
    print(f"📊 F1 Score:       {fmt_metric(metrics.get('metrics/f1(B)'))}")
    
    # Print per-class metrics if available
    if hasattr(results, 'perf_per_class'):
        print("\n📋 Per-Class Performance:")
        for cls_name, cls_metrics in results.perf_per_class.items():
            print(f"  {cls_name}: mAP50={fmt_metric(cls_metrics.get('mAP50'))}")
    
    # Save confusion matrix
    if plots:
        confusion_matrix_path = os.path.join("runs/detect", "confusion_matrix.png")
        if hasattr(confusion_matrix, 'savefig'):
            confusion_matrix.plot(save_dir="runs/detect")
            print(f"\n📁 Confusion matrix saved to: runs/detect/confusion_matrix.png")
    
    # Summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Validation Set: {data_yaml}")
    print(f"mAP50: {fmt_metric(metrics.get('metrics/mAP50(B)'))}")
    print(f"mAP50-95: {fmt_metric(metrics.get('metrics/mAP50-95(B)'))}")
    
    return results

if __name__ == "__main__":
    args = parse_args()
    
    try:
        results = evaluate_model(
            model_path=args.model,
            data_yaml=args.data,
            save=args.save,
            plots=args.plots,
            conf=args.conf,
            iou=args.iou
        )
        if results is not None:
            print("\n✅ Evaluation completed successfully!")
        else:
            print("\n❌ Evaluation failed!")
            exit(1)
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {str(e)}")
        raise
