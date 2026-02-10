"""
YOLOv11 Leaf Detection Test Script
====================================
This script runs inference on test images and saves predictions.

Usage:
    python test.py --model path/to/best.pt
    python test.py  # Uses default model path
"""

from ultralytics import YOLO
import argparse
import os
import glob
from pathlib import Path
import torch
from datetime import datetime

# Default model paths
DEFAULT_MODEL = "runs/detect/leaf_detection/weights/best.pt"
DEFAULT_TEST_DIR = "Deteksi_Daun.v2i.yolov11/test/images"
DEFAULT_OUTPUT_DIR = "runs/detect/test"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test YOLOv11 Leaf Detection Model")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Path to trained model")
    parser.add_argument("--source", type=str, default=DEFAULT_TEST_DIR,
                        help="Path to test images directory or single image")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save prediction images")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for detection")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS")
    parser.add_argument("--save-txt", action="store_true", default=True,
                        help="Save predictions in YOLO format (.txt)")
    parser.add_argument("--save-conf", action="store_true", default=True,
                        help="Save confidence scores in .txt files")
    parser.add_argument("--save-crop", action="store_true", default=False,
                        help="Save cropped detection images")
    parser.add_argument("--show-labels", action="store_true", default=True,
                        help="Show labels on images")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print detailed results")
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

def get_image_files(source):
    """Get list of image files from source."""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    files = []
    
    if os.path.isfile(source):
        # Single image file
        if source.lower().endswith(tuple(['.jpg', '.jpeg', '.png', '.bmp', '.webp'])):
            files = [source]
    elif os.path.isdir(source):
        # Directory
        for ext in image_extensions:
            files.extend(glob.glob(os.path.join(source, ext)))
    
    return sorted(files)

def run_inference(model, source, output_dir, conf=0.25, iou=0.45, 
                  save_txt=True, save_conf=True, save_crop=False,
                  show_labels=True, verbose=True):
    """Run inference on images."""
    print("=" * 60)
    print("Running Inference on Test Images")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "crops"), exist_ok=True)
    
    # Get image files
    image_files = get_image_files(source)
    
    if not image_files:
        print(f"❌ No images found in: {source}")
        return []
    
    print(f"\n📁 Found {len(image_files)} images to process")
    
    # Run inference
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        save=True,
        save_dir=output_dir,
        save_txt=save_txt,
        save_conf=save_conf,
        save_crop=save_crop,
        show_labels=show_labels,
        verbose=verbose
    )
    
    return results

def print_inference_summary(results, verbose=True):
    """Print summary of inference results."""
    total_images = len(results)
    total_detections = 0
    detections_by_class = {}
    
    for result in results:
        if result.boxes is not None:
            num_dets = len(result.boxes)
            total_detections += num_dets
            
            # Get class names
            cls = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            for i, c in enumerate(cls):
                class_name = result.names.get(int(c), f"class_{c}")
                if class_name not in detections_by_class:
                    detections_by_class[class_name] = {"count": 0, "avg_conf": []}
                detections_by_class[class_name]["count"] += 1
                detections_by_class[class_name]["avg_conf"].append(confs[i])
    
    print("\n" + "=" * 60)
    print("Inference Summary")
    print("=" * 60)
    print(f"📊 Total Images Processed: {total_images}")
    print(f"📊 Total Detections: {total_detections}")
    print(f"📊 Average Detections per Image: {total_detections/total_images:.2f}")
    
    if verbose and detections_by_class:
        print("\n📋 Detections by Class:")
        for class_name, data in sorted(detections_by_class.items()):
            avg_conf = sum(data["avg_conf"]) / len(data["avg_conf"]) if data["avg_conf"] else 0
            print(f"  {class_name}: {data['count']} detections (avg conf: {avg_conf:.4f})")
    
    return {
        "total_images": total_images,
        "total_detections": total_detections,
        "detections_by_class": detections_by_class
    }

def save_predictions_json(results, output_dir):
    """Save predictions to JSON file."""
    import json
    
    predictions = []
    for result in results:
        if result.boxes is not None:
            img_path = result.path
            img_name = os.path.basename(img_path)
            boxes = result.boxes
            
            for i in range(len(boxes)):
                pred = {
                    "image": img_name,
                    "detected_class": result.names[int(boxes.cls[i])],
                    "confidence": float(boxes.conf[i]),
                    "bounding_box": {
                        "x1": float(boxes.xyxy[i][0]),
                        "y1": float(boxes.xyxy[i][1]),
                        "x2": float(boxes.xyxy[i][2]),
                        "y2": float(boxes.xyxy[i][3])
                    }
                }
                predictions.append(pred)
    
    json_path = os.path.join(output_dir, "predictions.json")
    with open(json_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\n📁 Predictions saved to: {json_path}")
    return json_path

def main():
    """Main function."""
    args = parse_args()
    
    print("\n🕐 Starting test inference...")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check device
    device = check_device()
    
    # Validate model path
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("Please train the model first or provide a valid model path.")
        return
    
    # Load model
    print(f"\n📦 Loading model from: {args.model}")
    model = YOLO(args.model)
    
    # Run inference
    results = run_inference(
        model=model,
        source=args.source,
        output_dir=args.output,
        conf=args.conf,
        iou=args.iou,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        show_labels=args.show_labels,
        verbose=args.verbose
    )
    
    # Print summary
    summary = print_inference_summary(results, verbose=args.verbose)
    
    # Save predictions to JSON
    save_predictions_json(results, args.output)
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    print(f"📁 Predictions saved to: {args.output}")
    print(f"📁 Label files saved to: {args.output}/labels/")
    
    # Return summary for further processing
    return summary

if __name__ == "__main__":
    try:
        summary = main()
        print("\n✅ Test inference completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        raise
