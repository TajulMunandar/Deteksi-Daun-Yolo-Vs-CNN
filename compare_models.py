"""
YOLOv11 vs CNN Model Comparison Script
======================================
This script provides comprehensive comparison between YOLOv11 (detection) 
and CNN (classification) models for leaf analysis.

Usage:
    python compare_models.py
    python compare_models.py --yolo-model runs/detect/leaf_detection/weights/best.pt
    python compare_models.py --cnn-model runs/cnn/best_model.pth
    python compare_models.py --compare-all --save-results

This comparison is designed for thesis research on leaf detection/classification.

Features:
    - Architecture comparison
    - Parameter count comparison
    - Inference time comparison
    - Accuracy metrics comparison
    - Per-class performance analysis
    - Visualization of results
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

# Import models
from ultralytics import YOLO
from cnn_model import create_model, load_model, count_parameters


# Configuration
DATA_DIR = "Deteksi_Daun.v4i.yolov11"
CLASS_NAMES = ['daun jeruk', 'daun kari', 'daun kunyit', 'daun pandan', 'daun salam']
NUM_CLASSES = len(CLASS_NAMES)


class YOLOClassificationDataset(Dataset):
    """Dataset for evaluating YOLO as classifier."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path


def get_yolo_dataset(image_dir, label_dir):
    """Get dataset from YOLO format."""
    image_paths = []
    labels = []
    
    for img_file in os.listdir(image_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            base_name = os.path.splitext(img_file)[0]
            label_file = base_name + '.txt'
            label_path = os.path.join(label_dir, label_file)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        parts = lines[0].strip().split()
                        if len(parts) >= 5:
                            class_idx = int(parts[0])
                            image_paths.append(os.path.join(image_dir, img_file))
                            labels.append(class_idx)
    
    return image_paths, labels


def get_yolo_predictions(model, image_paths, labels, device, img_size=640):
    """Get predictions from YOLO model treating it as classifier."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    dataset = YOLOClassificationDataset(image_paths, labels, transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    all_preds = []
    all_labels = []
    all_confs = []
    
    for batch_idx, (inputs, targets, paths) in enumerate(tqdm(loader, desc="YOLO Inference")):
        inputs = inputs.to(device)
        
        # Run YOLO inference
        results = model.predict(inputs, verbose=False)
        
        for i, result in enumerate(results):
            target = targets[i].item()
            
            if result.boxes is not None and len(result.boxes) > 0:
                # Get most confident detection
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                # Find detection matching the ground truth or most confident
                best_idx = np.argmax(confs)
                pred = int(classes[best_idx])
                conf = float(confs[best_idx])
            else:
                # No detection, default to class 0
                pred = 0
                conf = 0.0
            
            all_preds.append(pred)
            all_labels.append(target)
            all_confs.append(conf)
    
    return np.array(all_preds), np.array(all_labels), np.array(all_confs)


def get_cnn_predictions(model, image_paths, labels, device, img_size=224):
    """Get predictions from CNN model."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = YOLOClassificationDataset(image_paths, labels, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for inputs, targets, _ in tqdm(loader, desc="CNN Inference"):
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def calculate_metrics(y_true, y_pred, y_proba=None, class_names=None):
    """Calculate comprehensive metrics for a model."""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    # Per-class metrics
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, 
                                  target_names=class_names,
                                  output_dict=True,
                                  zero_division=0)
    
    metrics['per_class'] = {
        class_names[i]: {
            'precision': report[class_names[i]]['precision'],
            'recall': report[class_names[i]]['recall'],
            'f1-score': report[class_names[i]]['f1-score'],
            'support': int(report[class_names[i]]['support'])
        }
        for i in range(len(class_names))
    }
    
    # ROC-AUC
    if y_proba is not None:
        try:
            y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
            metrics['roc_auc_ovr'] = roc_auc_score(y_true_bin, y_proba, multi_class='ovr')
        except:
            metrics['roc_auc_ovr'] = None
    
    return metrics


def measure_inference_time(model, image_paths, device, model_type='yolo', n_warmup=10, n_runs=30):
    """Measure average inference time."""
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])
    
    # Load a sample image
    sample_img = Image.open(image_paths[0]).convert('RGB')
    
    if model_type == 'yolo':
        input_tensor = transform(sample_img).unsqueeze(0).to(device)
        
        # Warmup
        for _ in range(n_warmup):
            _ = model.predict(input_tensor, verbose=False)
        
        # Measure
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model.predict(input_tensor, verbose=False)
            times.append(time.perf_counter() - start)
    
    else:  # cnn
        cnn_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = cnn_transform(sample_img).unsqueeze(0).to(device)
        
        model.eval()
        # Warmup
        for _ in range(n_warmup):
            with torch.no_grad():
                _ = model(input_tensor)
        
        # Measure
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(input_tensor)
            times.append(time.perf_counter() - start)
    
    return {
        'mean': np.mean(times) * 1000,  # ms
        'std': np.std(times) * 1000,   # ms
        'min': np.min(times) * 1000,   # ms
        'max': np.max(times) * 1000,   # ms
        'fps': 1 / np.mean(times)
    }


def plot_comparison_charts(yolo_metrics, cnn_metrics, save_dir='runs'):
    """Generate comparison charts."""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Overall metrics comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics_to_plot = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    yolo_values = [yolo_metrics[m] for m in metrics_to_plot]
    cnn_values = [cnn_metrics[m] for m in metrics_to_plot]
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, yolo_values, width, label='YOLOv11', color='#e41a1c', alpha=0.8)
    bars2 = axes[0].bar(x + width/2, cnn_values, width, label='CNN', color='#377eb8', alpha=0.8)
    
    axes[0].set_ylabel('Score')
    axes[0].set_title('Overall Metrics Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        axes[0].annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        axes[0].annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    
    # 2. Per-class F1-Score comparison
    per_class_metrics = CLASS_NAMES
    yolo_f1 = [yolo_metrics['per_class'][c]['f1-score'] for c in per_class_metrics]
    cnn_f1 = [cnn_metrics['per_class'][c]['f1-score'] for c in per_class_metrics]
    
    x = np.arange(len(per_class_metrics))
    
    bars1 = axes[1].bar(x - width/2, yolo_f1, width, label='YOLOv11', color='#e41a1c', alpha=0.8)
    bars2 = axes[1].bar(x + width/2, cnn_f1, width, label='CNN', color='#377eb8', alpha=0.8)
    
    axes[1].set_ylabel('F1-Score')
    axes[1].set_title('Per-Class F1-Score Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(per_class_metrics, rotation=45, ha='right')
    axes[1].legend()
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Confusion matrices side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    yolo_cm = np.array(yolo_metrics['confusion_matrix'])
    cnn_cm = np.array(cnn_metrics['confusion_matrix'])
    
    sns.heatmap(yolo_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    axes[0].set_title('YOLOv11 Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    sns.heatmap(cnn_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    axes[1].set_title('CNN Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrices_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Parameter count comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    
    params = ['Parameters\n(millions)', 'Inference Time\n(ms)', 'Model Size\n(MB)']
    yolo_vals = [yolo_metrics['total_params'] / 1e6, yolo_metrics['inference_time']['mean'], 0]
    cnn_vals = [cnn_metrics['total_params'] / 1e6, cnn_metrics['inference_time']['mean'], 0]
    
    # Note: Actual model sizes would need to be calculated from saved weights
    # Using estimated values
    yolo_model_size = os.path.getsize("runs/detect/leaf_detection/weights/best.pt") / (1024*1024) if os.path.exists("runs/detect/leaf_detection/weights/best.pt") else 0
    cnn_model_size = os.path.getsize("runs/cnn/best_model.pth") / (1024*1024) if os.path.exists("runs/cnn/best_model.pth") else 0
    
    x = np.arange(len(params))
    
    bars1 = ax.bar(x - width/2, [yolo_metrics['total_params']/1e6, yolo_metrics['inference_time']['mean'], yolo_model_size], 
                   width, label='YOLOv11', color='#e41a1c', alpha=0.8)
    bars2 = ax.bar(x + width/2, [cnn_metrics['total_params']/1e6, cnn_metrics['inference_time']['mean'], cnn_model_size], 
                   width, label='CNN', color='#377eb8', alpha=0.8)
    
    ax.set_ylabel('Value')
    ax.set_title('Model Complexity Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'complexity_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def compare_models(yolo_model_path, cnn_model_path, test_dir=None, label_dir=None, save_dir='runs'):
    """
    Comprehensive comparison between YOLOv11 and CNN models.
    
    Args:
        yolo_model_path: Path to YOLO model
        cnn_model_path: Path to CNN model
        test_dir: Test images directory
        label_dir: Test labels directory
        save_dir: Directory to save results
        
    Returns:
        dict: Comparison results
    """
    print("=" * 70)
    print("YOLOv11 vs CNN Model Comparison")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Device: {device}")
    
    # Load models
    print("\n📦 Loading models...")
    
    # YOLO
    if os.path.exists(yolo_model_path):
        yolo_model = YOLO(yolo_model_path)
        yolo_params, _ = count_parameters(yolo_model.model)
        print(f"   ✅ YOLO loaded: {yolo_model_path}")
    else:
        print(f"   ❌ YOLO model not found: {yolo_model_path}")
        return None
    
    # CNN
    if os.path.exists(cnn_model_path):
        from cnn_model import load_model as cnn_load_model
        cnn_model = cnn_load_model(cnn_model_path, model_type='custom', 
                                num_classes=NUM_CLASSES, device=device)
        cnn_params, _ = count_parameters(cnn_model)
        print(f"   ✅ CNN loaded: {cnn_model_path}")
    else:
        print(f"   ❌ CNN model not found: {cnn_model_path}")
        return None
    
    # Load test data
    if test_dir is None:
        test_dir = os.path.join(DATA_DIR, "test", "images")
    if label_dir is None:
        label_dir = os.path.join(DATA_DIR, "test", "labels")
    
    print(f"\n📂 Loading test data from: {test_dir}")
    test_paths, test_labels = get_yolo_dataset(test_dir, label_dir)
    print(f"   Test samples: {len(test_paths)}")
    
    # Get predictions
    print("\n🔍 Running YOLO inference...")
    yolo_preds, yolo_labels, yolo_confs = get_yolo_predictions(
        yolo_model, test_paths, test_labels, device
    )
    
    print("\n🔍 Running CNN inference...")
    cnn_preds, cnn_labels, cnn_probs = get_cnn_predictions(
        cnn_model, test_paths, test_labels, device
    )
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    yolo_metrics = calculate_metrics(yolo_labels, yolo_preds, yolo_confs, CLASS_NAMES)
    cnn_metrics = calculate_metrics(cnn_labels, cnn_preds, cnn_probs, CLASS_NAMES)
    
    # Add model info
    yolo_metrics['total_params'] = yolo_params
    cnn_metrics['total_params'] = cnn_params
    
    # Measure inference time
    print("\n⏱️  Measuring inference time...")
    yolo_metrics['inference_time'] = measure_inference_time(
        yolo_model, test_paths[:50], device, 'yolo'
    )
    cnn_metrics['inference_time'] = measure_inference_time(
        cnn_model, test_paths[:50], device, 'cnn'
    )
    
    # Print comparison results
    print("\n" + "=" * 70)
    print("Comparison Results")
    print("=" * 70)
    
    print("\n📊 Overall Performance:")
    print("-" * 70)
    print(f"{'Metric':<25} {'YOLOv11':>15} {'CNN':>15} {'Winner':>15}")
    print("-" * 70)
    
    comparisons = [
        ('Accuracy', 'accuracy'),
        ('Precision (macro)', 'precision_macro'),
        ('Recall (macro)', 'recall_macro'),
        ('F1-Score (macro)', 'f1_macro'),
        ('Inference Time (ms)', 'inference_time_mean'),
    ]
    
    for name, key in comparisons:
        yolo_val = yolo_metrics.get(key, 0)
        cnn_val = cnn_metrics.get(key, 0)
        
        if key == 'inference_time_mean':
            yolo_val = yolo_metrics['inference_time']['mean']
            cnn_val = cnn_metrics['inference_time']['mean']
            winner = "CNN" if cnn_val < yolo_val else "YOLO"
            print(f"{name:<25} {yolo_val:>15.2f} {cnn_val:>15.2f} {winner:>15}")
        else:
            winner = "YOLO" if yolo_val > cnn_val else "CNN" if cnn_val > yolo_val else "Tie"
            print(f"{name:<25} {yolo_val:>15.4f} {cnn_val:>15.4f} {winner:>15}")
    
    print("-" * 70)
    
    print("\n📏 Model Complexity:")
    print(f"   YOLOv11 Parameters: {yolo_params:,}")
    print(f"   CNN Parameters:     {cnn_params:,}")
    
    print("\n⏱️  Inference Time:")
    print(f"   YOLOv11: {yolo_metrics['inference_time']['mean']:.2f} ± {yolo_metrics['inference_time']['std']:.2f} ms")
    print(f"   CNN:     {cnn_metrics['inference_time']['mean']:.2f} ± {cnn_metrics['inference_time']['std']:.2f} ms")
    
    # Save results
    results = {
        'yolo': yolo_metrics,
        'cnn': cnn_metrics,
        'comparison': {
            'test_samples': len(test_paths),
            'num_classes': NUM_CLASSES,
            'class_names': CLASS_NAMES,
            'winner': 'CNN' if cnn_metrics['accuracy'] > yolo_metrics['accuracy'] else 'YOLO'
        }
    }
    
    results_path = os.path.join(save_dir, 'model_comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Results saved to: {results_path}")
    
    # Generate charts
    plot_comparison_charts(yolo_metrics, cnn_metrics, save_dir)
    print(f"📊 Charts saved to: {save_dir}/")
    
    return results


def print_architecture_comparison():
    """Print detailed architecture comparison."""
    print("\n" + "=" * 70)
    print("Architecture Comparison")
    print("=" * 70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                        YOLOv11 Architecture                         │
├─────────────────────────────────────────────────────────────────────┤
│ • Backbone: CSPDarknet-like architecture                             │
│ • Neck: PANet (Path Aggregation Network)                             │
│ • Head: Decoupled detection heads for box/class/centerness          │
│ • Task: Object Detection (bounding box + classification)            │
│ • Output: [x1, y1, x2, y2, confidence, class_probs] per detection  │
│ • Strengths:                                                        │
│   - Detects multiple objects in single forward pass                 │
│   - Provides localization with bounding boxes                       │
│   - Handles overlapping objects                                     │
│   - Real-time inference capability                                  │
│ • Weaknesses:                                                        │
│   - More parameters                                                  │
│   - Lower classification accuracy on single objects                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          CNN Architecture                            │
├─────────────────────────────────────────────────────────────────────┤
│ • Architecture: Custom CNN with residual connections                 │
│ • Components:                                                        │
│   - Initial conv layers for basic features                          │
│   - Residual blocks for deep features                               │
│   - Global Average Pooling                                          │
│   - Fully connected classifier head                                  │
│ • Task: Image Classification (single label per image)                │
│ • Output: Class probabilities for the entire image                   │
│ • Strengths:                                                        │
│   - Simpler architecture                                            │
│   - Faster inference for single objects                             │
│   - Higher accuracy for image-level classification                  │
│   - Fewer parameters (depending on backbone)                        │
│ • Weaknesses:                                                        │
│   - Single classification per image                                 │
│   - No localization information                                     │
│   - May miss small or overlapping objects                           │
└─────────────────────────────────────────────────────────────────────┘
""")

    print("\n" + "=" * 70)
    print("Key Differences Summary")
    print("=" * 70)
    
    print("""
┌─────────────────────┬────────────────────────┬────────────────────────┐
│     Aspect         │       YOLOv11          │         CNN            │
├─────────────────────┼────────────────────────┼────────────────────────┤
│ Primary Task       │ Object Detection       │ Image Classification   │
│ Output             │ Bounding Boxes +       │ Class Probabilities   │
│                    │ Class Labels           │                        │
│ Multi-Object       │ Yes                    │ No (single label)      │
│ Localization       │ Yes (bounding boxes)   │ No                     │
│ Speed              │ Real-time capable      │ Very fast              │
│ Accuracy (single)  │ Good                   │ Excellent              │
│ Architecture       │ Complex                │ Simpler                │
│ Parameters         │ More                   │ Fewer                  │
│ Use Case           │ Multiple leaves,       │ Single leaf image,     │
│                    │ localization needed    │ simple classification  │
└─────────────────────┴────────────────────────┴────────────────────────┘
""")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare YOLOv11 and CNN Models")
    
    parser.add_argument("--yolo-model", type=str,
                        default="runs/detect/leaf_detection/weights/best.pt",
                        help="Path to YOLOv11 model")
    parser.add_argument("--cnn-model", type=str,
                        default="runs/cnn/best_model.pth",
                        help="Path to CNN model")
    parser.add_argument("--test-dir", type=str, default=None,
                        help="Path to test images directory")
    parser.add_argument("--label-dir", type=str, default=None,
                        help="Path to test labels directory")
    parser.add_argument("--save-dir", type=str, default="runs",
                        help="Directory to save results")
    parser.add_argument("--show-arch", action="store_true",
                        help="Show architecture comparison")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Show architecture comparison
    if args.show_arch:
        print_architecture_comparison()
    
    # Run comparison
    try:
        results = compare_models(
            yolo_model_path=args.yolo_model,
            cnn_model_path=args.cnn_model,
            test_dir=args.test_dir,
            label_dir=args.label_dir,
            save_dir=args.save_dir
        )
        
        if results:
            print("\n" + "=" * 70)
            print("✅ Comparison completed successfully!")
            print("=" * 70)
            print(f"\n📊 Summary:")
            print(f"   Winner: {results['comparison']['winner']}")
            print(f"   Test Samples: {results['comparison']['test_samples']}")
            print(f"   YOLO Accuracy: {results['yolo']['accuracy']*100:.2f}%")
            print(f"   CNN Accuracy: {results['cnn']['accuracy']*100:.2f}%")
    
    except Exception as e:
        print(f"\n❌ Comparison failed: {str(e)}")
        raise
