"""
CNN Leaf Classification Evaluation Script
=========================================
This script evaluates a trained CNN model for leaf classification.

Usage:
    python cnn_evaluate.py --model runs/cnn/best_model.pth
    python cnn_evaluate.py --model runs/cnn/best_model.pth --plot
    python cnn_evaluate.py --model runs/cnn/best_model.pth --compare
"""

import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, cohen_kappa_score,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path

from cnn_model import create_model, load_model


# Configuration
DATA_DIR = "Deteksi_Daun.v4i.yolov11"
CLASS_NAMES = ['daun jeruk', 'daun kari', 'daun kunyit', 'daun pandan', 'daun salam']
NUM_CLASSES = len(CLASS_NAMES)


class SingleImageDataset(Dataset):
    """Dataset for single image evaluation."""
    
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_path


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


def create_test_loader(test_dir, label_dir, batch_size=16, img_size=224):
    """Create test dataloader."""
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_paths, test_labels = get_yolo_dataset(test_dir, label_dir)
    
    if len(test_paths) == 0:
        raise ValueError(f"No test images found in {test_dir}")
    
    test_dataset = SingleImageDataset(test_paths, transform=test_transform)
    
    # Get full labels for evaluation
    test_full_dataset = DatasetWrapper(test_paths, test_labels, test_transform)
    
    return DataLoader(test_full_dataset, batch_size=batch_size, shuffle=False), test_paths


class DatasetWrapper(Dataset):
    """Wrapper to include labels with images."""
    
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
        
        return image, label


def evaluate_model(model_path, model_type='custom', test_dir=None, label_dir=None, 
                  batch_size=16, img_size=224, device=None, save_results=True,
                  plot_dir='runs/cnn'):
    """
    Evaluate CNN model on test set.
    
    Args:
        model_path (str): Path to trained model weights
        model_type (str): Type of model architecture
        test_dir (str): Path to test images directory
        label_dir (str): Path to test labels directory
        batch_size (int): Batch size for evaluation
        img_size (int): Input image size
        device (torch.device): Device to use
        save_results (bool): Save results to JSON
        plot_dir (str): Directory to save plots
        
    Returns:
        dict: Evaluation metrics
    """
    print("=" * 60)
    print("CNN Leaf Classification Evaluation")
    print("=" * 60)
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 Using CPU")
    
    # Set default directories
    if test_dir is None:
        test_dir = os.path.join(DATA_DIR, "test", "images")
    if label_dir is None:
        label_dir = os.path.join(DATA_DIR, "test", "labels")
    
    # Load model
    print(f"\n📦 Loading model from: {model_path}")
    # Import here to use the fixed load_model
    from cnn_model import load_model as cnn_load_model
    model = cnn_load_model(model_path, model_type=model_type, 
                           num_classes=NUM_CLASSES, device=device)
    model.eval()
    
    # Create test dataloader
    print(f"\n📂 Loading test data from: {test_dir}")
    test_loader, test_paths = create_test_loader(
        test_dir, label_dir, batch_size, img_size
    )
    print(f"   Test samples: {len(test_paths)}")
    
    # Evaluation
    all_preds = []
    all_labels = []
    all_probs = []
    all_images = []
    
    print("\n🔍 Running evaluation...")
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_images.extend(inputs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(all_labels, all_preds)
    metrics['precision_macro'] = precision_score(all_labels, all_preds, average='macro')
    metrics['precision_weighted'] = precision_score(all_labels, all_preds, average='weighted')
    metrics['recall_macro'] = recall_score(all_labels, all_preds, average='macro')
    metrics['recall_weighted'] = recall_score(all_labels, all_preds, average='weighted')
    metrics['f1_macro'] = f1_score(all_labels, all_preds, average='macro')
    metrics['f1_weighted'] = f1_score(all_labels, all_preds, average='weighted')
    metrics['cohen_kappa'] = cohen_kappa_score(all_labels, all_preds)
    
    # Per-class metrics
    report = classification_report(all_labels, all_preds, 
                                   target_names=CLASS_NAMES,
                                   output_dict=True,
                                   digits=4)
    
    metrics['per_class'] = {
        CLASS_NAMES[i]: {
            'precision': report[CLASS_NAMES[i]]['precision'],
            'recall': report[CLASS_NAMES[i]]['recall'],
            'f1-score': report[CLASS_NAMES[i]]['f1-score'],
            'support': int(report[CLASS_NAMES[i]]['support'])
        }
        for i in range(NUM_CLASSES)
    }
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    metrics['confusion_matrix'] = cm.tolist()
    
    # ROC-AUC (One-vs-Rest)
    try:
        all_labels_bin = label_binarize(all_labels, classes=range(NUM_CLASSES))
        metrics['roc_auc_ovr'] = roc_auc_score(all_labels_bin, all_probs, multi_class='ovr')
        metrics['roc_auc_ovo'] = roc_auc_score(all_labels_bin, all_probs, multi_class='ovo')
    except Exception as e:
        print(f"⚠️  Could not compute ROC-AUC: {e}")
        metrics['roc_auc_ovr'] = None
        metrics['roc_auc_ovo'] = None
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy:          {metrics['accuracy']*100:.2f}%")
    print(f"   Precision (macro): {metrics['precision_macro']*100:.2f}%")
    print(f"   Recall (macro):     {metrics['recall_macro']*100:.2f}%")
    print(f"   F1 Score (macro):  {metrics['f1_macro']*100:.2f}%")
    print(f"   Cohen's Kappa:     {metrics['cohen_kappa']:.4f}")
    
    if metrics['roc_auc_ovr']:
        print(f"   ROC-AUC (OVR):     {metrics['roc_auc_ovr']:.4f}")
        print(f"   ROC-AUC (OVO):     {metrics['roc_auc_ovo']:.4f}")
    
    print(f"\n📋 Per-Class Performance:")
    print("-" * 60)
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 60)
    
    for i, class_name in enumerate(CLASS_NAMES):
        class_metrics = metrics['per_class'][class_name]
        print(f"{class_name:<15} {class_metrics['precision']:>10.4f} "
              f"{class_metrics['recall']:>10.4f} {class_metrics['f1-score']:>10.4f} "
              f"{class_metrics['support']:>10}")
    
    print("-" * 60)
    
    # Save plots
    os.makedirs(plot_dir, exist_ok=True)
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix - CNN Model', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(plot_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📁 Confusion matrix saved to: {cm_path}")
    
    # Normalized Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title('Normalized Confusion Matrix - CNN Model', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    cm_norm_path = os.path.join(plot_dir, 'confusion_matrix_normalized.png')
    plt.savefig(cm_norm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📁 Normalized confusion matrix saved to: {cm_norm_path}")
    
    # Per-class metrics bar chart
    plt.figure(figsize=(12, 6))
    x = np.arange(NUM_CLASSES)
    width = 0.25
    
    precision_vals = [metrics['per_class'][c]['precision'] for c in CLASS_NAMES]
    recall_vals = [metrics['per_class'][c]['recall'] for c in CLASS_NAMES]
    f1_vals = [metrics['per_class'][c]['f1-score'] for c in CLASS_NAMES]
    
    bars1 = plt.bar(x - width, precision_vals, width, label='Precision', color='#2196F3')
    bars2 = plt.bar(x, recall_vals, width, label='Recall', color='#4CAF50')
    bars3 = plt.bar(x + width, f1_vals, width, label='F1-Score', color='#FF9800')
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Per-Class Performance Metrics', fontsize=14)
    plt.xticks(x, CLASS_NAMES, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    per_class_path = os.path.join(plot_dir, 'per_class_metrics.png')
    plt.savefig(per_class_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📁 Per-class metrics saved to: {per_class_path}")
    
    # ROC Curves
    if metrics['roc_auc_ovr']:
        plt.figure(figsize=(10, 8))
        
        all_labels_bin = label_binarize(all_labels, classes=range(NUM_CLASSES))
        
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
        
        for i, (class_name, color) in enumerate(zip(CLASS_NAMES, colors)):
            fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
            auc = roc_auc_score(all_labels_bin[:, i], all_probs[:, i])
            plt.plot(fpr, tpr, color=color, lw=2,
                     label=f'{class_name} (AUC = {auc:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - CNN Model')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        roc_path = os.path.join(plot_dir, 'roc_curves.png')
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📁 ROC curves saved to: {roc_path}")
    
    # Save results to JSON
    if save_results:
        results_path = os.path.join(plot_dir, 'evaluation_results.json')
        
        # Convert numpy types to Python types for JSON serialization
        json_metrics = {
            'model_path': model_path,
            'model_type': model_type,
            'test_samples': len(all_labels),
            'accuracy': float(metrics['accuracy']),
            'precision_macro': float(metrics['precision_macro']),
            'recall_macro': float(metrics['recall_macro']),
            'f1_macro': float(metrics['f1_macro']),
            'cohen_kappa': float(metrics['cohen_kappa']),
            'roc_auc_ovr': float(metrics['roc_auc_ovr']) if metrics['roc_auc_ovr'] else None,
            'roc_auc_ovo': float(metrics['roc_auc_ovo']) if metrics['roc_auc_ovo'] else None,
            'confusion_matrix': metrics['confusion_matrix'],
            'per_class': metrics['per_class']
        }
        
        with open(results_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        print(f"📁 Results saved to: {results_path}")
    
    return metrics, all_preds, all_labels, all_probs


def predict_single_image(model, image_path, device, class_names, img_size=224):
    """
    Predict single image using CNN model.
    
    Args:
        model (nn.Module): Trained model
        image_path (str): Path to image
        device (torch.device): Device
        class_names (list): List of class names
        img_size (int): Image size
        
    Returns:
        dict: Prediction result
    """
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = outputs.max(1)
    
    # Get class name
    class_name = class_names[predicted.item()]
    confidence = confidence.item()
    
    # Get all probabilities
    probabilities = {
        class_names[i]: prob.item() 
        for i, prob in enumerate(probs[0])
    }
    
    return {
        'predicted_class': class_name,
        'confidence': confidence,
        'probabilities': probabilities
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate CNN for Leaf Classification")
    
    parser.add_argument("--model", type=str, 
                        default="runs/cnn/best_model.pth",
                        help="Path to trained model")
    parser.add_argument("--model-type", type=str, default="custom",
                        choices=['custom', 'efficient', 'resnet'],
                        help="Model architecture type")
    parser.add_argument("--test-dir", type=str, default=None,
                        help="Path to test images directory")
    parser.add_argument("--label-dir", type=str, default=None,
                        help="Path to test labels directory")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--plot-dir", type=str, default="runs/cnn",
                        help="Directory to save plots")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to single image for prediction")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Single image prediction
    if args.image:
        print("=" * 60)
        print("Single Image Prediction")
        print("=" * 60)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model(args.model, model_type=args.model_type,
                           num_classes=NUM_CLASSES, device=device)
        
        result = predict_single_image(
            model, args.image, device, CLASS_NAMES, args.img_size
        )
        
        print(f"\n📷 Image: {args.image}")
        print(f"   Predicted Class: {result['predicted_class']}")
        print(f"   Confidence: {result['confidence']*100:.2f}%")
        print("\n📊 Class Probabilities:")
        for class_name, prob in sorted(result['probabilities'].items(), 
                                        key=lambda x: x[1], reverse=True):
            print(f"   {class_name}: {prob*100:.2f}%")
    
    # Full evaluation
    else:
        try:
            metrics, preds, labels, probs = evaluate_model(
                model_path=args.model,
                model_type=args.model_type,
                test_dir=args.test_dir,
                label_dir=args.label_dir,
                batch_size=args.batch,
                img_size=args.img_size,
                plot_dir=args.plot_dir
            )
            
            print("\n" + "=" * 60)
            print("✅ Evaluation completed successfully!")
            print("=" * 60)
            print(f"\n📊 Summary:")
            print(f"   Accuracy: {metrics['accuracy']*100:.2f}%")
            print(f"   F1-Score (macro): {metrics['f1_macro']*100:.2f}%")
            
        except Exception as e:
            print(f"\n❌ Evaluation failed: {str(e)}")
            raise
