"""
CNN Leaf Classification Training Script
========================================
This script trains a CNN model for leaf classification.

Usage:
    python cnn_training.py
    python cnn_training.py --model custom --epochs 20 --batch 32
    python cnn_training.py --model efficient --pretrained --epochs 30

Comparison with YOLOv11:
    This CNN model provides image-level classification (single label per image),
    while YOLOv11 provides object detection (multiple objects with bounding boxes).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import json
from datetime import datetime
from pathlib import Path

# Import custom CNN model
from cnn_model import create_model, count_parameters


# Configuration
DATA_DIR = "Deteksi_Daun.v4i.yolov11"
TRAIN_DIR = os.path.join(DATA_DIR, "train", "images")
VAL_DIR = os.path.join(DATA_DIR, "valid", "images")
TEST_DIR = os.path.join(DATA_DIR, "test", "images")

# Class names (same as YOLOv11)
CLASS_NAMES = ['daun jeruk', 'daun kari', 'daun kunyit', 'daun pandan', 'daun salam']
NUM_CLASSES = len(CLASS_NAMES)

# Model configuration
MODEL_TYPES = ['custom', 'efficient', 'resnet']


class LeafDataset(Dataset):
    """Custom Dataset for Leaf Classification."""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Initialize dataset.
        
        Args:
            image_paths (list): List of image file paths
            labels (list): List of corresponding labels
            transform (callable): Optional transform to be applied on image
        """
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


def get_image_files_and_labels(data_dir):
    """
    Get image files and labels from directory structure.
    Assumes subdirectories are named after classes.
    
    Args:
        data_dir (str): Path to data directory
        
    Returns:
        tuple: (image_paths, labels)
    """
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    image_paths.append(os.path.join(class_dir, img_file))
                    labels.append(class_idx)
        else:
            # Try finding images in root of data_dir with label files
            # This handles YOLO format where labels are in separate files
            print(f"Warning: Class directory not found: {class_dir}")
            print("Trying alternative dataset structure...")
            return None, None
    
    return image_paths, labels


def get_yolo_dataset(image_dir, label_dir):
    """
    Get dataset from YOLO format (images and labels in separate directories).
    
    Args:
        image_dir (str): Path to images directory
        label_dir (str): Path to labels directory (txt files)
        
    Returns:
        tuple: (image_paths, labels)
    """
    image_paths = []
    labels = []
    
    # Get class from label file (assumes single label per image in YOLO format)
    for img_file in os.listdir(image_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            base_name = os.path.splitext(img_file)[0]
            label_file = base_name + '.txt'
            label_path = os.path.join(label_dir, label_file)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # Get class from first detection (most common class)
                        parts = lines[0].strip().split()
                        if len(parts) >= 5:
                            class_idx = int(parts[0])
                            image_paths.append(os.path.join(image_dir, img_file))
                            labels.append(class_idx)
    
    return image_paths, labels


def create_dataloaders(batch_size=16, img_size=224, val_split=0.2, random_state=42):
    """
    Create training and validation dataloaders.
    
    Args:
        batch_size (int): Batch size for training
        img_size (int): Input image size (square)
        val_split (float): Fraction of training data for validation
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_to_idx)
    """
    # Data transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    test_transform = val_transform
    
    # Get YOLO format dataset
    train_img_dir = os.path.join(DATA_DIR, "train", "images")
    train_lbl_dir = os.path.join(DATA_DIR, "train", "labels")
    val_img_dir = os.path.join(DATA_DIR, "valid", "images")
    val_lbl_dir = os.path.join(DATA_DIR, "valid", "labels")
    test_img_dir = os.path.join(DATA_DIR, "test", "images")
    test_lbl_dir = os.path.join(DATA_DIR, "test", "labels")
    
    # Load datasets
    print("📂 Loading datasets...")
    
    train_paths, train_labels = get_yolo_dataset(train_img_dir, train_lbl_dir)
    val_paths, val_labels = get_yolo_dataset(val_img_dir, val_lbl_dir)
    test_paths, test_labels = get_yolo_dataset(test_img_dir, test_lbl_dir)
    
    # If YOLO format not found, try class directory format
    if train_paths is None or len(train_paths) == 0:
        train_paths, train_labels = get_image_files_and_labels(TRAIN_DIR)
        val_paths, val_labels = get_image_files_and_labels(VAL_DIR)
        test_paths, test_labels = get_image_files_and_labels(TEST_DIR)
    
    print(f"   Training samples: {len(train_paths)}")
    print(f"   Validation samples: {len(val_paths)}")
    print(f"   Test samples: {len(test_paths)}")
    
    # Create datasets
    train_dataset = LeafDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = LeafDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = LeafDataset(test_paths, test_labels, transform=test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="[Validate]")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def test_model(model, test_loader, device, class_names):
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="[Test]"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
    
    # Calculate metrics
    test_acc = 100. * correct / total
    
    # Classification report
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, 
                                 target_names=class_names,
                                 digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix - CNN Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('runs/cnn_confusion_matrix.png', dpi=150)
    plt.close()
    
    return test_acc, all_preds, all_labels, cm


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot training history."""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_cnn_model(model_type='custom', epochs=20, batch_size=16, 
                    learning_rate=0.001, img_size=224, pretrained=True,
                    dropout_rate=0.5, save_dir='runs/cnn'):
    """
    Train CNN model for leaf classification.
    
    Args:
        model_type (str): Type of model (custom, efficient, resnet)
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        img_size (int): Input image size
        pretrained (bool): Use pretrained weights
        dropout_rate (float): Dropout rate
        save_dir (str): Directory to save model and results
    """
    print("=" * 60)
    print("CNN Leaf Classification Training")
    print("=" * 60)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 Using CPU")
    
    # Create model
    print(f"\n📦 Creating {model_type.upper()} model...")
    model = create_model(
        model_type=model_type,
        num_classes=NUM_CLASSES,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
    model.to(device)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=batch_size,
        img_size=img_size
    )
    
    # Loss function with class weights (if needed for imbalanced data)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer with different settings based on model type
    if model_type == 'custom':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        # For pretrained models, different learning rate for backbone and head
        optimizer = optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': learning_rate / 10},
            {'params': model.backbone.classifier.parameters(), 'lr': learning_rate}
        ], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    print(f"\n🔧 Training Configuration:")
    print(f"   - Model type: {model_type}")
    print(f"   - Epochs: {epochs}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Image size: {img_size}x{img_size}")
    print(f"   - Pretrained: {pretrained}")
    print()
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'class_names': CLASS_NAMES,
                'model_type': model_type
            }, model_path)
            print(f"   ✅ New best model saved! (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"   ⏳ Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered!")
                break
    
    # Plot training history
    history_path = os.path.join(save_dir, 'training_history.png')
    plot_training_history(train_losses, val_losses, train_accs, val_accs, history_path)
    print(f"\n📈 Training history saved to: {history_path}")
    
    # Test the model
    print("\n" + "=" * 60)
    print("Final Model Evaluation")
    print("=" * 60)
    
    # Load best model for testing
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_acc, preds, labels, cm = test_model(model, test_loader, device, CLASS_NAMES)
    
    # Save results summary
    results = {
        'model_type': model_type,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'epochs_trained': len(train_losses),
        'class_names': CLASS_NAMES,
        'training_timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(save_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {results_path}")
    print(f"📦 Model saved to: {save_dir}/best_model.pth")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    
    return model, results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CNN for Leaf Classification")
    parser.add_argument("--model", type=str, default="custom",
                        choices=MODEL_TYPES,
                        help="Model type (custom, efficient, resnet)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Don't use pretrained weights")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate")
    parser.add_argument("--save-dir", type=str, default="runs/cnn",
                        help="Directory to save model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    try:
        model, results = train_cnn_model(
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            learning_rate=args.lr,
            img_size=args.img_size,
            pretrained=not args.no_pretrained,
            dropout_rate=args.dropout,
            save_dir=args.save_dir
        )
        
        print("\n" + "=" * 60)
        print("✅ Training completed successfully!")
        print("=" * 60)
        print(f"\n📊 Final Results:")
        print(f"   Model Type: {results['model_type']}")
        print(f"   Test Accuracy: {results['test_acc']:.2f}%")
        print(f"   Best Validation Accuracy: {results['best_val_acc']:.2f}%")
        print(f"   Total Parameters: {results['total_params']:,}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        raise
