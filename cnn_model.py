"""
CNN Model for Leaf Classification
=================================
Custom CNN architecture for leaf classification (5 classes).
Designed to compare with YOLOv11 for thesis research.

Classes:
    - daun jeruk (citrus leaf)
    - daun kari (curry leaf)  
    - daun kunyit (turmeric leaf)
    - daun pandan (pandan leaf)
    - daun salam (bay leaf)

Usage:
    from cnn_model import LeafCNN, create_model
    model = LeafCNN(num_classes=5)
    model = create_model(num_classes=5, model_type='efficient')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os


class LeafCNN(nn.Module):
    """
    Custom CNN Architecture for Leaf Classification.
    
    This CNN uses a multi-scale feature extraction approach with:
    - Initial convolutional layers for basic feature extraction
    - Residual-like blocks for deeper feature learning
    - Global Average Pooling for dimension reduction
    - Fully connected layers for classification
    """
    
    def __init__(self, num_classes=5, dropout_rate=0.5):
        """
        Initialize the CNN model.
        
        Args:
            num_classes (int): Number of leaf classes to classify
            dropout_rate (float): Dropout probability for regularization
        """
        super(LeafCNN, self).__init__()
        
        # Feature Extraction Layers
        # Block 1: Initial convolution - 224x224 -> 112x112
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Block 2: 112x112 -> 56x56
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(64)
        
        # Block 3: 56x56 -> 28x28
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(128)
        
        # Residual connection for block 3
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )
        
        # Block 4: 28x28 -> 14x14
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(256)
        
        # Residual connection for block 4
        self.shortcut4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        )
        
        # Block 5: 14x14 -> 7x7
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5_2 = nn.BatchNorm2d(512)
        
        # Residual connection for block 5
        self.shortcut5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512)
        )
        
        # Global Average Pooling
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Classification Head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, 3, 224, 224)
            
        Returns:
            torch.Tensor: Output logits of shape (batch, num_classes)
        """
        # Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # Block 2
        identity = x
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.bn2_2(self.conv2_2(x))
        x += identity
        x = self.relu(x)
        
        # Block 3
        identity = self.shortcut3(x)
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.bn3_2(self.conv3_2(x))
        x += identity
        x = self.relu(x)
        
        # Block 4
        identity = self.shortcut4(x)
        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.bn4_2(self.conv4_2(x))
        x += identity
        x = self.relu(x)
        
        # Block 5
        identity = self.shortcut5(x)
        x = self.relu(self.bn5_1(self.conv5_1(x)))
        x = self.bn5_2(self.conv5_2(x))
        x += identity
        x = self.relu(x)
        
        # Global Average Pooling
        x = self.global_avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


class LeafEfficientNet(nn.Module):
    """
    EfficientNet-based model for leaf classification.
    Uses pretrained EfficientNet-B0 as backbone with custom head.
    """
    
    def __init__(self, num_classes=5, pretrained=True):
        """
        Initialize EfficientNet-based model.
        
        Args:
            num_classes (int): Number of leaf classes
            pretrained (bool): Whether to use pretrained weights
        """
        super(LeafEfficientNet, self).__init__()
        
        # Load pretrained EfficientNet-B0
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        # Get the number of features from the backbone
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        return self.backbone(x)


class LeafResNet(nn.Module):
    """
    ResNet-based model for leaf classification.
    Uses pretrained ResNet-34 as backbone with custom head.
    """
    
    def __init__(self, num_classes=5, pretrained=True):
        """
        Initialize ResNet-based model.
        
        Args:
            num_classes (int): Number of leaf classes
            pretrained (bool): Whether to use pretrained weights
        """
        super(LeafResNet, self).__init__()
        
        # Load pretrained ResNet-34
        if pretrained:
            weights = models.ResNet34_Weights.IMAGENET1K_V1
            self.backbone = models.resnet34(weights=weights)
        else:
            self.backbone = models.resnet34()
        
        # Get the number of features from the backbone
        num_features = self.backbone.fc.in_features
        
        # Replace fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        return self.backbone(x)


def create_model(model_type='custom', num_classes=5, pretrained=True, dropout_rate=0.5):
    """
    Factory function to create leaf classification model.
    
    Args:
        model_type (str): Type of model to create
            - 'custom': Custom CNN architecture
            - 'efficient': EfficientNet-B0
            - 'resnet': ResNet-34
        num_classes (int): Number of leaf classes
        pretrained (bool): Whether to use pretrained weights (for transfer learning)
        dropout_rate (float): Dropout probability
        
    Returns:
        nn.Module: Initialized model
        
    Examples:
        >>> model = create_model('custom', num_classes=5)
        >>> model = create_model('efficient', num_classes=5)
        >>> model = create_model('resnet', num_classes=5)
    """
    if model_type == 'custom':
        return LeafCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_type == 'efficient':
        return LeafEfficientNet(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'resnet':
        return LeafResNet(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'custom', 'efficient', or 'resnet'")


def load_model(model_path, model_type='custom', num_classes=5, device=None):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the saved model weights
        model_type (str): Type of model architecture
        num_classes (int): Number of classes
        device (torch.device): Device to load model on
        
    Returns:
        nn.Module: Loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_model(model_type=model_type, num_classes=num_classes)
    
    # Load weights
    if os.path.exists(model_path):
        print(f"📦 Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle both direct state_dict and checkpoint wrapper
        if 'model_state_dict' in checkpoint:
            # Checkpoint was saved with wrapper (from training script)
            state_dict = checkpoint['model_state_dict']
            if 'epoch' in checkpoint:
                print(f"   Trained for {checkpoint.get('epoch', 'unknown')} epochs")
            if 'val_acc' in checkpoint:
                print(f"   Validation accuracy: {checkpoint['val_acc']:.2f}%")
        else:
            # Direct state_dict
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully!")
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    return model


def count_parameters(model):
    """
    Count total and trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    # Test model creation
    print("=" * 60)
    print("Testing CNN Model Architectures")
    print("=" * 60)
    
    # Test input tensor
    test_input = torch.randn(2, 3, 224, 224)
    
    # Test Custom CNN
    print("\n📊 Custom CNN Model:")
    custom_model = create_model('custom', num_classes=5)
    total, trainable = count_parameters(custom_model)
    print(f"   Total parameters: {total:,}")
    print(f"   Trainable parameters: {trainable:,}")
    output = custom_model(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test EfficientNet
    print("\n📊 EfficientNet-B0 Model:")
    efficient_model = create_model('efficient', num_classes=5)
    total, trainable = count_parameters(efficient_model)
    print(f"   Total parameters: {total:,}")
    print(f"   Trainable parameters: {trainable:,}")
    output = efficient_model(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test ResNet
    print("\n📊 ResNet-34 Model:")
    resnet_model = create_model('resnet', num_classes=5)
    total, trainable = count_parameters(resnet_model)
    print(f"   Total parameters: {total:,}")
    print(f"   Trainable parameters: {trainable:,}")
    output = resnet_model(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    
    print("\n✅ All models tested successfully!")
