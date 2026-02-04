"""
3D U-Net / V-Net Implementation with Residual Connections
Volumetric CT Scan Analysis for Pulmonary Nodule Detection

Architecture based on:
- Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- Milletari et al. "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"

Input: 3D patches of size 64x64x32
Output: Binary segmentation mask for nodule detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    """
    Residual block with two 3D convolutions and skip connection.
    Uses PReLU activation and batch normalization for stable training.
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.prelu1 = nn.PReLU(out_channels)
        
        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.prelu2 = nn.PReLU(out_channels)
        
        # Skip connection with 1x1 conv if dimensions change
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity
        out = self.prelu2(out)
        
        return out


class EncoderBlock(nn.Module):
    """Encoder block with residual convolution and max pooling."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.residual = ResidualBlock3D(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> tuple:
        features = self.residual(x)
        pooled = self.pool(features)
        return pooled, features


class DecoderBlock(nn.Module):
    """Decoder block with transposed convolution and skip connection."""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channels, in_channels // 2,
            kernel_size=2, stride=2
        )
        self.residual = ResidualBlock3D(in_channels // 2 + skip_channels, out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        
        # Handle size mismatch due to odd dimensions
        if x.shape != skip.shape:
            diff_d = skip.shape[2] - x.shape[2]
            diff_h = skip.shape[3] - x.shape[3]
            diff_w = skip.shape[4] - x.shape[4]
            x = F.pad(x, [
                diff_w // 2, diff_w - diff_w // 2,
                diff_h // 2, diff_h - diff_h // 2,
                diff_d // 2, diff_d - diff_d // 2
            ])
        
        x = torch.cat([x, skip], dim=1)
        x = self.residual(x)
        return x


class UNet3D(nn.Module):
    """
    3D U-Net with Residual Connections for Volumetric Segmentation.
    
    Architecture:
    - Encoder: 4 downsampling stages (1→32→64→128→256 channels)
    - Bottleneck: 512 channels
    - Decoder: 4 upsampling stages with skip connections
    - Output: Binary segmentation mask via sigmoid
    
    Memory Optimization:
    - Designed for 64x64x32 patch input to fit in GPU memory
    - Uses batch normalization for training stability
    - PReLU activation for better gradient flow
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, init_features: int = 32):
        super().__init__()
        
        features = init_features
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.PReLU(features)
        )
        
        # Encoder path
        self.encoder1 = EncoderBlock(features, features * 2)      # 32 -> 64
        self.encoder2 = EncoderBlock(features * 2, features * 4)  # 64 -> 128
        self.encoder3 = EncoderBlock(features * 4, features * 8)  # 128 -> 256
        self.encoder4 = EncoderBlock(features * 8, features * 16) # 256 -> 512
        
        # Bottleneck
        self.bottleneck = ResidualBlock3D(features * 16, features * 16)
        
        # Decoder path - skip_channels must match encoder output channels at each level
        # encoder4 outputs 512ch (skip4), encoder3 outputs 256ch (skip3), etc.
        self.decoder4 = DecoderBlock(features * 16, features * 16, features * 8)  # 512 + 512 -> 256
        self.decoder3 = DecoderBlock(features * 8, features * 8, features * 4)    # 256 + 256 -> 128
        self.decoder2 = DecoderBlock(features * 4, features * 4, features * 2)    # 128 + 128 -> 64
        self.decoder1 = DecoderBlock(features * 2, features * 2, features)        # 64 + 64 -> 32
        
        # Output convolution
        self.out_conv = nn.Conv3d(features, out_channels, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (B, 1, D, H, W) where D=32, H=64, W=64
        
        Returns:
            Output tensor of shape (B, 1, D, H, W) with sigmoid activation
        """
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        
        # Output
        x = self.out_conv(x)
        x = torch.sigmoid(x)
        
        return x


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined BCE and Dice Loss for better training stability."""
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def get_model(device: str = 'cpu', pretrained_path: str = None) -> UNet3D:
    """
    Factory function to create and optionally load a pretrained model.
    
    Args:
        device: Device to load model on ('cpu' or 'cuda')
        pretrained_path: Path to pretrained weights (optional)
    
    Returns:
        UNet3D model instance
    """
    model = UNet3D(in_channels=1, out_channels=1, init_features=32)
    
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    return model


# Test the model architecture
if __name__ == "__main__":
    print("Testing 3D U-Net Architecture...")
    print("-" * 50)
    
    # Create model
    model = UNet3D(in_channels=1, out_channels=1, init_features=32)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Test forward pass with 64x64x32 input
    print("\nTesting forward pass...")
    dummy_input = torch.randn(1, 1, 32, 64, 64)  # (B, C, D, H, W)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print("\n✓ Model architecture test passed!")
