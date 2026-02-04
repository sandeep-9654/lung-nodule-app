"""
Unit tests for UNet3D model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from models.UNet3D import UNet3D, ResidualBlock3D, DiceLoss, CombinedLoss


class TestResidualBlock3D:
    """Tests for ResidualBlock3D."""
    
    def test_same_channels(self):
        """Test residual block with same input/output channels."""
        block = ResidualBlock3D(32, 32)
        x = torch.randn(1, 32, 8, 16, 16)
        out = block(x)
        assert out.shape == x.shape
    
    def test_different_channels(self):
        """Test residual block with different input/output channels."""
        block = ResidualBlock3D(32, 64)
        x = torch.randn(1, 32, 8, 16, 16)
        out = block(x)
        assert out.shape == (1, 64, 8, 16, 16)
    
    def test_stride(self):
        """Test residual block with stride."""
        block = ResidualBlock3D(32, 64, stride=2)
        x = torch.randn(1, 32, 8, 16, 16)
        out = block(x)
        assert out.shape == (1, 64, 4, 8, 8)


class TestUNet3D:
    """Tests for UNet3D model."""
    
    def test_forward_pass(self):
        """Test forward pass with expected input size."""
        model = UNet3D(in_channels=1, out_channels=1, init_features=32)
        x = torch.randn(1, 1, 32, 64, 64)
        
        model.eval()
        with torch.no_grad():
            out = model(x)
        
        assert out.shape == x.shape
        assert out.min() >= 0 and out.max() <= 1  # Sigmoid output
    
    def test_smaller_input(self):
        """Test with smaller input size."""
        model = UNet3D(in_channels=1, out_channels=1, init_features=16)
        x = torch.randn(1, 1, 16, 32, 32)
        
        model.eval()
        with torch.no_grad():
            out = model(x)
        
        assert out.shape == x.shape
    
    def test_batch_size(self):
        """Test with different batch sizes."""
        model = UNet3D(in_channels=1, out_channels=1, init_features=16)
        
        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 1, 16, 32, 32)
            model.eval()
            with torch.no_grad():
                out = model(x)
            assert out.shape[0] == batch_size
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        model = UNet3D(in_channels=1, out_channels=1, init_features=16)
        x = torch.randn(1, 1, 16, 32, 32, requires_grad=True)
        
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestLossFunctions:
    """Tests for loss functions."""
    
    def test_dice_loss_same(self):
        """Test Dice loss with identical inputs."""
        loss_fn = DiceLoss()
        pred = torch.ones(1, 1, 8, 8, 8)
        target = torch.ones(1, 1, 8, 8, 8)
        
        loss = loss_fn(pred, target)
        assert loss.item() < 0.01  # Should be close to 0
    
    def test_dice_loss_different(self):
        """Test Dice loss with different inputs."""
        loss_fn = DiceLoss()
        pred = torch.ones(1, 1, 8, 8, 8)
        target = torch.zeros(1, 1, 8, 8, 8)
        
        loss = loss_fn(pred, target)
        assert loss.item() > 0.9  # Should be close to 1
    
    def test_combined_loss(self):
        """Test combined BCE + Dice loss."""
        loss_fn = CombinedLoss()
        pred = torch.sigmoid(torch.randn(1, 1, 8, 8, 8))
        target = torch.zeros(1, 1, 8, 8, 8)
        
        loss = loss_fn(pred, target)
        assert not torch.isnan(loss)
        assert loss.item() >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
