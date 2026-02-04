"""
Training Script for 3D U-Net with Memory Optimization
Features: Mixed Precision (AMP), Gradient Accumulation, Learning Rate Scheduling
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.UNet3D import UNet3D, CombinedLoss, DiceLoss
from preprocessing import (
    load_ct_scan, normalize_hu, extract_patches_for_training,
    PATCH_SIZE
)
from database import init_database, save_metrics


class LUNADataset(Dataset):
    """
    LUNA16 Dataset for nodule detection.
    Extracts patches from CT volumes for training.
    """
    
    def __init__(self, data_dir: str, annotations_file: str = None, 
                 patches_per_scan: int = 10, augment: bool = True):
        """
        Args:
            data_dir: Directory containing .mhd files
            annotations_file: CSV file with nodule annotations (optional)
            patches_per_scan: Number of patches to extract per scan
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.patches_per_scan = patches_per_scan
        self.augment = augment
        
        # Find all .mhd files
        self.scan_files = list(self.data_dir.glob("**/*.mhd"))
        
        if len(self.scan_files) == 0:
            print(f"Warning: No .mhd files found in {data_dir}")
            # Create dummy data for testing
            self._create_dummy_data()
    
    def _create_dummy_data(self):
        """Create dummy data for testing when no real data is available."""
        self.dummy_mode = True
        self.length = 100  # Dummy samples
    
    def __len__(self):
        if hasattr(self, 'dummy_mode') and self.dummy_mode:
            return self.length
        return len(self.scan_files) * self.patches_per_scan
    
    def __getitem__(self, idx):
        if hasattr(self, 'dummy_mode') and self.dummy_mode:
            # Return random dummy patches
            patch = np.random.randn(*PATCH_SIZE).astype(np.float32) * 0.5 + 0.5
            
            # 30% chance of positive sample
            if np.random.random() < 0.3:
                # Add a nodule-like structure
                z, y, x = PATCH_SIZE[0]//2, PATCH_SIZE[1]//2, PATCH_SIZE[2]//2
                r = np.random.randint(4, 10)
                zz, yy, xx = np.ogrid[:PATCH_SIZE[0], :PATCH_SIZE[1], :PATCH_SIZE[2]]
                mask = ((zz-z)**2 + (yy-y)**2 + (xx-x)**2 <= r**2).astype(np.float32)
                patch = np.clip(patch + mask * 0.5, 0, 1)
            else:
                mask = np.zeros(PATCH_SIZE, dtype=np.float32)
            
            return (
                torch.from_numpy(patch).unsqueeze(0),  # Add channel dim
                torch.from_numpy(mask).unsqueeze(0)
            )
        
        # Real data loading
        scan_idx = idx // self.patches_per_scan
        patch_idx = idx % self.patches_per_scan
        
        scan_path = str(self.scan_files[scan_idx])
        volume, _ = load_ct_scan(scan_path)
        volume = normalize_hu(volume)
        
        # Extract random patch
        d, h, w = PATCH_SIZE
        z = np.random.randint(d//2, max(d//2+1, volume.shape[0] - d//2))
        y = np.random.randint(h//2, max(h//2+1, volume.shape[1] - h//2))
        x = np.random.randint(w//2, max(w//2+1, volume.shape[2] - w//2))
        
        patch = volume[
            max(0,z-d//2):z+d//2,
            max(0,y-h//2):y+h//2,
            max(0,x-w//2):x+w//2
        ]
        
        # Pad if necessary
        if patch.shape != PATCH_SIZE:
            padded = np.zeros(PATCH_SIZE, dtype=np.float32)
            pz, py, px = patch.shape
            padded[:pz, :py, :px] = patch
            patch = padded
        
        # For now, create empty mask (no ground truth available)
        mask = np.zeros(PATCH_SIZE, dtype=np.float32)
        
        # Augmentation
        if self.augment:
            patch, mask = self._augment(patch, mask)
        
        return (
            torch.from_numpy(patch).unsqueeze(0).float(),
            torch.from_numpy(mask).unsqueeze(0).float()
        )
    
    def _augment(self, patch, mask):
        """Apply random augmentations."""
        # Random flip
        if np.random.random() > 0.5:
            patch = np.flip(patch, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        if np.random.random() > 0.5:
            patch = np.flip(patch, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if np.random.random() > 0.5:
            patch = np.flip(patch, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()
        
        # Random intensity shift
        shift = np.random.uniform(-0.1, 0.1)
        patch = np.clip(patch + shift, 0, 1)
        
        return patch, mask


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: str,
    accumulation_steps: int = 4
) -> float:
    """Train for one epoch with gradient accumulation and AMP."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    for i, (patches, masks) in enumerate(dataloader):
        patches = patches.to(device)
        masks = masks.to(device)
        
        # Mixed precision forward pass
        with autocast():
            outputs = model(patches)
            loss = criterion(outputs, masks)
            loss = loss / accumulation_steps
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        
        # Update weights every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        num_batches += 1
        
        # Progress
        if (i + 1) % 10 == 0:
            print(f"  Batch {i+1}/{len(dataloader)}, Loss: {loss.item() * accumulation_steps:.4f}")
    
    return total_loss / num_batches


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> tuple:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0
    
    dice_metric = DiceLoss()
    
    with torch.no_grad():
        for patches, masks in dataloader:
            patches = patches.to(device)
            masks = masks.to(device)
            
            with autocast():
                outputs = model(patches)
                loss = criterion(outputs, masks)
            
            # Calculate Dice score (1 - DiceLoss)
            dice = 1 - dice_metric(outputs, masks).item()
            
            total_loss += loss.item()
            total_dice += dice
            num_batches += 1
    
    return total_loss / num_batches, total_dice / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train 3D U-Net for nodule detection')
    parser.add_argument('--data-dir', type=str, default='../data/LUNA16/raw/seg-lungs-LUNA16/seg-lungs-LUNA16',
                        help='Path to LUNA16 data directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size (small due to 3D data)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--accumulation-steps', type=int, default=4,
                        help='Gradient accumulation steps (effective batch = batch_size * steps)')
    parser.add_argument('--save-dir', type=str, default='weights',
                        help='Directory to save model weights')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize database for metrics logging
    init_database()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Dataset and DataLoader
    print(f"\nLoading data from: {args.data_dir}")
    dataset = LUNADataset(args.data_dir, augment=True)
    print(f"Dataset size: {len(dataset)} patches")
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,  # Set to 0 for compatibility
        pin_memory=True if device == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Model
    print("\nInitializing 3D U-Net model...")
    model = UNet3D(in_channels=1, out_channels=1, init_features=32)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Learning rate scheduler with warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Resume from checkpoint
    start_epoch = 0
    best_dice = 0.0
    
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"Effective batch size: {args.batch_size * args.accumulation_steps}")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, args.accumulation_steps
        )
        
        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Dice:   {val_dice:.4f}")
        print(f"  LR:         {current_lr:.2e}")
        print(f"  Time:       {epoch_time:.1f}s")
        
        # Save metrics to database
        save_metrics(
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            dice_score=val_dice
        )
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"  ✓ Saved new best model (Dice: {best_dice:.4f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        print()
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best Dice Score: {best_dice:.4f}")
    print(f"Model saved to: {args.save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
