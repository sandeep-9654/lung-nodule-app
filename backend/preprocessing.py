"""
Preprocessing Module for LUNA16 CT Scans
Handles loading, normalization, and patch extraction for 3D U-Net training.
"""

import os
import numpy as np
from typing import Tuple, List, Optional
import warnings

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False
    warnings.warn("SimpleITK not installed. Install with: pip install SimpleITK")


# Hounsfield Unit (HU) windowing for lung CT
HU_MIN = -1000  # Air
HU_MAX = 400    # Bone threshold
PATCH_SIZE = (32, 64, 64)  # (D, H, W)


def load_ct_scan(mhd_path: str) -> Tuple[np.ndarray, dict]:
    """
    Load a CT scan from .mhd file.
    
    Args:
        mhd_path: Path to the .mhd metadata file
    
    Returns:
        volume: 3D numpy array of shape (D, H, W)
        metadata: Dictionary with spacing, origin, and direction
    """
    if not HAS_SITK:
        raise ImportError("SimpleITK is required. Install with: pip install SimpleITK")
    
    # Read image
    image = sitk.ReadImage(mhd_path)
    volume = sitk.GetArrayFromImage(image)  # Returns (Z, Y, X) = (D, H, W)
    
    metadata = {
        'spacing': image.GetSpacing(),    # (X, Y, Z) spacing in mm
        'origin': image.GetOrigin(),
        'direction': image.GetDirection(),
        'size': image.GetSize()
    }
    
    return volume, metadata


def normalize_hu(volume: np.ndarray) -> np.ndarray:
    """
    Normalize Hounsfield Units to [0, 1] range.
    
    Args:
        volume: 3D numpy array with raw HU values
    
    Returns:
        Normalized volume in [0, 1] range
    """
    volume = np.clip(volume, HU_MIN, HU_MAX)
    volume = (volume - HU_MIN) / (HU_MAX - HU_MIN)
    return volume.astype(np.float32)


def extract_patch(
    volume: np.ndarray,
    center: Tuple[int, int, int],
    patch_size: Tuple[int, int, int] = PATCH_SIZE
) -> np.ndarray:
    """
    Extract a 3D patch centered at the given coordinates.
    
    Args:
        volume: 3D numpy array (D, H, W)
        center: (z, y, x) center coordinates
        patch_size: (depth, height, width) of the patch
    
    Returns:
        3D patch of shape patch_size
    """
    d, h, w = patch_size
    z, y, x = center
    
    # Calculate boundaries with padding
    z_start = max(0, z - d // 2)
    z_end = min(volume.shape[0], z + d // 2)
    y_start = max(0, y - h // 2)
    y_end = min(volume.shape[1], y + h // 2)
    x_start = max(0, x - w // 2)
    x_end = min(volume.shape[2], x + w // 2)
    
    # Extract patch
    patch = volume[z_start:z_end, y_start:y_end, x_start:x_end]
    
    # Pad if necessary
    if patch.shape != patch_size:
        padded = np.zeros(patch_size, dtype=patch.dtype)
        pz, py, px = patch.shape
        padded[:pz, :py, :px] = patch
        patch = padded
    
    return patch


def extract_patches_for_training(
    volume: np.ndarray,
    nodule_centers: List[Tuple[int, int, int]],
    num_negative: int = 5,
    patch_size: Tuple[int, int, int] = PATCH_SIZE
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extract positive (nodule) and negative (background) patches for training.
    
    Args:
        volume: Normalized 3D CT volume
        nodule_centers: List of (z, y, x) nodule center coordinates
        num_negative: Number of random negative patches per nodule
        patch_size: Size of patches to extract
    
    Returns:
        patches: List of 3D patches
        labels: List of corresponding segmentation masks
    """
    patches = []
    labels = []
    d, h, w = patch_size
    
    # Positive patches (containing nodules)
    for center in nodule_centers:
        patch = extract_patch(volume, center, patch_size)
        
        # Create simple spherical mask for nodule
        mask = create_spherical_mask(patch_size, radius=8)
        
        patches.append(patch)
        labels.append(mask)
    
    # Negative patches (random background)
    for _ in range(len(nodule_centers) * num_negative):
        # Random center with margin
        z = np.random.randint(d // 2, max(d // 2 + 1, volume.shape[0] - d // 2))
        y = np.random.randint(h // 2, max(h // 2 + 1, volume.shape[1] - h // 2))
        x = np.random.randint(w // 2, max(w // 2 + 1, volume.shape[2] - w // 2))
        
        # Skip if too close to any nodule
        is_near_nodule = any(
            abs(z - nz) < d and abs(y - ny) < h and abs(x - nx) < w
            for nz, ny, nx in nodule_centers
        )
        if is_near_nodule:
            continue
        
        patch = extract_patch(volume, (z, y, x), patch_size)
        mask = np.zeros(patch_size, dtype=np.float32)
        
        patches.append(patch)
        labels.append(mask)
    
    return patches, labels


def create_spherical_mask(
    shape: Tuple[int, int, int],
    radius: int = 8,
    center: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    Create a spherical mask for nodule annotation.
    
    Args:
        shape: (D, H, W) shape of the mask
        radius: Radius of the sphere in voxels
        center: Center of the sphere (defaults to center of volume)
    
    Returns:
        Binary mask with sphere
    """
    d, h, w = shape
    if center is None:
        center = (d // 2, h // 2, w // 2)
    
    z, y, x = np.ogrid[:d, :h, :w]
    cz, cy, cx = center
    
    distance = np.sqrt((z - cz)**2 + (y - cy)**2 + (x - cx)**2)
    mask = (distance <= radius).astype(np.float32)
    
    return mask


def sliding_window_inference(
    volume: np.ndarray,
    model,
    patch_size: Tuple[int, int, int] = PATCH_SIZE,
    stride: Tuple[int, int, int] = (32, 64, 64),
    device: str = 'cpu',
    batch_size: int = 8
) -> np.ndarray:
    """
    Perform sliding window inference on a full CT volume.
    
    Args:
        volume: Normalized 3D CT volume
        model: PyTorch model
        patch_size: Size of patches for inference
        stride: Stride between patches (larger = faster, less overlap)
        device: Device for inference
        batch_size: Number of patches to process in parallel
    
    Returns:
        Prediction volume with same shape as input
    """
    import torch
    import time
    
    model.eval()
    d, h, w = patch_size
    sd, sh, sw = stride
    
    # Store original shape before any padding
    orig_shape = volume.shape
    
    # Pad volume if necessary
    vd, vh, vw = volume.shape
    pad_d = max(0, d - vd)
    pad_h = max(0, h - vh)
    pad_w = max(0, w - vw)
    
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        volume = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
    
    vd, vh, vw = volume.shape
    
    # Pre-compute all patch positions
    positions = []
    for z in range(0, vd - d + 1, sd):
        for y in range(0, vh - h + 1, sh):
            for x in range(0, vw - w + 1, sw):
                positions.append((z, y, x))
    
    total_patches = len(positions)
    print(f"  Inference: {total_patches} patches (stride={stride}, batch_size={batch_size})")
    
    # Initialize output and count arrays
    output = np.zeros((vd, vh, vw), dtype=np.float32)
    count = np.zeros((vd, vh, vw), dtype=np.float32)
    
    start_time = time.time()
    
    # Process patches in batches
    with torch.no_grad():
        for batch_start in range(0, total_patches, batch_size):
            batch_end = min(batch_start + batch_size, total_patches)
            batch_positions = positions[batch_start:batch_end]
            
            # Extract batch of patches
            batch_patches = []
            for z, y, x in batch_positions:
                patch = volume[z:z+d, y:y+h, x:x+w]
                batch_patches.append(patch)
            
            # Stack into batch tensor
            batch_tensor = torch.from_numpy(np.stack(batch_patches)).float()
            batch_tensor = batch_tensor.unsqueeze(1)  # Add channel dim: (B, 1, D, H, W)
            batch_tensor = batch_tensor.to(device)
            
            # Predict entire batch
            preds = model(batch_tensor)
            preds = preds.squeeze(1).cpu().numpy()  # (B, D, H, W)
            
            # Accumulate results
            for i, (z, y, x) in enumerate(batch_positions):
                output[z:z+d, y:y+h, x:x+w] += preds[i]
                count[z:z+d, y:y+h, x:x+w] += 1
            
            # Progress logging
            done = batch_end
            elapsed = time.time() - start_time
            if done > 0 and elapsed > 0:
                speed = done / elapsed
                remaining = (total_patches - done) / speed
                print(f"  Progress: {done}/{total_patches} patches "
                      f"({done*100//total_patches}%) - {remaining:.0f}s remaining")
    
    elapsed = time.time() - start_time
    print(f"  Inference complete in {elapsed:.1f}s")
    
    # Average overlapping regions
    result = np.zeros_like(output)
    np.divide(output, count, where=count > 0, out=result)
    output = result
    
    # Remove padding
    output = output[:orig_shape[0], :orig_shape[1], :orig_shape[2]]
    
    return output


def find_nodule_candidates(
    prediction: np.ndarray,
    threshold: float = 0.5,
    min_size: int = 20
) -> List[dict]:
    """
    Find nodule candidates from prediction volume.
    
    Args:
        prediction: 3D prediction volume with probabilities
        threshold: Threshold for binary segmentation
        min_size: Minimum nodule size in voxels
    
    Returns:
        List of dictionaries with nodule information
    """
    from scipy import ndimage
    
    # Threshold
    binary = prediction > threshold
    
    # Connected component labeling
    labeled, num_features = ndimage.label(binary)
    
    nodules = []
    for i in range(1, num_features + 1):
        mask = labeled == i
        size = np.sum(mask)
        
        if size >= min_size:
            # Find centroid
            coords = np.where(mask)
            centroid = (
                int(np.mean(coords[0])),  # Z
                int(np.mean(coords[1])),  # Y
                int(np.mean(coords[2]))   # X
            )
            
            # Calculate probability
            prob = float(np.mean(prediction[mask]))
            
            nodules.append({
                'id': i,
                'centroid': centroid,
                'size_voxels': int(size),
                'probability': prob
            })
    
    # Sort by probability
    nodules.sort(key=lambda x: x['probability'], reverse=True)
    
    return nodules


# Demo/testing
if __name__ == "__main__":
    print("Preprocessing Module Test")
    print("-" * 50)
    
    # Test with dummy data
    dummy_volume = np.random.randn(64, 256, 256).astype(np.float32) * 500 - 500
    print(f"Created dummy volume: {dummy_volume.shape}")
    
    # Normalize
    normalized = normalize_hu(dummy_volume)
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # Extract patch
    patch = extract_patch(normalized, (32, 128, 128))
    print(f"Extracted patch shape: {patch.shape}")
    
    # Create mask
    mask = create_spherical_mask(PATCH_SIZE, radius=8)
    print(f"Created spherical mask: {mask.shape}, volume: {mask.sum():.0f} voxels")
    
    print("\n✓ Preprocessing module test passed!")
