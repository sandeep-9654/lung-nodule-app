#!/bin/bash
# ============================================================================
# LUNA16 Dataset Download and Setup Script
# Volumetric CT Scan Analysis for Pulmonary Nodule Detection
# ============================================================================

set -e

echo "======================================"
echo "LUNA16 Dataset Setup Script"
echo "======================================"

# Configuration
DOWNLOAD_DIR="$HOME/Downloads"
DATA_DIR="$(dirname "$0")/data/luna16"
ZIP_FILE="$DOWNLOAD_DIR/luna-lung-cancer-dataset.zip"

# Create data directory
echo "[1/4] Creating data directory..."
mkdir -p "$DATA_DIR"

# Download dataset from Kaggle
echo "[2/4] Downloading LUNA16 dataset from Kaggle..."
echo "      This may take a while depending on your connection speed."
echo ""

if [ -f "$ZIP_FILE" ]; then
    echo "      Dataset already downloaded at $ZIP_FILE"
else
    curl -L -o "$ZIP_FILE" \
        https://www.kaggle.com/api/v1/datasets/download/fanbyprinciple/luna-lung-cancer-dataset
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Download failed. You may need to:"
        echo "  1. Set up Kaggle API credentials (~/.kaggle/kaggle.json)"
        echo "  2. Or download manually from:"
        echo "     https://www.kaggle.com/datasets/fanbyprinciple/luna-lung-cancer-dataset"
        exit 1
    fi
fi

# Extract dataset
echo "[3/4] Extracting dataset..."
unzip -q -o "$ZIP_FILE" -d "$DATA_DIR"

# Verify extraction
echo "[4/4] Verifying extracted files..."
MHD_COUNT=$(find "$DATA_DIR" -name "*.mhd" 2>/dev/null | wc -l | tr -d ' ')
RAW_COUNT=$(find "$DATA_DIR" -name "*.raw" -o -name "*.zraw" 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo "Data directory: $DATA_DIR"
echo "Found $MHD_COUNT .mhd files"
echo "Found $RAW_COUNT .raw/.zraw files"
echo ""

if [ "$MHD_COUNT" -eq 0 ]; then
    echo "WARNING: No .mhd files found. Please check the download."
fi
