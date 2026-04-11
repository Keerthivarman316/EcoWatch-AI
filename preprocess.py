"""
EcoWatch-AI Preprocessing Pipeline (Multi-Zone Update)
===========================================
- Fixes NaN values in TIFF bands
- Supports processing multiple TIFF pairs (NE, NW, SE, SW)
- Patches to 128x128 with stride 64
"""

import rasterio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

RAW_DIR = r"e:\EcoWatch-AI\Data\Raw\drive-download-20260411T201058Z-3-001"
OUTPUT_DIR = r"e:\EcoWatch-AI\Data\Processed"
PATCH_SIZE = 128
STRIDE = 64

os.makedirs(OUTPUT_DIR, exist_ok=True)
ZONES = ["NE", "NW", "SE", "SW"]

print("=" * 60)
print("🎯 MULTI-ZONE PREPROCESSING STARTED")
print("=" * 60)

all_filtered_X1 = []
all_filtered_X2 = []
all_filtered_Y  = []
total_change_px = 0
total_raw_px = 0

def create_patches(t1_img, t2_img, mask, patch_size, stride):
    p1, p2, pm = [], [], []
    H, W, _ = t1_img.shape
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            p1.append(t1_img[i:i+patch_size, j:j+patch_size])
            p2.append(t2_img[i:i+patch_size, j:j+patch_size])
            pm.append(mask[i:i+patch_size, j:j+patch_size])
    return p1, p2, pm

for zone in ZONES:
    print(f"\n--- Processing Zone: {zone} ---")
    t1_path = os.path.join(RAW_DIR, f"Peenya_{zone}_T1_2019.tif")
    t2_path = os.path.join(RAW_DIR, f"Peenya_{zone}_T2_2023.tif")
    
    if not os.path.exists(t1_path) or not os.path.exists(t2_path):
        print(f"  ⚠️ Skipping {zone}: files not found.")
        continue

    # READ
    t1_ds = rasterio.open(t1_path)
    t2_ds = rasterio.open(t2_path)
    t1_raw = t1_ds.read()
    t2_raw = t2_ds.read()
    
    # HANDLE NANS & TRANSPOSE to (H,W,C)
    t1_raw = np.nan_to_num(t1_raw, nan=0.0)
    t2_raw = np.nan_to_num(t2_raw, nan=0.0)
    t1 = np.transpose(t1_raw, (1, 2, 0)).astype(np.float32)
    t2 = np.transpose(t2_raw, (1, 2, 0)).astype(np.float32)
    
    # NORMALIZE
    t1_max, t2_max = np.max(t1), np.max(t2)
    if t1_max > 0: t1 = t1 / t1_max
    if t2_max > 0: t2 = t2 / t2_max

    # Extract NDVI
    ndvi_t1 = t1[:, :, 3]
    ndvi_t2 = t2[:, :, 3]
    
    # Create Masks
    ndvi_diff = ndvi_t2 - ndvi_t1
    change_mask = (ndvi_diff < -0.15).astype(np.uint8)
    
    change_px = np.sum(change_mask)
    total_change_px += change_px
    total_raw_px += change_mask.size
    print(f"  Change pixels detected: {change_px} / {change_mask.size}")

    # PAD
    H, W, C = t1.shape
    pad_h = (PATCH_SIZE - H % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - W % PATCH_SIZE) % PATCH_SIZE
    
    t1_padded = np.pad(t1, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    t2_padded = np.pad(t2, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    change_mask_padded = np.pad(change_mask, ((0, pad_h), (0, pad_w)), mode='reflect')

    # GENERATE PATCHES
    X1, X2, Y = create_patches(t1_padded, t2_padded, change_mask_padded, PATCH_SIZE, STRIDE)
    print(f"  Generated {len(X1)} gross patches.")

    # FILTER
    zone_filtered = 0
    for i in range(len(Y)):
        if np.sum(Y[i]) > 50:
            all_filtered_X1.append(X1[i])
            all_filtered_X2.append(X2[i])
            all_filtered_Y.append(Y[i])
            zone_filtered += 1
            
    print(f"  Kept {zone_filtered} meaningful patches (>50 change px).")

print(f"\n--- AGGREGATING ALL ZONES ---")
if len(all_filtered_X1) < 20: 
    print("  ⚠️  Overall patches < 20. Dataset very sparse!")

# CONVERT TO NUMPY
X1_final = np.array(all_filtered_X1, dtype=np.float32)
X2_final = np.array(all_filtered_X2, dtype=np.float32)
Y_final  = np.array(all_filtered_Y, dtype=np.float32)

# CONVERT TO PYTORCH FORMAT (N, C, H, W)
X1_final = np.transpose(X1_final, (0, 3, 1, 2))
X2_final = np.transpose(X2_final, (0, 3, 1, 2))
Y_final  = np.expand_dims(Y_final, axis=1)

print(f"  Final X1 shape: {X1_final.shape}")
print(f"  Final X2 shape: {X2_final.shape}")
print(f"  Final Y  shape: {Y_final.shape}")

print("--- SAVING TO DISK ---")
np.save(os.path.join(OUTPUT_DIR, "X1.npy"), X1_final)
np.save(os.path.join(OUTPUT_DIR, "X2.npy"), X2_final)
np.save(os.path.join(OUTPUT_DIR, "Y.npy"), Y_final)
print("  ✅ All `.npy` datasets generated successfully!")
