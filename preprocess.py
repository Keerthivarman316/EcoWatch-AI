"""
EcoWatch-AI Preprocessing Pipeline (Fixed)
===========================================
Fixed for:
  - NaN values in TIFF bands
  - Small image size (446x669) with padding for 256x256 patches
  - Overlapping stride for more patch coverage
"""

import rasterio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

RAW_DIR = r"e:\EcoWatch-AI\Data\Raw\drive-download-20260411T172226Z-3-001"
OUTPUT_DIR = r"e:\EcoWatch-AI\Data\Processed"
PATCH_SIZE = 256
STRIDE = 128  # 50% overlap to get more patches from a small image

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STEP 1: Read TIFF Properly (bands, H, W) → (H, W, bands)
# ============================================================
print("=" * 60)
print("STEP 1: Reading TIFF files...")
t1_ds = rasterio.open(os.path.join(RAW_DIR, "T1_2019.tif"))
t2_ds = rasterio.open(os.path.join(RAW_DIR, "T2_2023.tif"))

t1_raw = t1_ds.read()
t2_raw = t2_ds.read()

print(f"  T1 band descriptions: {t1_ds.descriptions}")
print(f"  T2 band descriptions: {t2_ds.descriptions}")
print(f"  Raw T1 shape (bands, H, W): {t1_raw.shape}")
print(f"  Raw T2 shape (bands, H, W): {t2_raw.shape}")

# Handle NaN values
nan_t1 = np.sum(np.isnan(t1_raw))
nan_t2 = np.sum(np.isnan(t2_raw))
print(f"  T1 NaN count: {nan_t1}")
print(f"  T2 NaN count: {nan_t2}")

t1_raw = np.nan_to_num(t1_raw, nan=0.0)
t2_raw = np.nan_to_num(t2_raw, nan=0.0)

# Convert shape: (bands, H, W) → (H, W, bands)
t1 = np.transpose(t1_raw, (1, 2, 0)).astype(np.float32)
t2 = np.transpose(t2_raw, (1, 2, 0)).astype(np.float32)

print(f"  Transposed T1 shape (H, W, bands): {t1.shape}")
print(f"  Transposed T2 shape (H, W, bands): {t2.shape}")
print("  ✅ STEP 1 DONE\n")

# ============================================================
# STEP 2: Normalize (VERY IMPORTANT)
# ============================================================
print("STEP 2: Normalizing...")
print(f"  T1 range before: [{t1.min():.2f}, {t1.max():.2f}]")
print(f"  T2 range before: [{t2.min():.2f}, {t2.max():.2f}]")

t1 = t1 / np.max(t1)
t2 = t2 / np.max(t2)

print(f"  T1 range after:  [{t1.min():.4f}, {t1.max():.4f}]")
print(f"  T2 range after:  [{t2.min():.4f}, {t2.max():.4f}]")
print("  ✅ STEP 2 DONE\n")

# ============================================================
# STEP 3: Extract NDVI Band (4th band = index 3)
# ============================================================
print("STEP 3: Extracting NDVI band...")
ndvi_t1 = t1[:, :, 3]
ndvi_t2 = t2[:, :, 3]

print(f"  NDVI T1 shape: {ndvi_t1.shape}")
print(f"  NDVI T2 shape: {ndvi_t2.shape}")
print(f"  NDVI T1 range: [{ndvi_t1.min():.4f}, {ndvi_t1.max():.4f}], mean={ndvi_t1.mean():.4f}")
print(f"  NDVI T2 range: [{ndvi_t2.min():.4f}, {ndvi_t2.max():.4f}], mean={ndvi_t2.mean():.4f}")
print("  ✅ STEP 3 DONE\n")

# ============================================================
# STEP 4: Create Vegetation Mask
# ============================================================
print("STEP 4: Creating vegetation mask...")
veg_mask = (ndvi_t1 > 0.3).astype(np.uint8)

print(f"  Vegetation mask shape: {veg_mask.shape}")
print(f"  Vegetation pixels: {np.sum(veg_mask)} / {veg_mask.size} ({100*np.sum(veg_mask)/veg_mask.size:.2f}%)")
print("  ✅ STEP 4 DONE\n")

# ============================================================
# STEP 5: Create Change Mask (MAIN TASK)
# ============================================================
print("STEP 5: Creating change mask (ground truth)...")
ndvi_diff = ndvi_t2 - ndvi_t1
change_mask = (ndvi_diff < -0.15).astype(np.uint8)

print(f"  NDVI diff range: [{ndvi_diff.min():.4f}, {ndvi_diff.max():.4f}]")
print(f"  Change mask shape: {change_mask.shape}")
print(f"  Change pixels: {np.sum(change_mask)} / {change_mask.size} ({100*np.sum(change_mask)/change_mask.size:.2f}%)")
print("  ✅ STEP 5 DONE\n")

# ============================================================
# STEP 6: Pad image to make it evenly divisible by PATCH_SIZE
# ============================================================
print("STEP 6: Padding images for patch extraction...")
H, W, C = t1.shape

# Calculate padding needed
pad_h = (PATCH_SIZE - H % PATCH_SIZE) % PATCH_SIZE
pad_w = (PATCH_SIZE - W % PATCH_SIZE) % PATCH_SIZE

print(f"  Original size: {H}x{W}")
print(f"  Padding needed: H+{pad_h}, W+{pad_w}")

t1_padded = np.pad(t1, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
t2_padded = np.pad(t2, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
change_mask_padded = np.pad(change_mask, ((0, pad_h), (0, pad_w)), mode='reflect')

print(f"  Padded size: {t1_padded.shape[0]}x{t1_padded.shape[1]}")
print("  ✅ STEP 6 DONE\n")

# ============================================================
# STEP 7: Patch Generation with Overlapping Stride
# ============================================================
print("STEP 7: Generating patches (256x256, stride=128)...")

def create_patches(t1_img, t2_img, mask, patch_size, stride):
    patches_img1 = []
    patches_img2 = []
    patches_mask = []

    H, W, _ = t1_img.shape

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch1 = t1_img[i:i+patch_size, j:j+patch_size]
            patch2 = t2_img[i:i+patch_size, j:j+patch_size]
            mask_patch = mask[i:i+patch_size, j:j+patch_size]

            patches_img1.append(patch1)
            patches_img2.append(patch2)
            patches_mask.append(mask_patch)

    return patches_img1, patches_img2, patches_mask

X1, X2, Y = create_patches(t1_padded, t2_padded, change_mask_padded, PATCH_SIZE, STRIDE)

print(f"  Total patches: {len(X1)}")
print("  ✅ STEP 7 DONE\n")

# ============================================================
# STEP 8: Remove Bad Patches (filter out empty patches)
# ============================================================
print("STEP 8: Filtering bad patches...")

filtered_X1 = []
filtered_X2 = []
filtered_Y  = []

for i in range(len(Y)):
    if np.sum(Y[i]) > 50:   # keep only meaningful change
        filtered_X1.append(X1[i])
        filtered_X2.append(X2[i])
        filtered_Y.append(Y[i])

print(f"  Before filtering: {len(X1)} patches")
print(f"  After filtering:  {len(filtered_X1)} patches (with >50 change pixels)")

# If we got very few patches, also try a lower threshold
if len(filtered_X1) < 10:
    print("  ⚠️  Very few patches with threshold 50, trying threshold 10...")
    filtered_X1 = []
    filtered_X2 = []
    filtered_Y  = []
    for i in range(len(Y)):
        if np.sum(Y[i]) > 10:
            filtered_X1.append(X1[i])
            filtered_X2.append(X2[i])
            filtered_Y.append(Y[i])
    print(f"  After filtering (threshold=10): {len(filtered_X1)} patches")

print("  ✅ STEP 8 DONE\n")

# ============================================================
# STEP 9: Convert to Numpy Arrays
# ============================================================
print("STEP 9: Converting to numpy arrays...")

X1 = np.array(filtered_X1, dtype=np.float32)
X2 = np.array(filtered_X2, dtype=np.float32)
Y  = np.array(filtered_Y, dtype=np.float32)

print(f"  X1 shape: {X1.shape}  (N, 256, 256, 4)")
print(f"  X2 shape: {X2.shape}")
print(f"  Y  shape: {Y.shape}")
print("  ✅ STEP 9 DONE\n")

# ============================================================
# STEP 10: Convert to PyTorch Format (N, C, H, W)
# ============================================================
print("STEP 10: Converting to PyTorch format (N, C, H, W)...")

X1 = np.transpose(X1, (0, 3, 1, 2))
X2 = np.transpose(X2, (0, 3, 1, 2))
Y  = np.expand_dims(Y, axis=1)

print(f"  X1 shape: {X1.shape}  (N, 4, 256, 256)")
print(f"  X2 shape: {X2.shape}")
print(f"  Y  shape: {Y.shape}   (N, 1, 256, 256)")
print("  ✅ STEP 10 DONE\n")

# ============================================================
# STEP 11: Save Dataset
# ============================================================
print("STEP 11: Saving dataset to .npy files...")

np.save(os.path.join(OUTPUT_DIR, "X1.npy"), X1)
np.save(os.path.join(OUTPUT_DIR, "X2.npy"), X2)
np.save(os.path.join(OUTPUT_DIR, "Y.npy"), Y)

x1_size = os.path.getsize(os.path.join(OUTPUT_DIR, "X1.npy")) / (1024*1024)
x2_size = os.path.getsize(os.path.join(OUTPUT_DIR, "X2.npy")) / (1024*1024)
y_size  = os.path.getsize(os.path.join(OUTPUT_DIR, "Y.npy")) / (1024*1024)

print(f"  X1.npy saved: {x1_size:.2f} MB")
print(f"  X2.npy saved: {x2_size:.2f} MB")
print(f"  Y.npy  saved: {y_size:.2f} MB")
print("  ✅ STEP 11 DONE\n")

# ============================================================
# STEP 12: Quick Sanity Check (save plots)
# ============================================================
print("STEP 12: Generating sanity check plots...")

# --- Plot 1: Full NDVI difference map ---
fig_full, axes_full = plt.subplots(1, 3, figsize=(18, 6))

axes_full[0].imshow(ndvi_t1, cmap='YlGn', vmin=0, vmax=1)
axes_full[0].set_title("NDVI T1 (2019)", fontsize=12)
axes_full[0].axis('off')

axes_full[1].imshow(ndvi_t2, cmap='YlGn', vmin=0, vmax=1)
axes_full[1].set_title("NDVI T2 (2023)", fontsize=12)
axes_full[1].axis('off')

im = axes_full[2].imshow(ndvi_diff, cmap='RdYlGn', vmin=-0.5, vmax=0.5)
axes_full[2].set_title(f"NDVI Difference\n{np.sum(change_mask)} change pixels", fontsize=12)
axes_full[2].axis('off')
plt.colorbar(im, ax=axes_full[2], shrink=0.8)

plt.suptitle("EcoWatch-AI: Full Image NDVI Analysis", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "full_ndvi_analysis.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 2: Full change mask ---
fig_mask, ax_mask = plt.subplots(figsize=(8, 6))
ax_mask.imshow(change_mask, cmap='gray')
ax_mask.set_title(f"Full Change Mask (threshold: NDVI diff < -0.15)\n{np.sum(change_mask)} change pixels detected", fontsize=12)
ax_mask.axis('off')
plt.savefig(os.path.join(OUTPUT_DIR, "full_change_mask.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 3: Patch-level sanity checks ---
num_display = min(4, len(filtered_X1))
fig_patches, axes_p = plt.subplots(num_display, 4, figsize=(16, 4*num_display))
if num_display == 1:
    axes_p = axes_p[np.newaxis, :]

for idx in range(num_display):
    # T1 RGB
    t1_rgb = X1[idx][:3].transpose(1, 2, 0)
    axes_p[idx, 0].imshow(np.clip(t1_rgb, 0, 1))
    axes_p[idx, 0].set_title(f"T1 RGB (Patch {idx})")
    axes_p[idx, 0].axis('off')

    # T2 RGB
    t2_rgb = X2[idx][:3].transpose(1, 2, 0)
    axes_p[idx, 1].imshow(np.clip(t2_rgb, 0, 1))
    axes_p[idx, 1].set_title(f"T2 RGB (Patch {idx})")
    axes_p[idx, 1].axis('off')

    # Change mask
    axes_p[idx, 2].imshow(Y[idx][0], cmap='gray')
    axes_p[idx, 2].set_title(f"Change Mask\n(white px: {int(np.sum(Y[idx][0]))})")
    axes_p[idx, 2].axis('off')

    # NDVI diff
    ndvi_d = X2[idx][3] - X1[idx][3]
    im_d = axes_p[idx, 3].imshow(ndvi_d, cmap='RdYlGn', vmin=-0.5, vmax=0.5)
    axes_p[idx, 3].set_title(f"NDVI Diff (Patch {idx})")
    axes_p[idx, 3].axis('off')

plt.suptitle("EcoWatch-AI: Patch-Level Sanity Check", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "patch_sanity_check.png"), dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 4: Single change mask (original requirement) ---
fig_single, ax_single = plt.subplots(figsize=(6, 6))
ax_single.imshow(Y[0][0], cmap='gray')
ax_single.set_title("Change Mask (Patch 0)", fontsize=14)
ax_single.axis('off')
plt.savefig(os.path.join(OUTPUT_DIR, "change_mask_sample.png"), dpi=150, bbox_inches='tight')
plt.close()

print(f"  Saved: full_ndvi_analysis.png")
print(f"  Saved: full_change_mask.png")
print(f"  Saved: patch_sanity_check.png")
print(f"  Saved: change_mask_sample.png")
print("  ✅ STEP 12 DONE\n")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("=" * 60)
print("🎯 PREPROCESSING COMPLETE!")
print("=" * 60)
print(f"\n  📁 Output: {OUTPUT_DIR}")
print(f"  📦 X1.npy → T1 patches  | Shape: {X1.shape}")
print(f"  📦 X2.npy → T2 patches  | Shape: {X2.shape}")
print(f"  📦 Y.npy  → Change masks | Shape: {Y.shape}")
print(f"\n  🖼️  Image size: {H}×{W} (padded to {t1_padded.shape[0]}×{t1_padded.shape[1]})")
print(f"  🔲 Patch size: {PATCH_SIZE}×{PATCH_SIZE}, stride: {STRIDE}")
print(f"  📊 Total patches generated: {len(filtered_X1) + (len(X1) - len(filtered_X1))}")
print(f"  ✅ Filtered patches (meaningful change): {X1.shape[0]}")
print(f"  🌿 Change pixels detected: {np.sum(change_mask)} ({100*np.sum(change_mask)/change_mask.size:.2f}%)")
print("=" * 60)
