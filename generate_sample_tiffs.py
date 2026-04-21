"""
generate_sample_tiffs.py
Creates two realistic 4-band satellite GeoTIFF files for testing EcoWatch AI.
  - sample_T1_prior_2019.tif   (2019: healthy vegetation)
  - sample_T2_current_2023.tif (2023: construction/violation zones added)
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import os

OUT_DIR = r"e:/EcoWatch-AI/Data/Sample_TIFFs"
os.makedirs(OUT_DIR, exist_ok=True)

SIZE    = 512
BANDS   = 4
DTYPE   = np.float32
WEST, SOUTH, EAST, NORTH = 77.61, 12.80, 77.67, 12.86
transform = from_bounds(WEST, SOUTH, EAST, NORTH, SIZE, SIZE)
crs = "EPSG:4326"

rng = np.random.default_rng(42)

def save_tiff(path, data):
    with rasterio.open(path, "w", driver="GTiff",
                       height=SIZE, width=SIZE, count=BANDS, dtype=DTYPE,
                       crs=crs, transform=transform) as dst:
        for i in range(BANDS):
            dst.write(data[i], i + 1)
    print(f"  [OK] Saved: {path}")
t1 = np.zeros((BANDS, SIZE, SIZE), dtype=DTYPE)
t1[0] = np.clip(rng.normal(0.12, 0.04, (SIZE, SIZE)), 0, 1)  # Red   (low for veg)
t1[1] = np.clip(rng.normal(0.28, 0.05, (SIZE, SIZE)), 0, 1)  # Green (moderate)
t1[2] = np.clip(rng.normal(0.08, 0.03, (SIZE, SIZE)), 0, 1)  # Blue  (low)
t1[3] = np.clip(rng.normal(0.72, 0.08, (SIZE, SIZE)), 0, 1)  # NIR   (high = dense veg)
for _ in range(6):
    x, y = rng.integers(50, 430, 2)
    r, s = rng.integers(15, 40, 2)
    t1[:, y:y+r, x:x+s] = rng.uniform(0.35, 0.55)
for _ in range(4):
    y = rng.integers(50, 462)
    t1[:, y:y+3, :] = 0.45
for _ in range(4):
    x = rng.integers(50, 462)
    t1[:, :, x:x+3] = 0.45

t1 = np.clip(t1, 0, 1)
save_tiff(os.path.join(OUT_DIR, "sample_T1_prior_2019.tif"), t1)
t2 = t1.copy()

VIOLATION_ZONES = [
    (140, 110, 90, 70),
    (280, 200, 110, 80),
    (60,  320, 80,  60),
    (370, 280, 70,  90),
    (210, 380, 100, 55),
]
for (vx, vy, vw, vh) in VIOLATION_ZONES:
    # Concrete signature: high R/G/B, very low NIR
    t2[0, vy:vy+vh, vx:vx+vw] = rng.uniform(0.55, 0.75)
    t2[1, vy:vy+vh, vx:vx+vw] = rng.uniform(0.50, 0.65)
    t2[2, vy:vy+vh, vx:vx+vw] = rng.uniform(0.45, 0.60)
    t2[3, vy:vy+vh, vx:vx+vw] = rng.uniform(0.08, 0.18)
t2[3] *= rng.uniform(0.85, 0.95, (SIZE, SIZE))
t2[0] += rng.uniform(0.0,  0.04, (SIZE, SIZE))
t2 = np.clip(t2, 0, 1)
save_tiff(os.path.join(OUT_DIR, "sample_T2_current_2023.tif"), t2)

print(f"""
{'='*56}
  Sample TIFFs Generated in: {OUT_DIR}

  T1 (Prior)  : sample_T1_prior_2019.tif    -- 512x512, 4-Band
  T2 (Current): sample_T2_current_2023.tif  -- 512x512, 4-Band

  T2 contains 5 planted violation zones where NIR drops
  sharply and RGB rises (concrete signature over vegetation).

  Upload both files to Live Analysis to test real-time detection.
{'='*56}
""")
