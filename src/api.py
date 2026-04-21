import os
import json
import datetime
import math
import torch
import numpy as np
import cv2
import rasterio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import io
import matplotlib
matplotlib.use('Agg')

import train
# report_generator is imported locally inside /generate-report to avoid startup dependency

app = FastAPI(title="EcoWatch AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR     = r"e:/EcoWatch-AI/results"
COMPLAINTS_FILE = r"e:/EcoWatch-AI/Data/complaints.json"
CHECKPOINT_PATH = r"e:/EcoWatch-AI/checkpoints/ChangeDetection_best.pth"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(r"e:/EcoWatch-AI/Data", exist_ok=True)

dev   = "cuda" if torch.cuda.is_available() else "cpu"
model = None

def load_model():
    global model
    try:
        if model is None:
            print("Loading SiameseUNet...")
            model = train.SiameseUNet(4).to(dev)
            if os.path.exists(CHECKPOINT_PATH):
                ckpt = torch.load(CHECKPOINT_PATH, map_location=dev, weights_only=False)
                missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
                model.eval()
                best_f1 = ckpt.get('best_f1', 0.0)
                print(f"Model ready. Best F1: {best_f1:.4f}  ({len(missing)} new layers, {len(unexpected)} old keys skipped)")
            else:
                print("WARNING: No checkpoint found — model uses random weights until trained.")
                model.eval()
    except Exception as e:
        print(f"CRITICAL: Model load failed: {e}")
        import traceback; traceback.print_exc()
        model = None

@app.on_event("startup")
async def startup_event():
    load_model()
    if not os.path.exists(COMPLAINTS_FILE):
        with open(COMPLAINTS_FILE, 'w') as f:
            json.dump([], f)
    print("Startup complete.")

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None, "device": dev}
class Complaint(BaseModel):
    user_name: str; email: str; location: str; description: str
    lat: Optional[float] = None; lon: Optional[float] = None

@app.post("/complaints")
async def add_complaint(complaint: Complaint):
    try:
        data = []
        if os.path.exists(COMPLAINTS_FILE):
            with open(COMPLAINTS_FILE, 'r') as f:
                data = json.load(f)
        data.append(complaint.model_dump())
        with open(COMPLAINTS_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        return {"status": "success", "message": "Complaint registered."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/complaints")
async def get_complaints():
    if not os.path.exists(COMPLAINTS_FILE):
        return []
    with open(COMPLAINTS_FILE, 'r') as f:
        return json.load(f)
def compute_ndvi(bands_chw: np.ndarray) -> np.ndarray:
    """bands_chw: shape (C, H, W), expects band[0]=Red, band[3]=NIR (0-1 float32)."""
    red = bands_chw[0].astype(np.float32)
    nir = bands_chw[3].astype(np.float32) if bands_chw.shape[0] >= 4 else bands_chw[0]
    denom = nir + red + 1e-8
    return np.clip((nir - red) / denom, -1.0, 1.0)  # range -1..1

# ── Helper: pixel → lat/lon ─────────────────────────────────
def pixel_to_latlon(cy: float, cx: float, bounds, h: int, w: int):
    lon = bounds.left  + (cx / w) * (bounds.right - bounds.left)
    lat = bounds.top   - (cy / h) * (bounds.top   - bounds.bottom)
    return round(lat, 6), round(lon, 6)
def pixel_area_km2(bounds, h: int, w: int) -> float:
    mid_lat = (bounds.top + bounds.bottom) / 2.0
    deg_lat  = (bounds.top  - bounds.bottom) / h
    deg_lon  = (bounds.right - bounds.left)  / w
    km_lat   = deg_lat * 111.0
    km_lon   = deg_lon * 111.0 * math.cos(math.radians(mid_lat))
    return km_lat * km_lon
def make_comparison_png(
    t1_raw: np.ndarray,      # (C, H, W) float32 raw
    t2_raw: np.ndarray,      # (C, H, W) float32 raw
    heatmap_col: np.ndarray, # (H, W, 3) uint8 BGR colormap
    pred: np.ndarray,        # (H, W) float32 probability
    threshold: float,
) -> np.ndarray:
    """Return a BGR uint8 image: [T1 Before] | [Change Map] | [T2 After + violation outlines]."""
    PH, PW = 320, 400  # per-panel size

    def raw_to_bgr(arr: np.ndarray) -> np.ndarray:
        """Convert (C,H,W) float → (PH,PW,3) uint8 BGR, percentile-stretched."""
        c = arr.shape[0]
        r_idx = 0
        g_idx = 1 if c > 1 else 0
        b_idx = 2 if c > 2 else 0
        rgb = np.stack([arr[r_idx], arr[g_idx], arr[b_idx]], axis=-1).astype(np.float32)
        lo  = np.percentile(rgb, 2)
        hi  = np.percentile(rgb, 98)
        if hi > lo:
            rgb = np.clip((rgb - lo) / (hi - lo), 0, 1)
        bgr = (rgb[:, :, ::-1] * 255).astype(np.uint8)
        return cv2.resize(bgr, (PW, PH))

    t1_bgr   = raw_to_bgr(t1_raw)
    t2_bgr   = raw_to_bgr(t2_raw)
    heat_rsz = cv2.resize(heatmap_col, (PW, PH))
    mask_full = (pred > threshold).astype(np.uint8) * 255
    mask_rsz  = cv2.resize(mask_full, (PW, PH), interpolation=cv2.INTER_NEAREST)
    t2_ov = t2_bgr.copy()
    red_layer = np.zeros_like(t2_ov)
    red_layer[:, :, 2] = mask_rsz          # BGR: red = channel 2
    alpha_mask = (mask_rsz > 0).astype(np.float32) * 0.45
    for ch in range(3):
        t2_ov[:, :, ch] = np.clip(
            t2_bgr[:, :, ch] * (1 - alpha_mask) + red_layer[:, :, ch] * alpha_mask, 0, 255
        ).astype(np.uint8)
    contours, _ = cv2.findContours(mask_rsz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(t2_ov, contours, -1, (78, 222, 163), 2)  # neon green BGR
    font, fs, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
    bar_col  = (15, 20, 26)   # very dark
    txt_col  = (78, 222, 163) # neon green

    def label(img: np.ndarray, title: str, sub: str = "") -> np.ndarray:
        out = img.copy()
        cv2.rectangle(out, (0, 0), (PW, 28), bar_col, -1)
        cv2.putText(out, title, (8, 19), font, fs, txt_col, thick, cv2.LINE_AA)
        if sub:
            y = PH - 10
            cv2.rectangle(out, (0, y - 16), (PW, PH), bar_col, -1)
            cv2.putText(out, sub, (8, y), font, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
        return out

    p1 = label(t1_bgr, "BEFORE  (T1 – Prior)",   "Reference image")
    p2 = label(heat_rsz, "CHANGE MAP",            "AI analysis output")
    p3 = label(t2_ov, "AFTER  (T2 – Current)",   "Violations = neon outlines")
    div = np.full((PH, 3, 3), [78, 222, 163], dtype=np.uint8)
    return np.concatenate([p1, div, p2, div, p3], axis=1)
@app.post("/predict")
async def predict_change(
    t1_file      : UploadFile = File(...),
    t2_file      : UploadFile = File(...),
    analysis_type: str        = Form("Change Detection"),
    zone_name    : str        = Form("Unknown"),
):
    global model

    try:
        t1_bytes = await t1_file.read()
        t2_bytes = await t2_file.read()

        with rasterio.open(io.BytesIO(t1_bytes)) as src1:
            t1_raw  = src1.read()
            bounds  = src1.bounds
            h_full, w_full = src1.height, src1.width

        with rasterio.open(io.BytesIO(t2_bytes)) as src2:
            t2_raw  = src2.read()
        class DummyBounds:
            def __init__(self, t, b, l, r):
                self.top = t; self.bottom = b; self.left = l; self.right = r

        if zone_name == "Bommasandra":
            bounds = DummyBounds(12.825, 12.810, 77.670, 77.690)
        elif zone_name == "Peenya":
            bounds = DummyBounds(13.045, 13.030, 77.510, 77.530)
        elif zone_name == "Nanjangud":
            bounds = DummyBounds(12.125, 12.112, 76.670, 76.690)
        t1_raw = np.nan_to_num(t1_raw, nan=0.0).astype(np.float32)
        t2_raw = np.nan_to_num(t2_raw, nan=0.0).astype(np.float32)
        t1_for_ndvi = t1_raw.copy()
        t2_for_ndvi = t2_raw.copy()
        if t1_raw.shape[0] < 4:
            pad = np.zeros((4 - t1_raw.shape[0], *t1_raw.shape[1:]), dtype=np.float32)
            t1_raw      = np.concatenate([t1_raw,      pad], axis=0)
            t2_raw      = np.concatenate([t2_raw,      pad], axis=0)
            t1_for_ndvi = np.concatenate([t1_for_ndvi, pad], axis=0)
            t2_for_ndvi = np.concatenate([t2_for_ndvi, pad], axis=0)

        H, W = t1_raw.shape[1], t1_raw.shape[2]
        if H < 256 or W < 256:
            raise HTTPException(status_code=400, detail=f"Image too small ({H}x{W}). Minimum 256x256.")

        px_area = pixel_area_km2(bounds, H, W)
        if analysis_type == "Vegetation Segmentation":
            # Compute NDVI on RAW values — preserves NIR/Red ratio
            ndvi_t1 = compute_ndvi(t1_for_ndvi)  # range -1..1
            ndvi_t2 = compute_ndvi(t2_for_ndvi)
            raw_loss = ndvi_t1 - ndvi_t2          # -2..2
            loss_map = np.clip(raw_loss, 0, 2) / 2.0  # 0..1, higher = more loss

            pred      = loss_map
            threshold = 0.15       # flag areas with >15% NDVI loss
            min_blob_px = 25
            if pred.max() > 0:
                pred_vis = pred / (pred.max() + 1e-8)
            else:
                pred_vis = pred
            img_u8      = (pred_vis * 255).astype(np.uint8)
            heatmap_col = cv2.applyColorMap(img_u8, cv2.COLORMAP_INFERNO)

        else:
            if model is None:
                load_model()
            t1_norm = t1_raw.copy()
            t2_norm = t2_raw.copy()
            for b in range(t1_norm.shape[0]):
                mx = t1_norm[b].max()
                if mx > 0: t1_norm[b] /= mx
                mx = t2_norm[b].max()
                if mx > 0: t2_norm[b] /= mx

            ps   = 256
            t1_p = t1_norm[:4, :ps, :ps]
            t2_p = t2_norm[:4, :ps, :ps]

            pred_model = None
            if model is not None:
                t1_t = torch.from_numpy(t1_p).unsqueeze(0).to(dev)
                t2_t = torch.from_numpy(t2_p).unsqueeze(0).to(dev)
                with torch.no_grad():
                    pred_model = model(t1_t, t2_t).cpu().numpy()[0, 0]
                pred_model = cv2.resize(pred_model, (W, H), interpolation=cv2.INTER_LINEAR)

            # If model output is nearly flat (untrained / near-uniform) → use spectral diff
            model_is_flat = pred_model is None or (pred_model.max() - pred_model.min() < 0.05)

            if model_is_flat:
                diff_sq = np.zeros((H, W), dtype=np.float32)
                for b in range(min(t1_norm.shape[0], 4)):
                    diff_sq += (t1_norm[b] - t2_norm[b]) ** 2
                spectral_diff = np.sqrt(diff_sq / 4.0)          # 0..1
                spectral_diff = np.clip(spectral_diff * 4.0, 0, 1)
                pred = spectral_diff
            else:
                pred = pred_model

            threshold   = 0.20
            min_blob_px = 40
            if pred.max() > 0:
                pred_vis = pred / (pred.max() + 1e-8)
            else:
                pred_vis = pred
            img_u8      = (pred_vis * 255).astype(np.uint8)
            heatmap_col = cv2.applyColorMap(img_u8, cv2.COLORMAP_JET)
        heatmap_name = f"heatmap_{analysis_type.replace(' ', '_')}.png"
        heatmap_path = os.path.join(RESULTS_DIR, heatmap_name)
        cv2.imwrite(heatmap_path, heatmap_col)
        try:
            comp_img  = make_comparison_png(t1_raw, t2_raw, heatmap_col, pred, threshold)
            comp_name = f"comparison_{analysis_type.replace(' ', '_')}.png"
            comp_path = os.path.join(RESULTS_DIR, comp_name)
            cv2.imwrite(comp_path, comp_img)
        except Exception as ce:
            print(f"Comparison image failed (non-fatal): {ce}")
            comp_name = heatmap_name   # fallback to heatmap
        high_mask = pred > threshold
        if high_mask.any():
            confidence = float(pred[high_mask].mean()) * 100.0
        else:
            confidence = 0.0
        binary    = high_mask.astype(np.uint8)
        num_labels, labels_map, cc_stats, centroids = cv2.connectedComponentsWithStats(binary)

        blobs: List[dict] = []
        for i in range(1, num_labels):
            area_px = int(cc_stats[i, cv2.CC_STAT_AREA])
            if area_px < min_blob_px:
                continue
            area_km2  = round(area_px * px_area, 5)
            cx_px, cy_px = centroids[i]
            lat, lon = pixel_to_latlon(cy_px, cx_px, bounds, H, W)
            if area_km2 > 0.05:
                severity = "High"
            elif area_km2 > 0.015:
                severity = "Medium"
            else:
                severity = "Low"
            blobs.append({
                "id"       : f"VIO-{str(len(blobs)+1).zfill(3)}",
                "severity" : severity,
                "area_km2" : area_km2,
                "lat"      : lat,
                "lon"      : lon,
            })
        blobs.sort(key=lambda b: b["area_km2"], reverse=True)
        total_coverage = round(sum(b["area_km2"] for b in blobs), 4)

        return {
            "status"         : "success",
            "violations"     : len(blobs),
            "confidence"     : round(confidence, 1),
            "coverage_km2"   : total_coverage,
            "heatmap_url"    : f"/results/{heatmap_name}",
            "comparison_url" : f"/results/{comp_name}",
            "blobs"          : blobs[:20],
            "analysis_type"  : analysis_type,
            "zone"           : zone_name,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
@app.get("/results/{filename}")
async def get_result_file(filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="File not found")
@app.delete("/delete-report/{filename}")
async def delete_report(filename: str):
    if not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files can be deleted")
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    os.remove(path)
    return {"status": "deleted", "filename": filename}
@app.post("/generate-report")
async def generate_report_endpoint(
    zone_name    : str   = Form("Unknown"),
    analysis_type: str   = Form("Change Detection"),
    violations   : int   = Form(0),
    confidence   : float = Form(0.0),
    coverage_km2 : float = Form(0.0),
    blobs_json   : str   = Form("[]"),
    heatmap_url  : str   = Form(""),
):
    try:
        import json as _json
        from report_generator import generate_report as _gen

        blobs = _json.loads(blobs_json)
        heatmap_path = None
        if heatmap_url:
            fname = os.path.basename(heatmap_url)
            candidate = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(candidate):
                heatmap_path = candidate

        out_path = _gen(
            zone_name     = zone_name,
            analysis_type = analysis_type,
            violations    = violations,
            confidence    = confidence,
            coverage_km2  = coverage_km2,
            blobs         = blobs,
            heatmap_path  = heatmap_path,
        )

        return {
            "status"     : "success",
            "report_url" : f"/results/{os.path.basename(out_path)}",
            "filename"   : os.path.basename(out_path),
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
@app.get("/stats")
async def get_stats():
    complaint_count = 0
    if os.path.exists(COMPLAINTS_FILE):
        with open(COMPLAINTS_FILE, "r") as f:
            try: complaint_count = len(json.load(f))
            except: pass

    report_files = []
    if os.path.exists(RESULTS_DIR):
        for fname in sorted(os.listdir(RESULTS_DIR), reverse=True):
            if fname.endswith(".pdf"):
                fpath   = os.path.join(RESULTS_DIR, fname)
                size_kb = round(os.path.getsize(fpath) / 1024, 1)
                mtime   = os.path.getmtime(fpath)
                date_str = datetime.datetime.fromtimestamp(mtime).strftime("%b %d, %Y")
                report_files.append({"filename": fname, "url": f"/results/{fname}",
                                     "size_kb": size_kb, "date": date_str})

    return {"complaint_count": complaint_count, "report_count": len(report_files), "reports": report_files}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
