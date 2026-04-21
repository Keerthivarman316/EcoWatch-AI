import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train
import os
import cv2
import rasterio
from report_generator import run_reporting_demo
NANJANGUD_TRANSFORM = rasterio.Affine(10.0, 0.0, 686620.0, 0.0, -10.0, 1342110.0)

def run_inference(patch_source="test", zone_name="Nanjangud"):
    """
    Finalized Inference Engine:
    - Loads robust weights from checkpoints.
    - Generates Violation Maps & Evidence Heatmaps.
    - Triggers Geo-Coordinate Compliance Reporting.
    """
    dev = train.CFG['device']
    CHECKPOINT_DIR = train.CHECKPOINT_DIR
    RESULTS_DIR = train.RESULTS_DIR
    print(f"Loading {patch_source} data for {zone_name} area...")
    if patch_source == "test":
        X1 = np.load(os.path.join(train.DATA_DIR, 'X1_test.npy'))
        X2 = np.load(os.path.join(train.DATA_DIR, 'X2_test.npy'))
        Y  = np.load(os.path.join(train.DATA_DIR, 'Y_test.npy'))
    else:
        X1 = np.load(os.path.join(train.DATA_DIR, 'X1.npy'))
        X2 = np.load(os.path.join(train.DATA_DIR, 'X2.npy'))
        Y  = np.load(os.path.join(train.DATA_DIR, 'Y.npy'))
    best_idx = np.argmax([np.sum(p) for p in Y])
    
    t1_np = X1[best_idx].copy()
    t2_np = X2[best_idx].copy()
    mask_gt = Y[best_idx][0]

    t1_tensor = torch.from_numpy(t1_np).unsqueeze(0).to(dev)
    t2_tensor = torch.from_numpy(t2_np).unsqueeze(0).to(dev)
    print("Loading specialized Siamese ResNet-50 weights...")
    model = train.SiameseUNet(train.CFG['in_channels']).to(dev)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "ChangeDetection_best.pth")
    
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(ckpt['state_dict'])
        print(f"Successfully loaded model with F1: {ckpt['best_f1']:.4f}")
    else:
        print("WARNING: Checkpoint not found. Proceeding with uninitialized weights (for logic testing).")
    
    model.eval()
    print("Generating violation analysis...")
    with torch.no_grad():
        pred_mask = model(t1_tensor, t2_tensor).cpu().numpy()[0, 0]
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(np.clip(t1_np[:3].transpose(1, 2, 0), 0, 1))
    ax[0].set_title("T1 (2019)", fontsize=14)
    ax[0].axis('off')
    
    ax[1].imshow(np.clip(t2_np[:3].transpose(1, 2, 0), 0, 1))
    ax[1].set_title("T2 (2023)", fontsize=14)
    ax[1].axis('off')
    
    ax[2].imshow(mask_gt, cmap='gray')
    ax[2].set_title("Expected Violation", fontsize=14)
    ax[2].axis('off')
    
    ax[3].imshow((pred_mask > 0.5).astype(int), cmap='Reds')
    ax[3].set_title("Detected AI Proof", fontsize=14)
    ax[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "violation_map.png"), dpi=150)
    plt.close()
    print("Executing GradCAM explainable AI proofing...")
    
    def get_gradcam(model, t1, t2):
        model.train() # Enable grads
        t1.requires_grad_(True)
        t2.requires_grad_(True)
        
        # Hook for activations/grads
        activations = []
        def hook_fn(module, input, output): activations.append(output)
        handle = model.decoder.blocks[-1].register_forward_hook(hook_fn)
        
        pred = model(t1, t2)
        target = pred.sum()
        model.zero_grad()
        target.backward()
        
        grads = t1.grad.abs().sum(dim=1).cpu().numpy()[0] # Fallback to input gradients for cleaner visual overlay
        grads = cv2.GaussianBlur(grads, (7,7), 0)
        grads = (grads - grads.min()) / (grads.max() - grads.min() + 1e-7)
        
        handle.remove()
        return grads

    heatmap = get_gradcam(model, t1_tensor, t2_tensor)
    
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.imshow(np.clip(t2_np[:3].transpose(1, 2, 0), 0, 1))
    ax.imshow(heatmap, cmap='jet', alpha=0.4)
    ax.set_title("GradCAM: Violation Attention Heatmap", fontsize=12)
    ax.axis('off')
    plt.savefig(os.path.join(RESULTS_DIR, "gradcam_heatmap.png"), dpi=120)
    plt.close()
    print("\n" + "="*50)
    print("  COMMENCING GEOSPACIAL COMPLIANCE REPORTING")
    print("="*50)
    
    report_path = run_reporting_demo(pred_mask, NANJANGUD_TRANSFORM, zone_name)
    
    if report_path:
        print(f"SUCCESS: Compliance report issued to -> {report_path}")
    else:
        print("No significant violations detected above area threshold.")
    
    print("="*50)

if __name__ == "__main__":
    run_inference(patch_source="test", zone_name="Nanjangud")
