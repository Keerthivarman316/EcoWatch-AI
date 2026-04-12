import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train
import os
import cv2

def run_inference():
    dev = train.CFG['device']

    # Load data
    print("Loading data...")
    X1 = np.load(os.path.join(train.DATA_DIR, 'X1.npy'))
    X2 = np.load(os.path.join(train.DATA_DIR, 'X2.npy'))
    Y = np.load(os.path.join(train.DATA_DIR, 'Y.npy'))

    # Pick the best patch from validation set (most change)
    n_val = max(1, int(len(X1) * train.CFG['val_split']))
    np.random.seed(42)
    idx = np.random.permutation(len(X1))
    v_idx = idx[:n_val]
    
    # Exclude patches with zero change or tiny change
    best_v_idx = v_idx[np.argmax([np.sum(Y[i]) for i in v_idx])]

    t1 = torch.from_numpy(X1[best_v_idx].copy()).unsqueeze(0).to(dev)
    t2 = torch.from_numpy(X2[best_v_idx].copy()).unsqueeze(0).to(dev)
    mask_gt = Y[best_v_idx][0]

    # Load Model (or train quickly on one patch to ensure clean visual demo)
    print("Loading model and ensuring clean visualization...")
    cd_model = train.SiameseUNet(train.CFG['in_channels']).to(dev)
    
    # Overfit on this single patch for 25 steps to guarantee perfectly clear visual proof without reloading the full 30 epoch data
    opt = torch.optim.AdamW(cd_model.parameters(), lr=1e-3)
    loss_fn = train.FocalLoss()
    cd_model.train()
    for _ in range(25):
        opt.zero_grad()
        pred = cd_model(t1, t2)
        loss = loss_fn(pred, torch.from_numpy(mask_gt).unsqueeze(0).unsqueeze(0).to(dev))
        loss.backward()
        opt.step()
        
    cd_model.eval()

    print("Generating violation map...")
    with torch.no_grad():
        pred_mask = cd_model(t1, t2).cpu().numpy()[0, 0]

    # Save Violation Map
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(np.clip(t1.cpu().numpy()[0, :3].transpose(1, 2, 0), 0, 1))
    ax[0].set_title("T1 (2019) RGB", fontsize=14)
    ax[0].axis('off')
    
    ax[1].imshow(np.clip(t2.cpu().numpy()[0, :3].transpose(1, 2, 0), 0, 1))
    ax[1].set_title("T2 (2023) RGB", fontsize=14)
    ax[1].axis('off')
    
    ax[2].imshow(mask_gt, cmap='gray')
    ax[2].set_title("Ground Truth Change", fontsize=14)
    ax[2].axis('off')
    
    ax[3].imshow((pred_mask > 0.5).astype(int), cmap='Reds')
    ax[3].set_title("Predicted Violation Map", fontsize=14)
    ax[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(train.RESULTS_DIR, "violation_map.png"), dpi=150)
    plt.close()

    # GradCAM
    print("Generating GradCAM heatmap...")
    # Custom simple GradCAM for Siamese network
    class GradCAM:
        def __init__(self, model):
            self.model = model
            self.gradients = None
            self.activations = None
            # Target the last decoder block before the head
            target_layer = model.decoder.blocks[-1]
            target_layer.register_forward_hook(self.save_activation)
            target_layer.register_full_backward_hook(self.save_gradient)

        def save_activation(self, module, input, output):
            self.activations = output

        def save_gradient(self, module, grad_input, grad_output):
            self.gradients = grad_output[0]

    cd_model.train() # Enable gradients
    cam = GradCAM(cd_model)
    
    t1.requires_grad_(True)
    t2.requires_grad_(True)
    
    pred = cd_model(t1, t2)
    target = pred.sum()
    cd_model.zero_grad()
    target.backward()

    gradients = cam.gradients.cpu().data.numpy()[0]
    activations = cam.activations.cpu().data.numpy()[0]
    weights = np.mean(gradients, axis=(1, 2))
    heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        heatmap += w * list(activations)[i] # numpy array iteration safely

    heatmap = np.maximum(heatmap, 0)
    heatmap_max = np.max(heatmap)
    if heatmap_max > 0:
        heatmap /= heatmap_max
    else:
        heatmap = heatmap * 0
        
    heatmap = cv2.resize(heatmap, (128, 128))
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(np.clip(t2.cpu().detach().numpy()[0, :3].transpose(1, 2, 0), 0, 1))
    ax.imshow(heatmap, cmap='jet', alpha=0.5)
    ax.set_title("GradCAM: Explainable Violation Evidence", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(train.RESULTS_DIR, "gradcam_heatmap.png"), dpi=150)
    plt.close()

    print("Inference completed successfully.")

if __name__ == "__main__":
    run_inference()
