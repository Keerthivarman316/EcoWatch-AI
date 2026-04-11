"""
EcoWatch AI - Training Pipeline (Master Architecture Upgrade)
Models  : (1) Change Detection  - True Siamese U-Net (ResNet-50 shared encoder)
          (2) Vegetation Seg    - SegFormer (mit_b2 encoder)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, jaccard_score, cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')

DATA_DIR       = r"e:/EcoWatch-AI/Data/Processed"
CHECKPOINT_DIR = r"e:/EcoWatch-AI/checkpoints"
RESULTS_DIR    = r"e:/EcoWatch-AI/results"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)

CFG = {
    "patch_size"  : 128,
    "in_channels" : 4,
    "batch_size"  : 4,        # Lowered to 4 for mit_b2 & resnet50 Siamese on RTX 3050 6GB
    "epochs"      : 30,
    "lr"          : 5e-4,     # Slightly lower LR for larger pretrained models
    "val_split"   : 0.2,
    "num_workers" : 0,
    "device"      : "cuda" if torch.cuda.is_available() else "cpu",
    "seed"        : 42,
}

torch.manual_seed(CFG["seed"])

# =================================================================
#  DATASETS
# =================================================================
class ChangeDetectionDataset(Dataset):
    def __init__(self, X1, X2, Y, augment=False):
        self.X1, self.X2, self.Y = X1.astype(np.float32), X2.astype(np.float32), Y.astype(np.float32)
        self.augment = augment
        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.4), A.GaussNoise(p=0.3),
        ], additional_targets={"image2": "image", "mask": "mask"})

    def __len__(self): return len(self.X1)

    def __getitem__(self, idx):
        t1, t2, mask = self.X1[idx].transpose(1, 2, 0), self.X2[idx].transpose(1, 2, 0), self.Y[idx][0]
        if self.augment:
            res = self.aug(image=t1, image2=t2, mask=mask)
            t1, t2, mask = res["image"], res["image2"], res["mask"]
        return torch.from_numpy(t1.transpose(2,0,1).copy()), torch.from_numpy(t2.transpose(2,0,1).copy()), torch.from_numpy(mask.copy()).unsqueeze(0)

class SegmentationDataset(Dataset):
    def __init__(self, X2, augment=False):
        self.X2, self.augment = X2.astype(np.float32), augment
        self.veg_mask = (X2[:, 3, :, :] > 0.30).astype(np.float32)[:, np.newaxis, :, :]
        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        ], additional_targets={"mask": "mask"})

    def __len__(self): return len(self.X2)

    def __getitem__(self, idx):
        img, mask = self.X2[idx].transpose(1, 2, 0), self.veg_mask[idx][0]
        if self.augment:
            res = self.aug(image=img, mask=mask)
            img, mask = res["image"], res["mask"]
        return torch.from_numpy(img.transpose(2,0,1).copy()), torch.from_numpy(mask.copy()).unsqueeze(0)

# =================================================================
#  MODELS
# =================================================================
class SiameseUNet(nn.Module):
    """
    True Siamese Architecture decoding absolute feature differences.
    Matches Master Reference Prompt (Model 2).
    """
    def __init__(self, in_channels=4):
        super().__init__()
        self.encoder = smp.encoders.get_encoder("resnet50", in_channels=in_channels, weights="imagenet")
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5, add_center_block=False, attention_type=None
        )
        self.head = smp.base.SegmentationHead(in_channels=16, out_channels=1, activation="sigmoid", kernel_size=3)

    def forward(self, t1, t2):
        f1, f2 = self.encoder(t1), self.encoder(t2)
        diff = [torch.abs(x - y) for x, y in zip(f1, f2)]
        return self.head(self.decoder(diff))

def build_seg_model():
    """
    SegFormer B2 matching Master Reference Prompt (Model 1).
    """
    return smp.Unet(
        encoder_name="mit_b2", 
        encoder_weights="imagenet", 
        in_channels=4, 
        classes=1, 
        activation="sigmoid"
    )

# =================================================================
#  LOSS & METRICS
# =================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, pred, target):
        bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()

class DiceBCELoss(nn.Module):
    def forward(self, pred, target):
        intersection = (pred * target).sum()
        dice = 1 - (2. * intersection + 1.) / (pred.sum() + target.sum() + 1.)
        bce = nn.BCELoss()(pred, target)
        return 0.5 * dice + 0.5 * bce

def compute_metrics(p, t):
    p_b, t_b = (p > 0.5).astype(np.uint8).flatten(), t.astype(np.uint8).flatten()
    return {
        "f1": f1_score(t_b, p_b, zero_division=0),
        "iou": jaccard_score(t_b, p_b, zero_division=0),
        "kappa": cohen_kappa_score(t_b, p_b)
    }

# =================================================================
#  TRAINING
# =================================================================
def train_model(model, train_loader, val_loader, loss_fn, is_siamese, name, epochs, lr, dev):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_f1 = 0
    hist = {"train_loss":[],"val_loss":[],"val_f1":[],"val_iou":[],"val_kappa":[]}

    print(f"\nTraining {name} ({epochs} Epochs)")
    for ep in range(1, epochs+1):
        model.train()
        tl = 0
        for batch in train_loader:
            opt.zero_grad()
            if is_siamese:
                t1, t2, mask = [b.to(dev) for b in batch]
                pred = model(t1, t2)
            else:
                img, mask = [b.to(dev) for b in batch]
                pred = model(img)
            loss = loss_fn(pred, mask)
            loss.backward()
            opt.step()
            tl += loss.item()

        model.eval()
        vl, preds, masks = 0, [], []
        with torch.no_grad():
            for batch in val_loader:
                if is_siamese:
                    t1, t2, mask = [b.to(dev) for b in batch]
                    pred = model(t1, t2)
                else:
                    img, mask = [b.to(dev) for b in batch]
                    pred = model(img)
                vl += loss_fn(pred, mask).item()
                preds.append(pred.cpu().numpy()[:,0]); masks.append(mask.cpu().numpy()[:,0])

        m = compute_metrics(np.concatenate(preds), np.concatenate(masks))
        sched.step()
        
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save({"best_f1": best_f1, "metrics": m}, os.path.join(CHECKPOINT_DIR, f"{name}_best.pth"))
            
        print(f"Ep {ep:02d} | TL: {tl/len(train_loader):.4f} | VL: {vl/len(val_loader):.4f} | F1: {m['f1']: .4f} | IoU: {m['iou']:.4f}")

    return hist

if __name__ == "__main__":
    X1 = np.load(os.path.join(DATA_DIR, "X1.npy"))
    X2 = np.load(os.path.join(DATA_DIR, "X2.npy"))
    Y  = np.load(os.path.join(DATA_DIR, "Y.npy"))
    print(f"Loaded {len(X1)} total patches")
    
    n_val = max(1, int(len(X1) * CFG["val_split"]))
    idx = np.random.permutation(len(X1))
    t_idx, v_idx = idx[n_val:], idx[:n_val]
    
    dev = CFG["device"]
    
    # Model 1: Veg Seg
    seg_model = build_seg_model().to(dev)
    seg_dl_t = DataLoader(SegmentationDataset(X2[t_idx], True), batch_size=CFG["batch_size"], shuffle=True)
    seg_dl_v = DataLoader(SegmentationDataset(X2[v_idx], False), batch_size=CFG["batch_size"])
    train_model(seg_model, seg_dl_t, seg_dl_v, DiceBCELoss(), False, "VegSegmentation", CFG["epochs"], CFG["lr"], dev)
    
    # Model 2: Change Det
    cd_model = SiameseUNet(CFG["in_channels"]).to(dev)
    cd_dl_t = DataLoader(ChangeDetectionDataset(X1[t_idx], X2[t_idx], Y[t_idx], True), batch_size=CFG["batch_size"], shuffle=True)
    cd_dl_v = DataLoader(ChangeDetectionDataset(X1[v_idx], X2[v_idx], Y[v_idx], False), batch_size=CFG["batch_size"])
    train_model(cd_model, cd_dl_t, cd_dl_v, FocalLoss(), True, "ChangeDetection", CFG["epochs"], CFG["lr"], dev)

    # FINAL METRICS EVALUATION 
    cd_ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "ChangeDetection_best.pth"), map_location="cpu")
    sg_ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "VegSegmentation_best.pth"), map_location="cpu")
    
    print("\n" + "="*55)
    print("  FINAL MASTER ARCHITECTURE RESULTS")
    print("="*55)
    print(f"  Change Detection (Siamese ResNet50) -> F1: {cd_ckpt['best_f1']:.4f} | IoU: {cd_ckpt['metrics']['iou']:.4f}")
    print(f"  Veg Segmentation (U-Net mit_b2)     -> F1: {sg_ckpt['best_f1']:.4f} | IoU: {sg_ckpt['metrics']['iou']:.4f}")
    print(f"\n  Composite Score: {0.35 * cd_ckpt['best_f1'] + 0.25 * sg_ckpt['metrics']['iou']:.4f}")
    print("="*55)
