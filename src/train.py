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
    "patch_size"  : 256,      # Increased to 256 for more context
    "in_channels" : 4,
    "batch_size"  : 2,        # Lowered due to larger patches & Unet++
    "epochs"      : 100,      # Increased epochs with early stopping
    "patience"    : 15,       # Early-stopping patience
    "lr"          : 1e-4,     # Stable LR for fine-tuning
    "warmup_ep"   : 5,        # LR warmup epochs before cosine annealing
    "val_split"   : 0.2,
    "num_workers" : 2,
    "pin_memory"  : True,
    "amp"         : True,     # Mixed precision (fp16) for faster GPU training
    "device"      : "cuda" if torch.cuda.is_available() else "cpu",
    "seed"        : 42,
}

torch.manual_seed(CFG["seed"])
class ChangeDetectionDataset(Dataset):
    def __init__(self, X1, X2, Y, augment=False):
        self.X1, self.X2, self.Y = X1.astype(np.float32), X2.astype(np.float32), Y.astype(np.float32)
        self.augment = augment
        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.4), A.GaussNoise(p=0.3),
            A.OneOf([
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(p=0.3),
                A.ElasticTransform(p=0.3),
            ], p=0.4),
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
class SiameseUNet(nn.Module):
    """
    True Siamese Architecture decoding absolute feature differences.
    Matches Master Reference Prompt (Model 2).
    """
    def __init__(self, in_channels=4):
        super().__init__()
        self.encoder = smp.encoders.get_encoder("resnet50", in_channels=in_channels, weights="imagenet")
        self.decoder = smp.decoders.unetplusplus.decoder.UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5, attention_type="scse"
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
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        tp = (pred * target).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - tversky

class DiceBCELoss(nn.Module):
    def forward(self, pred, target):
        intersection = (pred * target).sum()
        dice = 1 - (2. * intersection + 1.) / (pred.sum() + target.sum() + 1.)
        bce = nn.functional.binary_cross_entropy(pred, target)
        return 0.5 * dice + 0.5 * bce

def compute_metrics(p, t):
    p_b, t_b = (p > 0.5).astype(np.uint8).flatten(), t.astype(np.uint8).flatten()
    return {
        "f1": f1_score(t_b, p_b, zero_division=0),
        "iou": jaccard_score(t_b, p_b, zero_division=0),
        "kappa": cohen_kappa_score(t_b, p_b)
    }
def train_model(model, train_loader, val_loader, loss_fn, is_siamese, name, epochs, lr, dev):
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2, eta_min=1e-6)
    scaler  = torch.amp.GradScaler(enabled=(CFG["amp"] and dev == "cuda"))
    best_f1 = 0
    no_imp  = 0  # Early stopping counter
    hist    = {"train_loss":[],"val_loss":[],"val_f1":[],"val_iou":[],"val_kappa":[]}

    print(f"\nTraining {name} ({epochs} Epochs | Early Stop Patience={CFG['patience']})")
    for ep in range(1, epochs+1):
        model.train()
        tl = 0
        for batch in train_loader:
            opt.zero_grad()
            with torch.amp.autocast(device_type=dev, enabled=(CFG["amp"] and dev == "cuda")):
                if is_siamese:
                    t1, t2, mask = [b.to(dev) for b in batch]
                    pred = model(t1, t2)
                else:
                    img, mask = [b.to(dev) for b in batch]
                    pred = model(img)
                loss = loss_fn(pred, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            tl += loss.item()

        model.eval()
        vl, preds, masks = 0, [], []
        with torch.no_grad():
            for batch in val_loader:
                with torch.amp.autocast(device_type=dev, enabled=(CFG["amp"] and dev == "cuda")):
                    if is_siamese:
                        t1, t2, mask = [b.to(dev) for b in batch]
                        pred = model(t1, t2)
                    else:
                        img, mask = [b.to(dev) for b in batch]
                        pred = model(img)
                vl += loss_fn(pred, mask).item()
                preds.append(pred.cpu().float().numpy()[:,0])
                masks.append(mask.cpu().float().numpy()[:,0])

        m = compute_metrics(np.concatenate(preds), np.concatenate(masks))
        sched.step()
        hist["val_f1"].append(m["f1"])

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            no_imp = 0
            torch.save({"state_dict": model.state_dict(), "best_f1": best_f1, "metrics": m}, os.path.join(CHECKPOINT_DIR, f"{name}_best.pth"))
        else:
            no_imp += 1

        print(f"Ep {ep:03d} | TL: {tl/len(train_loader):.4f} | VL: {vl/len(val_loader):.4f} | F1: {m['f1']:.4f} | IoU: {m['iou']:.4f} | LR: {opt.param_groups[0]['lr']:.2e} | No-Imp: {no_imp}")

        if no_imp >= CFG["patience"]:
            print(f"  >> Early stopping triggered at epoch {ep}. Best F1: {best_f1:.4f}")
            break

    return hist

if __name__ == "__main__":
    X1_train = np.load(os.path.join(DATA_DIR, "X1_train.npy"))
    X2_train = np.load(os.path.join(DATA_DIR, "X2_train.npy"))
    Y_train  = np.load(os.path.join(DATA_DIR, "Y_train.npy"))
    
    X1_val = np.load(os.path.join(DATA_DIR, "X1_val.npy"))
    X2_val = np.load(os.path.join(DATA_DIR, "X2_val.npy"))
    Y_val  = np.load(os.path.join(DATA_DIR, "Y_val.npy"))
    
    X1_test = np.load(os.path.join(DATA_DIR, "X1_test.npy"))
    X2_test = np.load(os.path.join(DATA_DIR, "X2_test.npy"))
    Y_test  = np.load(os.path.join(DATA_DIR, "Y_test.npy"))
    
    print(f"Loaded Geographical Splits -> Train: {len(X1_train)}, Val: {len(X1_val)}, Test: {len(X1_test)}")
    
    dev = CFG["device"]
    def get_dl(ds, batch_size, shuffle=False):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=CFG["num_workers"], pin_memory=CFG["pin_memory"])
    seg_model = build_seg_model().to(dev)
    if not os.path.exists(os.path.join(CHECKPOINT_DIR, "VegSegmentation_best.pth")):
        seg_dl_t = get_dl(SegmentationDataset(X2_train, True), batch_size=CFG["batch_size"], shuffle=True)
        seg_dl_v = get_dl(SegmentationDataset(X2_val, False), batch_size=CFG["batch_size"])
        train_model(seg_model, seg_dl_t, seg_dl_v, DiceBCELoss(), False, "VegSegmentation", CFG["epochs"], CFG["lr"], dev)
    else:
        print("\nSkipping VegSegmentation training - Checkpoint already exists.")
    cd_model = SiameseUNet(CFG["in_channels"]).to(dev)
    cd_dl_t = get_dl(ChangeDetectionDataset(X1_train, X2_train, Y_train, True), batch_size=CFG["batch_size"], shuffle=True)
    cd_dl_v = get_dl(ChangeDetectionDataset(X1_val, X2_val, Y_val, False), batch_size=CFG["batch_size"])
    train_model(cd_model, cd_dl_t, cd_dl_v, TverskyLoss(), True, "ChangeDetection", CFG["epochs"], CFG["lr"], dev)
    print("\n" + "="*55)
    print("  FINAL EVALUATION ON UNSEEN TEST ZONES (NANJANGUD)")
    print("="*55)
    
    def test_eval(model, is_siamese, X1, X2, Y, name):
        dl = DataLoader(ChangeDetectionDataset(X1, X2, Y, False) if is_siamese else SegmentationDataset(X2, False), batch_size=1)
        model.eval()
        preds, masks = [], []
        with torch.no_grad():
            for batch in dl:
                if is_siamese:
                    t1, t2, mask = [b.to(dev) for b in batch]
                    pred = model(t1, t2)
                else:
                    img, mask = [b.to(dev) for b in batch]
                    pred = model(img)
                preds.append(pred.cpu().numpy()[:,0]); masks.append(mask.cpu().numpy()[:,0])
        return compute_metrics(np.concatenate(preds), np.concatenate(masks))
    cd_ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "ChangeDetection_best.pth"), map_location=dev)
    sg_ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "VegSegmentation_best.pth"), map_location=dev)
    
    cd_model.load_state_dict(cd_ckpt["state_dict"])
    seg_model.load_state_dict(sg_ckpt["state_dict"])
    
    cd_m = test_eval(cd_model, True, X1_test, X2_test, Y_test, "ChangeDetection")
    sg_m = test_eval(seg_model, False, None, X2_test, Y_test, "VegSegmentation")
    
    print(f"  Change Detection (Siamese ResNet50) -> F1: {cd_m['f1']:.4f} | IoU: {cd_m['iou']:.4f}")
    print(f"  Veg Segmentation (U-Net mit_b2)     -> F1: {sg_m['f1']:.4f} | IoU: {sg_m['iou']:.4f}")
    print(f"\n  Final Generalization Composite: {0.35 * cd_m['f1'] + 0.25 * sg_m['iou']:.4f}")
    print("="*55)
