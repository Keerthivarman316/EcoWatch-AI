import train
import torch
import numpy as np
import os
from torch.utils.data import DataLoader

if __name__ == '__main__':
    dev = train.CFG['device']
    X1 = np.load(os.path.join(train.DATA_DIR, 'X1.npy'))
    X2 = np.load(os.path.join(train.DATA_DIR, 'X2.npy'))
    Y  = np.load(os.path.join(train.DATA_DIR, 'Y.npy'))
    n_val = max(1, int(len(X1) * train.CFG['val_split']))
    idx = np.random.permutation(len(X1))
    t_idx, v_idx = idx[n_val:], idx[:n_val]

    cd_model = train.SiameseUNet(train.CFG['in_channels']).to(dev)
    cd_dl_t = DataLoader(train.ChangeDetectionDataset(X1[t_idx], X2[t_idx], Y[t_idx], True), batch_size=train.CFG['batch_size'], shuffle=True)
    cd_dl_v = DataLoader(train.ChangeDetectionDataset(X1[v_idx], X2[v_idx], Y[v_idx], False), batch_size=train.CFG['batch_size'])
    train.train_model(cd_model, cd_dl_t, cd_dl_v, train.FocalLoss(), True, 'ChangeDetection', train.CFG['epochs'], train.CFG['lr'], dev)

    cd_ckpt = torch.load(os.path.join(train.CHECKPOINT_DIR, 'ChangeDetection_best.pth'), map_location='cpu')
    sg_ckpt = torch.load(os.path.join(train.CHECKPOINT_DIR, 'VegSegmentation_best.pth'), map_location='cpu')
    
    print('\n' + '='*55)
    print('  FINAL MASTER ARCHITECTURE RESULTS')
    print('='*55)
    print(f"  Change Detection (Siamese ResNet50) -> F1: {cd_ckpt['best_f1']:.4f} | IoU: {cd_ckpt['metrics']['iou']:.4f}")
    print(f"  Veg Segmentation (U-Net mit_b2)     -> F1: {sg_ckpt['best_f1']:.4f} | IoU: {sg_ckpt['metrics']['iou']:.4f}")
    print(f"\n  Composite Score: {0.35 * cd_ckpt['best_f1'] + 0.25 * sg_ckpt['metrics']['iou']:.4f}")
    print('='*55)
