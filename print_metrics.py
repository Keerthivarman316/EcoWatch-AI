import torch
import os
cd_ckpt = torch.load('e:/EcoWatch-AI/checkpoints/ChangeDetection_best.pth', map_location='cpu')
sg_ckpt = torch.load('e:/EcoWatch-AI/checkpoints/VegSegmentation_best.pth', map_location='cpu')
print('\n' + '='*55)
print('  FINAL MASTER ARCHITECTURE RESULTS')
print('='*55)
print(f"  Change Detection (Siamese ResNet50) -> F1: {cd_ckpt['best_f1']:.4f} | IoU: {cd_ckpt['metrics']['iou']:.4f}")
print(f"  Veg Segmentation (U-Net mit_b2)     -> F1: {sg_ckpt['best_f1']:.4f} | IoU: {sg_ckpt['metrics']['iou']:.4f}")
print(f"\n  Composite Score: {0.35 * cd_ckpt['best_f1'] + 0.25 * sg_ckpt['metrics']['iou']:.4f}")
print('='*55)
