import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import train

def run_final_eval():
    dev = train.CFG['device']
    DATA_DIR = train.DATA_DIR
    CHECKPOINT_DIR = train.CHECKPOINT_DIR
    print("Loading Nanjangud Test Data...")
    X1_test = np.load(os.path.join(DATA_DIR, "X1_test.npy"))
    X2_test = np.load(os.path.join(DATA_DIR, "X2_test.npy"))
    Y_test  = np.load(os.path.join(DATA_DIR, "Y_test.npy"))
    print("Loading best model checkpoints...")
    cd_model = train.SiameseUNet(train.CFG['in_channels']).to(dev)
    seg_model = train.build_seg_model().to(dev)

    cd_ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "ChangeDetection_best.pth"), map_location=dev)
    sg_ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "VegSegmentation_best.pth"), map_location=dev)

    cd_model.load_state_dict(cd_ckpt['state_dict'])
    seg_model.load_state_dict(sg_ckpt['state_dict'])
    def test_eval(model, is_siamese, X1, X2, Y):
        ds = train.ChangeDetectionDataset(X1, X2, Y, False) if is_siamese else train.SegmentationDataset(X2, False)
        dl = DataLoader(ds, batch_size=1)
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
        return train.compute_metrics(np.concatenate(preds), np.concatenate(masks))

    print("\n" + "="*55)
    print("  FINAL EVALUATION ON UNSEEN TEST ZONES (NANJANGUD)")
    print("="*55)
    
    cd_m = test_eval(cd_model, True, X1_test, X2_test, Y_test)
    sg_m = test_eval(seg_model, False, None, X2_test, Y_test)
    
    print(f"  Change Detection (Siamese ResNet50) -> F1: {cd_m['f1']:.4f} | IoU: {cd_m['iou']:.4f}")
    print(f"  Veg Segmentation (U-Net mit_b2)     -> F1: {sg_m['f1']:.4f} | IoU: {sg_m['iou']:.4f}")
    print(f"\n  Final Generalization Composite: {0.35 * cd_m['f1'] + 0.25 * sg_m['iou']:.4f}")
    print("="*55)

if __name__ == "__main__":
    run_final_eval()
