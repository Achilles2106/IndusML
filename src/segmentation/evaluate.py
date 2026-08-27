import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.segmentation.dataset import build_stratified_splits, TileSegmentationDataset
from src.segmentation.unet import UNet

# ---- HARDCODED CONFIG FOR TONIGHT ----
CHECKPOINT_PATH = "artifacts/checkpoints/unet_tile.pt"
REPORT_OUTPUT_PATH = "logs/eval_report_unet_tile.json"
THRESHOLD = 0.5
# ---------------------------------------


def compute_iou_dice(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    total = pred.sum() + gt.sum()

    iou = 1.0 if union == 0 else intersection / union
    dice = 1.0 if total == 0 else (2.0 * intersection) / total
    return iou, dice


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_items = build_stratified_splits()
    print(f"Test set size: {len(test_items)}")

    model = UNet(in_channels=3, out_channels=1, base_channels=32).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    # Group results by defect type
    results_by_type = {}

    dataset = TileSegmentationDataset(test_items, augment=False)

    with torch.no_grad():
        for idx, (image_tensor, mask_tensor) in enumerate(dataset):
            image_path, _ = test_items[idx]
            defect_type = os.path.basename(os.path.dirname(image_path))

            image_batch = image_tensor.unsqueeze(0).to(device)
            logits = model(image_batch)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_mask = (probs >= THRESHOLD).astype(np.uint8)

            gt_mask = mask_tensor.squeeze().numpy().astype(np.uint8)

            iou, dice = compute_iou_dice(pred_mask, gt_mask)

            results_by_type.setdefault(defect_type, {"iou": [], "dice": []})
            results_by_type[defect_type]["iou"].append(iou)
            results_by_type[defect_type]["dice"].append(dice)

    print("\n--- U-Net Test Set Results (per defect type) ---")
    report = {}
    all_ious = []
    all_dices = []

    for defect_type, scores in sorted(results_by_type.items()):
        mean_iou = float(np.mean(scores["iou"]))
        mean_dice = float(np.mean(scores["dice"]))
        n = len(scores["iou"])
        print(f"{defect_type} (n={n}): IoU={mean_iou:.4f}, Dice={mean_dice:.4f}")
        report[defect_type] = {"iou": mean_iou, "dice": mean_dice, "n": n}
        all_ious.extend(scores["iou"])
        all_dices.extend(scores["dice"])

    overall_iou = float(np.mean(all_ious))
    overall_dice = float(np.mean(all_dices))
    print(f"\nOverall: IoU={overall_iou:.4f}, Dice={overall_dice:.4f}")

    report["overall"] = {"iou": overall_iou, "dice": overall_dice, "n": len(all_ious)}

    os.makedirs(os.path.dirname(REPORT_OUTPUT_PATH), exist_ok=True)
    with open(REPORT_OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()