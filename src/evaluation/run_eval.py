import os
import json
import numpy as np
from src.anomaly_detection.score import AnomalyScorer
from src.anomaly_detection.threshold import compute_threshold
from src.evaluation.mask_utils import load_ground_truth_mask
from src.evaluation.segmentation_metrics import compute_iou, compute_dice
from src.evaluation.classification_metrics import compute_classification_metrics

# ---- HARDCODED PATHS FOR TONIGHT ----
TEST_ROOT = "data/mvtec/tile/test"
GROUND_TRUTH_ROOT = "data/mvtec/tile/ground_truth"
REPORT_OUTPUT_PATH = "logs/eval_report_tile.json"
OUTPUT_SIZE = (256, 256)
# --------------------------------------


def main():
    os.makedirs(os.path.dirname(REPORT_OUTPUT_PATH), exist_ok=True)

    scorer = AnomalyScorer()

    # Step 1: compute threshold from test/good distribution
    threshold = compute_threshold(scorer, output_size=OUTPUT_SIZE)

    # Step 2: iterate over all test subfolders (good + each defect type)
    defect_types = sorted([
        d for d in os.listdir(TEST_ROOT)
        if os.path.isdir(os.path.join(TEST_ROOT, d))
    ])
    print(f"\nFound test subfolders: {defect_types}")

    y_true = []       # 0 = good, 1 = defective
    y_pred = []       # 0/1 predicted label
    y_scores = []     # max anomaly score per image (used for AUROC)

    per_defect_type_iou = {}
    per_defect_type_dice = {}

    for defect_type in defect_types:
        folder = os.path.join(TEST_ROOT, defect_type)
        image_files = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        ious = []
        dices = []

        print(f"\nProcessing '{defect_type}' ({len(image_files)} images)...")

        for fname in image_files:
            path = os.path.join(folder, fname)

            anomaly_map = scorer.score_and_upsample(path, output_size=OUTPUT_SIZE)
            image_score = anomaly_map.max()  # image-level score for classification

            pred_mask = (anomaly_map >= threshold).astype(np.uint8)

            true_label = 0 if defect_type == "good" else 1
            pred_label = 1 if image_score >= threshold else 0

            y_true.append(true_label)
            y_pred.append(pred_label)
            y_scores.append(image_score)

            # Segmentation metrics only apply to defective images (have ground truth)
            gt_mask = load_ground_truth_mask(path, GROUND_TRUTH_ROOT, target_size=OUTPUT_SIZE)
            if gt_mask is not None:
                iou = compute_iou(pred_mask, gt_mask)
                dice = compute_dice(pred_mask, gt_mask)
                ious.append(iou)
                dices.append(dice)

        if ious:
            per_defect_type_iou[defect_type] = float(np.mean(ious))
            per_defect_type_dice[defect_type] = float(np.mean(dices))
            print(f"  Mean IoU: {np.mean(ious):.4f}, Mean Dice: {np.mean(dices):.4f}")

    # Step 3: classification metrics (across all images)
    cls_metrics = compute_classification_metrics(y_true, y_pred, y_scores)

    print("\n--- Classification Metrics (image-level) ---")
    print(f"Precision: {cls_metrics['precision']:.4f}")
    print(f"Recall:    {cls_metrics['recall']:.4f}")
    print(f"F1:        {cls_metrics['f1']:.4f}")
    print(f"AUROC:     {cls_metrics['auroc']:.4f}" if cls_metrics['auroc'] is not None else "AUROC: N/A")

    print("\n--- Segmentation Metrics (per defect type) ---")
    for dt in per_defect_type_iou:
        print(f"{dt}: IoU={per_defect_type_iou[dt]:.4f}, Dice={per_defect_type_dice[dt]:.4f}")

    # Step 4: save full report
    report = {
        "threshold": float(threshold),
        "classification_metrics": cls_metrics,
        "segmentation_metrics_per_defect_type": {
            dt: {"iou": per_defect_type_iou[dt], "dice": per_defect_type_dice[dt]}
            for dt in per_defect_type_iou
        }
    }

    with open(REPORT_OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nFull report saved to {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()