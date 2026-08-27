import json

HALF1_REPORT_PATH = "logs/eval_report_tile.json"
HALF2_REPORT_PATH = "logs/eval_report_unet_tile.json"
CONSOLIDATED_OUTPUT_PATH = "logs/consolidated_report.json"


def main():
    with open(HALF1_REPORT_PATH, "r") as f:
        half1 = json.load(f)

    with open(HALF2_REPORT_PATH, "r") as f:
        half2 = json.load(f)

    consolidated = {
        "half1_anomaly_detection": {
            "description": "Unsupervised anomaly detection (ResNet-18 backbone, "
                            "Mahalanobis distance over layer2+layer3 features). "
                            "Trained on train/good only, no labeled masks used.",
            "threshold": half1["threshold"],
            "classification_metrics": half1["classification_metrics"],
            "segmentation_metrics_per_defect_type": half1["segmentation_metrics_per_defect_type"],
        },
        "half2_unet_segmentation": {
            "description": "Supervised segmentation (U-Net). Trained on labeled "
                            "image/mask pairs, evaluated on held-out test split.",
            "segmentation_metrics_per_defect_type": {
                k: v for k, v in half2.items() if k != "overall"
            },
            "overall": half2["overall"],
        },
        "comparison_summary": {
            "note": "Half 1 achieves strong image-level classification (high AUROC, "
                    "perfect recall) but weak pixel-level localization on thin defects "
                    "(notably 'crack'), due to the coarse spatial resolution of "
                    "statistical feature-grid anomaly maps. Half 2 (U-Net) substantially "
                    "outperforms Half 1 on localization across every defect type, "
                    "most dramatically on 'crack', because direct pixel-level supervision "
                    "is not limited by this resolution bottleneck. This motivates using "
                    "Half 2 as the primary localization/severity source, with Half 1 "
                    "retained as a complementary detector for novel or uncharacterized "
                    "defect types it was never trained on."
        }
    }

    with open(CONSOLIDATED_OUTPUT_PATH, "w") as f:
        json.dump(consolidated, f, indent=2)

    print(f"Consolidated report saved to {CONSOLIDATED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()