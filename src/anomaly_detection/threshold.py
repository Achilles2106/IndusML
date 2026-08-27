import os
import numpy as np
from src.anomaly_detection.score import AnomalyScorer

# ---- HARDCODED PATHS FOR TONIGHT ----
TEST_GOOD_DIR = "data/mvtec/tile/test/good"
# --------------------------------------

PERCENTILE = 95  # threshold set at this percentile of good images' pixel-level scores


def compute_threshold(scorer, test_good_dir=TEST_GOOD_DIR, percentile=PERCENTILE, output_size=(256, 256)):
    """
    Runs all test/good images through the scorer, collects every
    pixel-level anomaly score across all of them, and returns the
    threshold at the given percentile of that combined distribution.
    """
    image_files = sorted([
        f for f in os.listdir(test_good_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    all_scores = []

    print(f"Computing threshold from {len(image_files)} test/good images...")
    for idx, fname in enumerate(image_files):
        path = os.path.join(test_good_dir, fname)
        anomaly_map = scorer.score_and_upsample(path, output_size=output_size)
        all_scores.append(anomaly_map.flatten())

        if (idx + 1) % 10 == 0 or (idx + 1) == len(image_files):
            print(f"  Processed {idx + 1}/{len(image_files)}")

    all_scores = np.concatenate(all_scores)
    threshold = np.percentile(all_scores, percentile)

    print(f"\nScore distribution over test/good pixels:")
    print(f"  mean={all_scores.mean():.4f}, std={all_scores.std():.4f}, "
          f"min={all_scores.min():.4f}, max={all_scores.max():.4f}")
    print(f"Threshold at {percentile}th percentile: {threshold:.4f}")

    return threshold