import os
import pickle
import numpy as np
import torch
from src.backbone.feature_extractor import FeatureExtractor

# ---- HARDCODED PATHS FOR TONIGHT ----
TRAIN_GOOD_DIR = "data/mvtec/tile/train/good"
STATS_OUTPUT_PATH = "artifacts/anomaly_stats/tile.pkl"
# --------------------------------------

EPSILON = 0.01  # regularization constant for covariance invertibility


def main():
    os.makedirs(os.path.dirname(STATS_OUTPUT_PATH), exist_ok=True)

    extractor = FeatureExtractor()
    print(f"Using device: {extractor.device}")

    image_files = sorted([
        f for f in os.listdir(TRAIN_GOOD_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    print(f"Found {len(image_files)} training images in {TRAIN_GOOD_DIR}")

    if len(image_files) == 0:
        raise RuntimeError("No training images found. Check TRAIN_GOOD_DIR path.")

    all_features = []  # will hold tensors of shape [C, H, W]

    for idx, fname in enumerate(image_files):
        path = os.path.join(TRAIN_GOOD_DIR, fname)
        combined = extractor.get_combined_features(path)  # [C, H, W]
        all_features.append(combined.cpu().numpy())
        if (idx + 1) % 20 == 0 or (idx + 1) == len(image_files):
            print(f"Processed {idx + 1}/{len(image_files)} images")

    # Stack into array of shape [N, C, H, W]
    all_features = np.stack(all_features, axis=0)
    N, C, H, W = all_features.shape
    print(f"Feature tensor shape: N={N}, C={C}, H={H}, W={W}")

    # Reshape to [N, C, H*W] for per-location processing
    all_features = all_features.reshape(N, C, H * W)

    NUM_SELECTED_CHANNELS = 100
    rng = np.random.default_rng(seed=42)
    selected_channels = rng.choice(C, size=NUM_SELECTED_CHANNELS, replace=False)
    selected_channels = np.sort(selected_channels)

    all_features = all_features[:, selected_channels, :]  # [N, 100, H*W]
    C = NUM_SELECTED_CHANNELS

    means = np.zeros((H * W, C), dtype=np.float32)
    inv_covariances = np.zeros((H * W, C, C), dtype=np.float32)

    print("Computing per-location mean and covariance...")
    for loc in range(H * W):
        # vectors at this location across all N training images: shape [N, C]
        vectors = all_features[:, :, loc]

        mean = np.mean(vectors, axis=0)  # [C]
        # rowvar=False because each row is a sample, each column a feature
        cov = np.cov(vectors, rowvar=False)  # [C, C]

        # Regularize for invertibility
        cov_reg = cov + EPSILON * np.identity(C)

        inv_cov = np.linalg.inv(cov_reg)

        means[loc] = mean
        inv_covariances[loc] = inv_cov

        if (loc + 1) % 200 == 0 or (loc + 1) == H * W:
            print(f"  Processed location {loc + 1}/{H * W}")

    stats = {
        "means": means,                    # [H*W, C]
        "inv_covariances": inv_covariances,  # [H*W, C, C]
        "grid_h": H,
        "grid_w": W,
        "num_channels": C,
        "num_training_images": N,
        "selected_channels": selected_channels,
    }

    with open(STATS_OUTPUT_PATH, "wb") as f:
        pickle.dump(stats, f)

    print(f"\nSaved anomaly stats to {STATS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()