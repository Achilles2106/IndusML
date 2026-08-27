import pickle
import numpy as np
import torch
from src.backbone.feature_extractor import FeatureExtractor

# ---- HARDCODED PATHS FOR TONIGHT ----
STATS_PATH = "artifacts/anomaly_stats/tile.pkl"
# --------------------------------------


class AnomalyScorer:
    def __init__(self, stats_path=STATS_PATH, extractor=None):
        with open(stats_path, "rb") as f:
            self.stats = pickle.load(f)

        self.means = self.stats["means"]                    # [H*W, C]
        self.inv_covariances = self.stats["inv_covariances"]  # [H*W, C, C]
        self.grid_h = self.stats["grid_h"]
        self.grid_w = self.stats["grid_w"]
        self.num_channels = self.stats["num_channels"]
        self.selected_channels = self.stats["selected_channels"]

        self.extractor = extractor if extractor is not None else FeatureExtractor()

    def score_image(self, image_path):
        """
        Returns a 2D numpy array of shape [grid_h, grid_w] containing
        the Mahalanobis distance (anomaly score) at each spatial location.
        """
        combined = self.extractor.get_combined_features(image_path)  # [C_full, H, W]
        combined = combined.cpu().numpy()

        C_full, H, W = combined.shape
        assert H == self.grid_h and W == self.grid_w, \
            f"Feature grid size mismatch: got {H}x{W}, expected {self.grid_h}x{self.grid_w}"

        # Reshape to [H*W, C_full]
        vectors = combined.reshape(C_full, H * W).transpose(1, 0)  # [H*W, C_full]

        # Select only the channels used when building stats (must match build_stats.py)
        vectors = vectors[:, self.selected_channels]  # [H*W, num_channels]

        anomaly_map = np.zeros(H * W, dtype=np.float32)

        for loc in range(H * W):
            x = vectors[loc]                  # [C]
            mean = self.means[loc]            # [C]
            inv_cov = self.inv_covariances[loc]  # [C, C]

            diff = x - mean
            # Mahalanobis distance: sqrt(diff^T * inv_cov * diff)
            mahalanobis_dist = np.sqrt(diff @ inv_cov @ diff.T)
            anomaly_map[loc] = mahalanobis_dist

        anomaly_map = anomaly_map.reshape(H, W)
        return anomaly_map

    def score_and_upsample(self, image_path, output_size=(256, 256)):
        """
        Returns the anomaly map upsampled to output_size (default 256x256,
        matching the model's input resolution) using bilinear interpolation.
        """
        anomaly_map = self.score_image(image_path)  # [H, W]

        anomaly_tensor = torch.from_numpy(anomaly_map).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        upsampled = torch.nn.functional.interpolate(
            anomaly_tensor, size=output_size, mode="bilinear", align_corners=False
        )
        return upsampled.squeeze().numpy()  # [output_h, output_w]