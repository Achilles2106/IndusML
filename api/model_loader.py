import torch
import os
import urllib.request

from src.backbone.feature_extractor import FeatureExtractor
from src.anomaly_detection.score import AnomalyScorer
from src.segmentation.unet import UNet

# ---- HARDCODED PATHS FOR TONIGHT ----
UNET_CHECKPOINT_PATH = "artifacts/checkpoints/unet_tile.pt"
ANOMALY_STATS_PATH = "artifacts/anomaly_stats/tile.pkl"
ANOMALY_STATS_URL = "https://github.com/Achilles2106/IndusML/releases/download/v1.0-artifacts/tile.pkl"
HALF1_THRESHOLD = 15.1819   # from your Half 1 baseline eval report
UNET_THRESHOLD = 0.5
# --------------------------------------

# Download tile.pkl if not present (not committed to repo — too large for GitHub's 100MB limit)
if not os.path.exists(ANOMALY_STATS_PATH):
    os.makedirs(os.path.dirname(ANOMALY_STATS_PATH), exist_ok=True)
    print(f"Downloading {ANOMALY_STATS_PATH} from release asset...")
    urllib.request.urlretrieve(ANOMALY_STATS_URL, ANOMALY_STATS_PATH)
    print("Download complete.")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Shared backbone + Half 1 scorer (loaded once at startup)
_feature_extractor = FeatureExtractor(device=_device)
_anomaly_scorer = AnomalyScorer(stats_path=ANOMALY_STATS_PATH, extractor=_feature_extractor)

# Half 2 U-Net (loaded once at startup)
_unet_model = UNet(in_channels=3, out_channels=1, base_channels=32).to(_device)
_unet_model.load_state_dict(torch.load(UNET_CHECKPOINT_PATH, map_location=_device))
_unet_model.eval()


def get_device():
    return _device


def get_feature_extractor():
    return _feature_extractor


def get_anomaly_scorer():
    return _anomaly_scorer


def get_unet_model():
    return _unet_model


def get_half1_threshold():
    return HALF1_THRESHOLD


def get_unet_threshold():
    return UNET_THRESHOLD