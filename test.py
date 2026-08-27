import numpy as np
import torch

from src.segmentation.unet import UNet
from src.segmentation.dataset import TileSegmentationDataset, build_stratified_splits
from src.severity.estimate import compute_severity, compute_severity_with_blob_breakdown

# ---- HARDCODED CONFIG ----
CHECKPOINT_PATH = "artifacts/checkpoints/unet_tile.pt"
THRESHOLD = 0.5
# ---------------------------


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, out_channels=1, base_channels=32).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    _, _, test_items = build_stratified_splits()
    dataset = TileSegmentationDataset(test_items, augment=False)

    print(f"Running severity estimation on {len(dataset)} test images...\n")

    with torch.no_grad():
        for idx in range(len(dataset)):
            image_tensor, mask_tensor = dataset[idx]
            image_path, _ = test_items[idx]

            image_batch = image_tensor.unsqueeze(0).to(device)
            logits = model(image_batch)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_mask = (probs >= THRESHOLD).astype(np.uint8)

            simple_severity = compute_severity(pred_mask)
            detailed = compute_severity_with_blob_breakdown(pred_mask)

            print(f"{image_path}")
            print(f"  Severity: {simple_severity:.2f}%")
            print(f"  Blobs: {detailed['num_defect_blobs']}, "
                  f"Largest blob: {detailed['blobs'][0]['severity_pct']:.2f}%"
                  if detailed['blobs'] else "  No blobs above minimum size")
            print()


if __name__ == "__main__":
    main()