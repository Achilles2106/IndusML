import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.segmentation.dataset import (
    build_stratified_splits,
    TileSegmentationDataset,
)
from src.segmentation.unet import UNet

# ---- HARDCODED CONFIG FOR TONIGHT ----
CHECKPOINT_PATH = "artifacts/checkpoints/unet_tile.pt"
BATCH_SIZE = 4
NUM_EPOCHS = 60
LEARNING_RATE = 1e-3
# ---------------------------------------


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_score.mean()


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.dice(logits, targets) + self.bce(logits, targets)


def compute_batch_iou(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = ((preds_flat + targets_flat) >= 1).float().sum(dim=1)

    # Avoid division by zero; treat empty union as perfect match
    iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
    return iou.mean().item()


def main():
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build splits
    train_items, val_items, test_items = build_stratified_splits()
    print(f"Train: {len(train_items)} | Val: {len(val_items)} | Test: {len(test_items)}")

    train_dataset = TileSegmentationDataset(train_items, augment=True)
    val_dataset = TileSegmentationDataset(val_items, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = UNet(in_channels=3, out_channels=1, base_channels=32).to(device)

    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_iou = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):
        # ---- Train ----
        model.train()
        train_loss_total = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * images.size(0)

        train_loss_avg = train_loss_total / len(train_dataset)

        # ---- Validate ----
        model.eval()
        val_loss_total = 0.0
        val_iou_total = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)

                val_loss_total += loss.item() * images.size(0)
                val_iou_total += compute_batch_iou(logits, masks) * images.size(0)

        val_loss_avg = val_loss_total / len(val_dataset)
        val_iou_avg = val_iou_total / len(val_dataset)

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss_avg:.4f} | "
              f"Val Loss: {val_loss_avg:.4f} | "
              f"Val IoU: {val_iou_avg:.4f}")

        # ---- Checkpoint on best val IoU ----
        if val_iou_avg > best_val_iou:
            best_val_iou = val_iou_avg
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> New best model saved (Val IoU: {best_val_iou:.4f})")

    print(f"\nTraining complete. Best Val IoU: {best_val_iou:.4f}")
    print(f"Best model saved at: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()