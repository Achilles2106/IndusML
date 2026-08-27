import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# ---- HARDCODED PATHS FOR TONIGHT ----
TEST_ROOT = "data/mvtec/tile/test"
GROUND_TRUTH_ROOT = "data/mvtec/tile/ground_truth"
# --------------------------------------

DEFECT_TYPES = ["crack", "glue_strip", "gray_stroke", "oil", "rough"]
IMAGE_SIZE = (256, 256)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _find_mask_path(image_path, defect_type):
    fname = os.path.basename(image_path)
    name_no_ext, ext = os.path.splitext(fname)
    mask_fname = f"{name_no_ext}_mask{ext}"
    return os.path.join(GROUND_TRUTH_ROOT, defect_type, mask_fname)


def build_stratified_splits(seed=42, train_frac=0.70, val_frac=0.15):
    """
    Returns three lists of (image_path, mask_path) tuples: train, val, test.
    Stratified by defect type so each split has a proportional mix.
    """
    random.seed(seed)

    train_items, val_items, test_items = [], [], []

    for defect_type in DEFECT_TYPES:
        folder = os.path.join(TEST_ROOT, defect_type)
        image_files = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        pairs = []
        for fname in image_files:
            image_path = os.path.join(folder, fname)
            mask_path = _find_mask_path(image_path, defect_type)
            if os.path.exists(mask_path):
                pairs.append((image_path, mask_path))

        random.shuffle(pairs)

        n = len(pairs)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train_items.extend(pairs[:n_train])
        val_items.extend(pairs[n_train:n_train + n_val])
        test_items.extend(pairs[n_train + n_val:])

    return train_items, val_items, test_items


class TileSegmentationDataset(Dataset):
    def __init__(self, items, augment=False):
        """
        items: list of (image_path, mask_path) tuples
        augment: if True, applies simple flips (only sensible for train split)
        """
        self.items = items
        self.augment = augment

        self.image_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_path, mask_path = self.items[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(IMAGE_SIZE, resample=Image.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        image_tensor = self.image_transform(image)

        mask_array = np.array(mask)
        mask_binary = (mask_array > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)  # [1, H, W]

        return image_tensor, mask_tensor
