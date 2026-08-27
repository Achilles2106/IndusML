import os
import numpy as np
from PIL import Image


def load_ground_truth_mask(test_image_path, ground_truth_root, target_size=(256, 256)):
    """
    Given a path like data/mvtec_ad/tile/test/crack/000.png,
    finds and loads the corresponding ground truth mask:
    data/mvtec_ad/tile/ground_truth/crack/000_mask.png

    Returns a binary numpy array of shape target_size, or None if
    the image is from the 'good' folder (no ground truth exists).
    """
    defect_type = os.path.basename(os.path.dirname(test_image_path))

    if defect_type == "good":
        return None

    fname = os.path.basename(test_image_path)
    name_no_ext, ext = os.path.splitext(fname)
    mask_fname = f"{name_no_ext}_mask{ext}"

    mask_path = os.path.join(ground_truth_root, defect_type, mask_fname)

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Expected ground truth mask not found: {mask_path}")

    mask = Image.open(mask_path).convert("L")  # grayscale
    # Nearest-neighbor resize to preserve binary values
    mask = mask.resize(target_size, resample=Image.NEAREST)
    mask_array = np.array(mask)

    # Binarize (MVTec masks are typically 0 or 255)
    binary_mask = (mask_array > 127).astype(np.uint8)
    return binary_mask