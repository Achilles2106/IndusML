import numpy as np


def compute_severity(binary_mask):
    """
    Given a binary defect mask (numpy array, values 0 or 1, any shape),
    returns the percentage of the image area flagged as defective.
    """
    total_pixels = binary_mask.size
    defect_pixels = binary_mask.sum()

    if total_pixels == 0:
        return 0.0

    severity_pct = (defect_pixels / total_pixels) * 100.0
    return float(severity_pct)


def compute_severity_with_blob_breakdown(binary_mask, min_blob_size=5):
    """
    Returns overall severity plus a breakdown of individual defect blobs
    (connected components), each with their own pixel count and percentage.
    Blobs smaller than min_blob_size are ignored as noise.

    Requires scipy.
    """
    from scipy import ndimage

    labeled_array, num_blobs = ndimage.label(binary_mask)

    total_pixels = binary_mask.size
    overall_severity = compute_severity(binary_mask)

    blobs = []
    for blob_id in range(1, num_blobs + 1):
        blob_mask = (labeled_array == blob_id)
        blob_pixel_count = blob_mask.sum()

        if blob_pixel_count < min_blob_size:
            continue

        blob_severity_pct = (blob_pixel_count / total_pixels) * 100.0
        blobs.append({
            "blob_id": blob_id,
            "pixel_count": int(blob_pixel_count),
            "severity_pct": float(blob_severity_pct)
        })

    # Sort largest blob first — usually the most relevant to report first
    blobs.sort(key=lambda b: b["pixel_count"], reverse=True)

    return {
        "overall_severity_pct": overall_severity,
        "num_defect_blobs": len(blobs),
        "blobs": blobs
    }