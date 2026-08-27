# Industrial Visual Inspection System

An automated visual inspection pipeline for detecting and localizing surface anomalies on industrial materials, combining unsupervised anomaly detection with supervised segmentation, served through a REST API and a web dashboard.

**Live demo:** `indusml-production.up.railway.app`

---

## Overview

This system inspects images of industrial surfaces and answers three questions:

1. **Is this surface defective?** (classification)
2. **Where exactly is the defect?** (pixel-level localization)
3. **How severe is it?** (affected-area percentage)

It does this using two complementary detection approaches running on a shared feature-extraction backbone, rather than a single model — a deliberate architectural choice that this project's own evaluation results end up justifying.

---

## Architecture

```
                          Input Image
                               |
                    Pretrained ResNet-18 Backbone
                    (ImageNet weights, frozen, transfer learning)
                               |
                +--------------+--------------+
                |                             |
      Half 1: Anomaly Detection      Half 2: U-Net Segmentation
      (unsupervised)                  (supervised)
                |                             |
                +--------------+--------------+
                               |
                  Severity Estimation (from U-Net mask)
                               |
                         FastAPI backend
                               |
                        React Dashboard
```

### Half 1 — Unsupervised Anomaly Detection

Learns a statistical model of "normal" surface appearance from defect-free images only, then flags regions that deviate from it. No labeled defect masks are needed to train this half.

- **Backbone:** ResNet-18 (ImageNet pretrained), frozen — used purely as a feature extractor.
- **Features:** intermediate activations from `layer2` (128ch, 32×32 grid) and `layer3` (256ch, 16×16 grid) are combined by upsampling `layer3` to 32×32 and concatenating channel-wise (384 channels total).
- **Statistical model:** PaDiM-style — a per-spatial-location Gaussian (mean + covariance) fit across all `train/good` images.
- **Scoring:** Mahalanobis distance between a new image's feature vector and the learned normal distribution, at every spatial location, producing an anomaly heatmap.
- **Deployment optimization:** the 384-channel covariance representation was ~577MB, too large for free-tier hosting memory limits. Reduced to **100 randomly-selected channels** (a standard PaDiM technique), shrinking the artifact to ~40MB with only a minor accuracy cost (see Results).

**Strength:** generalizes to defect types never seen during training, since it only ever learns "what normal looks like." **Weakness:** localization is only as fine as the feature grid's spatial resolution, so thin/hairline defects are poorly localized.

### Half 2 — Supervised Segmentation (U-Net)

A U-Net-style encoder-decoder trained directly on labeled image/mask pairs to predict pixel-precise defect masks.

- **Architecture:** standard U-Net, 4 encoder/decoder stages with skip connections, `base_channels=32`.
- **Loss:** combined Dice + BCE loss, to handle the class imbalance of small defect regions.
- **Training data:** MVTec AD only provides labeled masks in its test split (by design, since it's built for unsupervised anomaly detection) — so the ~84 labeled images were split 70/15/15 (train/val/test), stratified by defect type.

**Strength:** far more precise pixel-level boundaries, especially on defect types the statistical approach struggles with. **Weakness:** only generalizes to defect types it was trained on; does not detect novel/unseen defect types the way Half 1 does.

### Severity Estimation

Computed from Half 2's predicted mask (chosen as the primary localization source based on its stronger evaluation results): affected-pixel-count divided by total image area, expressed as a percentage. Connected-component analysis breaks the result down by individual defect blob, filtering out sub-5-pixel noise.

**Severity tiers** (used for the dashboard's pass/fail-style display):
| Tier | Range |
|---|---|
| Pass | 0% |
| Minor | 0–3% |
| Moderate | 3–15% |
| Severe | ≥15% |

---

## Dataset

**MVTec AD**, category `tile` — a real anomaly-detection benchmark with:

- `train/good/` — 230 defect-free images, used to build Half 1's normal-appearance model.
- `test/{good, crack, glue_strip, gray_stroke, oil, rough}/` — normal and defective test images across 5 distinct defect types.
- `ground_truth/<defect_type>/` — pixel-precise binary masks for every defective test image.

All images: native 840×840 resolution, resized to 256×256 for the pipeline (RGB, ImageNet-normalized).

---

## Results

### Half 1 — Anomaly Detection (memory-optimized, 100-channel deployed variant)

| Metric                                            | Value |
| ------------------------------------------------- | ----- |
| Threshold (95th percentile of `test/good` scores) | 7.53  |
| Precision                                         | 0.718 |
| Recall                                            | 1.000 |
| F1                                                | 0.836 |
| AUROC                                             | 0.916 |

Perfect recall means no defective image was missed — the model's false positives (roughly 1 in 4 flagged images being actually normal) are the more tolerable failure mode for a QC gate, since they route to a human double-check rather than let a defect through.

_(For reference, the original 384-channel, non-memory-constrained configuration scored AUROC 0.936 — the 100-channel deployment variant trades a small amount of accuracy for a ~14x reduction in memory footprint, enabling free-tier hosting.)_

#### Localization (IoU / Dice) by defect type

| Defect type | IoU       | Dice      |
| ----------- | --------- | --------- |
| oil         | 0.544     | 0.703     |
| rough       | 0.448     | 0.608     |
| glue_strip  | 0.311     | 0.465     |
| gray_stroke | 0.249     | 0.396     |
| **crack**   | **0.046** | **0.087** |

**Known limitation:** thin, hairline defects (`crack`) are poorly localized by this approach. This isn't a bug — it's a direct consequence of the anomaly map's spatial resolution (a 32×32 feature grid over a 256×256 image, i.e. each cell covers an 8×8 pixel patch). A crack that's only a few pixels wide is smaller than the grid's effective resolution, so while the model correctly detects that _something_ is wrong (hence perfect recall), it cannot draw a tight boundary around it. This finding directly motivated adding Half 2.

### Half 2 — U-Net Segmentation (held-out test set, 18 images)

| Defect type | IoU       | Dice      |
| ----------- | --------- | --------- |
| oil         | 0.909     | 0.952     |
| glue_strip  | 0.887     | 0.939     |
| gray_stroke | 0.853     | 0.920     |
| rough       | 0.826     | 0.904     |
| **crack**   | **0.759** | **0.862** |
| **Overall** | **0.848** | **0.916** |

U-Net outperforms Half 1 on every defect type, most dramatically on `crack` (0.046 → 0.759 IoU — a ~16x improvement), because direct pixel-level supervision isn't limited by the same feature-grid resolution bottleneck that constrains the unsupervised approach.

**Known limitation:** only ~84 labeled images were available (MVTec AD provides masks only for its test split), split further into train/val/test. The test set is just 18 images (3–4 per defect type), so these numbers, while genuinely strong, carry a wide margin of uncertainty — a different random split could shift individual defect-type scores by a nontrivial amount.

### Why Both Halves

This project's own results are the argument for the two-halves design: Half 1 catches defect types it was never explicitly trained on but localizes coarsely; Half 2 localizes precisely but only for defect types it has labeled examples of. A production system would run both — Half 1 as a safety net for novel defects, Half 2 for precise severity reporting on known ones.

---

## Tech Stack

| Layer              | Technology                                            |
| ------------------ | ----------------------------------------------------- |
| Feature extraction | PyTorch, torchvision (ResNet-18, ImageNet pretrained) |
| Anomaly detection  | NumPy, SciPy (PaDiM-style Mahalanobis distance)       |
| Segmentation       | PyTorch (custom U-Net)                                |
| Evaluation         | scikit-learn (precision/recall/F1/AUROC)              |
| API                | FastAPI, Uvicorn                                      |
| Dashboard          | React, Vite, Tailwind CSS                             |
| Deployment         | Docker, Railway                                       |

---

## API

### `GET /health`

Liveness check. Returns `{"status": "ok"}`.

### `GET /model-info`

Returns the consolidated evaluation report (both halves' metrics, side by side).

### `POST /inspect`

Accepts an image upload (`multipart/form-data`). Returns:

```json
{
  "classification": "defective",
  "anomaly_score": 12.4,
  "severity_pct": 3.36,
  "num_defect_blobs": 1,
  "blobs": [{ "blob_id": 1, "pixel_count": 220, "severity_pct": 3.35 }],
  "defect_tier": "moderate",
  "mask_overlay_base64": "<base64-encoded PNG>"
}
```

`classification` comes from Half 1; `severity_pct`, `num_defect_blobs`, and `mask_overlay_base64` come from Half 2's predicted mask.

---

## Project Structure

```
industrial-inspection/
├── data/mvtec/tile/              # dataset (not committed)
├── src/
│   ├── backbone/                 # shared ResNet-18 feature extractor
│   ├── anomaly_detection/        # Half 1: stats building, scoring, thresholding
│   ├── segmentation/             # Half 2: U-Net, dataset, training, evaluation
│   ├── severity/                 # mask -> severity % conversion
│   └── evaluation/                # metrics, ground-truth handling, report consolidation
├── artifacts/
│   ├── anomaly_stats/tile.pkl    # Half 1's learned "normal" statistics (~40MB)
│   └── checkpoints/unet_tile.pt  # Half 2's trained weights
├── api/                          # FastAPI backend
├── dashboard/                    # React frontend
├── logs/                         # evaluation reports (JSON)
├── Dockerfile
└── requirements.txt
```

---

## Known Limitations (Summary)

- **Single dataset category (`tile`)** — the pipeline is built to extend to other MVTec categories or real production data, but has only been trained/evaluated on this one.
- **Half 1 crack localization is weak** — a resolution bottleneck inherent to statistical anomaly detection, not a training issue. Documented above, and the direct motivation for Half 2.
- **Half 2's evaluation set is small (18 images)** — results are genuinely strong but should be read as indicative rather than statistically tight.
- **Half 1's deployed variant uses 100 randomly-selected channels**, not the full 384, to fit free-tier hosting memory limits — a small, deliberate, documented accuracy trade for deployability.
- **Grad-CAM explainability layer was scoped but not implemented** in this build — parked as a stretch goal since it required either training an additional classifier head or adapting Grad-CAM to a segmentation target, neither of which the core deliverables depended on.
- **Single-user demo assumptions** — e.g. the API writes uploaded images to a shared temp file path rather than per-request unique names, which would need addressing for concurrent multi-user traffic.

---

## Running Locally

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Build Half 1's statistics (one-time)
python -m src.anomaly_detection.build_stats

# Train Half 2 (one-time)
python -m src.segmentation.train

# Run the API (serves both the API and the built dashboard)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000`.
