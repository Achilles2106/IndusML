import io
import base64

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.schemas import InspectionResponse, HealthResponse, BlobInfo
from api.model_loader import (
    get_device,
    get_feature_extractor,
    get_anomaly_scorer,
    get_unet_model,
    get_half1_threshold,
    get_unet_threshold,
)
from src.severity.estimate import compute_severity_with_blob_breakdown
from torchvision import transforms

app = FastAPI(title="Industrial Surface Inspection API")

# Allow the dashboard (likely running on a different port/origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten this in a real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGE_SIZE = (256, 256)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_unet_preprocess = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def get_defect_tier(severity_pct):
    if severity_pct <= 0.0:
        return "pass"
    elif severity_pct < 3.0:
        return "minor"
    elif severity_pct < 15.0:
        return "moderate"
    else:
        return "severe"


def mask_to_base64_png(binary_mask):
    """Converts a 0/1 numpy mask into a base64-encoded PNG (white=defect, black=background)."""
    mask_image = Image.fromarray((binary_mask * 255).astype(np.uint8))
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.get("/model-info")
def model_info():
    import json
    with open("logs/consolidated_report.json", "r") as f:
        report = json.load(f)
    return report


@app.post("/inspect", response_model=InspectionResponse)
async def inspect(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded image.")

    device = get_device()

    # ---- Half 1: Anomaly detection (classification) ----
    scorer = get_anomaly_scorer()
    extractor = get_feature_extractor()

    # AnomalyScorer expects a file path; save the uploaded image to a temp buffer on disk
    # since our current implementation reads from a path, not a PIL image directly.
    temp_path = "artifacts/_temp_uploaded_image.png"
    image.save(temp_path)

    anomaly_map = scorer.score_and_upsample(temp_path, output_size=IMAGE_SIZE)
    anomaly_score = float(anomaly_map.max())

    threshold_half1 = get_half1_threshold()
    classification = "defective" if anomaly_score >= threshold_half1 else "normal"

    # ---- Half 2: U-Net segmentation (localization + severity) ----
    unet_model = get_unet_model()
    unet_threshold = get_unet_threshold()

    image_tensor = _unet_preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = unet_model(image_tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    pred_mask = (probs >= unet_threshold).astype(np.uint8)

    # ---- Severity estimation ----
    severity_result = compute_severity_with_blob_breakdown(pred_mask)
    severity_pct = severity_result["overall_severity_pct"]
    blobs = [BlobInfo(**b) for b in severity_result["blobs"]]

    defect_tier = get_defect_tier(severity_pct)

    # ---- Mask overlay for dashboard rendering ----
    mask_b64 = mask_to_base64_png(pred_mask)

    return InspectionResponse(
        classification=classification,
        anomaly_score=anomaly_score,
        severity_pct=severity_pct,
        num_defect_blobs=severity_result["num_defect_blobs"],
        blobs=blobs,
        defect_tier=defect_tier,
        mask_overlay_base64=mask_b64,
    )

app.mount("/", StaticFiles(directory="dashboard/dist", html=True), name="dashboard")