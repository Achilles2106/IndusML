from pydantic import BaseModel
from typing import List, Optional


class BlobInfo(BaseModel):
    blob_id: int
    pixel_count: int
    severity_pct: float


class InspectionResponse(BaseModel):
    classification: str              # "defective" or "normal"
    anomaly_score: float              # Half 1's max anomaly score
    severity_pct: float                # Half 2's overall severity
    num_defect_blobs: int
    blobs: List[BlobInfo]
    defect_tier: str                   # "pass" / "minor" / "moderate" / "severe"
    mask_overlay_base64: str           # base64-encoded PNG of the predicted mask


class HealthResponse(BaseModel):
    status: str