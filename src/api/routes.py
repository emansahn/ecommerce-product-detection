"""
API route definitions.

Routes access the ``detector`` and ``loader`` singletons via the
``src.api.app`` module at *call time* (not import time), so they pick up
any mutation made by tests or the lifespan manager.
"""

from fastapi import APIRouter, File, HTTPException, UploadFile
import numpy as np
import io
import logging
from datetime import datetime, timezone
from PIL import Image

from src.api.models import (
    DetectionResponse,
    HealthResponse,
    Detection,
    BoundingBox,
    ImageSize,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_detector():
    """Return the current detector singleton (resolves after app startup)."""
    import src.api.app as _app
    return _app.detector


def _get_config():
    import src.api.app as _app
    return _app.config


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health_check():
    """Return API health status and loaded model information."""
    det = _get_detector()
    cfg = _get_config()
    return HealthResponse(
        status="healthy" if det is not None else "unhealthy",
        version=cfg.version,
        model=cfg.detection.model_name,
        device=cfg.detection.device or "auto",
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

@router.post("/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect(file: UploadFile = File(..., description="Image file (JPEG / PNG / BMP)")):
    """Detect products in an uploaded image."""
    det = _get_detector()
    if det is None:
        raise HTTPException(status_code=503, detail="Detector is not available.")

    allowed = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Allowed: {sorted(allowed)}",
        )

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_rgb = np.array(pil_image)
        image_bgr = image_rgb[:, :, ::-1]

        start = datetime.now(timezone.utc)
        result = det.detect(image_bgr)
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()

        class_names = getattr(det, "class_names", {})

        detections = [
            Detection(
                bbox=BoundingBox(
                    x1=float(box[0]), y1=float(box[1]),
                    x2=float(box[2]), y2=float(box[3]),
                ),
                confidence=float(conf),
                class_id=int(cls_id),
                class_name=class_names.get(int(cls_id)),
            )
            for box, conf, cls_id in zip(result.boxes, result.confidences, result.class_ids)
        ]

        h, w = image_rgb.shape[:2]
        return DetectionResponse(
            detections=detections,
            count=len(detections),
            inference_time=elapsed,
            image_size=ImageSize(height=h, width=w),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Detection error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

@router.get("/categories", tags=["Detection"])
async def list_categories():
    """Return the class-name mapping of the loaded model."""
    det = _get_detector()
    if det is None:
        raise HTTPException(status_code=503, detail="Detector is not available.")
    return {"categories": det.class_names}
