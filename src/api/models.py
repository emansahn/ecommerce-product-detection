"""Pydantic models for API requests/responses"""

from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from datetime import datetime, timezone


class BoundingBox(BaseModel):
    """Bounding box in pixel coordinates (x1, y1) top-left, (x2, y2) bottom-right."""
    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


class Detection(BaseModel):
    """Single detection result."""
    bbox: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0, 1]")
    class_id: int = Field(..., ge=0, description="Predicted class ID")
    class_name: Optional[str] = Field(None, description="Human-readable class name")


class ImageSize(BaseModel):
    """Image dimensions."""
    height: int
    width: int


class DetectionResponse(BaseModel):
    """Full detection response returned by POST /detect."""
    detections: List[Detection]
    count: int = Field(..., description="Number of detections")
    inference_time: float = Field(..., description="Inference time in seconds")
    image_size: ImageSize = Field(..., description="Input image dimensions")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    model: str
    device: str


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
