"""Product detection model wrapper"""

import numpy as np
from typing import Dict, List, Optional
import logging

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

logger = logging.getLogger(__name__)


class DetectionResult:
    """Container for detection results"""

    def __init__(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
    ):
        """Initialize detection result.

        Args:
            boxes: Bounding boxes as (x1, y1, x2, y2) in pixel coordinates. Shape [N, 4].
            confidences: Confidence scores for each box. Shape [N].
            class_ids: Class IDs for each box. Shape [N].
        """
        self.boxes = boxes
        self.confidences = confidences
        self.class_ids = class_ids

    def __len__(self) -> int:
        return len(self.boxes)

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        detections = []
        for box, conf, cls_id in zip(self.boxes, self.confidences, self.class_ids):
            detections.append(
                {
                    "bbox": box.tolist(),
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                }
            )
        return {"detections": detections, "count": len(detections)}


class ProductDetector:
    """Product detection using YOLOv8."""

    def __init__(
        self,
        model: str = "yolov8m",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        """Initialize detector.

        Args:
            model: Model variant identifier (n, s, m, l, x) or path to custom .pt file.
            confidence_threshold: Minimum confidence score for a detection to be kept.
            iou_threshold: IoU threshold used in NMS.
            device: Inference device — ``"cpu"``, ``"cuda"``, ``"cuda:0"``, or ``None``
                    (auto-selects GPU when available, CPU otherwise).
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics is not installed. "
                "Install it with: pip install ultralytics"
            )

        # Resolve model path: bare size string -> "<size>.pt"
        model_path = model if model.endswith(".pt") else f"{model}.pt"
        self.model = YOLO(model_path)

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        # None lets ultralytics pick the best available device automatically.
        self.device = device

        logger.info(
            "ProductDetector initialised — model=%s, conf=%.2f, iou=%.2f, device=%s",
            model,
            confidence_threshold,
            iou_threshold,
            device or "auto",
        )

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Run detection on a single image.

        Args:
            image: Input image as a NumPy array in **BGR** channel order (OpenCV default).

        Returns:
            :class:`DetectionResult` containing boxes, confidences, and class IDs.
        """
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        result = results[0]

        if len(result.boxes) == 0:
            empty = np.empty((0, 4), dtype=np.float32)
            return DetectionResult(empty, np.array([]), np.array([], dtype=int))

        boxes = result.boxes.xyxy.cpu().numpy()           # [N, 4]  x1 y1 x2 y2
        confidences = result.boxes.conf.cpu().numpy()     # [N]
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # [N]

        return DetectionResult(boxes, confidences, class_ids)

    def detect_batch(self, images: List[np.ndarray]) -> List[DetectionResult]:
        """Run detection on a list of images.

        Args:
            images: List of BGR images.

        Returns:
            List of :class:`DetectionResult` objects, one per image.
        """
        return [self.detect(img) for img in images]

    @property
    def class_names(self) -> Dict[int, str]:
        """Return the class-name mapping from the loaded model."""
        return self.model.names  # type: ignore[return-value]
