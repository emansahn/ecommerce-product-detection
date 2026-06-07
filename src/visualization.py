"""
Visualization utilities for detection results.
"""
import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# Default color palette for classes
COLORS = [
    (255, 56, 56),   # Red
    (56, 255, 56),   # Green
    (56, 56, 255),   # Blue
    (255, 156, 56),  # Orange
    (156, 56, 255),  # Purple
    (56, 255, 156),  # Cyan-Green
    (255, 56, 156),  # Pink
    (156, 255, 56),  # Yellow-Green
    (56, 156, 255),  # Sky Blue
    (255, 255, 56),  # Yellow
]


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw detection boxes on image.

    Args:
        image: BGR image array
        boxes: Bounding boxes [N, 4] as [x1, y1, x2, y2]
        confidences: Confidence scores [N]
        class_ids: Class IDs [N]
        class_names: Optional dict mapping class_id -> class_name
        thickness: Box border thickness
        font_scale: Label font scale

    Returns:
        Annotated image copy
    """
    annotated = image.copy()

    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        color = COLORS[int(cls_id) % len(COLORS)]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = class_names.get(int(cls_id), str(cls_id)) if class_names else str(cls_id)
        text = f"{label}: {conf:.2f}"

        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        cv2.rectangle(
            annotated,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w, y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            text,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return annotated


def save_annotated_image(
    image: np.ndarray,
    output_path: str,
    boxes: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
) -> None:
    """Draw detections and save to disk."""
    annotated = draw_detections(image, boxes, confidences, class_ids, class_names)
    cv2.imwrite(output_path, annotated)
    logger.debug(f"Saved annotated image to {output_path}")
