"""Post-processing utilities for detection results"""

import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class NonMaximumSuppression:
    """Non-Maximum Suppression implementation"""
    
    @staticmethod
    def iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union (IoU)
        
        Args:
            box1: Box as [x1, y1, x2, y2]
            box2: Box as [x1, y1, x2, y2]
        
        Returns:
            IoU value (0-1)
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    @staticmethod
    def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
        """Apply Non-Maximum Suppression
        
        Args:
            boxes: Array of bounding boxes [N, 4] as [x1, y1, x2, y2]
            scores: Array of confidence scores [N]
            iou_threshold: IoU threshold for suppression
        
        Returns:
            Indices of boxes to keep
        """
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)
        
        # Sort by score in descending order
        sorted_indices = np.argsort(-scores)
        
        keep = []
        while len(sorted_indices) > 0:
            current_idx = sorted_indices[0]
            keep.append(current_idx)
            
            if len(sorted_indices) == 1:
                break
            
            current_box = boxes[current_idx]
            remaining_boxes = boxes[sorted_indices[1:]]
            
            # Calculate IoU with remaining boxes
            ious = np.array([
                NonMaximumSuppression.iou(current_box, box) 
                for box in remaining_boxes
            ])
            
            # Keep boxes with IoU below threshold
            keep_mask = ious < iou_threshold
            sorted_indices = sorted_indices[1:][keep_mask]
        
        return np.array(keep, dtype=np.int32)
    
    @staticmethod
    def soft_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5, 
                sigma: float = 0.5, score_threshold: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
        """Soft-NMS: More gentle suppression
        
        Args:
            boxes: Array of bounding boxes [N, 4]
            scores: Array of confidence scores [N]
            iou_threshold: IoU threshold
            sigma: Gaussian penalty parameter
            score_threshold: Minimum score to keep
        
        Returns:
            Tuple of (kept_indices, new_scores)
        """
        if len(boxes) == 0:
            return np.array([], dtype=np.int32), np.array([])
        
        new_scores = scores.copy().astype(np.float32)
        keep = []
        sorted_indices = np.argsort(-scores)
        
        while len(sorted_indices) > 0:
            current_idx = sorted_indices[0]
            
            if new_scores[current_idx] > score_threshold:
                keep.append(current_idx)
            
            if len(sorted_indices) == 1:
                break
            
            current_box = boxes[current_idx]
            remaining_indices = sorted_indices[1:]
            remaining_boxes = boxes[remaining_indices]
            
            # Calculate IoU
            ious = np.array([
                NonMaximumSuppression.iou(current_box, box)
                for box in remaining_boxes
            ])
            
            # Apply Gaussian penalty
            new_scores[remaining_indices] *= np.exp(-(ious ** 2) / sigma)
            
            # Re-sort
            sorted_indices = remaining_indices[np.argsort(-new_scores[remaining_indices])]
        
        return np.array(keep, dtype=np.int32), new_scores[keep]


class BoxFiltering:
    """Bounding box filtering and refinement"""
    
    @staticmethod
    def filter_by_confidence(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
                            confidence_threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter detections by confidence score
        
        Args:
            boxes: Bounding boxes [N, 4]
            scores: Confidence scores [N]
            class_ids: Class IDs [N]
            confidence_threshold: Minimum confidence
        
        Returns:
            Filtered boxes, scores, class_ids
        """
        mask = scores >= confidence_threshold
        return boxes[mask], scores[mask], class_ids[mask]
    
    @staticmethod
    def filter_by_area(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
                      min_area: float = 0.0, max_area: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter detections by area
        
        Args:
            boxes: Bounding boxes [N, 4]
            scores: Confidence scores [N]
            class_ids: Class IDs [N]
            min_area: Minimum area threshold
            max_area: Maximum area threshold
        
        Returns:
            Filtered boxes, scores, class_ids
        """
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        mask = areas >= min_area
        
        if max_area is not None:
            mask = mask & (areas <= max_area)
        
        return boxes[mask], scores[mask], class_ids[mask]
    
    @staticmethod
    def filter_by_aspect_ratio(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
                               min_ratio: float = 0.1, max_ratio: float = 10.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter detections by aspect ratio
        
        Args:
            boxes: Bounding boxes [N, 4]
            scores: Confidence scores [N]
            class_ids: Class IDs [N]
            min_ratio: Minimum width/height ratio
            max_ratio: Maximum width/height ratio
        
        Returns:
            Filtered boxes, scores, class_ids
        """
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ratios = widths / (heights + 1e-6)
        
        mask = (ratios >= min_ratio) & (ratios <= max_ratio)
        return boxes[mask], scores[mask], class_ids[mask]
    
    @staticmethod
    def clip_boxes(boxes: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
        """Clip boxes to image boundaries
        
        Args:
            boxes: Bounding boxes [N, 4]
            image_width: Image width
            image_height: Image height
        
        Returns:
            Clipped boxes
        """
        boxes = boxes.copy()
        boxes[:, 0] = np.clip(boxes[:, 0], 0, image_width - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, image_height - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, image_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, image_height)
        return boxes


class BoxConversion:
    """Bounding box format conversions"""
    
    @staticmethod
    def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
        """Convert from [x1, y1, x2, y2] to [x, y, w, h]
        
        Args:
            boxes: Boxes in [x1, y1, x2, y2] format
        
        Returns:
            Boxes in [x, y, w, h] format
        """
        converted = boxes.copy()
        converted[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
        converted[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
        return converted
    
    @staticmethod
    def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        """Convert from [x, y, w, h] to [x1, y1, x2, y2]
        
        Args:
            boxes: Boxes in [x, y, w, h] format
        
        Returns:
            Boxes in [x1, y1, x2, y2] format
        """
        converted = boxes.copy()
        converted[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x + w
        converted[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y + h
        return converted
    
    @staticmethod
    def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        """Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
        
        Args:
            boxes: Boxes in [cx, cy, w, h] format
        
        Returns:
            Boxes in [x1, y1, x2, y2] format
        """
        converted = np.zeros_like(boxes)
        converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        return converted
    
    @staticmethod
    def xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
        """Convert from [x1, y1, x2, y2] to [cx, cy, w, h]
        
        Args:
            boxes: Boxes in [x1, y1, x2, y2] format
        
        Returns:
            Boxes in [cx, cy, w, h] format
        """
        converted = np.zeros_like(boxes)
        converted[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # cx
        converted[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # cy
        converted[:, 2] = boxes[:, 2] - boxes[:, 0]  # w
        converted[:, 3] = boxes[:, 3] - boxes[:, 1]  # h
        return converted
