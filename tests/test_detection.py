"""Tests for the detection module (postprocessing & DetectionResult)."""

import numpy as np
import pytest

from src.detection.postprocessing import (
    NonMaximumSuppression,
    BoxFiltering,
    BoxConversion,
)
from src.detection.detector import DetectionResult


# ---------------------------------------------------------------------------
# DetectionResult
# ---------------------------------------------------------------------------

class TestDetectionResult:
    def _make(self, n=3):
        boxes = np.random.rand(n, 4).astype(np.float32)
        confs = np.random.rand(n).astype(np.float32)
        cls_ids = np.arange(n, dtype=int)
        return DetectionResult(boxes, confs, cls_ids)

    def test_len(self):
        assert len(self._make(5)) == 5

    def test_len_empty(self):
        empty = DetectionResult(
            np.empty((0, 4)), np.array([]), np.array([], dtype=int)
        )
        assert len(empty) == 0

    def test_to_dict_keys(self):
        d = self._make(2).to_dict()
        assert "detections" in d and "count" in d
        assert d["count"] == 2
        assert len(d["detections"]) == 2

    def test_to_dict_types(self):
        d = self._make(1).to_dict()
        det = d["detections"][0]
        assert isinstance(det["bbox"], list)
        assert isinstance(det["confidence"], float)
        assert isinstance(det["class_id"], int)


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

class TestIoU:
    def test_perfect_overlap(self):
        box = np.array([0, 0, 10, 10], dtype=float)
        assert NonMaximumSuppression.iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        b1 = np.array([0, 0, 5, 5], dtype=float)
        b2 = np.array([10, 10, 20, 20], dtype=float)
        assert NonMaximumSuppression.iou(b1, b2) == pytest.approx(0.0)

    def test_partial_overlap(self):
        b1 = np.array([0, 0, 10, 10], dtype=float)
        b2 = np.array([5, 5, 15, 15], dtype=float)
        # intersection = 5×5 = 25, union = 100+100-25 = 175
        assert NonMaximumSuppression.iou(b1, b2) == pytest.approx(25 / 175, abs=1e-5)


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

class TestNMS:
    def test_keeps_highest_score(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=float)
        scores = np.array([0.9, 0.8])
        kept = NonMaximumSuppression.nms(boxes, scores, iou_threshold=0.3)
        assert 0 in kept  # highest score survives
        assert 1 not in kept  # suppressed by heavy overlap

    def test_keeps_non_overlapping(self):
        boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
        scores = np.array([0.9, 0.8])
        kept = NonMaximumSuppression.nms(boxes, scores, iou_threshold=0.5)
        assert len(kept) == 2

    def test_empty_input(self):
        kept = NonMaximumSuppression.nms(
            np.empty((0, 4)), np.array([]), iou_threshold=0.5
        )
        assert len(kept) == 0

    def test_soft_nms_returns_fewer(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]], dtype=float)
        scores = np.array([0.9, 0.85, 0.7])
        kept, new_scores = NonMaximumSuppression.soft_nms(
            boxes, scores, iou_threshold=0.3, sigma=0.5, score_threshold=0.5
        )
        # Overlapping box should be penalised below 0.5
        assert len(kept) <= 3


# ---------------------------------------------------------------------------
# BoxFiltering
# ---------------------------------------------------------------------------

class TestBoxFiltering:
    BOXES = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], dtype=float)
    SCORES = np.array([0.9, 0.4, 0.7])
    CLS = np.array([0, 1, 2])

    def test_filter_by_confidence(self):
        b, s, c = BoxFiltering.filter_by_confidence(self.BOXES, self.SCORES, self.CLS, 0.5)
        assert len(b) == 2  # 0.9 and 0.7 survive

    def test_filter_by_area(self):
        # Each box is 10×10 = 100 area
        b, s, c = BoxFiltering.filter_by_area(self.BOXES, self.SCORES, self.CLS, min_area=50)
        assert len(b) == 3

    def test_filter_by_area_max(self):
        b, s, c = BoxFiltering.filter_by_area(
            self.BOXES, self.SCORES, self.CLS, min_area=0, max_area=99
        )
        assert len(b) == 0  # all boxes = 100 area, above max

    def test_clip_boxes(self):
        boxes = np.array([[-5, -5, 200, 200]], dtype=float)
        clipped = BoxFiltering.clip_boxes(boxes, 100, 100)
        assert clipped[0, 0] == 0
        assert clipped[0, 1] == 0
        assert clipped[0, 2] == 100
        assert clipped[0, 3] == 100


# ---------------------------------------------------------------------------
# BoxConversion
# ---------------------------------------------------------------------------

class TestBoxConversion:
    XYXY = np.array([[10, 20, 50, 80]], dtype=float)   # w=40, h=60
    XYWH = np.array([[10, 20, 40, 60]], dtype=float)
    CXCYWH = np.array([[30, 50, 40, 60]], dtype=float) # cx=30, cy=50

    def test_xyxy_to_xywh(self):
        out = BoxConversion.xyxy_to_xywh(self.XYXY)
        np.testing.assert_allclose(out, self.XYWH)

    def test_xywh_to_xyxy(self):
        out = BoxConversion.xywh_to_xyxy(self.XYWH)
        np.testing.assert_allclose(out, self.XYXY)

    def test_xyxy_to_cxcywh(self):
        out = BoxConversion.xyxy_to_cxcywh(self.XYXY)
        np.testing.assert_allclose(out, self.CXCYWH)

    def test_cxcywh_to_xyxy(self):
        out = BoxConversion.cxcywh_to_xyxy(self.CXCYWH)
        np.testing.assert_allclose(out, self.XYXY)

    def test_roundtrip_xyxy_xywh(self):
        out = BoxConversion.xywh_to_xyxy(BoxConversion.xyxy_to_xywh(self.XYXY))
        np.testing.assert_allclose(out, self.XYXY)

    def test_roundtrip_xyxy_cxcywh(self):
        out = BoxConversion.cxcywh_to_xyxy(BoxConversion.xyxy_to_cxcywh(self.XYXY))
        np.testing.assert_allclose(out, self.XYXY)
