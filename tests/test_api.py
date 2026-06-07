"""Tests for the FastAPI application (no live model required)."""

import sys
import numpy as np
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient


def _png_bytes(h=100, w=100):
    """Return minimal valid PNG bytes (no external deps)."""
    import struct, zlib
    pixels = np.full((h, w, 3), 128, dtype=np.uint8)
    raw = b"".join(b"\x00" + row.tobytes() for row in pixels)
    compressed = zlib.compress(raw)

    def chunk(name, data):
        c = name + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)

    ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
    return (
        b'\x89PNG\r\n\x1a\n'
        + chunk(b'IHDR', ihdr)
        + chunk(b'IDAT', compressed)
        + chunk(b'IEND', b'')
    )


def _make_mock_detector():
    from src.detection.detector import DetectionResult
    det = MagicMock()
    det.class_names = {0: "Electronics", 1: "Clothing"}
    det.detect.return_value = DetectionResult(
        boxes=np.array([[10, 20, 50, 80], [100, 100, 150, 200]], dtype=np.float32),
        confidences=np.array([0.92, 0.75], dtype=np.float32),
        class_ids=np.array([0, 1]),
    )
    return det


def _get_app_module():
    """Return the actual src.api.app module object (not the FastAPI app instance)."""
    import src.api.app  # noqa: F401
    return sys.modules["src.api.app"]


@pytest.fixture()
def client():
    mod = _get_app_module()
    original = mod.detector
    from src.api.app import app
    with TestClient(app, raise_server_exceptions=False) as c:
        mod.detector = _make_mock_detector()  # patch AFTER lifespan startup
        yield c
    mod.detector = original


@pytest.fixture()
def client_no_detector():
    mod = _get_app_module()
    original = mod.detector
    from src.api.app import app
    with TestClient(app, raise_server_exceptions=False) as c:
        mod.detector = None  # patch AFTER lifespan startup
        yield c
    mod.detector = original


# ---------------------------------------------------------------------------

class TestRootEndpoint:
    def test_root_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_root_has_docs_link(self, client):
        assert "docs" in client.get("/").json()


class TestHealthEndpoint:
    def test_health_200(self, client):
        assert client.get("/api/v1/health").status_code == 200

    def test_health_schema(self, client):
        data = client.get("/api/v1/health").json()
        for k in ("status", "version", "model", "device"):
            assert k in data

    def test_health_reports_healthy(self, client):
        assert client.get("/api/v1/health").json()["status"] == "healthy"

    def test_health_unhealthy_when_no_detector(self, client_no_detector):
        assert client_no_detector.get("/api/v1/health").json()["status"] == "unhealthy"


class TestDetectEndpoint:
    def test_detect_png_200(self, client):
        r = client.post("/api/v1/detect", files={"file": ("t.png", _png_bytes(), "image/png")})
        assert r.status_code == 200

    def test_detect_response_schema(self, client):
        data = client.post("/api/v1/detect", files={"file": ("t.png", _png_bytes(), "image/png")}).json()
        for k in ("detections", "count", "inference_time", "image_size"):
            assert k in data

    def test_detect_count_matches(self, client):
        data = client.post("/api/v1/detect", files={"file": ("t.png", _png_bytes(), "image/png")}).json()
        assert data["count"] == len(data["detections"]) == 2

    def test_detect_bbox_fields(self, client):
        det = client.post("/api/v1/detect", files={"file": ("t.png", _png_bytes(), "image/png")}).json()["detections"][0]
        for k in ("bbox", "confidence", "class_id"):
            assert k in det
        for coord in ("x1", "y1", "x2", "y2"):
            assert coord in det["bbox"]

    def test_detect_no_file_422(self, client):
        assert client.post("/api/v1/detect").status_code == 422

    def test_detect_returns_class_name(self, client):
        data = client.post("/api/v1/detect", files={"file": ("t.png", _png_bytes(), "image/png")}).json()
        assert data["detections"][0]["class_name"] == "Electronics"

    def test_detect_503_when_no_detector(self, client_no_detector):
        r = client_no_detector.post("/api/v1/detect", files={"file": ("t.png", _png_bytes(), "image/png")})
        assert r.status_code == 503

    def test_detect_image_size_fields(self, client):
        data = client.post("/api/v1/detect", files={"file": ("t.png", _png_bytes(80, 120), "image/png")}).json()
        assert data["image_size"]["height"] == 80
        assert data["image_size"]["width"] == 120


class TestCategoriesEndpoint:
    def test_categories_200(self, client):
        assert client.get("/api/v1/categories").status_code == 200

    def test_categories_has_data(self, client):
        data = client.get("/api/v1/categories").json()
        assert "categories" in data and len(data["categories"]) > 0

    def test_categories_503_when_no_detector(self, client_no_detector):
        assert client_no_detector.get("/api/v1/categories").status_code == 503