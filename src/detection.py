"""
Detection module entry point.
Import from src.detection package instead.
"""
from src.detection.detector import ProductDetector, DetectionResult
from src.detection.postprocessing import NonMaximumSuppression, BoxFiltering, BoxConversion

__all__ = [
    "ProductDetector",
    "DetectionResult",
    "NonMaximumSuppression",
    "BoxFiltering",
    "BoxConversion",
]
