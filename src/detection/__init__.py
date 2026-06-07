"""Detection package"""
from src.detection.detector import ProductDetector, DetectionResult
from src.detection.postprocessing import NonMaximumSuppression, BoxFiltering, BoxConversion

__all__ = [
    "ProductDetector",
    "DetectionResult",
    "NonMaximumSuppression",
    "BoxFiltering",
    "BoxConversion",
]
