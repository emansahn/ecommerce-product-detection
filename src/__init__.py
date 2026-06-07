"""Ecommerce Product Detection Package"""

__version__ = "0.1.0"
__author__ = "emansahn"

# Lazy imports — only raise ImportError when the user actually tries to use the class,
# not when the package is first imported (avoids hard crash if ultralytics is missing).
def __getattr__(name):
    if name == "ProductDetector":
        from src.detection.detector import ProductDetector
        return ProductDetector
    if name == "ImageLoader":
        from src.preprocessing.image_loader import ImageLoader
        return ImageLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ProductDetector",
    "ImageLoader",
]
