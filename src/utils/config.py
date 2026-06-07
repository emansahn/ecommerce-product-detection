"""Configuration management for the application"""

import os
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ImageConfig:
    """Image processing configuration"""
    size: int = 640
    max_size: int = 2048
    min_size: int = 320
    normalize: bool = True

    def __post_init__(self):
        self.size = int(os.getenv("IMAGE_SIZE", self.size))
        self.max_size = int(os.getenv("MAX_IMAGE_SIZE", self.max_size))
        self.min_size = int(os.getenv("MIN_IMAGE_SIZE", self.min_size))


@dataclass
class DetectionConfig:
    """Detection model configuration"""
    model_name: str = "yolov8m"
    model_path: str = "models/weights/yolov8m.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: Optional[str] = None  # None = auto (ultralytics picks best device)

    def __post_init__(self):
        self.model_name = os.getenv("MODEL_NAME", self.model_name)
        self.model_path = os.getenv("MODEL_PATH", self.model_path)
        self.confidence_threshold = float(
            os.getenv("CONFIDENCE_THRESHOLD", self.confidence_threshold)
        )
        self.iou_threshold = float(os.getenv("IOU_THRESHOLD", self.iou_threshold))
        device_env = os.getenv("DEVICE", "")
        if device_env:
            self.device = device_env  # e.g. "cpu", "cuda:0"
        # else: keep None → ultralytics auto-selects


@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False

    def __post_init__(self):
        self.host = os.getenv("API_HOST", self.host)
        self.port = int(os.getenv("API_PORT", self.port))
        self.workers = int(os.getenv("API_WORKERS", self.workers))
        self.reload = os.getenv("DEBUG", "False").lower() == "true"


@dataclass
class AppConfig:
    """Main application configuration"""
    app_name: str = "ecommerce-product-detection"
    version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    image: ImageConfig = field(default_factory=ImageConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    api: APIConfig = field(default_factory=APIConfig)

    def __post_init__(self):
        self.app_name = os.getenv("APP_NAME", self.app_name)
        self.version = os.getenv("APP_VERSION", self.version)
        self.debug = os.getenv("DEBUG", "False").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)


def get_config() -> AppConfig:
    """Return a fully populated :class:`AppConfig` instance."""
    return AppConfig()
