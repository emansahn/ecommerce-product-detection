"""Image loading and validation"""

import numpy as np
from pathlib import Path
from typing import Union
import cv2
import logging

logger = logging.getLogger(__name__)


class ImageLoader:
    """Load and validate images from various sources"""

    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    MAX_FILE_SIZE_MB = 100

    def __init__(self, validate: bool = True):
        self.validate = validate

    def load(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load image from file path (BGR).

        Raises:
            ValueError: If the file extension is not supported.
            FileNotFoundError: If the file does not exist.
            ValueError: If OpenCV cannot decode the file.
        """
        image_path = Path(image_path)

        # Extension check FIRST (before filesystem access)
        if image_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported image format: '{image_path.suffix}'. "
                f"Supported: {sorted(self.SUPPORTED_FORMATS)}"
            )

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        file_size_mb = image_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.MAX_FILE_SIZE_MB:
            raise ValueError(
                f"Image file too large: {file_size_mb:.2f} MB "
                f"(max {self.MAX_FILE_SIZE_MB} MB)"
            )

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to decode image: {image_path}")

        logger.debug("Loaded image: %s  shape: %s", image_path, image.shape)
        return image

    def load_from_array(self, array: np.ndarray) -> np.ndarray:
        """Validate and return an image already in memory."""
        if not isinstance(array, np.ndarray):
            raise ValueError(f"Expected np.ndarray, got {type(array)}")
        if array.ndim not in (2, 3):
            raise ValueError(
                f"Invalid image shape {array.shape}. Expected 2-D or 3-D array."
            )
        if array.size == 0:
            raise ValueError("Empty image array.")
        return array

    def load_batch(self, image_dir: Union[str, Path]) -> dict:
        """Load all supported images from a directory."""
        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {image_dir}")

        images = {}
        for p in image_dir.iterdir():
            if p.suffix.lower() in self.SUPPORTED_FORMATS:
                try:
                    images[p.name] = self.load(p)
                except Exception as exc:
                    logger.warning("Skipping %s: %s", p, exc)

        logger.info("Loaded %d images from %s", len(images), image_dir)
        return images
