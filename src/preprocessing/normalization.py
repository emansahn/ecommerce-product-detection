"""Image normalization and resizing utilities"""

import numpy as np
import cv2
from typing import Tuple, Optional


def normalize_image(
    image: np.ndarray,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    scale: float = 255.0,
) -> np.ndarray:
    """Normalize image using z-score normalization or simple [0, 1] scaling.

    Args:
        image: Input image — BGR (H, W, C) or grayscale (H, W).
        mean: Per-channel mean values for z-score normalization.
              Must match the number of channels. Ignored when ``std`` is ``None``.
        std: Per-channel standard deviation. Must match the number of channels.
             Ignored when ``mean`` is ``None``.
        scale: Divisor applied before z-score (typically 255 for uint8 images).

    Returns:
        Normalized image as ``np.float32``.
    """
    img = image.astype(np.float32)

    if mean is not None and std is not None:
        # Validate channel count
        n_channels = img.shape[2] if img.ndim == 3 else 1
        if len(mean) != n_channels or len(std) != n_channels:
            raise ValueError(
                f"mean/std length ({len(mean)}/{len(std)}) must match "
                f"image channels ({n_channels})."
            )

        img /= scale

        if img.ndim == 3:
            for i in range(n_channels):
                img[:, :, i] = (img[:, :, i] - mean[i]) / (std[i] + 1e-7)
        else:  # grayscale
            img = (img - mean[0]) / (std[0] + 1e-7)
    else:
        img /= scale

    return img


def resize_image(
    image: np.ndarray,
    size: int,
    maintain_aspect_ratio: bool = True,
    interpolation: int = cv2.INTER_LINEAR,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Resize image to a square canvas of ``size × size``.

    Args:
        image: Input image (BGR or grayscale).
        size: Target canvas side length in pixels.
        maintain_aspect_ratio: When ``True`` the image is scaled so its longest
            side equals ``size`` and the rest is zero-padded (letterbox). When
            ``False`` the image is stretched to fill ``size × size``.
        interpolation: OpenCV interpolation flag.

    Returns:
        Tuple of (resized_image, (scale_x, scale_y)).
        Both scale factors are identical when ``maintain_aspect_ratio=True``.
    """
    original_h, original_w = image.shape[:2]
    is_color = image.ndim == 3

    if maintain_aspect_ratio:
        scale = min(size / original_w, size / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

        # Create zero-padded canvas with the correct number of channels
        if is_color:
            canvas = np.zeros((size, size, image.shape[2]), dtype=image.dtype)
        else:
            canvas = np.zeros((size, size), dtype=image.dtype)

        top = (size - new_h) // 2
        left = (size - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = resized

        return canvas, (scale, scale)
    else:
        resized = cv2.resize(image, (size, size), interpolation=interpolation)
        scale_x = size / original_w
        scale_y = size / original_h
        return resized, (scale_x, scale_y)


def convert_color_space(
    image: np.ndarray,
    from_space: str = "BGR",
    to_space: str = "RGB",
) -> np.ndarray:
    """Convert an image between color spaces.

    Supported conversions: BGR ↔ RGB, BGR/RGB → GRAY, BGR/RGB → HSV.

    Args:
        image: Input image.
        from_space: Source color space identifier (``"BGR"``, ``"RGB"``,
                    ``"GRAY"``, ``"HSV"``).
        to_space: Target color space identifier.

    Returns:
        Converted image.

    Raises:
        ValueError: If the requested conversion is not supported.
    """
    if from_space == to_space:
        return image.copy()

    _CODES = {
        ("BGR", "RGB"): cv2.COLOR_BGR2RGB,
        ("RGB", "BGR"): cv2.COLOR_RGB2BGR,
        ("BGR", "GRAY"): cv2.COLOR_BGR2GRAY,
        ("RGB", "GRAY"): cv2.COLOR_RGB2GRAY,
        ("BGR", "HSV"): cv2.COLOR_BGR2HSV,
        ("RGB", "HSV"): cv2.COLOR_RGB2HSV,
        ("HSV", "BGR"): cv2.COLOR_HSV2BGR,
        ("HSV", "RGB"): cv2.COLOR_HSV2RGB,
        ("GRAY", "BGR"): cv2.COLOR_GRAY2BGR,
        ("GRAY", "RGB"): cv2.COLOR_GRAY2RGB,
    }

    code = _CODES.get((from_space, to_space))
    if code is None:
        raise ValueError(
            f"Unsupported conversion: {from_space} → {to_space}. "
            f"Supported pairs: {list(_CODES.keys())}"
        )

    return cv2.cvtColor(image, code)
