"""Tests for the preprocessing module."""

import numpy as np
import pytest
from pathlib import Path

from src.preprocessing.image_loader import ImageLoader
from src.preprocessing.normalization import normalize_image, resize_image, convert_color_space
from src.preprocessing.augmentation import ImageAugmentor

FIXTURES = Path(__file__).parent / "fixtures" / "sample_images"


# ---------------------------------------------------------------------------
# ImageLoader
# ---------------------------------------------------------------------------

class TestImageLoader:
    def test_load_from_array_rgb(self):
        loader = ImageLoader()
        arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        assert loader.load_from_array(arr).shape == (100, 100, 3)

    def test_load_from_array_grayscale(self):
        loader = ImageLoader()
        arr = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        assert loader.load_from_array(arr).shape == (50, 50)

    def test_load_from_array_4d_raises(self):
        loader = ImageLoader()
        with pytest.raises(ValueError):
            loader.load_from_array(np.zeros((10, 10, 3, 2)))

    def test_load_from_array_empty_raises(self):
        loader = ImageLoader()
        with pytest.raises(ValueError):
            loader.load_from_array(np.array([]))

    def test_load_from_file(self):
        loader = ImageLoader()
        img = loader.load(FIXTURES / "product_blue.png")
        assert img.ndim == 3
        assert img.shape[2] == 3

    def test_load_missing_file_raises(self):
        loader = ImageLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent.jpg")

    def test_load_unsupported_ext_raises(self):
        loader = ImageLoader()
        with pytest.raises(ValueError):
            loader.load("image.gif")

    def test_load_batch(self, tmp_path):
        loader = ImageLoader()
        # Copy fixtures to tmp dir
        import shutil
        for p in FIXTURES.glob("*.png"):
            shutil.copy(p, tmp_path / p.name)
        images = loader.load_batch(tmp_path)
        assert len(images) == 3
        for arr in images.values():
            assert arr.ndim == 3


# ---------------------------------------------------------------------------
# normalize_image
# ---------------------------------------------------------------------------

class TestNormalizeImage:
    def test_scale_only(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        out = normalize_image(img)
        assert out.dtype == np.float32
        assert np.allclose(out, 128 / 255.0, atol=1e-5)

    def test_zscore_rgb(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        out = normalize_image(img, mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2))
        assert out.dtype == np.float32
        # (128/255 - 0.5) / 0.2 ≈ 0.01
        assert out.min() > -5 and out.max() < 5

    def test_zscore_grayscale(self):
        img = np.full((10, 10), 128, dtype=np.uint8)
        out = normalize_image(img, mean=(0.5,), std=(0.2,))
        assert out.dtype == np.float32

    def test_mean_std_channel_mismatch_raises(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            normalize_image(img, mean=(0.5,), std=(0.2,))


# ---------------------------------------------------------------------------
# resize_image
# ---------------------------------------------------------------------------

class TestResizeImage:
    def test_letterbox(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        out, (sx, sy) = resize_image(img, 512, maintain_aspect_ratio=True)
        assert out.shape == (512, 512, 3)
        assert sx == sy  # uniform scale for letterbox

    def test_stretch(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        out, (sx, sy) = resize_image(img, 256, maintain_aspect_ratio=False)
        assert out.shape == (256, 256, 3)

    def test_grayscale_letterbox(self):
        img = np.zeros((100, 200), dtype=np.uint8)
        out, _ = resize_image(img, 128, maintain_aspect_ratio=True)
        assert out.shape == (128, 128)


# ---------------------------------------------------------------------------
# convert_color_space
# ---------------------------------------------------------------------------

class TestConvertColorSpace:
    def test_bgr_to_rgb(self):
        img = np.array([[[255, 0, 0]]], dtype=np.uint8)
        out = convert_color_space(img, "BGR", "RGB")
        assert out[0, 0, 0] == 0  # R channel (was B)
        assert out[0, 0, 2] == 255  # B channel (was R)

    def test_identity(self):
        img = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        out = convert_color_space(img, "BGR", "BGR")
        np.testing.assert_array_equal(out, img)

    def test_unsupported_raises(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            convert_color_space(img, "BGR", "XYZ")

    def test_bgr_to_gray(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        out = convert_color_space(img, "BGR", "GRAY")
        assert out.ndim == 2


# ---------------------------------------------------------------------------
# ImageAugmentor
# ---------------------------------------------------------------------------

class TestImageAugmentor:
    IMG = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    def test_flip_horizontal(self):
        aug = ImageAugmentor(seed=0)
        out = aug.flip(self.IMG.copy(), horizontal=True)
        assert out.shape == self.IMG.shape

    def test_rotate(self):
        aug = ImageAugmentor(seed=0)
        out = aug.rotate(self.IMG.copy(), 45)
        assert out.shape == self.IMG.shape

    def test_brightness(self):
        aug = ImageAugmentor(seed=0)
        out = aug.brightness(self.IMG.copy(), factor=1.5)
        assert out.dtype == self.IMG.dtype

    def test_compose(self):
        aug = ImageAugmentor(seed=42)
        pipeline = [
            {"type": "flip", "params": {"horizontal": True}},
            {"type": "brightness", "params": {"factor": 1.1}},
        ]
        out = aug.compose(self.IMG.copy(), pipeline)
        assert out.shape == self.IMG.shape
