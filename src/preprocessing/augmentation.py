"""Image augmentation pipelines for data augmentation"""

import numpy as np
import cv2
from typing import List, Optional, Tuple
import random
import logging

logger = logging.getLogger(__name__)


class ImageAugmentor:
    """Image augmentation utilities for training data"""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize augmentor
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.seed = seed
    
    def rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by specified angle
        
        Args:
            image: Input image
            angle: Rotation angle in degrees (-180 to 180)
        
        Returns:
            Rotated image
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (width, height))
        return rotated
    
    def flip(self, image: np.ndarray, horizontal: bool = True, vertical: bool = False) -> np.ndarray:
        """Flip image horizontally or vertically
        
        Args:
            image: Input image
            horizontal: Whether to flip horizontally
            vertical: Whether to flip vertically
        
        Returns:
            Flipped image
        """
        if horizontal:
            image = cv2.flip(image, 1)
        if vertical:
            image = cv2.flip(image, 0)
        return image
    
    def brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness
        
        Args:
            image: Input image
            factor: Brightness factor (0.5-2.0). 1.0 = original
        
        Returns:
            Brightness-adjusted image
        """
        image_float = image.astype(np.float32)
        image_float = image_float * factor
        image_float = np.clip(image_float, 0, 255)
        return image_float.astype(image.dtype)
    
    def contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image contrast
        
        Args:
            image: Input image
            factor: Contrast factor (0.5-2.0). 1.0 = original
        
        Returns:
            Contrast-adjusted image
        """
        image_float = image.astype(np.float32)
        mean = np.mean(image_float)
        image_float = (image_float - mean) * factor + mean
        image_float = np.clip(image_float, 0, 255)
        return image_float.astype(image.dtype)
    
    def gaussian_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply Gaussian blur
        
        Args:
            image: Input image
            kernel_size: Blur kernel size (must be odd)
        
        Returns:
            Blurred image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def gaussian_noise(self, image: np.ndarray, std: float = 25.0) -> np.ndarray:
        """Add Gaussian noise to image
        
        Args:
            image: Input image
            std: Standard deviation of noise
        
        Returns:
            Noisy image
        """
        noise = np.random.normal(0, std, image.shape)
        noisy_image = image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image.astype(image.dtype)
    
    def saturation(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust color saturation
        
        Args:
            image: Input image (BGR)
            factor: Saturation factor (0.5-2.0). 1.0 = original
        
        Returns:
            Saturation-adjusted image
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return bgr
    
    def hue_shift(self, image: np.ndarray, shift: int) -> np.ndarray:
        """Shift hue values
        
        Args:
            image: Input image (BGR)
            shift: Hue shift value (-180 to 180)
        
        Returns:
            Hue-shifted image
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
        bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return bgr
    
    def perspective_transform(self, image: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """Apply random perspective transformation
        
        Args:
            image: Input image
            strength: Strength of perspective distortion (0.0-0.3)
        
        Returns:
            Perspective-transformed image
        """
        height, width = image.shape[:2]
        strength = min(max(strength, 0.0), 0.3)
        
        # Generate random perspective points
        offset_x = int(width * strength)
        offset_y = int(height * strength)
        
        src_points = np.array([
            [0, 0],
            [width - 1, 0],
            [0, height - 1],
            [width - 1, height - 1]
        ], dtype=np.float32)
        
        dst_points = src_points + np.random.uniform(-offset_x, offset_x, src_points.shape).astype(np.float32)
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(image, matrix, (width, height))
        return warped
    
    def elastic_deform(self, image: np.ndarray, alpha: float = 30, sigma: float = 3) -> np.ndarray:
        """Apply elastic deformation
        
        Args:
            image: Input image
            alpha: Deformation strength
            sigma: Gaussian filter sigma
        
        Returns:
            Elastically deformed image
        """
        height, width = image.shape[:2]
        
        # Generate random displacement fields
        dx = np.random.randn(height, width) * sigma
        dy = np.random.randn(height, width) * sigma
        
        # Smooth displacement fields
        dx = cv2.GaussianBlur(dx, (5, 5), 0) * alpha
        dy = cv2.GaussianBlur(dy, (5, 5), 0) * alpha
        
        # Create coordinate maps
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x_deformed = np.clip(x + dx, 0, width - 1).astype(np.float32)
        y_deformed = np.clip(y + dy, 0, height - 1).astype(np.float32)
        
        # Remap image
        deformed = cv2.remap(image, x_deformed, y_deformed, cv2.INTER_LINEAR)
        return deformed
    
    def random_crop(self, image: np.ndarray, crop_size: int) -> np.ndarray:
        """Random crop from image
        
        Args:
            image: Input image
            crop_size: Size of crop region
        
        Returns:
            Cropped image
        """
        height, width = image.shape[:2]
        
        if crop_size >= min(height, width):
            return image
        
        x = random.randint(0, width - crop_size)
        y = random.randint(0, height - crop_size)
        
        return image[y:y+crop_size, x:x+crop_size]
    
    def random_erasing(self, image: np.ndarray, probability: float = 0.5, scale: Tuple[float, float] = (0.02, 0.33), 
                      ratio: Tuple[float, float] = (0.3, 3.0)) -> np.ndarray:
        """Random erasing augmentation
        
        Args:
            image: Input image
            probability: Probability of applying erasing
            scale: Range of erasing area size
            ratio: Range of erasing aspect ratio
        
        Returns:
            Image with random erasing
        """
        if random.random() > probability:
            return image
        
        height, width = image.shape[:2]
        area = height * width
        
        # Random erase area size and ratio
        erase_area = random.uniform(scale[0], scale[1]) * area
        erase_ratio = random.uniform(ratio[0], ratio[1])
        
        erase_h = int(np.sqrt(erase_area / erase_ratio))
        erase_w = int(np.sqrt(erase_area * erase_ratio))
        
        erase_h = min(erase_h, height)
        erase_w = min(erase_w, width)
        
        # Random position
        y = random.randint(0, height - erase_h)
        x = random.randint(0, width - erase_w)
        
        # Erase with random value
        erase_color = random.randint(0, 255)
        image[y:y+erase_h, x:x+erase_w] = erase_color
        
        return image
    
    def compose(self, image: np.ndarray, augmentations: List[dict]) -> np.ndarray:
        """Apply multiple augmentations in sequence
        
        Args:
            image: Input image
            augmentations: List of augmentation dicts with 'type' and 'params'
        
        Returns:
            Augmented image
        """
        for aug in augmentations:
            aug_type = aug.get('type')
            params = aug.get('params', {})
            
            if aug_type == 'rotate':
                image = self.rotate(image, **params)
            elif aug_type == 'flip':
                image = self.flip(image, **params)
            elif aug_type == 'brightness':
                image = self.brightness(image, **params)
            elif aug_type == 'contrast':
                image = self.contrast(image, **params)
            elif aug_type == 'blur':
                image = self.gaussian_blur(image, **params)
            elif aug_type == 'noise':
                image = self.gaussian_noise(image, **params)
            elif aug_type == 'saturation':
                image = self.saturation(image, **params)
            elif aug_type == 'hue_shift':
                image = self.hue_shift(image, **params)
            elif aug_type == 'perspective':
                image = self.perspective_transform(image, **params)
            elif aug_type == 'elastic':
                image = self.elastic_deform(image, **params)
            elif aug_type == 'crop':
                image = self.random_crop(image, **params)
            elif aug_type == 'erasing':
                image = self.random_erasing(image, **params)
        
        return image


class AugmentationPipeline:
    """Pre-configured augmentation pipelines"""
    
    @staticmethod
    def light_augmentation() -> List[dict]:
        """Light augmentation for validation/test"""
        return [
            {'type': 'flip', 'params': {'horizontal': True}},
        ]
    
    @staticmethod
    def moderate_augmentation() -> List[dict]:
        """Moderate augmentation for training"""
        return [
            {'type': 'flip', 'params': {'horizontal': True}},
            {'type': 'rotate', 'params': {'angle': random.uniform(-15, 15)}},
            {'type': 'brightness', 'params': {'factor': random.uniform(0.8, 1.2)}},
            {'type': 'contrast', 'params': {'factor': random.uniform(0.8, 1.2)}},
        ]
    
    @staticmethod
    def strong_augmentation() -> List[dict]:
        """Strong augmentation for robust training"""
        return [
            {'type': 'flip', 'params': {'horizontal': True}},
            {'type': 'rotate', 'params': {'angle': random.uniform(-25, 25)}},
            {'type': 'brightness', 'params': {'factor': random.uniform(0.7, 1.3)}},
            {'type': 'contrast', 'params': {'factor': random.uniform(0.7, 1.3)}},
            {'type': 'blur', 'params': {'kernel_size': random.choice([3, 5])}},
            {'type': 'noise', 'params': {'std': random.uniform(5, 15)}},
            {'type': 'saturation', 'params': {'factor': random.uniform(0.8, 1.2)}},
            {'type': 'perspective', 'params': {'strength': random.uniform(0.05, 0.15)}},
        ]
    
    @staticmethod
    def extreme_augmentation() -> List[dict]:
        """Extreme augmentation for regularization"""
        return [
            {'type': 'flip', 'params': {'horizontal': random.choice([True, False])}},
            {'type': 'flip', 'params': {'vertical': random.choice([True, False])}},
            {'type': 'rotate', 'params': {'angle': random.uniform(-45, 45)}},
            {'type': 'brightness', 'params': {'factor': random.uniform(0.5, 1.5)}},
            {'type': 'contrast', 'params': {'factor': random.uniform(0.5, 1.5)}},
            {'type': 'blur', 'params': {'kernel_size': random.choice([3, 5, 7])}},
            {'type': 'noise', 'params': {'std': random.uniform(10, 25)}},
            {'type': 'saturation', 'params': {'factor': random.uniform(0.5, 1.5)}},
            {'type': 'hue_shift', 'params': {'shift': random.randint(-30, 30)}},
            {'type': 'perspective', 'params': {'strength': random.uniform(0.1, 0.25)}},
            {'type': 'erasing', 'params': {'probability': 0.5}},
        ]
