"""Feature extraction for classification"""

import numpy as np
import cv2
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from detected regions"""
    
    def __init__(self, feature_dim: int = 256):
        """Initialize feature extractor
        
        Args:
            feature_dim: Output feature dimension
        """
        self.feature_dim = feature_dim
    
    def extract_color_features(self, image: np.ndarray) -> np.ndarray:
        """Extract color-based features
        
        Args:
            image: Input image (BGR)
        
        Returns:
            Color feature vector
        """
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Compute color histograms
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        # Normalize histograms
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()
        
        # Concatenate features
        features = np.concatenate([h_hist, s_hist, v_hist])
        return features
    
    def extract_shape_features(self, image: np.ndarray) -> np.ndarray:
        """Extract shape-based features
        
        Args:
            image: Input image (BGR)
        
        Returns:
            Shape feature vector
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute edge map
        edges = cv2.Canny(gray, 50, 150)
        
        # Compute contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract shape features
        num_contours = len(contours)
        perimeter = 0
        area = 0
        
        for contour in contours:
            perimeter += cv2.arcLength(contour, True)
            area += cv2.contourArea(contour)
        
        # Compute Hu moments for shape description
        hu_moments = cv2.HuMoments(edges).flatten()
        
        # Build feature vector
        features = np.array([
            num_contours,
            perimeter,
            area,
            *hu_moments[:7]  # Use first 7 Hu moments
        ])
        
        return features
    
    def extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract texture features using LBP
        
        Args:
            image: Input image (BGR)
        
        Returns:
            Texture feature vector
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple texture features: compute statistics on gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude and direction
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Statistics
        features = np.array([
            np.mean(magnitude),
            np.std(magnitude),
            np.min(magnitude),
            np.max(magnitude),
            np.mean(sobelx),
            np.std(sobelx),
            np.mean(sobely),
            np.std(sobely),
        ])
        
        return features
    
    def extract_sift_features(self, image: np.ndarray, max_features: int = 100) -> np.ndarray:
        """Extract SIFT features
        
        Args:
            image: Input image (BGR)
            max_features: Maximum number of SIFT features
        
        Returns:
            SIFT feature vector
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect SIFT keypoints and descriptors
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        if descriptors is None:
            # No features found
            return np.zeros(128)
        
        # Pool features (mean)
        if len(descriptors) > max_features:
            indices = np.random.choice(len(descriptors), max_features, replace=False)
            descriptors = descriptors[indices]
        
        # Aggregate descriptors
        features = np.mean(descriptors, axis=0)
        
        return features
    
    def extract_orb_features(self, image: np.ndarray, max_features: int = 100) -> np.ndarray:
        """Extract ORB features
        
        Args:
            image: Input image (BGR)
            max_features: Maximum number of ORB features
        
        Returns:
            ORB feature vector
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect ORB keypoints and descriptors
        orb = cv2.ORB_create(nfeatures=max_features)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        if descriptors is None:
            # No features found
            return np.zeros(32)
        
        # Aggregate descriptors
        features = np.mean(descriptors.astype(np.float32), axis=0)
        
        return features
    
    def extract_all_features(self, image: np.ndarray) -> np.ndarray:
        """Extract all available features
        
        Args:
            image: Input image (BGR)
        
        Returns:
            Combined feature vector
        """
        color_features = self.extract_color_features(image)
        shape_features = self.extract_shape_features(image)
        texture_features = self.extract_texture_features(image)
        
        # Normalize features
        color_features = color_features / (np.linalg.norm(color_features) + 1e-6)
        shape_features = shape_features / (np.linalg.norm(shape_features) + 1e-6)
        texture_features = texture_features / (np.linalg.norm(texture_features) + 1e-6)
        
        # Concatenate
        all_features = np.concatenate([
            color_features,
            shape_features,
            texture_features,
        ])
        
        return all_features
