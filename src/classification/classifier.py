"""Product classification module"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ProductClassifier:
    """Feature-based product classifier with an optional scikit-learn backend.

    In the absence of a trained model file the classifier falls back to a
    simple argmax over the first ``num_classes`` dimensions of the feature
    vector — useful for demos and unit tests.
    """

    CATEGORIES: Dict[int, str] = {
        0: "Electronics",
        1: "Clothing",
        2: "Accessories",
        3: "Home & Garden",
        4: "Sports & Outdoors",
        5: "Beauty & Personal Care",
        6: "Food & Beverage",
        7: "Books & Media",
        8: "Furniture",
        9: "Toys & Games",
    }

    def __init__(self, model_path: Optional[str] = None, num_classes: int = 10):
        """Initialise classifier.

        Args:
            model_path: Optional path to a pickled scikit-learn (or compatible)
                model saved with ``joblib.dump``.
            num_classes: Number of product categories.
        """
        self.num_classes = num_classes
        self.model = None
        self.model_path = model_path

        if model_path:
            self.load_model(model_path)

        logger.info("ProductClassifier initialised — %d categories", num_classes)

    # ------------------------------------------------------------------
    # Model I/O
    # ------------------------------------------------------------------

    def load_model(self, model_path: str) -> None:
        """Load a serialised scikit-learn classifier from disk.

        Args:
            model_path: Path to the ``.joblib`` (or ``.pkl``) file.

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError: If joblib is not installed or loading fails.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Classifier model not found: {path}")

        try:
            import joblib  # type: ignore
            self.model = joblib.load(path)
            self.model_path = str(path)
            logger.info("Loaded classifier from %s", path)
        except ImportError as exc:
            raise RuntimeError(
                "joblib is required to load the classifier. "
                "Install with: pip install joblib"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load model: {exc}") from exc

    def save_model(self, output_path: str) -> None:
        """Serialise the current model to disk with joblib.

        Args:
            output_path: Destination path (e.g. ``"models/weights/classifier.joblib"``).

        Raises:
            RuntimeError: If no model is loaded or joblib is missing.
        """
        if self.model is None:
            raise RuntimeError("No model to save — train or load a model first.")
        try:
            import joblib  # type: ignore
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, output_path)
            logger.info("Classifier saved to %s", output_path)
        except ImportError as exc:
            raise RuntimeError("joblib is required. Install with: pip install joblib") from exc

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify(self, features: np.ndarray) -> Dict:
        """Classify a single feature vector.

        Args:
            features: 1-D float array produced by :class:`~src.classification.features.FeatureExtractor`.

        Returns:
            Dict with ``class_id`` (int), ``class_name`` (str), and
            ``confidence`` (float in [0, 1]).
        """
        features = np.asarray(features, dtype=np.float32).ravel()

        if self.model is not None:
            # scikit-learn classifier
            proba = self.model.predict_proba(features.reshape(1, -1))[0]
            class_id = int(np.argmax(proba))
            confidence = float(proba[class_id])
        else:
            # Fallback: treat first num_classes dims as pseudo-probabilities
            pseudo = features[: self.num_classes] if len(features) >= self.num_classes else features
            if pseudo.size == 0 or pseudo.max() == pseudo.min():
                class_id, confidence = 0, 1.0 / max(self.num_classes, 1)
            else:
                # Softmax-normalise so confidence is meaningful
                exp = np.exp(pseudo - pseudo.max())
                proba = exp / exp.sum()
                class_id = int(np.argmax(proba))
                confidence = float(proba[class_id])

        return {
            "class_id": class_id,
            "class_name": self.CATEGORIES.get(class_id, f"class_{class_id}"),
            "confidence": confidence,
        }

    def classify_batch(self, features_batch: np.ndarray) -> List[Dict]:
        """Classify multiple feature vectors.

        Args:
            features_batch: 2-D array of shape ``[N, F]``.

        Returns:
            List of classification result dicts.
        """
        return [self.classify(f) for f in features_batch]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_class_name(self, class_id: int) -> str:
        return self.CATEGORIES.get(class_id, f"class_{class_id}")

    def get_all_categories(self) -> Dict[int, str]:
        return self.CATEGORIES.copy()
