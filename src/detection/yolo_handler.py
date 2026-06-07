"""YOLOv8 handler — fine-tuning, export, and advanced inference utilities."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

logger = logging.getLogger(__name__)


class YOLOHandler:
    """High-level wrapper around Ultralytics YOLO for training, evaluation, and export.

    Use :class:`~src.detection.detector.ProductDetector` for plain inference.
    Use this class when you need to fine-tune, validate, or convert models.
    """

    SUPPORTED_EXPORT_FORMATS = {"onnx", "torchscript", "tflite", "pb", "coreml", "saved_model"}

    def __init__(self, model_path: str = "yolov8m.pt"):
        """Load a YOLO model from a weights file or a pretrained model tag.

        Args:
            model_path: Path to a ``.pt`` file or an Ultralytics model tag such as
                ``"yolov8m"`` (the ``.pt`` suffix is added automatically).
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics is not installed. Install with: pip install ultralytics"
            )

        if not model_path.endswith(".pt"):
            model_path = f"{model_path}.pt"

        self.model = YOLO(model_path)
        self.model_path = model_path
        logger.info("YOLOHandler loaded: %s", model_path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        device: Optional[str] = None,
        patience: int = 20,
        augment: bool = True,
        save_period: int = 10,
        project: str = "models/weights",
        name: Optional[str] = None,
    ):
        """Fine-tune the model on a custom dataset.

        Args:
            data_yaml: Path to the dataset YAML file (Ultralytics format).
            epochs: Maximum number of training epochs.
            batch_size: Mini-batch size.
            img_size: Input image resolution (square).
            device: Inference device (``None`` = auto).
            patience: Early-stopping patience in epochs.
            augment: Enable built-in Ultralytics augmentation.
            save_period: Save a checkpoint every *N* epochs.
            project: Parent directory for run outputs.
            name: Sub-directory name for this run (auto-generated when ``None``).

        Returns:
            Ultralytics training results object.
        """
        logger.info(
            "Starting training — data=%s, epochs=%d, batch=%d, img=%d",
            data_yaml,
            epochs,
            batch_size,
            img_size,
        )
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            patience=patience,
            augment=augment,
            save_period=save_period,
            project=project,
            name=name,
            exist_ok=False,
        )
        logger.info("Training complete.")
        return results

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        data_yaml: str,
        img_size: int = 640,
        device: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> Dict[str, float]:
        """Run validation and return a summary metrics dict.

        Returns:
            Dict with keys ``mAP@0.5``, ``mAP@0.5:0.95``, ``precision``, ``recall``.
        """
        results = self.model.val(
            data=data_yaml,
            imgsz=img_size,
            device=device,
            conf=conf,
            iou=iou,
        )
        metrics = {
            "mAP@0.5": float(results.box.map50),
            "mAP@0.5:0.95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        }
        logger.info("Validation — %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, fmt: str = "onnx", output_path: Optional[str] = None) -> str:
        """Export the model to a deployment format.

        Args:
            fmt: Target format — one of ``onnx``, ``torchscript``, ``tflite``,
                 ``pb``, ``coreml``, ``saved_model``.
            output_path: Optional override for the exported file location.

        Returns:
            Path to the exported model.

        Raises:
            ValueError: If the requested format is not supported.
        """
        if fmt not in self.SUPPORTED_EXPORT_FORMATS:
            raise ValueError(
                f"Unsupported format '{fmt}'. "
                f"Choose from: {sorted(self.SUPPORTED_EXPORT_FORMATS)}"
            )
        exported = self.model.export(format=fmt)
        logger.info("Model exported to %s", exported)
        return str(exported)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def predict(
        self,
        source: Union[str, np.ndarray],
        conf: float = 0.5,
        iou: float = 0.45,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> List[Dict]:
        """Run inference and return a list of structured result dicts.

        Args:
            source: Image path, directory path, URL, or BGR NumPy array.
            conf: Confidence threshold.
            iou: IoU threshold for NMS.
            device: Device override.
            verbose: Print Ultralytics progress output.

        Returns:
            List of dicts, one per image, each containing:
            ``{"boxes": [...], "confidences": [...], "class_ids": [...], "class_names": [...]}``.
        """
        raw_results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            device=device,
            verbose=verbose,
        )

        output = []
        for r in raw_results:
            boxes = r.boxes.xyxy.cpu().numpy().tolist()
            confs = r.boxes.conf.cpu().numpy().tolist()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int).tolist()
            output.append(
                {
                    "boxes": boxes,
                    "confidences": confs,
                    "class_ids": cls_ids,
                    "class_names": [self.model.names[c] for c in cls_ids],
                }
            )
        return output

    @property
    def class_names(self) -> Dict[int, str]:
        """Class-name mapping from the loaded model."""
        return self.model.names
