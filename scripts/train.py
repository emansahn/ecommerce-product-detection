"""Training script for product detection model"""

import argparse
import os
import logging
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from src.utils.config import get_config
from src.utils.logger import setup_logger
from src.utils.helpers import set_random_seed, ensure_directory

logger = setup_logger(__name__)


class ModelTrainer:
    """Training manager for product detection models"""
    
    def __init__(self, config_path: str = None, output_dir: str = 'models/weights'):
        """Initialize trainer
        
        Args:
            config_path: Path to training configuration
            output_dir: Directory for saving models
        """
        self.config = get_config()
        self.output_dir = ensure_directory(output_dir)
        self.model = None
        self.best_metrics = {}
        
        logger.info(f"Initialized ModelTrainer, output: {self.output_dir}")
    
    def setup_model(self, model_name: str = 'yolov8m'):
        """Setup YOLO model for training
        
        Args:
            model_name: YOLOv8 model size (n, s, m, l, x)
        """
        if not YOLO_AVAILABLE:
            raise ImportError("YOLOv8 not installed. Install: pip install ultralytics")
        
        self.model = YOLO(f'{model_name}.pt')
        logger.info(f"Loaded model: {model_name}")
    
    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        device: str = '0',
        patience: int = 20,
        augment: bool = True,
        save_period: int = 10,
    ):
        """Train detection model
        
        Args:
            data_yaml: Path to dataset YAML file
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Input image size
            device: GPU device ID or 'cpu'
            patience: Early stopping patience
            augment: Enable augmentation
            save_period: Save checkpoint every N epochs
        
        Returns:
            Training results
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call setup_model() first.")
        
        logger.info("Starting training...")
        logger.info(f"Dataset: {data_yaml}")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        try:
            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                device=device,
                patience=patience,
                augment=augment,
                save_period=save_period,
                project=self.output_dir,
                name=f"yolov8_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                exist_ok=False,
            )
            
            self.best_metrics = {
                'epochs': epochs,
                'batch_size': batch_size,
                'img_size': img_size,
                'device': device,
                'timestamp': datetime.now().isoformat(),
            }
            
            logger.info(f"Training completed. Results saved to {self.output_dir}")
            return results
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def validate(self, model_path: str, data_yaml: str, img_size: int = 640, device: str = '0'):
        """Validate model on validation set
        
        Args:
            model_path: Path to trained model
            data_yaml: Path to dataset YAML
            img_size: Input image size
            device: GPU device ID
        
        Returns:
            Validation results
        """
        try:
            model = YOLO(model_path)
            results = model.val(data=data_yaml, imgsz=img_size, device=device)
            
            logger.info(f"Validation completed")
            logger.info(f"mAP@0.5: {results.box.map50:.4f}")
            logger.info(f"mAP@0.5:0.95: {results.box.map:.4f}")
            
            return results
        
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
    
    def export_model(self, model_path: str, format: str = 'onnx'):
        """Export trained model to different format
        
        Args:
            model_path: Path to trained model
            format: Export format (pytorch, onnx, tflite, pb, etc.)
        """
        try:
            model = YOLO(model_path)
            exported_path = model.export(format=format)
            logger.info(f"Model exported to {exported_path}")
            return exported_path
        
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise


def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(description='Train product detection model')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset YAML')
    parser.add_argument('--model', type=str, default='yolov8m', help='Model size (n, s, m, l, x)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--img', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='0', help='GPU device ID')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--output', type=str, default='models/weights', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-augment', action='store_true', help='Disable augmentation')
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Initialize trainer
    trainer = ModelTrainer(output_dir=args.output)
    trainer.setup_model(args.model)
    
    # Train model
    trainer.train(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img,
        device=args.device,
        patience=args.patience,
        augment=not args.no_augment,
    )


if __name__ == '__main__':
    main()
