"""Evaluation script for model performance assessment"""

import argparse
import logging
from pathlib import Path
import numpy as np
import json

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from src.utils.logger import setup_logger
from src.utils.helpers import ensure_directory, dict_to_json_file

logger = setup_logger(__name__)


class ModelEvaluator:
    """Evaluation manager for detection models"""
    
    def __init__(self, model_path: str, output_dir: str = 'results'):
        """Initialize evaluator
        
        Args:
            model_path: Path to trained model
            output_dir: Directory for saving results
        """
        if not YOLO_AVAILABLE:
            raise ImportError("YOLOv8 not installed")
        
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.output_dir = ensure_directory(output_dir)
        self.results = {}
        
        logger.info(f"Initialized ModelEvaluator with model: {model_path}")
    
    def evaluate_on_dataset(self, data_yaml: str, img_size: int = 640, device: str = '0', 
                           conf: float = 0.25, iou: float = 0.45):
        """Evaluate model on dataset
        
        Args:
            data_yaml: Path to dataset YAML
            img_size: Input image size
            device: GPU device ID
            conf: Confidence threshold
            iou: IoU threshold for NMS
        
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating on dataset: {data_yaml}")
        
        try:
            results = self.model.val(
                data=data_yaml,
                imgsz=img_size,
                device=device,
                conf=conf,
                iou=iou,
            )
            
            # Extract metrics
            self.results['validation'] = {
                'mAP@0.5': float(results.box.map50) if hasattr(results, 'box') else 0.0,
                'mAP@0.5:0.95': float(results.box.map) if hasattr(results, 'box') else 0.0,
                'precision': float(results.box.mp) if hasattr(results, 'box') else 0.0,
                'recall': float(results.box.mr) if hasattr(results, 'box') else 0.0,
            }
            
            logger.info(f"Validation Results:")
            logger.info(f"  mAP@0.5: {self.results['validation']['mAP@0.5']:.4f}")
            logger.info(f"  mAP@0.5:0.95: {self.results['validation']['mAP@0.5:0.95']:.4f}")
            logger.info(f"  Precision: {self.results['validation']['precision']:.4f}")
            logger.info(f"  Recall: {self.results['validation']['recall']:.4f}")
            
            return results
        
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def evaluate_on_images(self, image_dir: str, img_size: int = 640, device: str = '0',
                          conf: float = 0.25, visualize: bool = False):
        """Evaluate on directory of images
        
        Args:
            image_dir: Directory containing test images
            img_size: Input image size
            device: GPU device ID
            conf: Confidence threshold
            visualize: Save annotated images
        
        Returns:
            List of results per image
        """
        logger.info(f"Evaluating on image directory: {image_dir}")
        
        try:
            results = self.model.predict(
                source=image_dir,
                imgsz=img_size,
                device=device,
                conf=conf,
                save=visualize,
                project=self.output_dir,
                name='predictions',
            )
            
            logger.info(f"Processed {len(results)} images")
            
            # Aggregate statistics
            num_detections = sum([len(r.boxes) for r in results])
            avg_conf = np.mean([r.boxes.conf.mean().cpu().numpy() for r in results if len(r.boxes) > 0])
            
            self.results['image_evaluation'] = {
                'num_images': len(results),
                'num_detections': int(num_detections),
                'avg_confidence': float(avg_conf) if not np.isnan(avg_conf) else 0.0,
            }
            
            logger.info(f"Average confidence: {self.results['image_evaluation']['avg_confidence']:.4f}")
            
            return results
        
        except Exception as e:
            logger.error(f"Image evaluation failed: {e}")
            raise
    
    def compute_metrics_per_class(self, data_yaml: str, img_size: int = 640, device: str = '0'):
        """Compute metrics per class
        
        Args:
            data_yaml: Path to dataset YAML
            img_size: Input image size
            device: GPU device ID
        
        Returns:
            Per-class metrics
        """
        logger.info("Computing per-class metrics")
        
        try:
            results = self.model.val(
                data=data_yaml,
                imgsz=img_size,
                device=device,
            )
            
            # Extract per-class results
            if hasattr(results, 'class_result'):
                per_class_metrics = {}
                for class_id, metric in results.class_result.items():
                    per_class_metrics[str(class_id)] = {
                        'mAP@0.5': float(metric[0]) if len(metric) > 0 else 0.0,
                        'mAP@0.5:0.95': float(metric[1]) if len(metric) > 1 else 0.0,
                    }
                
                self.results['per_class'] = per_class_metrics
                logger.info(f"Computed metrics for {len(per_class_metrics)} classes")
            
            return per_class_metrics
        
        except Exception as e:
            logger.error(f"Per-class evaluation failed: {e}")
            raise
    
    def save_results(self, filename: str = 'evaluation_results.json'):
        """Save evaluation results to JSON
        
        Args:
            filename: Output filename
        """
        output_path = Path(self.output_dir) / filename
        dict_to_json_file(self.results, str(output_path))
        logger.info(f"Results saved to {output_path}")
    
    def generate_report(self):
        """Generate evaluation report
        
        Returns:
            Report string
        """
        report = "\n" + "="*60 + "\n"
        report += "EVALUATION REPORT\n"
        report += "="*60 + "\n"
        
        for section, metrics in self.results.items():
            report += f"\n{section.upper()}:\n"
            for key, value in metrics.items():
                if isinstance(value, float):
                    report += f"  {key}: {value:.4f}\n"
                else:
                    report += f"  {key}: {value}\n"
        
        report += "\n" + "="*60 + "\n"
        return report


def main():
    """Main evaluation entry point"""
    parser = argparse.ArgumentParser(description='Evaluate product detection model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, help='Path to dataset YAML')
    parser.add_argument('--images', type=str, help='Path to image directory')
    parser.add_argument('--img', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='0', help='GPU device ID')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Save annotated images')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model, args.output)
    
    # Evaluate on dataset
    if args.data:
        evaluator.evaluate_on_dataset(args.data, args.img, args.device, args.conf, args.iou)
    
    # Evaluate on images
    if args.images:
        evaluator.evaluate_on_images(args.images, args.img, args.device, args.conf, args.visualize)
    
    # Save and print results
    evaluator.save_results()
    print(evaluator.generate_report())


if __name__ == '__main__':
    main()
