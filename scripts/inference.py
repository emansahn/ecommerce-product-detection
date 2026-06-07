"""Batch inference script for product detection"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import json
from tqdm import tqdm
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from src.utils.logger import setup_logger
from src.utils.helpers import ensure_directory, dict_to_json_file, list_files, format_time

logger = setup_logger(__name__)


class BatchInference:
    """Batch inference manager for product detection"""
    
    def __init__(self, model_path: str, output_dir: str = 'inference_results'):
        """Initialize inference manager
        
        Args:
            model_path: Path to trained model
            output_dir: Directory for saving results
        """
        if not YOLO_AVAILABLE:
            raise ImportError("YOLOv8 not installed")
        
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.output_dir = ensure_directory(output_dir)
        self.results = []
        self.statistics = {}
        
        logger.info(f"Initialized BatchInference with model: {model_path}")
    
    def infer_single_image(self, image_path: str, conf: float = 0.5, iou: float = 0.45,
                          device: str = '0') -> Dict:
        """Run inference on single image
        
        Args:
            image_path: Path to input image
            conf: Confidence threshold
            iou: IoU threshold for NMS
            device: GPU device ID
        
        Returns:
            Detection results dictionary
        """
        try:
            result = self.model.predict(
                source=image_path,
                conf=conf,
                iou=iou,
                device=device,
                verbose=False,
            )
            
            result = result[0]
            
            # Extract detections
            detections = []
            for i in range(len(result.boxes)):
                box = result.boxes[i]
                detection = {
                    'class_id': int(box.cls[0]),
                    'class_name': self.model.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': {
                        'x1': float(box.xyxy[0][0]),
                        'y1': float(box.xyxy[0][1]),
                        'x2': float(box.xyxy[0][2]),
                        'y2': float(box.xyxy[0][3]),
                    },
                    'area': float((box.xyxy[0][2] - box.xyxy[0][0]) * 
                                 (box.xyxy[0][3] - box.xyxy[0][1])),
                }
                detections.append(detection)
            
            result_dict = {
                'image_path': str(image_path),
                'image_size': [int(result.orig_shape[0]), int(result.orig_shape[1])],
                'num_detections': len(detections),
                'detections': detections,
                'inference_time': float(result.speed['inference']),
            }
            
            return result_dict
        
        except Exception as e:
            logger.error(f"Inference failed for {image_path}: {e}")
            return None
    
    def infer_batch(self, image_dir: str, conf: float = 0.5, iou: float = 0.45,
                   device: str = '0', save_txt: bool = False, save_images: bool = False) -> List[Dict]:
        """Run inference on batch of images
        
        Args:
            image_dir: Directory containing images
            conf: Confidence threshold
            iou: IoU threshold
            device: GPU device ID
            save_txt: Save detections to text files
            save_images: Save annotated images
        
        Returns:
            List of detection results
        """
        logger.info(f"Running batch inference on {image_dir}")
        
        # Get image list
        image_paths = list_files(image_dir, extension='.jpg')
        image_paths.extend(list_files(image_dir, extension='.png'))
        
        if not image_paths:
            logger.warning(f"No images found in {image_dir}")
            return []
        
        logger.info(f"Found {len(image_paths)} images")
        
        # Run inference
        start_time = time.time()
        results = []
        
        for image_path in tqdm(image_paths, desc="Processing images"):
            result = self.infer_single_image(image_path, conf, iou, device)
            if result:
                results.append(result)
                
                # Save text annotations
                if save_txt:
                    self._save_txt_annotation(image_path, result)
        
        total_time = time.time() - start_time
        
        # Save annotated images if requested
        if save_images:
            self._save_annotated_images(image_dir, results)
        
        # Compute statistics
        self.statistics = self._compute_statistics(results, total_time)
        self.results = results
        
        logger.info(f"Batch inference completed in {format_time(total_time)}")
        logger.info(f"Processed {len(results)} images")
        logger.info(f"Average inference time: {self.statistics['avg_inference_time']:.2f}ms")
        logger.info(f"Total detections: {self.statistics['total_detections']}")
        
        return results
    
    def _save_txt_annotation(self, image_path: str, result: Dict):
        """Save detections to text file (YOLO format)
        
        Args:
            image_path: Path to image
            result: Detection result
        """
        txt_path = Path(self.output_dir) / (Path(image_path).stem + '.txt')
        
        with open(txt_path, 'w') as f:
            for detection in result['detections']:
                # YOLO format: class_id center_x center_y width height confidence
                bbox = detection['bbox']
                center_x = (bbox['x1'] + bbox['x2']) / 2 / result['image_size'][1]
                center_y = (bbox['y1'] + bbox['y2']) / 2 / result['image_size'][0]
                width = (bbox['x2'] - bbox['x1']) / result['image_size'][1]
                height = (bbox['y2'] - bbox['y1']) / result['image_size'][0]
                
                f.write(f"{detection['class_id']} {center_x} {center_y} {width} {height} {detection['confidence']}\n")
    
    def _save_annotated_images(self, image_dir: str, results: List[Dict]):
        """Save annotated images
        
        Args:
            image_dir: Source image directory
            results: Detection results
        """
        import cv2
        
        logger.info("Saving annotated images...")
        
        for result in results:
            image_path = result['image_path']
            image = cv2.imread(image_path)
            
            if image is None:
                continue
            
            # Draw detections
            for detection in result['detections']:
                bbox = detection['bbox']
                x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
                conf = detection['confidence']
                class_name = detection['class_name']
                
                # Draw box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 2)
            
            # Save annotated image
            output_path = Path(self.output_dir) / f"annotated_{Path(image_path).name}"
            cv2.imwrite(str(output_path), image)
        
        logger.info(f"Saved {len(results)} annotated images")
    
    def _compute_statistics(self, results: List[Dict], total_time: float) -> Dict:
        """Compute inference statistics
        
        Args:
            results: Detection results
            total_time: Total processing time
        
        Returns:
            Statistics dictionary
        """
        total_detections = sum([r['num_detections'] for r in results])
        inference_times = [r['inference_time'] for r in results]
        
        return {
            'num_images': len(results),
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / len(results) if results else 0,
            'avg_inference_time': np.mean(inference_times) * 1000 if inference_times else 0,  # ms
            'total_time': total_time,
            'throughput': len(results) / total_time if total_time > 0 else 0,  # images/sec
        }
    
    def save_results(self, filename: str = 'inference_results.json'):
        """Save inference results to JSON
        
        Args:
            filename: Output filename
        """
        output = {
            'statistics': self.statistics,
            'results': self.results,
        }
        
        output_path = Path(self.output_dir) / filename
        dict_to_json_file(output, str(output_path))
        logger.info(f"Results saved to {output_path}")
    
    def generate_report(self) -> str:
        """Generate inference report
        
        Returns:
            Report string
        """
        report = "\n" + "="*60 + "\n"
        report += "INFERENCE REPORT\n"
        report += "="*60 + "\n"
        report += f"Model: {self.model_path}\n"
        report += f"Output Directory: {self.output_dir}\n\n"
        
        report += "STATISTICS:\n"
        for key, value in self.statistics.items():
            if 'time' in key or 'throughput' in key:
                if 'throughput' in key:
                    report += f"  {key}: {value:.2f} images/sec\n"
                else:
                    report += f"  {key}: {value:.2f}ms\n"
            else:
                report += f"  {key}: {value}\n"
        
        report += "\n" + "="*60 + "\n"
        return report


def main():
    """Main inference entry point"""
    parser = argparse.ArgumentParser(description='Run batch inference on product images')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--source', type=str, required=True, help='Image directory or single image')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='0', help='GPU device ID')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--output', type=str, default='inference_results', help='Output directory')
    parser.add_argument('--save-txt', action='store_true', help='Save text annotations')
    parser.add_argument('--save-images', action='store_true', help='Save annotated images')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = BatchInference(args.model, args.output)
    
    # Run inference
    if Path(args.source).is_dir():
        results = inference.infer_batch(
            args.source,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            save_txt=args.save_txt,
            save_images=args.save_images,
        )
    else:
        result = inference.infer_single_image(
            args.source,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
        )
        results = [result] if result else []
    
    # Save and print results
    inference.save_results()
    print(inference.generate_report())


if __name__ == '__main__':
    main()
