"""Dataset preparation script for product detection"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import shutil
import json
import numpy as np
from tqdm import tqdm

from src.utils.logger import setup_logger
from src.utils.helpers import ensure_directory, list_files

logger = setup_logger(__name__)


class DatasetPreparer:
    """Prepare datasets for training"""
    
    def __init__(self, output_dir: str = 'data/prepared'):
        """Initialize dataset preparer
        
        Args:
            output_dir: Output directory for prepared dataset
        """
        self.output_dir = ensure_directory(output_dir)
        self.dataset_info = {}
        
        logger.info(f"Initialized DatasetPreparer, output: {output_dir}")
    
    def create_train_val_test_split(self, image_dir: str, train_ratio: float = 0.7,
                                   val_ratio: float = 0.15, test_ratio: float = 0.15,
                                   seed: int = 42):
        """Create train/val/test splits from image directory
        
        Args:
            image_dir: Directory containing all images
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed for reproducibility
        """
        logger.info(f"Creating train/val/test splits from {image_dir}")
        
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # Get image list
        image_paths = list_files(image_dir, extension='.jpg')
        image_paths.extend(list_files(image_dir, extension='.png'))
        
        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")
        
        logger.info(f"Found {len(image_paths)} images")
        
        # Shuffle with seed
        np.random.seed(seed)
        indices = np.random.permutation(len(image_paths))
        image_paths = [image_paths[i] for i in indices]
        
        # Calculate split indices
        train_count = int(len(image_paths) * train_ratio)
        val_count = int(len(image_paths) * val_ratio)
        
        train_images = image_paths[:train_count]
        val_images = image_paths[train_count:train_count + val_count]
        test_images = image_paths[train_count + val_count:]
        
        # Create directories
        splits_dir = Path(self.output_dir) / 'images'
        train_dir = splits_dir / 'train'
        val_dir = splits_dir / 'val'
        test_dir = splits_dir / 'test'
        
        for split_dir in [train_dir, val_dir, test_dir]:
            ensure_directory(str(split_dir))
        
        # Copy images
        self._copy_images(train_images, train_dir, 'Training')
        self._copy_images(val_images, val_dir, 'Validation')
        self._copy_images(test_images, test_dir, 'Test')
        
        # Save split info
        split_info = {
            'train': len(train_images),
            'val': len(val_images),
            'test': len(test_images),
            'total': len(image_paths),
            'seed': seed,
        }
        
        self.dataset_info['splits'] = split_info
        logger.info(f"Split info: {split_info}")
        
        return split_info
    
    def create_coco_annotations(self, annotations_file: str, output_file: str):
        """Convert annotations to COCO format
        
        Args:
            annotations_file: Input annotations file (JSON)
            output_file: Output COCO format file
        """
        logger.info(f"Converting annotations to COCO format")
        
        try:
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
            
            # Create COCO format structure
            coco_format = {
                'images': [],
                'annotations': [],
                'categories': [],
            }
            
            # Add images and annotations
            for image_id, (image_path, image_annots) in enumerate(annotations.items()):
                coco_format['images'].append({
                    'id': image_id,
                    'file_name': image_path,
                })
                
                for annot_id, annot in enumerate(image_annots.get('annotations', [])):
                    coco_format['annotations'].append({
                        'id': len(coco_format['annotations']),
                        'image_id': image_id,
                        'category_id': annot.get('class_id', 0),
                        'bbox': annot.get('bbox', []),
                        'area': annot.get('area', 0),
                        'iscrowd': 0,
                    })
            
            # Add categories
            categories = set(a['category_id'] for a in coco_format['annotations'])
            for cat_id in sorted(categories):
                coco_format['categories'].append({
                    'id': cat_id,
                    'name': f'class_{cat_id}',
                })
            
            # Save COCO format
            with open(output_file, 'w') as f:
                json.dump(coco_format, f, indent=2)
            
            logger.info(f"Saved COCO format to {output_file}")
            logger.info(f"Images: {len(coco_format['images'])}, Annotations: {len(coco_format['annotations'])}")
        
        except Exception as e:
            logger.error(f"Failed to convert annotations: {e}")
            raise
    
    def create_yolo_yaml(self, dataset_name: str = 'product_detection', num_classes: int = 10):
        """Create YOLO dataset YAML file
        
        Args:
            dataset_name: Dataset name
            num_classes: Number of classes
        
        Returns:
            Path to YAML file
        """
        logger.info(f"Creating YOLO YAML for {dataset_name}")
        
        yaml_content = f"""# {dataset_name} dataset
path: {self.output_dir}/images
train: train
val: val
test: test

nc: {num_classes}
names: 
"""
        
        for i in range(num_classes):
            yaml_content += f"  {i}: class_{i}\n"
        
        yaml_path = Path(self.output_dir) / f'{dataset_name}.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        logger.info(f"Saved YAML to {yaml_path}")
        return yaml_path
    
    def validate_dataset_structure(self, dataset_dir: str) -> Dict:
        """Validate dataset directory structure
        
        Args:
            dataset_dir: Dataset directory to validate
        
        Returns:
            Validation report
        """
        logger.info(f"Validating dataset structure: {dataset_dir}")
        
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'images_count': {},
            'annotations_count': {},
        }
        
        dataset_path = Path(dataset_dir)
        
        # Check for required directories
        required_dirs = ['images', 'annotations']
        for required_dir in required_dirs:
            dir_path = dataset_path / required_dir
            if not dir_path.exists():
                report['valid'] = False
                report['errors'].append(f"Missing directory: {required_dir}")
        
        # Count images
        images_dir = dataset_path / 'images'
        if images_dir.exists():
            for split in ['train', 'val', 'test']:
                split_dir = images_dir / split
                if split_dir.exists():
                    images = list_files(str(split_dir), extension='.jpg')
                    images.extend(list_files(str(split_dir), extension='.png'))
                    report['images_count'][split] = len(images)
                    
                    if len(images) == 0:
                        report['warnings'].append(f"No images in {split} split")
        
        logger.info(f"Validation report: {report}")
        return report
    
    def _copy_images(self, image_paths: List[str], dest_dir: Path, split_name: str):
        """Copy images to destination directory
        
        Args:
            image_paths: List of image paths
            dest_dir: Destination directory
            split_name: Name of split (for logging)
        """
        logger.info(f"Copying {len(image_paths)} {split_name} images...")
        
        for image_path in tqdm(image_paths, desc=f"{split_name} images"):
            dest_path = dest_dir / Path(image_path).name
            shutil.copy2(image_path, dest_path)
    
    def save_dataset_info(self, filename: str = 'dataset_info.json'):
        """Save dataset information
        
        Args:
            filename: Output filename
        """
        output_path = Path(self.output_dir) / filename
        with open(output_path, 'w') as f:
            json.dump(self.dataset_info, f, indent=2)
        logger.info(f"Saved dataset info to {output_path}")


def main():
    """Main dataset preparation entry point"""
    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument('--input', type=str, required=True, help='Input image directory')
    parser.add_argument('--output', type=str, default='data/prepared', help='Output directory')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Test set ratio')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--dataset-name', type=str, default='product_detection', help='Dataset name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--validate', action='store_true', help='Validate dataset structure')
    
    args = parser.parse_args()
    
    # Initialize preparer
    preparer = DatasetPreparer(args.output)
    
    # Create splits
    preparer.create_train_val_test_split(
        args.input,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    
    # Create YOLO YAML
    preparer.create_yolo_yaml(args.dataset_name, args.num_classes)
    
    # Save dataset info
    preparer.save_dataset_info()
    
    # Validate if requested
    if args.validate:
        report = preparer.validate_dataset_structure(args.output)
        logger.info(f"Validation result: {'PASSED' if report['valid'] else 'FAILED'}")
    
    logger.info(f"Dataset preparation completed!")


if __name__ == '__main__':
    main()
