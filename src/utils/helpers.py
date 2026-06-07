"""Utility helper functions"""

import numpy as np
from typing import Tuple, List, Optional, Any, Dict
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def create_directory_structure(base_path: str, structure: Dict[str, Any]) -> None:
    """Create nested directory structure
    
    Args:
        base_path: Base directory path
        structure: Dictionary defining directory structure
    """
    for key, value in structure.items():
        path = os.path.join(base_path, key)
        os.makedirs(path, exist_ok=True)
        logger.debug(f"Created directory: {path}")
        
        if isinstance(value, dict):
            create_directory_structure(path, value)


def ensure_directory(path: str) -> str:
    """Ensure directory exists, create if needed
    
    Args:
        path: Directory path
    
    Returns:
        Absolute path
    """
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    return path


def get_file_size(file_path: str) -> int:
    """Get file size in bytes
    
    Args:
        file_path: Path to file
    
    Returns:
        File size in bytes
    """
    if os.path.exists(file_path):
        return os.path.getsize(file_path)
    return 0


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB
    
    Args:
        file_path: Path to file
    
    Returns:
        File size in MB
    """
    return get_file_size(file_path) / (1024 * 1024)


def list_files(directory: str, extension: Optional[str] = None) -> List[str]:
    """List files in directory
    
    Args:
        directory: Directory path
        extension: Optional file extension filter (e.g., '.jpg')
    
    Returns:
        List of file paths
    """
    files = []
    if not os.path.isdir(directory):
        return files
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            if extension is None or filename.endswith(extension):
                files.append(filepath)
    
    return sorted(files)


def dict_to_json_file(data: Dict, file_path: str) -> None:
    """Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        file_path: Output file path
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved JSON to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")
        raise


def json_file_to_dict(file_path: str) -> Dict:
    """Load dictionary from JSON file
    
    Args:
        file_path: Input file path
    
    Returns:
        Loaded dictionary
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.debug(f"Loaded JSON from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON: {e}")
        raise


def normalize_vector(vector: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Normalize vector to unit length
    
    Args:
        vector: Input vector
        axis: Axis for normalization
    
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(vector, axis=axis, keepdims=True)
    return vector / (norm + 1e-8)


def compute_statistics(values: np.ndarray) -> Dict[str, float]:
    """Compute statistics for array
    
    Args:
        values: Input array
    
    Returns:
        Dictionary with statistics
    """
    return {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75)),
    }


def flatten_nested_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten nested dictionary
    
    Args:
        d: Nested dictionary
        parent_key: Parent key prefix
        sep: Separator for keys
    
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def merge_dicts(*dicts: Dict) -> Dict:
    """Merge multiple dictionaries
    
    Args:
        *dicts: Variable number of dictionaries
    
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def format_time(seconds: float) -> str:
    """Format seconds to human-readable time
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_size(bytes_size: int) -> str:
    """Format bytes to human-readable size
    
    Args:
        bytes_size: Size in bytes
    
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f}TB"


def batch_iterator(items: List[Any], batch_size: int):
    """Iterate over items in batches
    
    Args:
        items: List of items
        batch_size: Size of each batch
    
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    except ImportError:
        pass
    logger.debug(f"Set random seed to {seed}")


def get_config_path() -> str:
    """Get configuration directory path
    
    Args:
    
    Returns:
        Config directory path
    """
    config_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'config')
    return os.path.abspath(config_dir)


def get_data_path() -> str:
    """Get data directory path
    
    Returns:
        Data directory path
    """
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    return os.path.abspath(data_dir)


def get_models_path() -> str:
    """Get models directory path
    
    Returns:
        Models directory path
    """
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    return os.path.abspath(models_dir)
