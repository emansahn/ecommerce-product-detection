# Ecommerce Product Detection

A computer vision system for automated product detection and classification in ecommerce images using NumPy-based image processing and deep learning models.

## 🎯 Project Overview

This project aims to build a robust product detection pipeline for ecommerce platforms, capable of:
- **Product Detection**: Identifying products in cluttered scenes
- **Bounding Box Regression**: Precise localization of product regions
- **Multi-class Classification**: Categorizing products into ecommerce categories
- **Quality Assessment**: Evaluating image quality and product visibility
- **Batch Processing**: Handling large-scale product image catalogs

## 🏗️ Project Structure

```
ecommerce-product-detection/
├── data/
│   ├── raw/                    # Raw images from ecommerce platforms
│   ├── processed/              # Preprocessed images
│   ├── annotations/            # YOLO/COCO format annotations
│   └── splits/                 # Train/val/test splits
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── image_loader.py     # Load and validate images
│   │   ├── augmentation.py     # Image augmentation pipelines
│   │   └── normalization.py    # Image normalization & resizing
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── detector.py         # Main detection model wrapper
│   │   ├── yolo_handler.py     # YOLOv8 integration
│   │   └── postprocessing.py   # NMS, confidence filtering
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── classifier.py       # Product classification
│   │   └── features.py         # Feature extraction
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py              # FastAPI application
│   │   ├── models.py           # Request/response schemas
│   │   └── routes.py           # API endpoints
│   └── utils/
│       ├── __init__.py
│       ├── logger.py           # Logging configuration
│       ├── config.py           # Configuration management
│       └── helpers.py          # Utility functions
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_pipeline.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation_analysis.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_detection.py
│   ├── test_api.py
│   └── fixtures/
│       └── sample_images/
├── models/
│   ├── weights/                # Pre-trained model weights
│   └── configs/                # Model configuration files
├── scripts/
│   ├── train.py                # Training script
│   ├── evaluate.py             # Model evaluation
│   ├── inference.py            # Batch inference
│   └── prepare_dataset.py       # Dataset preparation
├── config/
│   ├── default.yaml            # Default configuration
│   ├── development.yaml        # Dev environment config
│   └── production.yaml         # Production config
├── docs/
│   ├── ARCHITECTURE.md         # System architecture
│   ├── API.md                  # API documentation
│   ├── SETUP.md                # Setup instructions
│   └── MODELS.md               # Model documentation
├── .github/
│   ├── workflows/
│   │   ├── tests.yml           # CI tests
│   │   └── deploy.yml          # CD pipeline
│   └── ISSUE_TEMPLATE/
│       └── bug_report.md
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── .gitignore
├── .env.example
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── LICENSE
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- GPU (CUDA 11.8+) - Optional but recommended
- 8GB RAM minimum, 16GB+ recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/emansahn/ecommerce-product-detection.git
cd ecommerce-product-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.detection.detector import ProductDetector
from src.preprocessing.image_loader import ImageLoader

# Initialize detector
detector = ProductDetector(model='yolov8m')

# Load image
loader = ImageLoader()
image = loader.load('path/to/product_image.jpg')

# Run detection
results = detector.detect(image)
print(results)
```

### Start API Server

```bash
cd src/api
uvicorn app:app --reload --port 8000
```

API will be available at `http://localhost:8000`

## 📊 Model Architecture

### Detection Pipeline
1. **Image Preprocessing**: Normalization, resizing, augmentation
2. **Feature Extraction**: YOLOv8 backbone (CSPDarknet)
3. **Detection Head**: Anchor-free detection with confidence scores
4. **Post-processing**: NMS, confidence filtering, bounding box refinement

### Technologies
- **Detection**: YOLOv8 (Ultralytics)
- **Image Processing**: NumPy, OpenCV
- **Classification**: PyTorch/TensorFlow
- **API**: FastAPI
- **Database**: PostgreSQL (optional)

## 📈 Performance Metrics

| Metric | Target | Current |
|--------|--------|----------|
| mAP@0.5 | >85% | - |
| mAP@0.5:0.95 | >70% | - |
| Inference Speed | <100ms | - |
| Model Size | <500MB | - |

## 🔄 Data Pipeline

```
Raw Images → Validation → Preprocessing → Augmentation → Model Training → Evaluation → Deployment
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_detection.py -v
```

## 📚 Documentation

- [Architecture](docs/ARCHITECTURE.md) - System design and components
- [API Documentation](docs/API.md) - Endpoint specifications
- [Setup Guide](docs/SETUP.md) - Detailed setup instructions
- [Model Details](docs/MODELS.md) - Model specifications and training

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Commit changes: `git commit -am 'Add feature'`
3. Push to branch: `git push origin feature/your-feature`
4. Submit pull request

## 📋 Roadmap

- [ ] Phase 1: Data collection & annotation (Weeks 1-2)
- [ ] Phase 2: Baseline model training (Weeks 3-4)
- [ ] Phase 3: Model optimization & tuning (Weeks 5-6)
- [ ] Phase 4: API development & deployment (Weeks 7-8)
- [ ] Phase 5: Production monitoring & scaling (Weeks 9+)

## 🔐 Environment Variables

See `.env.example` for required environment variables:

```bash
cp .env.example .env
# Edit .env with your configuration
```

## 📦 Deployment

### Docker

```bash
# Build image
docker build -f docker/Dockerfile -t product-detector:latest .

# Run container
docker run -p 8000:8000 product-detector:latest
```

### Docker Compose

```bash
docker-compose -f docker/docker-compose.yml up
```

## 📊 Results & Benchmarks

Detailed results and benchmarks will be documented in [RESULTS.md](docs/RESULTS.md)

## 🐛 Troubleshooting

Common issues and solutions are documented in [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 👥 Contributors

- [@emansahn](https://github.com/emansahn) - Project Lead

## 📞 Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/emansahn/ecommerce-product-detection/issues)
- Discussions: [GitHub Discussions](https://github.com/emansahn/ecommerce-product-detection/discussions)

## 🎓 References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [NumPy Image Processing Guide](https://numpy.org/)
- [Computer Vision Best Practices](https://paperswithcode.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Last Updated**: June 2026  
**Status**: 🟡 In Active Development
