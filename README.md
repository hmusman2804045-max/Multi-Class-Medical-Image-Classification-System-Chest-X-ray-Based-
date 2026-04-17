# Multi-Class Medical Image Classification System (Chest X-ray Based)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.15%2B-orange.svg)

## Overview
This repository contains a professional deep learning pipeline for multi-class classification of Chest X-rays. The system leverages a fine-tuned **ResNet50** architecture to classify images into four diagnostic categories: **COVID-19, Normal, Pneumonia, and Tuberculosis**.

## Key Features
- **Dynamic Fine-Tuning**: Configurable unfreezing of backbone layers to specialize the model for medical features.
- **Robust Data Pipeline**: Optimized TensorFlow data loading with advanced augmentation (Horizontal Flip, Rotation, Zoom, Brightness).
- **Overfitting Prevention**: Implemented strong regularization using **0.5 Dropout** and adaptive learning rates.
- **Visualization Tools**: Real-time evaluation script to compare **Actual vs. Predicted** labels on test images.
- **Class Imbalance Handling**: Automatic class weighting to ensure minority classes (e.g., Tuberculosis) are properly prioritized.

## Project Structure
- `train.py`: Main entry point for model training and fine-tuning.
- `data_pipeline.py`: Handles dataset loading, augmentation, and preprocessing.
- `model_builder.py`: Defines the ResNet50-based model architecture.
- `visualize_results.py`: Utility to display sample predictions and model performance visually.
- `smoke_test.py`: Lightweight script to verify the entire pipeline in under 60 seconds.
- `requirements.txt`: Project dependencies.

## Installation & Setup

### 1. Environment
We recommend using Python 3.12 within a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

### 2. Dataset Configuration
The system expects the following directory structure:
```
dataset/
└── train/
    ├── covid/
    ├── normal/
    ├── pneumonia/
    └── tuberculosis/
```

## Usage

### Stage 1: Verification
Run the smoke test to ensure your environment and data paths are correct:
```bash
.\venv\Scripts\python smoke_test.py
```

### Stage 2: Training & Fine-Tuning
Execute the main training script. The current configuration is optimized for **Stage 2 Fine-Tuning** (Unfrozen layers + 1e-5 Learning Rate):
```bash
.\venv\Scripts\python train.py
```

### Stage 3: Visualization
After training, evaluate the model's performance on random validation samples:
```bash
.\venv\Scripts\python visualize_results.py
```

## Performance Logs
- **Models**: Saved in `models/medical_resnet_v1.h5`.
- **Metrics**: Detailed per-epoch logs available in `logs/training_log.csv`.
- **Visuals**: Latest prediction samples saved in `logs/latest_visualization.png`.

## License
Distributed under the MIT License. See `LICENSE` for more information.
