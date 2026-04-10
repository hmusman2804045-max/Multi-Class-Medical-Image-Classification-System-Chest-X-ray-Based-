# Multi-Class Medical Image Classification System (Chest X-ray Based)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.10%2B-orange.svg)

## Overview
This repository contains a professional deep learning pipeline for multi-class classification of medical images, specifically focused on Chest X-rays. The system utilizes a fine-tuned ResNet50 architecture to classify X-rays into multiple diagnostic categories.

## Key Features
- **Robust Data Pipeline**: Optimized TensorFlow data loading with built-in augmentation (rotation, zoom, translation, brightness) and automated normalization.
- **Architectural Excellence**: Transfer learning implementation using ResNet50 with configurable fine-tuning depth and Global Average Pooling.
- **Performance Optimized**: Integration of `tf.data.AUTOTUNE` and prefetching for maximum GPU utilization.
- **Evaluation Metrics**: Tracks Accuracy and AUC (Area Under Curve) for balanced diagnostic assessment.

## Project Structure
- `data_pipeline.py`: Handles dataset loading, augmentation, and preprocessing.
- `model_builder.py`: defines the ResNet50-based model architecture and compilation logic.
- `LICENSE`: MIT License information.

## Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Dependencies
Install the required packages using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Dataset Configuration
While the full dataset is not included in this repository due to size constraints, the system expects a directory structure as follows:
```
dataset/
└── train/
    ├── Normal/
    ├── Viral Pneumonia/
    ├── COVID-19/
    └── ... (other classes)
```
The pipeline automatically detects class names from the folder structure.

## Usage
To test the data pipeline:
```bash
python data_pipeline.py
```
To visualize the model architecture:
```bash
python model_builder.py
```

## License
Distributed under the MIT License. See `LICENSE` for more information.
