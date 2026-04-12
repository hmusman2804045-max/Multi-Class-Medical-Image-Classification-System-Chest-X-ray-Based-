# Multi-Class Medical Image Classification System (Chest X-ray Based)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.10%2B-orange.svg)

## Overview
This repository contains a professional deep learning pipeline for multi-class classification of medical images, specifically focused on Chest X-rays. The system utilizes a fine-tuned ResNet50 architecture to classify X-rays into diagnostic categories: **COVID-19, Normal, Pneumonia, and Tuberculosis**.

## Key Features
- **Robust Data Pipeline**: Optimized TensorFlow data loading with built-in augmentation (rotation, zoom, translation, brightness) and automated normalization (+ [Rescaling](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Rescaling)).
- **Architectural Excellence**: Transfer learning implementation using **ResNet50** with configurable fine-tuning depth and Global Average Pooling.
- **Class Imbalance Handling**: Automated calculation of class weights to ensure minority classes like Tuberculosis are properly weighted during training.
- **Advanced Callbacks**: Integrated `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`, and `CSVLogger` for robust training management.
- **Performance Optimized**: Integration of `tf.data.AUTOTUNE` and prefetching for maximum CPU/GPU utilization.
- **Evaluation Metrics**: Tracks Accuracy and AUC (Area Under Curve) for balanced diagnostic assessment.

## Project Structure
- `train.py`: The main entry point for executing model training, managing callbacks, and class weights.
- `data_pipeline.py`: Handles dataset loading, augmentation, and preprocessing.
- `model_builder.py`: Defines the ResNet50-based model architecture and compilation logic.
- `requirements.txt`: List of dependencies required to run the project.
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
    ├── covid/
    ├── normal/
    ├── pneumonia/
    └── tuberculosis/
```
The pipeline automatically detects class names and counts from the folder structure.

## Usage

### To begin training:
```bash
python train.py
```
This will:
1. Initialize the data pipelines.
2. Build the ResNet50 model with a frozen base.
3. Calculate class weights based on the dataset distribution.
4. Execute training with performance-tracking callbacks.

### To test the data pipeline:
```bash
python data_pipeline.py
```

### To visualize the model architecture:
```bash
python model_builder.py
```

## Outputs
- **Models**: Saved in the `models/` directory (e.g., `medical_resnet_v1.h5`).
- **Logs**: Training metrics are recorded in `logs/training_log.csv`.

## License
Distributed under the MIT License. See `LICENSE` for more information.
