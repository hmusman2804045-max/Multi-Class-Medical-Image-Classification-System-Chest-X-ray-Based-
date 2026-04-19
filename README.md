# Multi-Class Medical Image Classification System (Chest X-ray Based)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![Accuracy](https://img.shields.io/badge/accuracy-80.57%25-green.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.15%2B-orange.svg)

## Overview
This repository contains a professional deep learning pipeline for multi-class classification of Chest X-rays. The system leverages a fine-tuned **ResNet50** architecture to classify images into four diagnostic categories: **COVID-19, Normal, Pneumonia, and Tuberculosis**.

## Project Status: COMPLETED ✅
The system has been successfully optimized through three stages of development:
- **Phase 1**: Environment setup and baseline ResNet50.
- **Phase 2**: Initial fine-tuning (20 layers). Faced overfitting but reached 77% peak.
- **Phase 3 (Final)**: Refinement stage with 10 unfrozen layers and aggressive regularization (0.5 Dropout). Reached a stable **80.57%** validation accuracy.

## Key Features
- **Keep-Alive Protection**: Includes a wrapper script to prevent Windows OS from sleeping during long training runs.
- **Resume-from-Best**: Intelligent loading logic that automatically resumes training from the highest accuracy checkpoint.
- **Robust Regularization**: 0.5 Dropout and 10-layer refinement to ensure high generalization and prevent overfitting.
- **Adaptive Learning**: Uses `ReduceLROnPlateau` for precision tuning in final epochs.

## Project Structure
- `train.py`: Main entry point with "Resume & Refine" logic.
- `keep_awake_train.py`: Windows Sleep prevention wrapper for `train.py`.
- `data_pipeline.py`: Handles dataset loading, augmentation, and preprocessing.
- `model_builder.py`: Defines the ResNet50-based model architecture.
- `visualize_results.py`: Utility to display sample predictions visually.
- `logs/training_log.csv`: Full history of the 15-epoch refinement run.

## Installation & Setup
1. **Environment**: Recommended Python 3.12.
2. **Setup**:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate
   pip install -r requirements.txt
   ```

## Usage
To start the refinement process without the computer sleeping:
```bash
.\venv\Scripts\python keep_awake_train.py
```

To visualize results after training:
```bash
.\venv\Scripts\python visualize_results.py
```

## Results Summary
- **Accuracy**: 80.57%
- **Loss (Val)**: 0.47 (Stable)
- **Top Metrics**: Consistently high AUC (>0.95) across all medical classes.

## License
Distributed under the MIT License. See `LICENSE` for more information.
