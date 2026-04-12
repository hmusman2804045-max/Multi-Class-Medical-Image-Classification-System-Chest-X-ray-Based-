import os
import tensorflow as tf
import numpy as np
from data_pipeline import create_data_pipelines
from model_builder import build_resnet_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

def main():
    DATA_DIR = "dataset/train"
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 1
    MODEL_SAVE_PATH = "models/medical_resnet_v1.h5"
    LOGS_PATH = "logs/training_log.csv"
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("--- Initializing Data Pipelines ---")
    train_ds, val_ds, class_names = create_data_pipelines(
        data_dir=DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    num_classes = len(class_names)
    print(f"Target Classes: {class_names}")

    class_counts = {
        'covid': 3616,
        'normal': 15275,
        'pneumonia': 4273,
        'tuberculosis': 700
    }
    
    total_samples = sum(class_counts.values())
    class_weight = {}
    
    for i, name in enumerate(class_names):
        count = class_counts.get(name.lower(), 1)
        weight = total_samples / (num_classes * count)
        class_weight[i] = weight
    
    print(f"Calculated Class Weights: {class_weight}")

    print("--- Building Model (ResNet50 Base Frozen) ---")
    model = build_resnet_model(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        num_classes=num_classes,
        fine_tune_layers=0
    )

    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        CSVLogger(LOGS_PATH)
    ]

    print(f"--- Starting Training on CPU (Epochs: {EPOCHS}) ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    print(f"\nTraining Complete. Best model saved to: {MODEL_SAVE_PATH}")
    return history

if __name__ == "__main__":
    main()
