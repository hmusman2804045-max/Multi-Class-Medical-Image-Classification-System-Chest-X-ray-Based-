import os
import tensorflow as tf
from data_pipeline import create_data_pipelines
from model_builder import build_resnet_model

"""
SMOKE TEST SCRIPT
This script verifies the end-to-end ML pipeline with a minimal run.
It ensures that data loading, model building, and basic training logic are correct.
"""

def run_smoke_test():
    # Configuration for smoke test
    DATA_DIR = "dataset/train"
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 8  # Smaller batch for faster loading
    EPOCHS = 1
    SMOKE_MODEL_PATH = "models/smoke_test.h5"
    
    os.makedirs("models", exist_ok=True)

    print("\n[SMOKE TEST] Step 1: Initializing Data Pipelines...")
    # We use a 0.2 split just like in the real training
    train_ds, val_ds, class_names = create_data_pipelines(
        data_dir=DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    
    print(f"[SMOKE TEST] Classes detected: {class_names}")

    print("\n[SMOKE TEST] Step 2: Building Model...")
    model = build_resnet_model(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        num_classes=len(class_names)
    )

    print("\n[SMOKE TEST] Step 3: Executing Micro-Training Run (2 steps)...")
    try:
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            steps_per_epoch=2,    # LIMIT training steps
            validation_steps=1,   # LIMIT validation steps
            verbose=1
        )
        print("\n[SMOKE TEST] Step 4: Verifying Model Saving...")
        model.save(SMOKE_MODEL_PATH)
        
        if os.path.exists(SMOKE_MODEL_PATH):
            print(f"SUCCESS: Smoke test model saved to {SMOKE_MODEL_PATH}")
        else:
            print("FAILURE: Model file was not created.")
            
        print("\n==================================================")
        print("          SMOKE TEST COMPLETED SUCCESSFULLYFUL          ")
        print("==================================================\n")
        
    except Exception as e:
        print(f"\nERROR during smoke test: {str(e)}")
        raise e

if __name__ == "__main__":
    run_smoke_test()
