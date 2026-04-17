import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from data_pipeline import create_data_pipelines

"""
VISUALIZATION TOOL
This script loads the trained model and displays random images from the dataset
along with the model's predictions compared to the actual labels.
"""

def visualize_model_performance(model_path="models/medical_resnet_v1.h5", data_dir="dataset/train"):
    # Configurations
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 9  # We will visualize 9 images (3x3 grid)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}. Please train the model first.")
        return

    print(f"--- Loading Model: {model_path} ---")
    model = tf.keras.models.load_model(model_path)
    
    print("--- Preparing Data for Visualization ---")
    # We use create_data_pipelines but only take a small sample
    _, val_ds, class_names = create_data_pipelines(
        data_dir=data_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Get one batch of images and labels
    for images, labels in val_ds.take(1):
        # Make predictions
        predictions = model.predict(images)
        
        plt.figure(figsize=(15, 15))
        plt.suptitle("Medical Image Classification: Actual vs. Predicted", fontsize=20)
        
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            
            # Show the image
            # Images are normalized (0-1), so we don't need to rescale
            plt.imshow(images[i])
            
            # Get actual and predicted labels
            actual_idx = np.argmax(labels[i])
            pred_idx = np.argmax(predictions[i])
            confidence = predictions[i][pred_idx] * 100
            
            actual_label = class_names[actual_idx]
            pred_label = class_names[pred_idx]
            
            # Set title color based on correctness
            title_color = 'green' if actual_idx == pred_idx else 'red'
            
            plt.title(
                f"Actual: {actual_label}\nPred: {pred_label} ({confidence:.1f}%)",
                color=title_color,
                fontsize=12
            )
            plt.axis('off')
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save the visualization for review
        save_path = "logs/latest_visualization.png"
        os.makedirs("logs", exist_ok=True)
        plt.savefig(save_path)
        print(f"\nSUCCESS: Visualization saved to {save_path}")
        
        # Show the plot
        plt.show()
        break

if __name__ == "__main__":
    visualize_model_performance()
