import os
import glob
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image

# 1. Download/Load the model
print("Loading model from Hugging Face Hub...")
REPO_ID = "usman-ai-dev/healthscan-model"
FILENAME = "medical_resnet_v1.h5"

try:
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

# 2. Create the feature extractor sub-model
# We intercept at the Global Average Pooling layer before the Dense layers
# In standard ResNet50, this is often named 'avg_pool' or similar. 
# Let's dynamically find the GlobalAveragePooling2D layer.
pooling_layer_name = None
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
        pooling_layer_name = layer.name
        break

if not pooling_layer_name:
    print("Error: Could not find GlobalAveragePooling2D layer in the model.")
    exit(1)

print(f"Creating sub-model intercepting at layer: {pooling_layer_name}")
feature_extractor = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=model.get_layer(pooling_layer_name).output
)

# 3. Collect 50 random valid X-rays from the dataset
# We take a mix from different classes to create a robust average X-ray signature
train_dir = 'dataset/train'
image_paths = []
for class_folder in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_folder)
    if os.path.isdir(class_path):
        # Get all jpg/png files
        files = glob.glob(os.path.join(class_path, '*.*'))
        # Grab up to 15 images from each class
        image_paths.extend(files[:15])

if not image_paths:
    print("No images found in dataset/train! Cannot generate reference.")
    exit(1)

print(f"Found {len(image_paths)} X-ray images for reference generation.")

# 4. Extract features
xray_features = []
for path in image_paths:
    try:
        # Preprocess exactly how app.py does it
        img = Image.open(path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Extract the 2048-dimensional vector
        features = feature_extractor.predict(img_array, verbose=0)
        xray_features.append(features.flatten())
    except Exception as e:
        print(f"Skipping {path} due to error: {e}")

if not xray_features:
    print("Failed to extract features from any images.")
    exit(1)

# 5. Calculate Golden Reference (Mean)
golden_reference = np.mean(xray_features, axis=0)

# Save it to the models directory
os.makedirs('models', exist_ok=True)
save_path = os.path.join('models', 'xray_reference.npy')
np.save(save_path, golden_reference)

print(f"Success! Golden Reference Vector saved to: {save_path}")
print(f"Vector shape: {golden_reference.shape}")
