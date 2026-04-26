import os
import io
import cv2
import numpy as np
from scipy.spatial.distance import cosine
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image
from huggingface_hub import hf_hub_download

app = Flask(__name__)

HF_MODEL_REPO = "usman-ai-dev/healthscan-model"
MODEL_FILENAME = "medical_resnet_v1.h5"
LOCAL_MODEL_PATH = "models/medical_resnet_v1.h5"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['covid', 'normal', 'pneumonia', 'tuberculosis']
LAST_CONV_LAYER = "conv5_block3_out"
UPLOAD_FOLDER = 'static/uploads'

os.makedirs("models", exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if not os.path.exists(LOCAL_MODEL_PATH):
    print(f"--- [BACKEND] Model not found locally. Downloading from HuggingFace Hub... ---")
    LOCAL_MODEL_PATH = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=MODEL_FILENAME,
        local_dir="models"
    )
    print(f"--- [BACKEND] Model downloaded to {LOCAL_MODEL_PATH} ---")

print(f"--- [BACKEND] Loading AI Brain from {LOCAL_MODEL_PATH} ---")
model = tf.keras.models.load_model(LOCAL_MODEL_PATH)

print("--- [BACKEND] Setting up Security Check (OOD Detection) ---")
try:
    golden_reference = np.load("models/xray_reference.npy")
    pooling_layer_name = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            pooling_layer_name = layer.name
            break
            
    if pooling_layer_name:
        feature_extractor = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=model.get_layer(pooling_layer_name).output
        )
        print("--- [BACKEND] Security Check Active. ---")
    else:
        raise ValueError("Could not find GlobalAveragePooling2D layer")
except Exception as e:
    print(f"--- [WARNING] Security Check setup failed: {e}. System will run without OOD detection. ---")
    golden_reference = None
    feature_extractor = None

print("--- [BACKEND] AI Brain is loaded. Running wake-up scan... ---")

dummy_input = np.zeros((1, 224, 224, 3))
_ = model.predict(dummy_input)

print("--- [BACKEND] AI Brain is fully awake and ready for scans. ---")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    base_model = model.layers[0]
    target_layer = base_model.get_layer(last_conv_layer_name)

    feature_extractor = tf.keras.models.Model(
        inputs=base_model.inputs,
        outputs=[target_layer.output, base_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_features = feature_extractor(img_array)
        tape.watch(conv_outputs)

        x = base_features
        for layer in model.layers[1:]:
            x = layer(x)

        if pred_index is None:
            pred_index = tf.argmax(x[0])
        class_channel = x[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def save_and_display_gradcam(img_bytes, heatmap, cam_path, alpha=0.65):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))

    superimposed_img = jet * alpha + img * (1 - alpha * 0.3)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')

    cv2.imwrite(cam_path, superimposed_img)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        img_bytes = file.read()
        processed_img = preprocess_image(img_bytes)

        # 1. SECURITY CHECK (Out-of-Distribution Detection)
        if golden_reference is not None and feature_extractor is not None:
            features = feature_extractor.predict(processed_img, verbose=0)
            similarity = 1 - cosine(features.flatten(), golden_reference)
            print(f"--- [SECURITY] Image Similarity Score: {similarity:.4f} ---")
            
            if similarity < 0.70: # 70% similarity threshold
                print("--- [SECURITY] Image REJECTED! Not a valid X-ray. ---")
                return jsonify({"error": "Security Check Failed: The uploaded image does not appear to be a valid Chest X-ray. Please upload a medical scan."}), 400

        # 2. INFERENCE
        predictions = model.predict(processed_img)
        pred_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_idx])

        heatmap = make_gradcam_heatmap(processed_img, model, LAST_CONV_LAYER, pred_index=pred_idx)

        heatmap_filename = "latest_heatmap.png"
        heatmap_path = os.path.join(UPLOAD_FOLDER, heatmap_filename)
        save_and_display_gradcam(img_bytes, heatmap, heatmap_path)

        result = {
            "diagnosis": CLASS_NAMES[pred_idx],
            "confidence": f"{confidence * 100:.2f}%",
            "heatmap_url": f"/static/uploads/{heatmap_filename}?t={os.urandom(8).hex()}",
            "all_scores": {CLASS_NAMES[i]: f"{float(predictions[0][i]) * 100:.2f}%" for i in range(len(CLASS_NAMES))}
        }

        print(f"--- [INFERENCE] Detected: {result['diagnosis']} ({result['confidence']}) ---")
        return jsonify(result)

    except Exception as e:
        print(f"--- [ERROR] Inference failed: {e} ---")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
