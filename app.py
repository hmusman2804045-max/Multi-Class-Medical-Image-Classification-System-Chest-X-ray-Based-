import os
import io
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image

app = Flask(__name__)

MODEL_PATH = "models/medical_resnet_v1.h5"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['covid', 'normal', 'pneumonia', 'tuberculosis']
LAST_CONV_LAYER = "conv5_block3_out"
UPLOAD_FOLDER = 'static/uploads'

print(f"--- [BACKEND] Loading AI Brain from {MODEL_PATH} ---")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run training first!")

model = tf.keras.models.load_model(MODEL_PATH)
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
    app.run(host="0.0.0.0", port=5000, debug=True)
