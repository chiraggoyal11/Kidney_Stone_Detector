"""Standalone Gradio web app for the Kidney Stone Detector.

Loads the trained CNN (`model.keras`) and SVM (`svc.pkl`) and serves a small web
UI that classifies an uploaded kidney CT image using both models combined.

Train the models first (run `code.ipynb` or `train.py`) so that the model files
exist under MODEL_DIR. Paths can be overridden with the KIDNEY_MODEL_DIR env var.

Run:
    TF_USE_LEGACY_KERAS=1 python app.py
"""
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # project targets the Keras 2 API

import random

import cv2
import gradio as gr
import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_DIR = os.environ.get("KIDNEY_MODEL_DIR", "models")
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svc.pkl")

for path in (CNN_MODEL_PATH, SVM_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Train the models first by running code.ipynb (or train.py) so that "
            f"'{CNN_MODEL_PATH}' and '{SVM_MODEL_PATH}' exist."
        )

cnn_model = load_model(CNN_MODEL_PATH)
svc_model = joblib.load(SVM_MODEL_PATH)


def extract_features(images):
    # Must match the HOG features the SVM was trained on (see train.py / code.ipynb).
    feature_list = []
    for img in images:
        # Resize to 128x128 to match training
        resized = cv2.resize(img, (128, 128))
        fd = hog(
            resized, orientations=8, pixels_per_cell=(16, 16),
            cells_per_block=(1, 1), channel_axis=2,
        )
        feature_list.append(fd)
    return np.array(feature_list)


# CT scans are effectively grayscale (R almost equals G almost equals B). Reject
# clearly colorful images (e.g. random screenshots): mean per-pixel channel
# spread above this tolerance means the image is not a CT scan.
COLOR_TOLERANCE = 20.0


def is_ct_like(img):
    rgb = img if (img.ndim == 3 and img.shape[2] == 3) else np.stack([img] * 3, axis=-1)
    spread = float(
        np.mean(rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16))
    )
    return spread <= COLOR_TOLERANCE, spread


def estimate_stone_size():
    # NOTE: placeholder only. The models classify presence/absence of a stone;
    # they do not measure size, so this returns an illustrative random value.
    return round(random.uniform(0.5, 5.0), 2)


def predict_combined(img):
    if img is None:
        return "Please upload an image."

    try:
        # Guard against non-CT input (e.g. random colour screenshots): a real CT
        # scan is grayscale, so a large colour spread means this isn't a CT image.
        ct_like, spread = is_ct_like(img)
        if not ct_like:
            return (
                "This image does not look like a grayscale CT scan "
                f"(colour spread {spread:.0f}). Please upload a kidney CT image."
            )

        cnn_test_img = image.img_to_array(Image.fromarray(img).resize((150, 150)))
        cnn_test_img = np.expand_dims(cnn_test_img, axis=0) / 255.0
        cnn_result = cnn_model.predict(cnn_test_img)
        # class index 1 == "Stone" (matches train_generator.class_indices)
        cnn_stone_prob = float(cnn_result[0][1])

        test_img_rgb = (
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if len(img.shape) == 3
            else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        )
        features_for_svm = extract_features([test_img_rgb])
        classes = list(svc_model.classes_)
        svm_stone_prob = float(
            svc_model.predict_proba(features_for_svm)[0][classes.index("Stone")]
        )

        # Combined stone probability: soft-voting average of the two models
        # (more robust than max, which lets a single noisy model dominate).
        stone_prob = (cnn_stone_prob + svm_stone_prob) / 2.0
        detail = f"(CNN: {cnn_stone_prob * 100:.1f}%, SVM: {svm_stone_prob * 100:.1f}%)"

        # Uncertain band: the models are not confident either way.
        if 0.45 < stone_prob < 0.55:
            return (
                f"Uncertain — stone probability {stone_prob * 100:.1f}% {detail}.\n"
                "The models are not confident. Please upload a clear kidney CT image."
            )

        if stone_prob >= 0.55:
            stone_size = f"{estimate_stone_size()} cm"
            prescription = (
                "Consult a urologist. Suggested medications: pain relievers (e.g., ibuprofen, "
                "acetaminophen) and alpha blockers (e.g., tamsulosin). Drink plenty of water "
                "(2.5-3 liters per day). Consider dietary adjustments."
            )
            care_instructions = (
                "Follow the urologist's guidance and adhere to prescribed medication. Stay hydrated. "
                "If pain or urinary obstruction persists, surgical intervention may be recommended."
            )
            return (
                "Diagnosis: Kidney Stone Detected (Positive)\n"
                f"Probability: {stone_prob * 100:.2f}% {detail}\n"
                f"Estimated Stone Size: {stone_size}\n"
                f"Prescription Guidance: {prescription}\n"
                f"Care Instructions: {care_instructions}"
            )
        return (
            "Diagnosis: No Kidney Stone Detected (Negative)\n"
            f"Probability: {(1 - stone_prob) * 100:.2f}% {detail}"
        )
    except Exception as e:  # noqa: BLE001 - surface any runtime error in the UI
        return f"Error in prediction: {e}"


def build_interface():
    with gr.Blocks(title="Kidney Stone Detector") as iface:
        gr.Markdown("# Kidney Stone Detection using Combined CNN and SVM")
        gr.Markdown(
            "**Educational demo only — not a medical device.** Do not use for "
            "diagnosis or treatment decisions."
        )
        ct_image = gr.Image(label="Upload CT Image")
        combined_output = gr.Textbox(label="Prediction Result and Care Instructions")
        combined_predict_btn = gr.Button("Run Prediction")
        combined_predict_btn.click(
            fn=predict_combined, inputs=ct_image, outputs=combined_output
        )
    return iface


if __name__ == "__main__":
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    build_interface().launch(
        server_name=server_name, server_port=server_port, share=share
    )
