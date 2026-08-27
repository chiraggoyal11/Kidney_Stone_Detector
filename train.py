"""Train and save the CNN and SVM models for the Kidney Stone Detector.

This mirrors the training pipeline in `code.ipynb` as a headless script so the
models can be produced without opening the notebook.

Expected dataset layout (override the root with the KIDNEY_DATA_DIR env var):
    data/CT_SCAN/Train/Normal/*.jpg
    data/CT_SCAN/Train/Stone/*.jpg
    data/CT_SCAN/Test/Normal/*.jpg   (optional, for evaluation)
    data/CT_SCAN/Test/Stone/*.jpg

Run:
    TF_USE_LEGACY_KERAS=1 python train.py
"""
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # project targets the Keras 2 API

import cv2
import joblib
import numpy as np
from skimage.feature import hog
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten,
    GlobalAveragePooling2D, Input, MaxPooling2D, Rescaling,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = os.environ.get("KIDNEY_DATA_DIR", "data/CT_SCAN")
MODEL_DIR = os.environ.get("KIDNEY_MODEL_DIR", "models")
TRAIN_DIR = os.path.join(DATA_DIR, "Train")
TEST_DIR = os.path.join(DATA_DIR, "Test")
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svc.pkl")
EPOCHS = int(os.environ.get("KIDNEY_EPOCHS", "50"))
BATCH_SIZE = 15
# "transfer" = MobileNetV2 transfer learning (default, stronger); "simple" = the
# small from-scratch CNN.
MODEL_ARCH = os.environ.get("KIDNEY_MODEL_ARCH", "transfer").lower()
INPUT_SIZE = (150, 150)


def build_simple_cnn():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def build_transfer_cnn():
    # MobileNetV2 pretrained on ImageNet as a frozen feature extractor + a small
    # classification head. Inputs arrive rescaled to [0, 1] (matching the app and
    # the ImageDataGenerator rescale); the Rescaling layer maps them to [-1, 1],
    # which is what MobileNetV2 expects. Keeping preprocessing inside the model
    # means the app needs no special handling.
    inputs = Input(shape=(*INPUT_SIZE, 3))
    x = Rescaling(scale=2.0, offset=-1.0)(inputs)
    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(*INPUT_SIZE, 3))
    base.trainable = False
    x = base(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(2, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def build_cnn():
    if MODEL_ARCH == "transfer":
        try:
            print("Building MobileNetV2 transfer-learning model ...")
            return build_transfer_cnn()
        except Exception as e:  # e.g. no network for pretrained weights
            print(f"[warn] transfer model unavailable ({e}); using simple CNN")
    return build_simple_cnn()


def evaluate_cnn(model):
    if not os.path.isdir(TEST_DIR):
        return
    gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        TEST_DIR, target_size=INPUT_SIZE, class_mode="categorical",
        batch_size=BATCH_SIZE, shuffle=False,
    )
    y_pred = model.predict(gen).argmax(axis=1)
    names = list(gen.class_indices.keys())
    print("CNN test-set metrics:")
    print(classification_report(gen.classes, y_pred, target_names=names, zero_division=0))


def train_cnn():
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.2,
    )
    train_gen = datagen.flow_from_directory(
        TRAIN_DIR, target_size=(150, 150), class_mode="categorical",
        batch_size=BATCH_SIZE, subset="training",
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2).flow_from_directory(
        TRAIN_DIR, target_size=(150, 150), class_mode="categorical",
        batch_size=BATCH_SIZE, subset="validation",
    )
    print("class_indices:", train_gen.class_indices)

    callbacks = [
        EarlyStopping(patience=10),
        ReduceLROnPlateau(monitor="val_accuracy", patience=2, verbose=1, factor=0.5, min_lr=1e-5),
    ]
    model = build_cnn()
    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        steps_per_epoch=max(1, train_gen.samples // BATCH_SIZE),
        validation_steps=max(1, val_gen.samples // BATCH_SIZE),
        callbacks=callbacks,
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(CNN_MODEL_PATH)
    print(f"Saved CNN -> {CNN_MODEL_PATH}")
    evaluate_cnn(model)


def read_images(path):
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            images.append(img)
    return images


def extract_features(images):
    # Resize each image to a fixed size so the HOG descriptor has a consistent
    # length across images, then extract HOG features.
    feats = []
    for img in images:
        resized = cv2.resize(img, (128, 128))
        fd = hog(
            resized, orientations=8, pixels_per_cell=(16, 16),
            cells_per_block=(1, 1), channel_axis=2,
        )
        feats.append(fd)
    return feats


def train_svm():
    normal = read_images(os.path.join(TRAIN_DIR, "Normal"))
    stone = read_images(os.path.join(TRAIN_DIR, "Stone"))
    features = extract_features(normal) + extract_features(stone)
    labels = ["Normal"] * len(normal) + ["Stone"] * len(stone)

    X_train, X_valid, y_train, y_valid = train_test_split(
        features, labels, test_size=0.2, random_state=0
    )
    # Standardize the HOG features, then an RBF SVM wrapped in
    # CalibratedClassifierCV so predict_proba returns well-calibrated
    # probabilities (the modern replacement for SVC(probability=True)).
    svc = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(SVC(kernel="rbf", C=1, gamma="scale"), cv=3),
    )
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_valid)
    print("SVM validation accuracy:", accuracy_score(y_valid, y_pred))
    print("SVM validation metrics:")
    print(classification_report(y_valid, y_pred, zero_division=0))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(svc, SVM_MODEL_PATH)
    print(f"Saved SVM -> {SVM_MODEL_PATH}")


if __name__ == "__main__":
    print(f"Training from dataset at: {TRAIN_DIR}")
    train_cnn()
    train_svm()
    print("Done.")
