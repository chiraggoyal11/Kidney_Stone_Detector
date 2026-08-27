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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = os.environ.get("KIDNEY_DATA_DIR", "data/CT_SCAN")
MODEL_DIR = os.environ.get("KIDNEY_MODEL_DIR", "models")
TRAIN_DIR = os.path.join(DATA_DIR, "Train")
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svc.pkl")
EPOCHS = int(os.environ.get("KIDNEY_EPOCHS", "50"))
BATCH_SIZE = 15


def build_cnn():
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
    model.add(Dense(2))
    model.add(Activation("sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


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
    # Standardize the HOG features before the RBF SVM (gamma="scale" is the
    # modern default and works well with standardized inputs).
    svc = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1, gamma="scale"))
    svc.fit(X_train, y_train)
    print("SVM validation accuracy:", accuracy_score(y_valid, svc.predict(X_valid)))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(svc, SVM_MODEL_PATH)
    print(f"Saved SVM -> {SVM_MODEL_PATH}")


if __name__ == "__main__":
    print(f"Training from dataset at: {TRAIN_DIR}")
    train_cnn()
    train_svm()
    print("Done.")
