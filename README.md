# Kidney Stone Detector

Detects whether a kidney stone is present in a kidney CT scan by combining a
**Convolutional Neural Network (CNN)** and a **Support Vector Machine (SVM)**.
An image is classified as a stone if *either* model predicts a stone.

The project ships:

- `code.ipynb` — end-to-end notebook: data loading, CNN training, SVM training,
  evaluation, and two demo apps (a Tkinter desktop GUI and a Gradio web app).
- `train.py` — headless training script that produces the two model files.
- `app.py` — standalone Gradio web app for running predictions.

## Requirements

- Python 3.10+
- Dependencies in `requirements.txt` (TensorFlow/Keras, scikit-learn,
  scikit-image, OpenCV, Gradio, JupyterLab, …).

The notebook was written against the **Keras 2** API. On modern TensorFlow
(Keras 3 by default) this is handled by installing `tf-keras` and setting
`TF_USE_LEGACY_KERAS=1`. The notebook, `train.py`, and `app.py` set this
environment variable automatically.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

(On Cloud Agents this is done automatically by `.cursor/install.sh`.)

## Dataset

The dataset is **not** included in this repository. Use a kidney CT dataset with
`Normal` and `Stone` classes, for example the public
[CT KIDNEY DATASET (Normal / Cyst / Tumor / Stone) on Kaggle](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone),
keeping only the `Normal` and `Stone` categories.

Arrange it like this (default root is `data/CT_SCAN`, override with the
`KIDNEY_DATA_DIR` env var):

```
data/CT_SCAN/
├── Train/
│   ├── Normal/   *.jpg
│   └── Stone/    *.jpg
└── Test/
    ├── Normal/   *.jpg
    └── Stone/    *.jpg
```

## Train the models

Either run `code.ipynb` top to bottom, or use the script:

```bash
python train.py            # writes models/model.keras and models/svc.pkl
KIDNEY_EPOCHS=10 python train.py   # fewer epochs for a quick run
```

Trained models are written to `models/` by default (override with
`KIDNEY_MODEL_DIR`). The `data/` and `models/` directories are git-ignored.

## Run the web app

Once the models exist:

```bash
python app.py              # serves on http://localhost:7860
```

Upload a kidney CT image and click **Run Prediction** to get a diagnosis,
an estimated probability, and (for positive cases) illustrative guidance.

> Note: `estimate_stone_size()` returns an illustrative random value — the
> models classify stone presence/absence, they do not measure stone size.

## Configuration

| Env var | Default | Purpose |
| --- | --- | --- |
| `KIDNEY_DATA_DIR` | `data/CT_SCAN` | Dataset root (contains `Train/` and `Test/`). |
| `KIDNEY_MODEL_DIR` | `models` | Where trained models are read/written. |
| `KIDNEY_EPOCHS` | `50` | CNN training epochs (`train.py`). |
| `TF_USE_LEGACY_KERAS` | `1` | Use the Keras 2 API (set automatically). |

## Disclaimer

This is an educational project and **not** a medical device. Do not use it for
diagnosis or treatment decisions.
