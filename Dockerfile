# Container image for the Kidney Stone Detector Gradio web app.
#
# The app needs trained models at /app/models (model.keras + svc.pkl). Train them
# first (python train.py) so the models/ directory exists in the build context,
# or mount them at runtime with `-v $(pwd)/models:/app/models`.
FROM python:3.12-slim

ENV TF_USE_LEGACY_KERAS=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

# System libs required by OpenCV (headless) at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py train.py ./
# Trained models (must exist in the build context; see note above).
COPY models/ ./models/

EXPOSE 7860
CMD ["python", "app.py"]
