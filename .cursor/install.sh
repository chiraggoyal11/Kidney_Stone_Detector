#!/usr/bin/env bash
# Idempotent setup for the Kidney Stone Detector notebook project.
set -euo pipefail

cd "$(dirname "$0")/.."

# System packages: venv support + Tk runtime (needed by tkinter/customtkinter).
if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y python3-venv python3-tk
fi

# Create the virtualenv once; reuse it on subsequent runs.
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

echo "Setup complete. Activate with: source .venv/bin/activate"
echo "The notebook uses the Keras 2 API; export TF_USE_LEGACY_KERAS=1 before running it."
