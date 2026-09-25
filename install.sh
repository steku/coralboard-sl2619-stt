#!/usr/bin/env bash
# ==============================================================================
# Wyoming Speech-to-Text Installer for Coralboard SL2619 (Astra SL2619 SoC)
# Target Environment: Linux aarch64, Python 3.12.9
# ==============================================================================

set -euo pipefail

echo "=========================================================="
echo "Installing Wyoming STT for Coralboard SL2619 (Torq NPU)"
echo "=========================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 1. Verify Python version
PYTHON_BIN="${PYTHON_BIN:-python3}"
PY_VER=$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
echo "[1/6] Detected Python version: ${PY_VER}"

if [[ ! "${PY_VER}" =~ ^3\.12 ]]; then
    echo "Warning: Expected Python 3.12.x, but found ${PY_VER}."
    echo "Continuing with ${PYTHON_BIN}..."
fi

# 2. Setup Virtual Environment
VENV_DIR="${SCRIPT_DIR}/.venv"
echo "[2/6] Creating Python 3.12 virtual environment at ${VENV_DIR}..."
${PYTHON_BIN} -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# 3. Upgrade pip and build tools
echo "[3/6] Upgrading pip and wheel..."
pip install --upgrade pip setuptools wheel

# 4. Install base Python dependencies
echo "[4/6] Installing Python dependencies from requirements.txt..."
pip install -r "${SCRIPT_DIR}/requirements.txt"

# 5. Install Torq NPU Runtime Wheel (Python 3.12 / aarch64)
TORQ_WHEEL_URL="https://github.com/synaptics-torq/torq-compiler/releases/download/v2.1.0/torq_runtime-2.1.0-cp312-cp312-manylinux_2_28_aarch64.whl"
echo "[5/6] Installing Synaptics Torq NPU Runtime wheel for SL2619..."

if pip install "${TORQ_WHEEL_URL}"; then
    echo "Successfully installed Torq Runtime for Coralboard SL2619 NPU."
else
    echo "Notice: Could not automatically download Torq runtime wheel from GitHub."
    echo "If offline, place the torq_runtime cp312 .whl file in this directory and install manually."
fi

# 6. Prepare models directory and download initial models
mkdir -p "${SCRIPT_DIR}/models"
echo "[6/6] Downloading speech models, tokenizers, and Torq NPU artifacts..."
python3 "${SCRIPT_DIR}/tools/download_models.py" --models-dir "${SCRIPT_DIR}/models" || {
    echo "Warning: Model download script encountered an error or network limitation."
    echo "You can re-run 'python3 tools/download_models.py' once connected to the internet."
}

echo "=========================================================="
echo "Installation completed successfully!"
echo ""
echo "To test the server manually:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python3 -m coral_stt --port 10300"
echo ""
echo "To install as a persistent systemd service:"
echo "  sudo cp wyoming-coral-stt.service /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable --now wyoming-coral-stt"
echo "=========================================================="
