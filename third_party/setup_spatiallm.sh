#!/bin/bash
# Set up the SpatialLM1.1 (Sonata path) environment at ~/spatiallm_env for the
# cloud_slam_icp pipeline. All no-sudo, Python 3.10 venv, WSL-native.
#
# Replays the steps from the user's 2026-03-31 working setup_wsl.sh,
# minus the 3.12 apt install and torchsparse/libsparsehash (not needed for SpatialLM1.1).

set -e

VENV_DIR="${VENV_DIR:-$HOME/spatiallm_env}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")" && pwd)/SpatialLM}"

echo "=== SpatialLM env setup ==="
echo "venv : $VENV_DIR"
echo "repo : $REPO_DIR"

if [[ ! -d "$REPO_DIR" ]]; then
    echo "[FATAL] SpatialLM repo not found at $REPO_DIR"
    exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "=== creating venv (virtualenv bundles pip; python3-venv apt not needed) ==="
    if ! command -v virtualenv >/dev/null 2>&1 && ! python3 -m virtualenv --version >/dev/null 2>&1; then
        pip install --user --quiet virtualenv
    fi
    PATH="$HOME/.local/bin:$PATH" virtualenv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python --version
pip --version

echo "=== poetry install (pulls torch 2.4.1+cu124 and SpatialLM deps) ==="
pip install --quiet poetry
cd "$REPO_DIR"
poetry config virtualenvs.create false --local
poetry install --no-interaction

echo "=== Sonata encoder deps (flash-attn compile is the slow one, ~15 min) ==="
pip install --quiet ninja psutil timm
pip install flash-attn --no-build-isolation
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install spconv-cu120

echo "=== verify ==="
python - <<'PY'
import torch, transformers
print(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}, dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")
print(f"transformers {transformers.__version__}")
try:
    import spatiallm
    print(f"spatiallm import: OK")
except Exception as e:
    print(f"spatiallm import FAILED: {e}")
PY

echo "=== DONE ==="
