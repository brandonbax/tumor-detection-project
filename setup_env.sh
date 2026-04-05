#!/usr/bin/env bash
# setup_env.sh — Create the conda environment and verify CUDA.
#
# Usage:
#   bash setup_env.sh            # full setup
#   bash setup_env.sh --check    # only check existing environment

set -euo pipefail

ENV_NAME="cv-cw"
ENV_FILE="environment.yml"
PYTHON_VER="3.11"

# ── Colours ──────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }

# ── Parse args ───────────────────────────────────
CHECK_ONLY=false
if [[ "${1:-}" == "--check" ]]; then
    CHECK_ONLY=true
fi

# ── Check conda is available ─────────────────────
if ! command -v conda &> /dev/null; then
    fail "conda not found. Install Miniconda or Anaconda first."
    exit 1
fi
ok "conda found: $(conda --version)"

# ── Create / update environment ──────────────────
if [[ "$CHECK_ONLY" == false ]]; then
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "Environment '${ENV_NAME}' exists. Updating..."
        conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
    else
        echo "Creating environment '${ENV_NAME}'..."
        conda env create -f "$ENV_FILE"
    fi
    ok "Environment '${ENV_NAME}' is ready"

    # Install the project in editable mode
    echo "Installing project in editable mode..."
    conda run -n "$ENV_NAME" --live-stream pip install -e . --quiet
    ok "Project installed"
fi

# ── Activate for checks ─────────────────────────
# Use conda run for the remaining checks so we don't need to
# source activate (which doesn't work reliably in scripts).
RUN="conda run -n $ENV_NAME --live-stream"

# ── Python version ───────────────────────────────
PY_VER=$($RUN python --version 2>&1)
echo ""
echo "=== Environment Check ==="
ok "Python: $PY_VER"

# ── PyTorch ──────────────────────────────────────
TORCH_VER=$($RUN python -c "import torch; print(torch.__version__)" 2>&1)
ok "PyTorch: $TORCH_VER"

# ── CUDA ─────────────────────────────────────────
echo ""
echo "=== CUDA Check ==="
if $RUN python check_cuda.py; then
    ok "CUDA check passed"
else
    fail "CUDA check failed"
    echo ""
    echo "  Fix: install the CUDA build of PyTorch:"
    echo "    conda install -n $ENV_NAME -c conda-forge pytorch=*=cuda*"
    exit 1
fi

# ── Key packages ─────────────────────────────────
echo ""
echo "=== Key Packages ==="
$RUN python -c "
packages = [
    'albumentations', 'monai', 'torchmetrics', 'rasterio',
    'geopandas', 'tifffile', 'tqdm', 'seaborn',
]
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'ok')
        print(f'  {pkg:.<25s} {ver}')
    except ImportError:
        print(f'  {pkg:.<25s} MISSING')
"

echo ""
ok "Setup complete. Activate with: conda activate $ENV_NAME"
