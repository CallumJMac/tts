#!/usr/bin/env bash
# Run ON the g5.xlarge instance after cloning the repo.
# Sets up the environment and launches Phase 3 inside tmux.
#
# Usage:
#   cd tts
#   bash scripts/setup_gpu.sh

set -euo pipefail

echo "============================================"
echo "Phase 3 GPU Setup"
echo "============================================"

# --- Verify GPU ---
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Is this a GPU instance with drivers?"
    exit 1
fi
echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# --- System deps ---
echo "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip ffmpeg > /dev/null

# --- Python venv ---
PYTHON=${PYTHON:-python3}
echo "Using: $($PYTHON --version)"
echo "Creating virtual environment..."
$PYTHON -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q

# --- Install PyTorch with CUDA ---
echo "Installing PyTorch (CUDA)..."
pip install --no-cache-dir -q \
    torch==2.10.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu126

# --- Install project dependencies ---
echo "Installing project dependencies..."
pip install --no-cache-dir -q -r requirements.txt

# --- flash-attn (optional, skip by default — compiling from source is slow/fragile) ---
if [[ "${INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
    echo "Installing flash-attn (this may take 15-30 min)..."
    pip install --no-cache-dir -q wheel setuptools
    pip install --no-cache-dir flash-attn --no-build-isolation || {
        echo "WARNING: flash-attn failed to install. Continuing without it."
    }
else
    echo "Skipping flash-attn (set INSTALL_FLASH_ATTN=1 to enable)."
fi

# --- Verify CUDA works ---
echo
echo "Verifying CUDA..."
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# --- Verify imports ---
echo
echo "Verifying project imports..."
python -c "from src.experiment.run_fewshot import main; print('run_fewshot: OK')"
python -c "from src.experiment.analyze import main; print('analyze: OK')"

# --- Data check ---
MANIFEST="data/libritts_r_aligned/manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
    echo
    echo "WARNING: $MANIFEST not found."
    echo "You need to copy or download the data before running."
    echo "  scp -r data/libritts_r_aligned/ ubuntu@<this-ip>:~/tts/data/"
fi

# --- Launch Phase 3 in tmux ---
echo
echo "============================================"
echo "Setup complete. Ready to launch Phase 3."
echo "============================================"
echo
echo "To run (in tmux so it survives SSH disconnect):"
echo
cat <<'CMD'
tmux new -s phase3

source .venv/bin/activate
python scripts/run_fewshot.py \
    --manifest data/libritts_r_aligned/manifest.json \
    --approaches single_baseline embed_avg concat_audio \
    --num-refs 1 2 3 5 \
    --strategies random longest \
    --seeds 42 123 456 \
    --held-out-targets 5 \
    --device cuda:0 --dtype float16 \
    --skip-speechbertscore

# Detach tmux: Ctrl-b then d
# Reattach later: tmux attach -t phase3
CMD
echo
echo "After it finishes, run analysis:"
echo "  python scripts/analyze_fewshot.py"
echo
echo "Then pull results to your laptop:"
echo "  scp ubuntu@<ip>:~/tts/outputs/fewshot/results.csv outputs/fewshot/results.csv"
