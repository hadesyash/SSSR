#!/bin/bash
##############################################################################
# SilentSpeechLLM — One-time HPC environment setup (Imperial College)
#
# Run this ONCE on a login node:
#   bash ~/aml_lab/hpc/setup_env.sh
#
# Prerequisites:
#   - miniforge3 installed at ~/miniforge3
#   - huggingface-cli login done (for gated LLaMA model)
##############################################################################

set -e  # Exit on any error

echo "============================================"
echo "SilentSpeechLLM HPC Environment Setup"
echo "============================================"

# ---- 1. Create conda environment ----
echo "[1/5] Creating conda environment 'emg_llm' (Python 3.11)..."
source "$HOME/miniforge3/etc/profile.d/conda.sh"

# Remove old env if it exists (clean start)
if conda env list | grep -q "emg_llm"; then
    echo "  Removing existing emg_llm environment..."
    conda deactivate 2>/dev/null || true
    conda env remove -n emg_llm -y
fi

conda create -n emg_llm python=3.11 -y
conda activate emg_llm

echo "  Python: $(which python)"
echo "  Version: $(python --version)"

# ---- 2. Install PyTorch with CUDA 12.1 ----
echo "[2/5] Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ---- 3. Install transformers + ML deps (matching repo versions) ----
echo "[3/5] Installing transformers, peft, and other deps..."
pip install "transformers>=4.44.0" "peft>=0.10.0" jiwer unidecode librosa soundfile huggingface_hub

# ---- 4. Verify installation ----
echo "[4/5] Verifying installation..."
python -c "
import sys
print(f'Python: {sys.version}')
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA compiled: {torch.version.cuda}')
import transformers
print(f'Transformers: {transformers.__version__}')
import peft
print(f'PEFT: {peft.__version__}')
import jiwer
print(f'jiwer: {jiwer.__version__}')
print('All imports OK')
"

# ---- 5. Create directories ----
echo "[5/5] Creating directories..."
PROJECT_DIR="/rds/general/user/yma25/home/aml_lab"
mkdir -p "$PROJECT_DIR/hpc/logs"
mkdir -p "$PROJECT_DIR/checkpoints"
mkdir -p "$PROJECT_DIR/models"

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Download the HC checkpoint from Google Drive:"
echo "     gdown 1xtq_UkSTn1rwMRt8KD_-Vz3chABIaJnR -O ~/aml_lab/models/full_checkpoint.pth"
echo "     (or manually upload it to ~/aml_lab/models/)"
echo ""
echo "  2. Extract the transNet weights (run once):"
echo "     python ~/aml_lab/hpc/extract_transnet.py"
echo ""
echo "  3. Run inference to verify WER:"
echo "     qsub ~/aml_lab/hpc/run_inference.pbs"
echo "============================================"
