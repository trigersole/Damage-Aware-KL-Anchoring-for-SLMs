#!/bin/bash
#SBATCH --job-name=math10k-div-restart
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/math10k-diverse-restart-%j.out
#SBATCH --error=logs/math10k-diverse-restart-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
nvidia-smi
uv run --locked --no-sync python -m kl_anchors.train_math10k \
  --protocol configs/protocol.math10k-llama32-3b-diverse-kl-fullpool-restart.json \
  --mode diverse_kl --experiment-id math10k-llama32-3b-diverse-kl-fullpool-02
