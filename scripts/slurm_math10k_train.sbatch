#!/bin/bash
#SBATCH --job-name=math10k-full
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/math10k-full-%j.out
#SBATCH --error=logs/math10k-full-%j.err

set -euo pipefail

cd /home/msai/srihari004/Damage-Aware-KL-Anchoring-for-SLMs

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi

uv run --locked --no-sync python -m kl_anchors.train_math10k \
  --protocol configs/protocol.math10k-llama32-3b.json \
  --experiment-id math10k-llama32-3b-full-01
