#!/bin/bash
#SBATCH --job-name=diverse-128-eligible
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/diverse-128-eligible-%j.out
#SBATCH --error=logs/diverse-128-eligible-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
uv run --locked --no-sync python -m kl_anchors.prepare_diverse_prompts \
  --protocol configs/protocol.math10k-llama32-3b-diverse-kl-exploratory.json \
  --pool data/anchors/eligible_anchors.jsonl \
  --output-dir artifacts/math10k/full-eligible-128-v1
