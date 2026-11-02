#!/bin/bash
#SBATCH --job-name=anchor-cache-200
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

uv run --locked --no-sync python \
  -m kl_anchors.anchor_pipeline cache \
  --protocol configs/protocol.anchor_part1.json \
  --anchors data/processed/anchors/anchors_part1.jsonl \
  --output teacher_cache_part1.jsonl