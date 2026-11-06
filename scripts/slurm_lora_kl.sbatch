#!/bin/bash
#SBATCH --job-name=anchor-score-part2
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

uv run --locked --no-sync python \
  -m kl_anchors.anchor_pipeline score \
  --protocol configs/protocol.anchor_part2.json \
  --cache artifacts/math10k/part2/teacher_cache_part2.jsonl \
  --discovery-adapter results/runs/math10k-llama32-3b-full-01/checkpoints/step_000600 \
  --allow-final-checkpoint \
  --batch-size 1 \
  --output artifacts/math10k/part2/scores_part2.jsonl
