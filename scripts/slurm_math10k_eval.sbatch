#!/bin/bash
#SBATCH --job-name=math10k-eval-best
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
  -m kl_anchors.evaluate_math10k \
  --training-protocol configs/protocol.math10k-llama32-3b.json \
  --datasets-config configs/math10k_eval_datasets.json \
  --model best \
  --run-dir results/runs/math10k-llama32-3b-full-01 \
  --benchmarks gsm8k arc_challenge openbookqa social_iqa \
  --limit 10 \
  --output-dir results/eval/math10k-best-smoke-04
