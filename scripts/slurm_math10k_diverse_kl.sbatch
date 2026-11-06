#!/bin/bash
#SBATCH --job-name=math10k-diverse-kl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/math10k-diverse-kl-%j.out
#SBATCH --error=logs/math10k-diverse-kl-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAGE="${STAGE:?set STAGE to rank or train}"
PROTOCOL="${PROTOCOL:-$PWD/configs/protocol.math10k-llama32-3b-diverse-kl-exploratory.json}"
POOL="$PWD/data/processed/anchors/anchors_part2.jsonl"
CACHE="$PWD/artifacts/math10k/part2/teacher_cache_part2.jsonl"
RANKS="$PWD/artifacts/math10k/part2/diversity_ranks_part2.jsonl"
EXPERIMENT_ID="${EXPERIMENT_ID:-math10k-llama32-3b-diverse-kl-exp01}"
EMBED_MODEL_ID="sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL_REVISION="1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

nvidia-smi

case "$STAGE" in
  rank)
    uv run --locked --no-sync python -m kl_anchors.diversity \
      --pool "$POOL" --teacher-cache "$CACHE" --output "$RANKS" \
      --model-id "$EMBED_MODEL_ID" --model-revision "$EMBED_MODEL_REVISION" \
      --algorithm farthest_first --seed 42 --device cuda:0
    ;;
  train)
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      uv run --locked --no-sync python -m kl_anchors.train_math10k \
      --protocol "$PROTOCOL" --mode diverse_kl \
      --experiment-id "$EXPERIMENT_ID"
    ;;
  *)
    echo "unknown STAGE=$STAGE; use rank or train" >&2
    exit 2
    ;;
esac
