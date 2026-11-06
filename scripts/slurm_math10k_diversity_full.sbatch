#!/bin/bash
#SBATCH --job-name=math10k-div-full
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/math10k-diversity-full-%j.out
#SBATCH --error=logs/math10k-diversity-full-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

POOL="$PWD/data/anchors/eligible_anchors.jsonl"
RANKS="$PWD/artifacts/math10k/full/diversity_ranks_full.jsonl"
SELECTED="$PWD/artifacts/math10k/full/diverse_prompts_top128.jsonl"
EMBED_MODEL_ID="sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL_REVISION="1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

for output in "$RANKS" "$SELECTED"; do
  if [[ -e "$output" ]]; then
    echo "refusing to overwrite existing output: $output" >&2
    exit 2
  fi
done

nvidia-smi

uv run --locked --no-sync python -m kl_anchors.diversity \
  --pool "$POOL" --output "$RANKS" \
  --model-id "$EMBED_MODEL_ID" --model-revision "$EMBED_MODEL_REVISION" \
  --algorithm farthest_first --seed 42 --device cuda:0 \
  --top-k 128 --selected-output "$SELECTED"
