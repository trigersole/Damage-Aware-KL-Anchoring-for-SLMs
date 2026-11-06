#!/bin/bash
#SBATCH --job-name=anchor-cache-part2
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

PROTOCOL="${PROTOCOL:-$PWD/configs/protocol.anchor_part2.json}"
ANCHORS="${ANCHORS:-$PWD/data/processed/anchors/anchors_part2.jsonl}"
OUTPUT="${OUTPUT:-$PWD/artifacts/math10k/part2/teacher_cache_part2.jsonl}"

mkdir -p "$(dirname "$OUTPUT")"

args=(
  --protocol "$PROTOCOL"
  --anchors "$ANCHORS"
  --output "$OUTPUT"
)
if [[ -n "${ANCHOR_LIMIT:-}" ]]; then
  args+=(--limit "$ANCHOR_LIMIT")
fi

uv run --locked --no-sync python \
  -m kl_anchors.anchor_pipeline cache \
  "${args[@]}"
