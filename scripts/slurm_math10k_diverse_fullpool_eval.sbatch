#!/bin/bash
#SBATCH --job-name=div-fullpool-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/math10k-diverse-fullpool-eval-%j.out
#SBATCH --error=logs/math10k-diverse-fullpool-eval-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR="$PWD/results/runs/math10k-llama32-3b-diverse-kl-fullpool-02"
SMOKE="$PWD/results/eval/math10k-diverse-fullpool-02-best-all-smoke-01"
FULL="$PWD/results/eval/math10k-diverse-fullpool-02-best-all-full-01"
for output in "$SMOKE" "$FULL"; do
  if [[ -e "$output" ]]; then
    echo "refusing to overwrite existing evaluation: $output" >&2
    exit 2
  fi
done

common=(
  --training-protocol "$RUN_DIR/protocol.json"
  --datasets-config "$PWD/configs/math10k_eval_datasets.json"
  --model best --run-dir "$RUN_DIR"
  --benchmarks gsm8k arc_challenge arc_easy openbookqa social_iqa
)
nvidia-smi
uv run --locked --no-sync python -m kl_anchors.evaluate_math10k \
  "${common[@]}" --limit 10 --output-dir "$SMOKE"
uv run --locked --no-sync python -m kl_anchors.evaluate_math10k \
  "${common[@]}" --output-dir "$FULL"
