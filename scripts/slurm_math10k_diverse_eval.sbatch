#!/bin/bash
#SBATCH --job-name=math10k-div-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/math10k-diverse-eval-%j.out
#SBATCH --error=logs/math10k-diverse-eval-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCOPE="${SCOPE:-smoke}"
PROTOCOL="$PWD/configs/protocol.math10k-llama32-3b-diverse-kl-exploratory.json"
DATASETS="$PWD/configs/math10k_eval_datasets.json"
RUN_DIR="$PWD/results/runs/math10k-llama32-3b-diverse-kl-exp01"

common_args=(
  --training-protocol "$PROTOCOL"
  --datasets-config "$DATASETS"
  --model best
  --run-dir "$RUN_DIR"
)

case "$SCOPE" in
  smoke)
    output="$PWD/results/eval/math10k-diverse-kl-best-smoke-01"
    benchmarks=(gsm8k arc_challenge openbookqa social_iqa)
    extra_args=(--limit 10)
    ;;
  full)
    BENCHMARK="${BENCHMARK:?set BENCHMARK for a full evaluation}"
    case "$BENCHMARK" in
      gsm8k|arc_challenge|openbookqa|social_iqa) ;;
      *) echo "unsupported BENCHMARK=$BENCHMARK" >&2; exit 2 ;;
    esac
    output="$PWD/results/eval/math10k-diverse-kl-best-${BENCHMARK}-full"
    benchmarks=("$BENCHMARK")
    extra_args=()
    ;;
  *)
    echo "unknown SCOPE=$SCOPE; use smoke or full" >&2
    exit 2
    ;;
esac

if [[ -e "$output" ]]; then
  echo "refusing to overwrite existing output: $output" >&2
  exit 2
fi

nvidia-smi

uv run --locked --no-sync python -m kl_anchors.evaluate_math10k \
  "${common_args[@]}" \
  --benchmarks "${benchmarks[@]}" \
  "${extra_args[@]}" \
  --output-dir "$output"
