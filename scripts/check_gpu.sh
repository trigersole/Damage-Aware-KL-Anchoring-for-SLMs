#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="${GPU_CHECK_LOG:-$REPO_ROOT/logs/gpu-check-$timestamp.log}"

if [[ "${1:-}" != "--allocated" && -z "${SLURM_JOB_ID:-}" ]]; then
  if ! command -v srun >/dev/null 2>&1; then
    echo "srun is unavailable. Run this script on a Slurm login node or inside a GPU allocation." >&2
    exit 2
  fi
  srun_args=(
    --job-name=kl-anchor-gpu-check
    --gres=gpu:1
    --cpus-per-task=2
    --mem=8G
    --time=00:10:00
  )
  if [[ -n "${GPU_PARTITION:-}" ]]; then
    srun_args+=(--partition "$GPU_PARTITION")
  fi
  exec srun "${srun_args[@]}" env GPU_CHECK_LOG="$log_file" "$0" --allocated
fi

exec > >(tee -a "$log_file") 2>&1

echo "GPU check started: $(date -u --iso-8601=seconds)"
echo "Repository: $REPO_ROOT"
echo "Host: $(hostname)"
echo "Log: $log_file"
echo "Slurm job: ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo

echo "=== nvidia-smi ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi is unavailable on this node"
fi
echo

echo "=== Torch CUDA check ==="
if command -v uv >/dev/null 2>&1; then
  uv run --locked --no-sync python - <<'PY'
import sys

print("Python:", sys.executable)
try:
    import torch
except Exception as exc:
    print("Torch import failed:", repr(exc))
    raise SystemExit(1)

print("Torch:", torch.__version__)
print("Torch compiled CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        print(f"GPU {index}:", torch.cuda.get_device_name(index))
    tensor = torch.ones(1, device="cuda")
    print("CUDA tensor check:", tensor.device, tensor.item())
PY
else
  echo "uv is unavailable on this node"
fi

echo
echo "GPU check finished: $(date -u --iso-8601=seconds)"