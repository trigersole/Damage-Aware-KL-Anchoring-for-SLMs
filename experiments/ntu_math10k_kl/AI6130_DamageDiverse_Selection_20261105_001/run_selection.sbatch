#!/usr/bin/env bash
#SBATCH --job-name=damage-diverse
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH --output=logs/damage-diverse-select-%j.out
#SBATCH --error=logs/damage-diverse-select-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
cd "${SLURM_SUBMIT_DIR:?Submit from the separate selection folder}"
SELECTION_PYTHON="/home/msai/hemanth005/AI6130_Srihari_Step5_20261102/code_git_ab02b28b/.venv/bin/python"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "[LAUNCH] job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-unknown} scope=approved-cached-vector-selection-only"
exec "$SELECTION_PYTHON" -B -u select_cached.py --seconds 3300 "$@"
