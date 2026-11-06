#!/usr/bin/env bash
#SBATCH --job-name=damage-A-export
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --export=ALL
#SBATCH --output=logs/damage-A-%j.out
#SBATCH --error=logs/damage-A-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
cd "${SLURM_SUBMIT_DIR:?Submit from the separate export folder}"
DAMAGE_EXPORT_PYTHON="/home/msai/hemanth005/AI6130_Srihari_Step5_20261102/code_git_ab02b28b/.venv/bin/python"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "[LAUNCH] job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-unknown} scope=existing-cache-and-scores-top128-export"
exec "$DAMAGE_EXPORT_PYTHON" -B -u export_damage_A.py
