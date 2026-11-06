#!/usr/bin/env bash
#SBATCH --job-name=damage-A-full
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --export=ALL
#SBATCH --output=logs/damage-A-full-%j.out
#SBATCH --error=logs/damage-A-full-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using Slurm}"
cd "${SLURM_SUBMIT_DIR:?Submit from the separate full-training folder}"
FULL_PYTHON="/home/msai/hemanth005/AI6130_Srihari_Step5_20261102/code_git_ab02b28b/.venv/bin/python"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "[LAUNCH] job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-unknown} stage=damage_A_full_lambda001"
exec "$FULL_PYTHON" -B -u full_train.py train
