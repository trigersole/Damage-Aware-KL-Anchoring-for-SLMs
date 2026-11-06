#!/usr/bin/env bash
#SBATCH --job-name=damage-A-calib
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --export=ALL
#SBATCH --output=logs/kl-calibration-%j.out
#SBATCH --error=logs/kl-calibration-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
cd "${SLURM_SUBMIT_DIR:?Submit from the separate calibration folder}"
CALIBRATION_STAGE="${1:-smoke}"
case "$CALIBRATION_STAGE" in smoke|calibrate) ;; *) echo 'Use smoke or calibrate'; exit 2 ;; esac
CALIBRATION_PYTHON="/home/msai/hemanth005/AI6130_Srihari_Step5_20261102/code_git_ab02b28b/.venv/bin/python"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "[LAUNCH] job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-unknown} stage=$CALIBRATION_STAGE"
exec "$CALIBRATION_PYTHON" -B -u calibrate.py "$CALIBRATION_STAGE"
