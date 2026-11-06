#!/usr/bin/env bash
#SBATCH --job-name=random-anchors
#SBATCH --partition=MGPU-TC2
#SBATCH --nodelist=TC2N01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --export=ALL
#SBATCH --output=logs/random-anchors-%j.out
#SBATCH --error=logs/random-anchors-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
cd "${SLURM_SUBMIT_DIR:?Submit from the separate random-anchors package}"
RANDOM_ANCHOR_STAGE="${1:-smoke}"
case "$RANDOM_ANCHOR_STAGE" in smoke|train|evaluate) ;; *) echo 'Use smoke, train or evaluate'; exit 2 ;; esac
if [[ $# -gt 2 || ( $# -eq 2 && "$2" != "--resume" ) ]]; then
  echo 'Optional second argument is --resume'; exit 2
fi
RANDOM_ANCHOR_PYTHON="/home/msai/hemanth005/AI6130_Srihari_Step5_20261102/code_git_ab02b28b/.venv/bin/python"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "[LAUNCH] job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-unknown} stage=$RANDOM_ANCHOR_STAGE"
if [[ "$RANDOM_ANCHOR_STAGE" == "evaluate" ]]; then
  exec "$RANDOM_ANCHOR_PYTHON" -B -u evaluate_random.py run "${@:2}"
else
  exec "$RANDOM_ANCHOR_PYTHON" -B -u run_random.py "$RANDOM_ANCHOR_STAGE" "${@:2}"
fi
