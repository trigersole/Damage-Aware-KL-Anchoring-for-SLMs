#!/usr/bin/env bash
#SBATCH --job-name=damage-diverse
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
#SBATCH --output=logs/damage-diverse-%j.out
#SBATCH --error=logs/damage-diverse-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
cd "${SLURM_SUBMIT_DIR:?Submit from the separate damage-diverse package}"
DAMAGE_DIVERSE_STAGE="${1:-smoke}"
case "$DAMAGE_DIVERSE_STAGE" in smoke|train|evaluate) ;; *) echo 'Use smoke, train or evaluate'; exit 2 ;; esac
if [[ $# -gt 2 || ( $# -eq 2 && "$2" != "--resume" ) ]]; then
  echo 'Optional second argument is --resume'; exit 2
fi
DAMAGE_DIVERSE_PYTHON="/home/msai/hemanth005/AI6130_Srihari_Step5_20261102/code_git_ab02b28b/.venv/bin/python"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "[LAUNCH] job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-unknown} stage=$DAMAGE_DIVERSE_STAGE"
if [[ "$DAMAGE_DIVERSE_STAGE" == "evaluate" ]]; then
  exec "$DAMAGE_DIVERSE_PYTHON" -B -u evaluate_dd.py run "${@:2}"
else
  exec "$DAMAGE_DIVERSE_PYTHON" -B -u run_dd.py "$DAMAGE_DIVERSE_STAGE" "${@:2}"
fi
