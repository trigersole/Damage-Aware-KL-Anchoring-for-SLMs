#!/usr/bin/env bash
#SBATCH --job-name=ai6130-kl-anchors
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Submit with site-specific resources, for example:
# sbatch --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00 \
#   --export=ALL,STAGE=all,PROTOCOL=/abs/protocol.json,DATA_CONFIG=/abs/data.json,... \
#   scripts/slurm_pipeline.sbatch
# No partition or GPU model is assumed here; NTU cluster details remain unknown.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export TOKENIZERS_PARALLELISM=false

: "${STAGE:?set STAGE to prepare, semantic, base, discovery, cache, score, diversity, merge, train, evaluate, analyze, or all}"
: "${PROTOCOL:?set PROTOCOL to a completed shared protocol JSON path}"
: "${DATA_DIR:?set DATA_DIR to a new absolute prepared-data directory}"
: "${ARTIFACT_DIR:?set ARTIFACT_DIR to an absolute anchor-artifact directory}"
: "${RESULTS_ROOT:?set RESULTS_ROOT to match protocol.output_directory}"
: "${RUN_PREFIX:?set RUN_PREFIX to a unique identifier for this study batch}"
if [[ "$STAGE" == "all" ]]; then
  : "${DATA_CONFIG:?set DATA_CONFIG}"
  : "${DISCOVERY_PROTOCOL:?set DISCOVERY_PROTOCOL}"
  : "${DISCOVERY_STEP:?set DISCOVERY_STEP}"
  : "${EMBED_MODEL_ID:?set EMBED_MODEL_ID}"
  : "${EMBED_MODEL_REVISION:?set EMBED_MODEL_REVISION}"
  : "${SEMANTIC_THRESHOLD:?set SEMANTIC_THRESHOLD}"
  : "${SEED:?set SEED}"
  : "${DAMAGE_WEIGHT:?set DAMAGE_WEIGHT}"
fi

UV_BIN="${UV_BIN:-uv}"
if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  echo "uv is unavailable; install uv on the login node before submission" >&2
  exit 2
fi

run_py() {
  "$UV_BIN" run --locked --no-sync python -m "$@"
}

validate_paths() {
  local config_path="${1:-$PROTOCOL}"
  python - "$config_path" "$RESULTS_ROOT" <<'PY'
import json
import pathlib
import sys
protocol = json.loads(pathlib.Path(sys.argv[1]).read_text())
if pathlib.Path(protocol["output_directory"]).resolve() != pathlib.Path(sys.argv[2]).resolve():
    raise SystemExit("RESULTS_ROOT must equal protocol.output_directory")
PY
}

prepare() {
  : "${DATA_CONFIG:?set DATA_CONFIG to the explicit data-preparation JSON}"
  run_py kl_anchors.prepare_data --config "$DATA_CONFIG" --output-dir "$DATA_DIR"
}

base() {
  run_py kl_anchors.evaluate_study --base-protocol "$PROTOCOL" --experiment-id "${RUN_PREFIX}-base"
}

semantic() {
  : "${DATA_CONFIG:?set DATA_CONFIG to the explicit data-preparation JSON}"
  : "${EMBED_MODEL_ID:?set the pinned sentence-embedding model ID}"
  : "${EMBED_MODEL_REVISION:?set its full commit revision}"
  : "${SEMANTIC_THRESHOLD:?set the predeclared cosine-overlap threshold}"
  mkdir -p "$ARTIFACT_DIR"
  run_py kl_anchors.semantic_audit --data-config "$DATA_CONFIG" --data-dir "$DATA_DIR" \
    --candidates "$DATA_DIR/eligible_anchors.jsonl" \
    --model-id "$EMBED_MODEL_ID" --model-revision "$EMBED_MODEL_REVISION" \
    --threshold "$SEMANTIC_THRESHOLD" --device "${EMBED_DEVICE:-cuda:0}" \
    --output "$ARTIFACT_DIR/semantic_eligible_anchors.jsonl" \
    --exclusions "$ARTIFACT_DIR/semantic_exclusions.jsonl"
}

discovery() {
  : "${DISCOVERY_PROTOCOL:?set a separate, predeclared discovery adaptation protocol}"
  validate_paths "$DISCOVERY_PROTOCOL"
  run_py kl_anchors.train --protocol "$DISCOVERY_PROTOCOL" --mode lora --condition discovery --experiment-id "${RUN_PREFIX}-discovery"
}

cache() {
  mkdir -p "$ARTIFACT_DIR"
  run_py kl_anchors.anchor_pipeline cache --protocol "$PROTOCOL" \
    --anchors "$ARTIFACT_DIR/semantic_eligible_anchors.jsonl" --output "$ARTIFACT_DIR/teacher_cache.jsonl"
}

score() {
  : "${DISCOVERY_STEP:?set the predeclared discovery checkpoint step (integer)}"
  local checkpoint_name
  printf -v checkpoint_name 'step_%06d' "$DISCOVERY_STEP"
  run_py kl_anchors.anchor_pipeline score --protocol "$PROTOCOL" \
    --cache "$ARTIFACT_DIR/teacher_cache.jsonl" \
    --discovery-adapter "$RESULTS_ROOT/${RUN_PREFIX}-discovery/checkpoints/$checkpoint_name" \
    --output "$ARTIFACT_DIR/vulnerability_scores.jsonl"
}

diversity() {
  : "${EMBED_MODEL_ID:?set the pinned sentence-embedding model ID}"
  : "${EMBED_MODEL_REVISION:?set its full commit revision}"
  : "${SEED:?set the locked seed}"
  run_py kl_anchors.diversity --pool "$ARTIFACT_DIR/semantic_eligible_anchors.jsonl" \
    --output "$ARTIFACT_DIR/diversity_ranks.jsonl" \
    --model-id "$EMBED_MODEL_ID" --model-revision "$EMBED_MODEL_REVISION" \
    --algorithm farthest_first --seed "$SEED" --device "${EMBED_DEVICE:-cuda:0}"
}

merge() {
  run_py kl_anchors.anchor_pipeline merge \
    --cache "$ARTIFACT_DIR/teacher_cache.jsonl" \
    --scores "$ARTIFACT_DIR/vulnerability_scores.jsonl" \
    --ranks "$ARTIFACT_DIR/diversity_ranks.jsonl" \
    --output "$ARTIFACT_DIR/scored_pool.jsonl"
}

train_one() {
  local mode="$1"
  local args=(--protocol "$PROTOCOL" --condition final --mode "$mode" --experiment-id "${RUN_PREFIX}-${mode}")
  if [[ "$mode" == "damage_diverse_kl" ]]; then
    : "${DAMAGE_WEIGHT:?set a predeclared rank-fusion weight in [0,1]}"
    args+=(--damage-weight "$DAMAGE_WEIGHT")
  fi
  if [[ -n "${BEST_METRIC:-}" ]]; then
    args+=(--best-metric "$BEST_METRIC")
  fi
  run_py kl_anchors.train "${args[@]}"
}

train_all() {
  for mode in lora random_kl diverse_kl damage_kl damage_diverse_kl; do
    train_one "$mode"
  done
}

evaluate_all() {
  for mode in lora random_kl diverse_kl damage_kl damage_diverse_kl; do
    local run_dir="$RESULTS_ROOT/${RUN_PREFIX}-${mode}"
    run_py kl_anchors.evaluate_study --run-dir "$run_dir"
    if [[ "${EVALUATE_ALL_CHECKPOINTS:-1}" == "1" ]]; then
      for checkpoint in "$run_dir"/checkpoints/step_*; do
        [[ -d "$checkpoint" ]] || continue
        run_py kl_anchors.evaluate_study --run-dir "$run_dir" --checkpoint "$(basename "$checkpoint")"
      done
    fi
  done
}

analyze() {
  local args=(
    --base-run "$RESULTS_ROOT/${RUN_PREFIX}-base" \
    --adapted-run "$RESULTS_ROOT/${RUN_PREFIX}-lora" \
    --adapted-run "$RESULTS_ROOT/${RUN_PREFIX}-random_kl" \
    --adapted-run "$RESULTS_ROOT/${RUN_PREFIX}-diverse_kl" \
    --adapted-run "$RESULTS_ROOT/${RUN_PREFIX}-damage_kl" \
    --adapted-run "$RESULTS_ROOT/${RUN_PREFIX}-damage_diverse_kl" \
    --output-dir "$RESULTS_ROOT/${RUN_PREFIX}-analysis"
  )
  if [[ -n "${MATCH_TOLERANCE:-}" ]]; then
    args+=(--match-tolerance "$MATCH_TOLERANCE")
  fi
  if [[ "${EVALUATE_ALL_CHECKPOINTS:-1}" == "1" ]]; then
    for mode in lora random_kl diverse_kl damage_kl damage_diverse_kl; do
      for evaluation_dir in "$RESULTS_ROOT/${RUN_PREFIX}-${mode}"/evaluations/step_*; do
        [[ -d "$evaluation_dir" ]] || continue
        args+=(--adapted-run "$evaluation_dir")
      done
    done
  fi
  run_py kl_anchors.analyze "${args[@]}"
}

validate_paths
case "$STAGE" in
  prepare) prepare ;;
  semantic) semantic ;;
  base) base ;;
  discovery) discovery ;;
  cache) cache ;;
  score) score ;;
  diversity) diversity ;;
  merge) merge ;;
  train) : "${MODE:?set MODE for a single training condition}"; train_one "$MODE" ;;
  evaluate) : "${MODE:?set MODE for a single trained condition}"; run_py kl_anchors.evaluate_study --run-dir "$RESULTS_ROOT/${RUN_PREFIX}-${MODE}" ;;
  analyze) analyze ;;
  all) prepare; semantic; base; discovery; cache; score; diversity; merge; train_all; evaluate_all; analyze ;;
  *) echo "unknown STAGE: $STAGE" >&2; exit 2 ;;
esac
