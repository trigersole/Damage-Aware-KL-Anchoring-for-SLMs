#!/usr/bin/env bash
#SBATCH --job-name=ai6130-kl-anchors
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# Flag reference and copy-paste commands: scripts/README.md
# The resource defaults above suit one observed single-condition A40 run. Override
# them at submission time when the site, stage, or model requires different limits.
# Run the multi-stage study as separate jobs unless one allocation is known to fit.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export TOKENIZERS_PARALLELISM=false

: "${STAGE:?see scripts/README.md and set STAGE}"

UV_BIN="${UV_BIN:-uv}"
if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  echo "uv is unavailable; install uv on the login node before submission" >&2
  exit 2
fi

run_py() {
  "$UV_BIN" run --locked --no-sync python -m "$@"
}

require_value() {
  local name="$1"
  local guidance="$2"
  if [[ -z "${!name:-}" ]]; then
    echo "$name is required: $guidance" >&2
    exit 2
  fi
}

validate_paths() {
  local config_path="$1"
  require_value RESULTS_ROOT "set it to the same absolute directory as protocol.output_directory"
  python - "$config_path" "$RESULTS_ROOT" <<'PY'
import json
import pathlib
import sys
protocol = json.loads(pathlib.Path(sys.argv[1]).read_text())
if pathlib.Path(protocol["output_directory"]).resolve() != pathlib.Path(sys.argv[2]).resolve():
    raise SystemExit("RESULTS_ROOT must equal protocol.output_directory")
PY
}

require_protocol_context() {
  require_value PROTOCOL "set an absolute path to a complete protocol JSON"
  validate_paths "$PROTOCOL"
}

require_run_context() {
  require_protocol_context
  require_value RUN_PREFIX "choose a unique experiment prefix; never reuse a completed run ID"
}

validate_mode_name() {
  case "$1" in
    lora|random_kl|diverse_kl|damage_kl|damage_diverse_kl) ;;
    *) echo "invalid MODE: $1" >&2; exit 2 ;;
  esac
}

validate_training_mode_config() {
  local mode="$1"
  validate_mode_name "$mode"
  [[ "$mode" == "lora" ]] && return
  python - "$PROTOCOL" <<'PY'
import json
import pathlib
import sys

protocol = json.loads(pathlib.Path(sys.argv[1]).read_text())
checks = {
    "anchor_selection.token_budget": protocol["anchor_selection"].get("token_budget"),
    "training.anchor_interleave_every_steps": protocol["training"].get("anchor_interleave_every_steps"),
    "runtime.anchor_tokens_per_update": protocol["runtime"].get("anchor_tokens_per_update"),
    "kl.coefficient": protocol["kl"].get("coefficient"),
}
invalid = [name for name, value in checks.items() if not isinstance(value, (int, float)) or value <= 0]
for name, value in (
    ("anchor_selection.candidate_pool_manifest", protocol["anchor_selection"].get("candidate_pool_manifest")),
    ("kl.teacher_continuation", protocol["kl"].get("teacher_continuation")),
):
    if not isinstance(value, str) or not value or not pathlib.Path(value).is_file():
        invalid.append(name + " (must name an existing file)")
if invalid:
    raise SystemExit("anchored MODE requires a completed anchored protocol: " + ", ".join(invalid))
PY
}

prepare() {
  : "${DATA_CONFIG:?set DATA_CONFIG to the explicit data-preparation JSON}"
  run_py kl_anchors.prepare_data --config "$DATA_CONFIG" --output-dir "$DATA_DIR"
}

base() {
  local args=(--base-protocol "$PROTOCOL" --experiment-id "${RUN_PREFIX}-base")
  if [[ "${FULL_DATASET_EVALUATION:-0}" == "1" ]]; then
    args+=(--full-dataset)
  fi
  run_py kl_anchors.evaluate_study "${args[@]}"
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
  validate_training_mode_config "$mode"
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

case "$STAGE" in
  prepare)
    require_value DATA_CONFIG "set the absolute data-preparation config path"
    require_value DATA_DIR "set a new absolute output directory"
    prepare
    ;;
  semantic)
    require_value DATA_CONFIG "set the absolute data-preparation config path"
    require_value DATA_DIR "set the prepared-data directory"
    require_value ARTIFACT_DIR "set the absolute anchor-artifact directory"
    semantic
    ;;
  base)
    require_run_context
    base
    ;;
  discovery)
    require_value DISCOVERY_PROTOCOL "set the separately approved discovery protocol"
    require_value RESULTS_ROOT "set it to the discovery protocol output_directory"
    require_value RUN_PREFIX "choose a unique study prefix"
    discovery
    ;;
  cache)
    require_protocol_context
    require_value ARTIFACT_DIR "set the absolute anchor-artifact directory"
    cache
    ;;
  score)
    require_run_context
    require_value ARTIFACT_DIR "set the absolute anchor-artifact directory"
    score
    ;;
  diversity)
    require_value ARTIFACT_DIR "set the absolute anchor-artifact directory"
    diversity
    ;;
  merge)
    require_value ARTIFACT_DIR "set the absolute anchor-artifact directory"
    merge
    ;;
  train)
    require_run_context
    require_value MODE "use lora, random_kl, diverse_kl, damage_kl, or damage_diverse_kl"
    train_one "$MODE"
    ;;
  evaluate)
    require_run_context
    require_value MODE "set the mode used by the completed training run"
    validate_mode_name "$MODE"
    args=(--run-dir "$RESULTS_ROOT/${RUN_PREFIX}-${MODE}")
    if [[ "${FULL_DATASET_EVALUATION:-0}" == "1" ]]; then
      args+=(--full-dataset)
    fi
    run_py kl_anchors.evaluate_study "${args[@]}"
    ;;
  analyze)
    require_value RESULTS_ROOT "set the absolute results/runs directory"
    require_value RUN_PREFIX "set the shared study prefix"
    analyze
    ;;
  all)
    require_run_context
    require_value DATA_CONFIG "set the absolute data-preparation config path"
    require_value DATA_DIR "set a new absolute prepared-data directory"
    require_value ARTIFACT_DIR "set the absolute anchor-artifact directory"
    require_value DISCOVERY_PROTOCOL "set the separately approved discovery protocol"
    require_value DISCOVERY_STEP "set the predeclared discovery checkpoint step"
    require_value EMBED_MODEL_ID "set the pinned embedding-model repository ID"
    require_value EMBED_MODEL_REVISION "set the immutable embedding-model revision"
    require_value SEMANTIC_THRESHOLD "set the approved cosine-overlap threshold"
    require_value SEED "set the locked random seed"
    require_value DAMAGE_WEIGHT "set the optional combined-selector rank-fusion weight"
    prepare; semantic; base; discovery; cache; score; diversity; merge; train_all; evaluate_all; analyze
    ;;
  *) echo "unknown STAGE: $STAGE" >&2; exit 2 ;;
esac
