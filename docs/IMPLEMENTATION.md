# Running the KL anchor study

This repository implements the Team G4 experiment. It has **no reported model runs yet**. The CPU tests use synthetic examples. The exact Qwen2.5 commit, anchor source, candidate filters, discovery duration, KL settings, anchor budget, evaluation details, seeds, and NTU GPU allocation remain protocol decisions in [`DECISIONS.md`](../DECISIONS.md). Fill and approve them before interpreting final test results.

## 1. Environment

Use Python 3.11 or 3.12. The project has a `pyproject.toml`, a human-readable `requirements.txt`, and an exact transitive `uv.lock`. On the NTU login node:

```bash
uv sync --locked
# Only when the approved configuration uses 4-bit or 8-bit quantization:
uv sync --locked --extra cuda-quantized
uv run --locked python -m unittest discover -s tests -v
```

`uv run --locked --no-sync` is used inside the Slurm job to avoid dependency resolution on a compute node. The [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) gives installation options. The lockfile was resolved for Python 3.12; the training stack has not been installed or tested on NTU hardware yet. Record the final Python/CUDA/device and installed package versions in the completed protocol.

## 2. Freeze data and the protocol

1. Copy [`data.template.json`](../configs/data.template.json) to a new named JSON file. Fill source commit revisions, a validation count and seed, the approved anchor source/columns, n-gram overlap thresholds, maximum prompt length, target-like and unsafe regexes, and manual review exclusions. Set `candidate_filters.review_completed=true` only after that review; the preparation CLI requires it. Keep the validation carve-out within GSM8K **train**; never train or tune on GSM8K test.
2. Copy [`protocol.template.json`](../configs/protocol.template.json) to a named study JSON file. Fill every applicable `null`. The model repository is already set to `Qwen/Qwen2.5-1.5B-Instruct`; pin its full 40-character commit. Point paths to the future frozen manifests and outputs. Leave the same target data, LoRA settings, optimizer, seed, training steps, KL settings and anchor-token budget in the shared configuration for all final modes.
   For this code's evaluator, set `evaluation.harness` to `internal_kl_anchors_v1` and set `evaluation.harness_revision` to `uv run --locked python -m kl_anchors.provenance --digest-only`. The evaluator checks that source digest before scoring. Record package versions with `python -m kl_anchors.provenance` after the NTU environment is installed.
3. Record team-approved material choices in [`DECISIONS.md`](../DECISIONS.md). `python -m kl_anchors.protocol --config <study.json>` lists missing or structurally invalid fields. A structurally valid file is not evidence of team approval or scientific validity.

The runner records a new immutable `protocol.json` under `output_directory/experiment_id`. It refuses to reuse an experiment ID. For task-only LoRA, it records zero anchor and KL budgets. Discovery uses its own experiment ID and never initializes a final condition.

The preparation command is:

```bash
uv run --locked python -m kl_anchors.prepare_data \
  --config configs/data.study.json --output-dir data/processed/study-01
```

This writes `gsm8k_train.jsonl`, `gsm8k_validation.jsonl`, `gsm8k_test.jsonl`, `eligible_anchors.jsonl`, exclusions and a data manifest. Each target row is `{example_id, prompt, response}`. This format is reusable for a later target task; task-specific evaluation adapters can be added separately. Candidate rows are `{example_id, prompt}`. Final test answers are saved for final evaluation but never enter the anchor selector.

Run the embedding overlap screen on the preliminary eligible pool, with an approved embedding model/revision and cosine threshold:

```bash
uv run --locked python -m kl_anchors.semantic_audit \
  --data-config configs/data.study.json --data-dir data/processed/study-01 \
  --candidates data/processed/study-01/eligible_anchors.jsonl \
  --model-id <embedding-repo> --model-revision <40-hex-commit> \
  --threshold <approved-threshold> --device cuda:0 \
  --output artifacts/study-01/semantic_eligible_anchors.jsonl \
  --exclusions artifacts/study-01/semantic_exclusions.jsonl
```

Set `datasets.anchors.eligible_ids_manifest` to that output and retain both surface and semantic exclusion manifests. The semantic screen uses held-out inputs and choices, never labels. Pattern filters and a manual review list cover target-like and unsafe prompts; these need a human review before freezing the pool. Prompts that exceed the configured token sequence limit fail the teacher-cache stage instead of being silently truncated.

## 3. Base and discovery

Evaluate the untouched base before adaptation:

```bash
uv run --locked python -m kl_anchors.evaluate_study \
  --base-protocol configs/protocol.study.json --experiment-id study-01-base
```

Run a separately configured, task-only discovery adaptation. Use a distinct protocol file if it has a shorter target subset or step count. Its training manifest and exact duration must be frozen before scoring anchors:

```bash
uv run --locked python -m kl_anchors.train \
  --protocol configs/protocol.discovery.json --condition discovery \
  --mode lora --experiment-id study-01-discovery
```

Each training run saves `checkpoints/step_XXXXXX/`, `best_model/`, `training_history.jsonl`, `validation_history.jsonl`, `loss_vs_epoch.png`, `status.json`, and the immutable protocol. `best_model` is copied from the checkpoint with the best **validation** GSM8K exact match or lowest validation target loss, as specified by `evaluation.checkpoint_selection_rule`. A tie on GSM8K accuracy uses validation loss. Test sets are never used to choose a checkpoint. The complete learning-forgetting curve can be obtained by evaluating saved step checkpoints separately.

## 4. Build the common teacher cache and anchor rankings

The frozen teacher generates each eligible prompt's continuation once. Discovery scoring then forces teacher and adapted student to score the **same token IDs**. KL is averaged over aligned continuation positions. The score describes behavioural drift; it is not proof of factual damage or parameter localization.

```bash
uv run --locked python -m kl_anchors.anchor_pipeline cache \
  --protocol configs/protocol.study.json \
  --anchors artifacts/study-01/semantic_eligible_anchors.jsonl \
  --output artifacts/study-01/teacher_cache.jsonl

uv run --locked python -m kl_anchors.anchor_pipeline score \
  --protocol configs/protocol.study.json \
  --cache artifacts/study-01/teacher_cache.jsonl \
  --discovery-adapter results/runs/study-01-discovery/checkpoints/step_<predeclared> \
  --output artifacts/study-01/vulnerability_scores.jsonl

uv run --locked python -m kl_anchors.diversity \
  --pool artifacts/study-01/semantic_eligible_anchors.jsonl \
  --model-id <embedding-repo> --model-revision <40-hex-commit> \
  --algorithm farthest_first --seed <locked-seed> --device cuda:0 \
  --output artifacts/study-01/diversity_ranks.jsonl

uv run --locked python -m kl_anchors.anchor_pipeline merge \
  --cache artifacts/study-01/teacher_cache.jsonl \
  --scores artifacts/study-01/vulnerability_scores.jsonl \
  --ranks artifacts/study-01/diversity_ranks.jsonl \
  --output artifacts/study-01/scored_pool.jsonl
```

The shared protocol's `anchor_selection.candidate_pool_manifest` points to the merged pool and `kl.teacher_continuation` points to the cache. Random, diversity, vulnerability, and combined selectors read **the same pool**. They select an exact number of scored response tokens. The final selected continuation can be shortened to fill the budget. Training also schedules an exact configured number of anchor tokens per KL update, so equal step counts and interleave frequency give equal scored-token exposure. Prompt overhead and wall time can still differ and are recorded separately.

## 5. Final comparison with one mode flag

Use a new experiment ID for each independent run. The runner reloads the untouched base and creates a new LoRA adapter every time.

| `--mode` | Condition |
|---|---|
| `lora` | Task-only LoRA |
| `random_kl` | Uniform random prompt order, KL anchors |
| `diverse_kl` | Pinned farthest-first semantic order, KL anchors |
| `damage_kl` | Highest discovery drift first, KL anchors |
| `damage_diverse_kl` | Weighted rank fusion of drift and diversity, KL anchors; optional extension |

For example:

```bash
uv run --locked python -m kl_anchors.train \
  --protocol configs/protocol.study.json --condition final \
  --mode damage_kl --experiment-id study-01-damage-seed0

uv run --locked python -m kl_anchors.train \
  --protocol configs/protocol.study.json --condition final \
  --mode damage_diverse_kl --damage-weight <approved-weight> \
  --experiment-id study-01-combined-seed0
```

The combined method averages normalized damage rank and diversity rank according to its explicit weight in `[0,1]`. It is an extra analysis; the original base, task-only, random, diversity and vulnerability conditions remain required. If compute permits, repeat the full final matrix with additional predeclared seeds. Do not select method parameters using final benchmark scores.

## 6. Evaluate and analyze

```bash
uv run --locked python -m kl_anchors.evaluate_study \
  --run-dir results/runs/study-01-damage-seed0

# To build a full trade-off curve, evaluate a saved step adapter in a new result folder:
uv run --locked python -m kl_anchors.evaluate_study \
  --run-dir results/runs/study-01-damage-seed0 --checkpoint step_000100

uv run --locked python -m kl_anchors.analyze \
  --base-run results/runs/study-01-base \
  --adapted-run results/runs/study-01-lora-seed0 \
  --adapted-run results/runs/study-01-random-seed0 \
  --adapted-run results/runs/study-01-diverse-seed0 \
  --adapted-run results/runs/study-01-damage-seed0 \
  --output-dir results/runs/study-01-analysis
```

Evaluation saves per-item predictions and multiple-choice likelihoods plus aggregate accuracy. Analysis refuses incompatible prompts, scoring, model revisions or item IDs, then writes forgetting per benchmark, item loss/gain counts, and target-versus-retention plots. Add `--adapted-run results/runs/<id>/evaluations/step_000100` for every evaluated checkpoint to plot a fuller learning-forgetting curve. Pass a team-approved `--match-tolerance <fraction>` to report comparisons at approximately matched **validation** GSM8K accuracy; absent a match, the analysis reports no matched pair. Report the full checkpoint trade-off and all three benchmark changes. Negative forgetting means improvement. Only these selected benchmarks support the retention claim.

## 7. Slurm

[`scripts/slurm_pipeline.sbatch`](../scripts/slurm_pipeline.sbatch) accepts `STAGE=all` for the full sequence or one stage at a time (`prepare`, `semantic`, `base`, `discovery`, `cache`, `score`, `diversity`, `merge`, `train`, `evaluate`, `analyze`). For a single method set `STAGE=train MODE=damage_kl`. The full sequence evaluates saved checkpoints for the trade-off curve by default; set `EVALUATE_ALL_CHECKPOINTS=0` only for an exploratory run. Set `MATCH_TOLERANCE` after the team fixes its validation matching rule. Install/sync the locked environment on the login node first. Submit from the repository root with NTU-specific resource flags:

```bash
sbatch --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00 \
  --export=ALL,STAGE=all,PROTOCOL=/absolute/protocol.study.json,DISCOVERY_PROTOCOL=/absolute/protocol.discovery.json,DATA_CONFIG=/absolute/data.study.json,DATA_DIR=/absolute/data/processed/study-01,ARTIFACT_DIR=/absolute/artifacts/study-01,RESULTS_ROOT=/absolute/results/runs,RUN_PREFIX=study-01,EMBED_MODEL_ID=<repo>,EMBED_MODEL_REVISION=<commit>,SEMANTIC_THRESHOLD=<threshold>,SEED=<seed>,DISCOVERY_STEP=<step>,DAMAGE_WEIGHT=<weight> \
  scripts/slurm_pipeline.sbatch
```

Ask NTU for the actual partition, GPU type/memory, wall-time limit, local storage and CUDA driver before submitting. Keep downloaded datasets, cached models, adapters and large outputs on cluster storage outside version control. Use stage-by-stage jobs with unique experiment IDs if the full sequence exceeds one allocation.

## Current limits

- The code paths that need Hugging Face models and NTU GPUs have not been run here. The local machine has no accessible CUDA or MPS device, and no model weights were downloaded.
- The GSM8K answer parser is a documented numeric heuristic. Lock and inspect it on validation examples before final evaluation.
- The current target-training interface accepts prompt/response JSONL and is reusable for another supervised target. The supplied final evaluator implements GSM8K, BoolQ, HellaSwag and ARC-Easy only; an additional target requires a task-specific evaluation adapter.
- The fixed budget equalizes scored continuation tokens. Variable prompt length still affects compute and should be reported.
- A single discovery seed may make rankings unstable. Use additional seeds when compute permits and label any one-seed result preliminary.
