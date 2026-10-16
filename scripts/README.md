# Slurm launcher

Run `slurm_pipeline.sbatch` from the repository root. The launcher accepts one `STAGE` through `sbatch --export` and fails before starting project code when a variable required by that stage is missing.

The checked-in resource defaults request one GPU, eight CPUs, 32 GB of system memory and six hours. They were sufficient for the observed single-condition Qwen2.5-1.5B LoRA runs on an NVIDIA A40. They are not a guarantee for another cluster, anchored training, checkpoint-wide evaluation or `STAGE=all`; override them with site-specific `sbatch` flags.

## Current task-only Study-02 setup

From the repository root:

```bash
export PROTOCOL="$PWD/configs/protocol.study-02.json"
export DATA_CONFIG="$PWD/configs/data.study-02.json"
export DATA_DIR="$PWD/data/processed/study-02"
export ARTIFACT_DIR="$PWD/artifacts/study-02"
export RESULTS_ROOT="$PWD/results/runs"
export RUN_PREFIX="study-02-repro-seed42-01"
```

`RUN_PREFIX` must be new. The commands below produce run IDs such as `${RUN_PREFIX}-lora`; never reuse a completed experiment ID. The paths embedded in `PROTOCOL` must be valid absolute paths on the execution host, and `RESULTS_ROOT` must equal its `output_directory`.

Prepare the deterministic 300-example GSM8K validation split:

```bash
sbatch --export=ALL,STAGE=prepare scripts/slurm_pipeline.sbatch
```

Train task-only LoRA:

```bash
sbatch --export=ALL,STAGE=train,MODE=lora scripts/slurm_pipeline.sbatch
```

Evaluate the selected task-only checkpoint:

```bash
sbatch --export=ALL,STAGE=evaluate,MODE=lora scripts/slurm_pipeline.sbatch
```

For reported results, evaluate every item in the locked labelled splits. This
writes `evaluation_full/` and preserves any earlier capped `evaluation/`:

```bash
sbatch --time=12:00:00 \
  --export=ALL,STAGE=evaluate,MODE=lora,FULL_DATASET_EVALUATION=1 \
  scripts/slurm_pipeline.sbatch
```

Evaluate the untouched base model under the same benchmark protocol:

```bash
sbatch --export=ALL,STAGE=base scripts/slurm_pipeline.sbatch
```

Use a new `RUN_PREFIX` and `FULL_DATASET_EVALUATION=1` for the matching full
base evaluation. The four full denominators are GSM8K 1,319, BoolQ 3,270,
HellaSwag 10,042 and ARC-Easy 2,376.

Add cluster-specific flags when required, for example:

```bash
sbatch --partition=<partition> --account=<account> \
  --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=06:00:00 \
  --export=ALL,STAGE=train,MODE=lora \
  scripts/slurm_pipeline.sbatch
```

## Stages and required variables

Variables exported in the shell are inherited because every example uses `--export=ALL`.

| `STAGE` | Required variables | Purpose |
|---|---|---|
| `prepare` | `DATA_CONFIG`, `DATA_DIR` | Create frozen target splits and the surface-audited candidate pool. |
| `semantic` | `DATA_CONFIG`, `DATA_DIR`, `ARTIFACT_DIR`, `EMBED_MODEL_ID`, `EMBED_MODEL_REVISION`, `SEMANTIC_THRESHOLD` | Remove semantic overlap from the candidate pool. |
| `base` | `PROTOCOL`, `RESULTS_ROOT`, `RUN_PREFIX` | Evaluate the untouched base checkpoint. |
| `discovery` | `DISCOVERY_PROTOCOL`, `RESULTS_ROOT`, `RUN_PREFIX` | Train the separate diagnostic task-only adapter. |
| `cache` | `PROTOCOL`, `RESULTS_ROOT`, `ARTIFACT_DIR` | Cache frozen-teacher continuations. |
| `score` | `PROTOCOL`, `RESULTS_ROOT`, `RUN_PREFIX`, `ARTIFACT_DIR`, `DISCOVERY_STEP` | Measure teacher-to-discovery behavioural divergence. |
| `diversity` | `ARTIFACT_DIR`, `EMBED_MODEL_ID`, `EMBED_MODEL_REVISION`, `SEED` | Produce semantic-diversity ranks. |
| `merge` | `ARTIFACT_DIR` | Merge teacher targets, vulnerability scores and diversity ranks. |
| `train` | `PROTOCOL`, `RESULTS_ROOT`, `RUN_PREFIX`, `MODE` | Train one fresh final condition. |
| `evaluate` | `PROTOCOL`, `RESULTS_ROOT`, `RUN_PREFIX`, `MODE` | Evaluate the selected checkpoint for one condition. |
| `analyze` | `RESULTS_ROOT`, `RUN_PREFIX` | Compare the expected base and adapted run directories. |
| `all` | All variables above plus `DAMAGE_WEIGHT` | Run the complete pipeline and all five implemented modes sequentially. |

Optional variables:

| Variable | Meaning |
|---|---|
| `UV_BIN` | `uv` executable; defaults to `uv`. |
| `EMBED_DEVICE` | Embedding device; defaults to `cuda:0`. |
| `BEST_METRIC` | Override checkpoint selection with `min_validation_loss` or `max_gsm8k_exact_match`. Do not change this for reported comparisons without approval and a decision-log entry. |
| `EVALUATE_ALL_CHECKPOINTS` | For `STAGE=all`, defaults to `1`; set to `0` only for an explicitly exploratory run. |
| `MATCH_TOLERANCE` | Approved validation-accuracy tolerance used by analysis for approximately matched target performance. |
| `FULL_DATASET_EVALUATION` | Set to `1` to remove all four evaluation caps and write separate full-evaluation outputs. |

Allowed `MODE` values:

```text
lora  random_kl  diverse_kl  damage_kl  damage_diverse_kl
```

`damage_diverse_kl` is an optional extension and needs `DAMAGE_WEIGHT`. The core study requires untouched base, task-only LoRA, random KL, diversity KL and vulnerability-guided (`damage_kl`) conditions.

## Anchored runs are not configured yet

`configs/protocol.study-02.json` is the immutable task-only protocol used for the completed Study-02 LoRA run. Its KL coefficient, anchor budget and anchor mixing fields are intentionally zero, and its scored-pool paths are unset. Do not use it to launch `random_kl`, `diverse_kl`, `damage_kl`, `damage_diverse_kl` or `STAGE=all`.

The launcher performs an anchored-mode preflight and rejects non-positive budgets/mixing values or missing scored-pool and teacher-cache files before loading a model.

Before anchored runs, the team must approve the open KL, anchor-budget, discovery and diversity choices in `DECISIONS.md`. Record those decisions and create a new named protocol rather than altering the completed Study-02 protocol.

## Monitoring and outputs

```bash
squeue -u "$USER"
sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed
tail -f "slurm-<job-id>.err"
```

Training writes `results/runs/${RUN_PREFIX}-${MODE}/`. Inspect `status.json`, `protocol.json`, `validation_history.jsonl` and `best_model/`. Capped evaluation adds `evaluation_status.json` and `evaluation/`; full evaluation adds `evaluation_full_status.json` and `evaluation_full/`. Generated datasets, adapters, predictions and Slurm logs are ignored by Git; preserve compact report artifacts separately.
