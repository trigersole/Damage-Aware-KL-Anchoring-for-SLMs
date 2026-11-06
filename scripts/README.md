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

## Exploratory Math10K diversity-KL run

This separate path preserves the existing Llama-3.2-3B Math10K baseline and
uses the schedule approved on 2026-11-05. First reproduce the CPU-side target
manifests using the commands in `data/README.md`. Then rank on the GPU:

```bash
mkdir -p logs
sbatch --export=ALL,STAGE=rank scripts/slurm_math10k_diverse_kl.sbatch
```

The rank stage aligns the part-2 prompt pool to the completed teacher cache,
embeds it with the pinned `sentence-transformers/all-MiniLM-L6-v2` revision and
writes deterministic farthest-first ranks. Run preflight directly without
requesting a GPU allocation:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --locked --no-sync python -m kl_anchors.train_math10k \
  --protocol configs/protocol.math10k-llama32-3b-diverse-kl-exploratory.json \
  --mode diverse_kl --experiment-id math10k-llama32-3b-diverse-kl-exp01 \
  --preflight
```

It must report `Anchor preflight OK: prompts=128 positions=255`. Then submit:

```bash
sbatch --export=ALL,STAGE=train scripts/slurm_math10k_diverse_kl.sbatch
```

Training applies 64 scored positions every four updates with KL coefficient
0.01. The expected total is 902 KL events and 57,728 scored-position exposures
over 3,609 updates. Use a new `EXPERIMENT_ID` if the default output directory
already exists; never overwrite a completed run.

### Full-pool Math10K diversity prompt export

Evaluate the completed corrected `fullpool-02` best adapter on all five pinned
Math10K evaluator datasets with
`sbatch scripts/slurm_math10k_diverse_fullpool_eval.sbatch`.
The single six-hour job runs ten items per benchmark first, then the full
GSM8K, ARC-Challenge, ARC-Easy, OpenBookQA and Social IQA splits. Outputs are
`results/eval/math10k-diverse-fullpool-02-best-all-smoke-01` and
`results/eval/math10k-diverse-fullpool-02-best-all-full-01`, each containing
`items.jsonl` and `aggregate.json`. Existing output directories cause a refusal
to overwrite. ARC-Easy is an additional benchmark relative to the earlier
four-benchmark Math10K evaluation suite.

After the quota-failed `fullpool-01`, restart with
`sbatch scripts/slurm_math10k_diverse_fullpool_restart.sbatch`.
The new `fullpool-02` run uses
`configs/protocol.math10k-llama32-3b-diverse-kl-fullpool-restart.json` and
identical seed/training settings from the untouched base. The optional
`runtime.keep_last_checkpoints=1` retains the best plus latest checkpoint,
pruning older weights only after successful saving and validation; histories
remain available. Zero/unset keeps the historical retain-all behavior.

Train with the corrected length-eligible 128-prompt selection using
`sbatch scripts/slurm_math10k_diverse_fullpool_train.sbatch` (six-hour limit).
Configuration: `configs/protocol.math10k-llama32-3b-diverse-kl-fullpool.json`.
Run: `results/runs/math10k-llama32-3b-diverse-kl-fullpool-01`.
It uses seed 42, KL weight 0.01 and the existing three-epoch Math10K protocol.
The training manifest contains the already selected 128 prompts with verified
continuation lengths; the configuration also records the complete eligible
pool and ranks used to select them.

For the KL-length-eligible 128-prompt export, use
`sbatch scripts/slurm_math10k_diversity_eligible.sbatch` (six-hour limit).
It screens the complete 13,769-row source using the existing Llama prompt length
policy before reranking. Output `artifacts/math10k/full-eligible-128-v1/`
contains exclusions, eligible pool, full ranks, selected prompt text, selected
teacher continuations, exact 255-position allocation and completion manifest.
Teacher generation runs only for the selected 128; all candidates undergo the
same tokenizer-based eligibility check. This does not launch training.

The earlier exploratory run ranked only the part-2 shard. Do not use its
selection as the full-pool diverse condition. Rank all 13,769 prompts in the
frozen combined manifest and export the first 128 ranks with:

```bash
mkdir -p logs
sbatch scripts/slurm_math10k_diversity_full.sbatch
```

The six-hour job writes the complete ranking to
`artifacts/math10k/full/diversity_ranks_full.jsonl` and the shareable selected
prompt set to `artifacts/math10k/full/diverse_prompts_top128.jsonl`. Both retain
the pinned embedding-model revision, seed, pool fingerprint and prompt text.

After the diverse-KL run completes, evaluate its validation-selected
`best_model` checkpoint. Start with the four-benchmark, ten-item smoke job:

```bash
sbatch --export=ALL,SCOPE=smoke scripts/slurm_math10k_diverse_eval.sbatch
```

After it succeeds, submit one full job per benchmark. Every job retains the
six-hour compute limit and writes a separate immutable output directory:

```bash
for benchmark in gsm8k arc_challenge openbookqa social_iqa; do
  sbatch --export=ALL,SCOPE=full,BENCHMARK="$benchmark" \
    scripts/slurm_math10k_diverse_eval.sbatch
done
```

These use the pinned dataset revisions in `configs/math10k_eval_datasets.json`.
The current Math10K evaluator is an exploratory, study-specific protocol; do
not present these outputs as the final core GSM8K/Qwen comparison.
