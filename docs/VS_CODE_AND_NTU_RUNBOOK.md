# Where to run the project

Open the **repository root**, `AI6130_KL_Anchors_Project`, in VS Code. Run commands in a terminal whose current directory is that root. The `src/` directory is a Python package, not the directory from which to launch the whole study.

There are three useful levels of execution:

| Level | Where | Purpose |
|---|---|---|
| CPU smoke | VS Code on a laptop | Check the training math, method switch, checkpoints and plot using tiny synthetic data. No Qwen download or research result. |
| Exploratory Qwen run | VS Code connected to a suitable GPU machine, or an NTU interactive GPU allocation | Check that the actual model, tokenizer and a few GSM8K updates work together. Use a separate configuration and an `exploratory-` experiment ID. |
| Reported study | NTU GPU Slurm jobs | Run the frozen data, anchor-selection, training and evaluation protocol across all required conditions. |

This Mac's ordinary `python3` currently reports Python 3.14, while the project declares Python 3.11 or 3.12. Select 3.12 with `uv` rather than relying on `python3`.

## Folder map

```text
AI6130_KL_Anchors_Project/            <-- open this folder in VS Code
  README.md, PROJECT_CONTEXT.md       research overview
  DECISIONS.md                        approved and open scientific choices
  pyproject.toml, uv.lock              install definition and exact resolved versions
  requirements.txt                    human-readable direct dependencies
  configs/
    data.template.json                copy to a named data configuration
    protocol.template.json            copy to a named study protocol
  src/kl_anchors/                      reusable Python implementation
    prepare_data.py                    GSM8K split and eligible-pool preparation
    semantic_audit.py                  held-out semantic overlap screen
    anchor_pipeline.py                 teacher cache, discovery scores, merge
    diversity.py                       diversity order
    train.py                           LoRA and four KL modes
    evaluate_study.py, analyze.py      benchmark scoring and comparison
  scripts/
    local_smoke.py                     tiny local functional check
    slurm_pipeline.sbatch              NTU job launcher
  tests/                               CPU unit and integration tests
  data/processed/<study-id>/           generated, ignored by Git
  artifacts/<study-id>/                generated anchor cache and rankings, ignored
  results/runs/<experiment-id>/        generated adapters and metrics, ignored
  reports/                             selected report material
```

Keep code, configuration templates, the named **approved** configurations, and documentation in this root folder. `.venv/`, downloaded datasets, model weights, generated artifacts and `results/runs/` are intentionally ignored. This folder does not currently have a `.git` repository or remote; sharing it with teammates or transferring it to NTU requires copying it or setting up version control separately. Keep a backup of generated run outputs on permitted cluster storage.

## First run in VS Code on your computer

1. In VS Code choose **File → Open Folder** and open the path above. Open **Terminal → New Terminal**. Confirm that `pwd` ends in `AI6130_KL_Anchors_Project`.
2. Install `uv` if `uv --version` fails. On macOS with Homebrew, `brew install uv` is one option; the [official uv installation page](https://docs.astral.sh/uv/getting-started/installation/) lists others. Reopen the terminal after installation if the command is not found.
3. From the project root, run:

   ```bash
   uv python install 3.12
   uv sync --locked --python 3.12
   uv run --locked python --version
   uv run --locked python -m unittest discover -s tests -q
   uv run --locked python scripts/local_smoke.py
   ```

   `uv sync` creates the project `.venv/` and installs the locked environment. It may download a large PyTorch stack. `--locked` prevents an unnoticed lockfile change. The local smoke command creates a new ignored folder under `results/runs/local-smoke/` and prints its path; it does not create a research result. Add `--all-modes` to check every method flag. See [`LOCAL_SMOKE.md`](LOCAL_SMOKE.md) for its exact scope and output files.

4. In VS Code use **Python: Select Interpreter** from the Command Palette and choose this project's `.venv/bin/python`. On Windows the corresponding path is `.venv\\Scripts\\python.exe`. The [VS Code environment guide](https://code.visualstudio.com/docs/python/environments) describes this selection. The terminal commands above use `uv run`, so they do not depend on the editor selection.

If `uv run --locked python --version` prints 3.14, recreate or reselect the environment with `uv sync --locked --python 3.12` before proceeding.

## Preparing an actual study

The templates contain `null` values intentionally. A full Qwen run cannot start until the team approves the open choices in `DECISIONS.md`. Do not fill research values just to make validation pass and then report the result as the study.

1. Copy `configs/data.template.json` to a named file such as `configs/data.study-01.json`. Fill the exact dataset commits, a GSM8K train/validation split seed and count, the approved anchor source, prompt columns, overlap thresholds and reviewed filters. Set `candidate_filters.review_completed` to `true` **after** reviewing the source and filter/exclusion list.
2. Copy `configs/protocol.template.json` to `configs/protocol.study-01.json`. Fill the model's 40-character commit, dataset IDs/revisions/splits, absolute NTU paths, LoRA/optimizer/steps, KL and token budget, prompts/scoring, and hardware/software record. The repository ID is already `Qwen/Qwen2.5-1.5B-Instruct`; its commit remains open. Set the internal evaluator digest as described in `IMPLEMENTATION.md`.
3. Make a separate `configs/protocol.discovery-01.json` for the predeclared diagnostic adaptation. A short exploratory trial should have yet another named config and an `exploratory-` output directory or experiment ID.
4. Check the completed protocol:

   ```bash
   uv run --locked python -m kl_anchors.protocol --config configs/protocol.study-01.json
   ```

   The validator lists missing fields. Passing it checks structure, not team approval.

Use **absolute paths on the machine that will execute the run**. A Mac path such as `/Users/...` is not a valid NTU path. For example, `training.training_ids_manifest` must point to the future `data/processed/study-01/gsm8k_train.jsonl` on NTU, `datasets.gsm8k.validation_ids_manifest` to its validation file, `datasets.anchors.eligible_ids_manifest` to the semantic-screened pool, `anchor_selection.candidate_pool_manifest` to the merged scored pool, and `kl.teacher_continuation` to the teacher cache. `output_directory` must equal the Slurm `RESULTS_ROOT` variable.

## A small **real Qwen** batch

Use a GPU allocation or VS Code Remote SSH connected to NTU. A laptop CPU test is possible in principle but loading a 1.5B model and a separate teacher for KL is slow and memory intensive; the synthetic local smoke is the immediate laptop check.

For a technical trial, copy the approved protocol to a clearly named exploratory config and use, for example, `training.max_steps=2`, `training.per_device_batch_size=1`, `training.gradient_accumulation_steps=1`, and `runtime.checkpoint_every_steps=1`. These numbers are **illustrative smoke settings**, not approved scientific settings. Keep a small, separate GSM8K training manifest and validation carve-out, update their paths and recorded counts, and use a new experiment ID. First try task-only LoRA because it does not need the teacher cache or anchor pool:

```bash
uv run --locked python -m kl_anchors.train \
  --protocol configs/protocol.exploratory-01.json \
  --condition final --mode lora --experiment-id exploratory-lora-001
```

The CLI uses `--condition final` for an ordinary adaptation run. The `exploratory-` config and ID mark this short technical trial as **unreported**, and it must not be mixed with the final comparison. All protocol fields still need to be valid, including benchmark metadata and evaluation templates. `--mode lora` sets its KL budget to zero and does not load anchor data. If the trial fails, inspect `results/runs/<id>/status.json`; use a new ID for the retry.

## Full NTU sequence

Transfer this source folder to NTU (or set up a repository there), then install `uv` and run `uv sync --locked --python 3.12` from the NTU copy's root. If the approved protocol uses 4-bit or 8-bit quantization, use `uv sync --locked --extra cuda-quantized --python 3.12`. Check `nvidia-smi` and `uv run --locked python -c 'import torch; print(torch.cuda.is_available())'` inside the GPU allocation. Record the GPU and software versions. The exact partition, resource limits and storage path depend on NTU's cluster configuration.

Run stages in this order, using the exact commands in `IMPLEMENTATION.md` or `scripts/slurm_pipeline.sbatch`:

| Order | Stage | Main output |
|---:|---|---|
| 1 | `prepare` | GSM8K train, validation and test JSONL; preliminary eligible anchors; audit manifest |
| 2 | `semantic` | Final eligible anchor JSONL and semantic exclusions |
| 3 | `base` | Untouched base benchmark predictions and metrics |
| 4 | `discovery` | Separate task-only LoRA diagnostic adapter |
| 5 | `cache` | Frozen teacher continuations on the eligible pool |
| 6 | `score` | Behavioural drift scores using a **predeclared discovery step checkpoint** |
| 7 | `diversity` | Pinned embedding-based diversity ranks on the same pool |
| 8 | `merge` | One scored candidate pool for every anchored method |
| 9 | `train` | Fresh `lora`, `random_kl`, `diverse_kl`, `damage_kl`, and optional `damage_diverse_kl` adapters |
| 10 | `evaluate` | Per-item GSM8K, BoolQ, HellaSwag and ARC-Easy results |
| 11 | `analyze` | Forgetting tables and learning-versus-retention plots |

The Slurm script accepts `STAGE=all` or one stage at a time. Its `STAGE=all` path includes **all five training modes and evaluation of every saved checkpoint by default**. The resource flags in `IMPLEMENTATION.md` are examples, not measured NTU requirements. Use separate jobs when one allocation cannot contain the whole sequence. Submit from the repository root so `SLURM_SUBMIT_DIR` points there.

For example, after filling the protocol with NTU paths, set the variables required by the launcher in the **NTU** terminal and submit one condition:

```bash
export PROTOCOL="$PWD/configs/protocol.study-01.json"
export DATA_DIR="$PWD/data/processed/study-01"
export ARTIFACT_DIR="$PWD/artifacts/study-01"
export RESULTS_ROOT="$PWD/results/runs"     # must equal protocol.output_directory
export RUN_PREFIX="study-01-seed0"
sbatch --gres=gpu:1 --export=ALL,STAGE=train,MODE=lora scripts/slurm_pipeline.sbatch
```

Add NTU's required partition/account, CPU, memory and wall-time flags to `sbatch`; the example does not establish a sufficient allocation. `STAGE=train` expects data and, for anchored modes, the scored pool from earlier stages. Replace `MODE=lora` with another mode to submit its own run. The `DISCOVERY_PROTOCOL`, `DISCOVERY_STEP`, `EMBED_MODEL_ID`, `EMBED_MODEL_REVISION`, `SEMANTIC_THRESHOLD`, `SEED` and `DAMAGE_WEIGHT` variables are additionally required by their corresponding stages or by `STAGE=all`; see the script header and `IMPLEMENTATION.md`.

Each training condition is switched with one flag:

```bash
uv run --locked python -m kl_anchors.train --protocol configs/protocol.study-01.json \
  --condition final --mode damage_kl --experiment-id study-01-damage-seed0
```

Allowed values are `lora`, `random_kl`, `diverse_kl`, `damage_kl`, and `damage_diverse_kl`. The combined mode also needs a predeclared `--damage-weight` or a weight in the protocol. The untouched base is evaluated separately.

## Inspecting and keeping results

After a run, open `results/runs/<experiment-id>/`:

```text
protocol.json                    immutable configuration used for this run
status.json                      complete/failed state, counters and best checkpoint
checkpoints/step_000001/        saved LoRA adapter at a training step
best_model/                     copy of the best validation checkpoint
training_history.jsonl          per-step target loss, KL and epoch
validation_history.jsonl        checkpoint validation loss/accuracy
loss_vs_epoch.png               report-ready training curve
anchor_selection.json           selected IDs and budget for anchored modes
evaluation/                     per-item predictions and aggregate metrics after evaluation
evaluations/step_XXXXXX/        optional checkpoint-by-checkpoint evaluations
```

Generated runs, data and artifacts are excluded from version control; copy the **small final plots/tables** into `reports/` and back up the full raw outputs where NTU permits. Never reuse an experiment ID or overwrite a reported run. Record team-approved changes in `DECISIONS.md` and keep the exact named JSON configs alongside the code.

For implementation details and exact commands for every stage, continue with [`IMPLEMENTATION.md`](IMPLEMENTATION.md).
