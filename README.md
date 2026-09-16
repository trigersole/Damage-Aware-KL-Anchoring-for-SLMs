# AI6130 Course Project: Vulnerability-Guided KL Anchors

**Course:** AI6130 Large Language Models  
**Team:** G4  
**Instructor:** Prof. Luu Anh Tuan  
**Project title:** *Can Vulnerability-Guided KL Anchors Reduce Catastrophic Forgetting in Parameter-Efficient Fine-Tuning?*

## Team

| Member | Matriculation number |
|---|---|
| Sridhar Srihari | G2510717L |
| Dadi Hemanth | G2510794D |
| Siddharth Paliwal | G2510801L |
| Balasubramanian Hariharan | G2610112K |

## Research question

When a language model is adapted to a target task using LoRA, can a limited set of vulnerability-guided KL anchors preserve performance on selected non-target benchmarks better than randomly or diversity-selected anchors?

The proposed contribution is the **anchor-selection strategy**, not KL regularization itself. Vulnerability-guided anchors are selected using observed behavioural drift between the frozen reference model and a task-adapted model.

## Initial experimental scope

- **Target task:** GSM8K mathematical reasoning
- **Retention benchmarks:** BoolQ, HellaSwag and ARC-Easy
- **Model repository:** `Qwen/Qwen2.5-1.5B-Instruct`; immutable revision remains to be pinned
- **Adaptation method:** LoRA
- **Reference model:** Frozen, untouched checkpoint

### Conditions to compare

1. Untouched base model
2. Task-only LoRA
3. LoRA with randomly selected KL anchors
4. LoRA with diversity-selected KL anchors
5. LoRA with vulnerability-guided KL anchors

All anchored conditions should use the same anchor-token budget and otherwise matched training settings. Comparisons should report both target-task learning and non-target retention.

## Evaluation

- **GSM8K:** final-answer exact-match accuracy
- **BoolQ:** accuracy
- **HellaSwag:** multiple-choice accuracy using a documented likelihood-scoring protocol
- **ARC-Easy:** multiple-choice accuracy using the same documented protocol
- **Forgetting per benchmark:** `base-model score - fine-tuned-model score`
- **Target-task gain:** `fine-tuned-model score - base-model score`

Claims must be limited to the evaluated benchmarks. Preserving these scores does not establish preservation of every general model capability.

## Repository structure

```text
.
├── AGENTS.md                 # Durable instructions for future Codex tasks
├── README.md                 # Project entry point
├── PROJECT_CONTEXT.md        # Full handover from the planning conversation
├── DECISIONS.md              # Confirmed, provisional and unresolved choices
├── configs/                  # Versioned experiment configurations
├── data/
│   └── README.md             # Acquisition, licensing and preprocessing notes
├── docs/
│   ├── proposal/             # Proposal, instructions and approval email
│   ├── literature/           # SOTA report and evidence files
│   ├── methodology/          # Detailed beginner and technical guides
│   └── history/              # Earlier ideas and research provenance
├── notebooks/                # Exploration only; reported runs should use scripts
├── src/                      # Training, selection and evaluation code
├── tests/
├── results/                  # Small summaries and tables, not model checkpoints
├── reports/
└── archive/                  # Builders for previously generated artifacts
```

## Starting a new Codex task

Before making changes:

1. Read `AGENTS.md`, `PROJECT_CONTEXT.md` and `DECISIONS.md`.
2. Read the latest proposal under `docs/proposal/`.
3. Identify whether the requested work depends on an unresolved decision.
4. Record approved material design changes in `DECISIONS.md`.
5. Keep experiments reproducible through saved configurations, seeds and exact checkpoint identifiers.

Suggested first message in the new Codex project:

> Read `AGENTS.md`, `PROJECT_CONTEXT.md`, `DECISIONS.md`, and the current proposal. Summarize the finalized research question, the open decisions, and the next safe implementation step before changing files.

## Current status

The topic and core comparison are finalized. A reusable implementation now exists for data preparation, contamination audits, teacher continuation caching, discovery scoring, diversity ranking, five LoRA modes, evaluation, checkpoint selection, and plots. It has synthetic CPU tests; no Qwen training or benchmark result has been reported. Resolve the open items in `DECISIONS.md` and fill the pinned configurations before expensive experiments.

## Run the code

Start with the [VS Code and NTU runbook](docs/VS_CODE_AND_NTU_RUNBOOK.md) for where to open the folder, a local smoke check, folder layout, and the complete run order. See the [implementation guide](docs/IMPLEMENTATION.md) for configuration fields, all stages, `uv`, and Slurm commands. The training runner changes method with `--mode`:

```text
lora  random_kl  diverse_kl  damage_kl  damage_diverse_kl
```

The untouched base is evaluated separately. `damage_diverse_kl` is an optional extension; base, task-only, random, diversity and vulnerability conditions remain the required comparison.
