# Project Decisions

This file is the authoritative record of agreed design choices and unresolved questions. Update it whenever the team approves a material change.

**Last reviewed:** 2026-10-16

## Confirmed decisions

| Item | Decision |
|---|---|
| Team | G4 |
| Title | *Can Vulnerability-Guided KL Anchors Reduce Catastrophic Forgetting in Parameter-Efficient Fine-Tuning?* |
| Main method | LoRA with vulnerability-guided KL anchors |
| Initial target task | GSM8K |
| Initial retention benchmarks | BoolQ, HellaSwag and ARC-Easy |
| Core baselines | Untouched base model and task-only LoRA |
| Anchor-selection comparisons | Random, diversity-based and vulnerability-guided |
| Initial scope | Mathematics adaptation first; additional domains only if time and resources permit |
| Contribution framing | Anchor selection based on observed behavioural vulnerability, not KL regularization itself |
| Claim scope | Retention on selected non-target benchmarks, not preservation of all general capabilities |
| Model repository ID | `Qwen/Qwen2.5-1.5B-Instruct`; exact immutable revision still to be pinned |
| Additional implementation mode | Combined damage-plus-diversity KL anchoring is supported as an optional extension; the original core baselines remain required |

The term **pilot** was intentionally removed from the professor-facing proposal and email. Internally, the method still requires a clearly specified vulnerability-identification stage. The code supports a separate discovery run, but its duration, data and checkpoint must be approved before final experiments.

## Provisional design requiring team confirmation

- GSM8K will use final-answer exact-match accuracy.
- BoolQ will use accuracy.
- HellaSwag and ARC-Easy will use a documented multiple-choice likelihood protocol.
- Forgetting will be calculated separately as `base-model score - fine-tuned-model score`.
- Anchored conditions will use equal anchor-token budgets.
- Training settings will be matched across final conditions.
- Primary analysis will compare retention at approximately matched GSM8K performance and report the learning-forgetting trade-off.
- The frozen reference and current student will be compared using token-level KL on aligned response positions.

These are scientifically recommended controls but should not be represented as team-approved until confirmed.

## Open decisions

1. **Exact model revision:** The repository ID is `Qwen/Qwen2.5-1.5B-Instruct`; pin a 40-character immutable commit before reported runs.
2. **Vulnerability-identification procedure:** Define the adaptation duration, data subset, seed and checkpoint used to measure behavioural drift.
3. **Anchor-pool source:** Select the general-purpose prompt dataset and confirm its licence.
4. **Anchor-pool cleaning:** Define duplicate, semantic-overlap, target-like, unsafe and excessive-length filters.
5. **Teacher-response protocol:** Decide whether to cache deterministic teacher continuations and define generation/scoring settings.
6. **KL formulation:** Choose KL direction, temperature, coefficient, response-token mask and normalization.
7. **Anchor budget:** Fix the total anchor tokens and sampling/mixing ratio.
8. **Diversity baseline:** Select the embedding model, clustering algorithm and sampling rule.
9. **Performance matching:** Decide between checkpoints, early stopping or a hyperparameter sweep for matched GSM8K performance.
10. **Evaluation implementation:** Select or build the evaluation harness and freeze prompt templates.
11. **Statistical reporting:** Decide the number of seeds, confidence intervals and aggregation method.
12. **Compute limits:** Record available GPU type, memory and maximum runtime/storage budget.
13. **Extension criterion:** Define when resources are sufficient to attempt a second target domain.

## Decision log

Add new entries in this format:

### YYYY-MM-DD — Short decision name

- **Status:** Approved / Rejected / Superseded
- **Decision:**
- **Reason:**
- **Alternatives considered:**
- **Consequences for existing experiments or documents:**
- **Approved by:**

### 2026-10-17 — Full held-out evaluation for final accuracy reporting

- **Status:** Approved
- **Decision:** Preserve the deterministic GSM8K split of 7,173 training and 300 validation examples, and keep the official 1,319-example GSM8K test split untouched. For final accuracy reporting, evaluate every labelled item in the locked retention splits: BoolQ validation (3,270), HellaSwag validation (10,042) and ARC-Easy test (2,376). Retain the earlier 100-example outputs as exploratory artifacts rather than overwriting them.
- **Reason:** The 100-example caps produce one-percentage-point resolution and are too noisy for final target-learning or retention conclusions. Training, checkpoint selection and final testing must remain separate.
- **Alternatives considered:** Increase each cap to another sample size or resample new subsets. Full labelled splits avoid sampling variance and make paired base-versus-adapted comparisons auditable.
- **Consequences for existing experiments or documents:** Reuse the completed Study-02 best checkpoint without retraining. Run matching full-split evaluations for both the untouched base and every adapted condition using the locked prompts and likelihood-scoring rules. Do not tune checkpoints or hyperparameters using these final test results.
- **Approved by:** User, in this task.

### 2026-10-16 — Larger GSM8K validation carve-out and lower learning rate

- **Status:** Approved
- **Decision:** Use a deterministic 300-example validation carve-out from the GSM8K training split with seed 42, leaving 7,173 target-training examples. Train all new final comparison conditions with AdamW learning rate `2e-5` for five target-data epochs. With per-device batch size 1 and four target microbatches per optimizer update, use `training.max_steps` 8,970.
- **Reason:** The 100-example validation loss was considered too noisy for checkpoint guidance, and the earlier `2e-4` task-only run reached its validation-loss minimum near 1.5 epochs before increasingly overfitting. The larger carve-out provides a more stable checkpoint-selection signal, while the lower learning rate tests a slower adaptation trajectory.
- **Alternatives considered:** Retain 100 validation examples; use 500 validation examples; retain learning rate `2e-4`; or use an intermediate learning rate. The user selected 300 examples and `2e-5`.
- **Consequences for existing experiments or documents:** Freeze a new prepared-data directory and a new protocol/configuration. Preserve all earlier runs unchanged and label them as belonging to the prior split and optimizer protocol. The new base, task-only, random-anchor, diversity-anchor and vulnerability-anchor comparisons must all use the same 300-example split, `2e-5` learning rate and five-epoch budget. Results from the earlier 100-example/`2e-4` run are not directly interchangeable with this comparison matrix.
- **Approved by:** User, in this task.

### 2026-09-25 — Five-epoch target-training budget

- **Status:** Approved
- **Decision:** Run the aligned GSM8K protocol for five target-data epochs. With 7,373 training examples, per-device batch size 1 and four target microbatches per optimizer update, use `training.max_steps` 9,220 and record `training.epochs` as 5.
- **Reason:** The user requested a longer adaptation run after the one-epoch task-only run showed only a modest validation-loss improvement.
- **Alternatives considered:** Retain one epoch or change checkpoint selection to validation GSM8K exact match. Checkpoint selection remains unchanged because only the training-duration increase was approved.
- **Consequences for existing experiments or documents:** Use `configs/protocol.study-01-aligned.json` with a new experiment ID. Preserve the completed one-epoch run unchanged. The five-epoch result is a different training budget and must not be treated as directly matched to one-epoch conditions.
- **Approved by:** User, in this task.

### 2026-09-24 — Align GSM8K training and evaluation prompts

- **Status:** Superseded
- **Decision:** Use the raw GSM8K question as the complete user message during both target training and GSM8K generation evaluation. The evaluation template is therefore {question}, with no added solve-step-by-step instruction or Question:/Answer: wrapper.
- **Reason:** The completed study-01-seed0-lora run trained on raw-question user messages but its evaluation protocol added an instruction and wrapper. Matching the prompts removes this train-evaluation distribution mismatch.
- **Alternatives considered:** Remove only the first instruction sentence while retaining the Question:/Answer: wrapper; instead, the whole wrapper was removed because retaining it would not make the prompts equal.
- **Consequences for existing experiments or documents:** Preserve study-01-seed0-lora/protocol.json unchanged. Use configs/protocol.study-01-aligned.json and new experiment IDs for future base evaluation, training, and comparison runs. Results produced with the old and aligned evaluation prompts must be labelled separately and are not directly comparable as one locked-protocol study.
- **Approved by:** User, in this task.

### 2026-09-24 — Use the full GSM8K instruction consistently

- **Status:** Approved
- **Decision:** Apply the same full prompt template during target training, validation-loss calculation and GSM8K generation evaluation: instruct the model to solve step by step and put the final numerical answer after `####`, followed by the labelled question and answer fields.
- **Reason:** This preserves train-evaluation consistency while making the expected reasoning and final-answer format explicit for the instruction-tuned model.
- **Alternatives considered:** Use the raw question alone in both phases. This was rejected because it leaves the response format implicit even though GSM8K targets end with `####`.
- **Consequences for existing experiments or documents:** The target-training runner now formats GSM8K questions through the locked evaluation template. Preserve the completed `study-01-seed0-lora` artifacts unchanged; use `configs/protocol.study-01-aligned.json` and new experiment IDs for every new base, task-only and anchored comparison. Previously trained and newly trained conditions must not be mixed in one locked-protocol comparison.
- **Approved by:** User, in this task.

### 2026-09-16 — Qwen2.5 model repository

- **Status:** Approved
- **Decision:** Use `Qwen/Qwen2.5-1.5B-Instruct` as the base model repository. The exact revision remains open and is required by the run validator.
- **Reason:** The user selected this checkpoint explicitly while planning implementation.
- **Alternatives considered:** `Qwen/Qwen2-1.5B-Instruct`, which appeared in an earlier discussion.
- **Consequences for existing experiments or documents:** Update executable configs to Qwen2.5; do not treat earlier Qwen2 proposal text as the final checkpoint. No model runs have been reported.
- **Approved by:** Dadi Hemanth, in this task.

### 2026-09-16 — Optional combined anchor selector

- **Status:** Approved
- **Decision:** Implement a damage-plus-diversity KL mode alongside the original task-only, random, diversity, and vulnerability modes. Keep the original core comparisons; combined selection is an optional extension. Its rank-fusion weight must be stated in the protocol.
- **Reason:** The user requested a reusable flag for this mode while explicitly retaining the project's core comparisons.
- **Alternatives considered:** Keeping the combined method only as a future idea.
- **Consequences for existing experiments or documents:** The runner and selector expose `damage_diverse_kl`; no outcome or additional run is claimed.
- **Approved by:** Dadi Hemanth, in this task.
