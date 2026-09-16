# Project Decisions

This file is the authoritative record of agreed design choices and unresolved questions. Update it whenever the team approves a material change.

**Last reviewed:** 2026-09-16

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
