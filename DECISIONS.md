# Project Decisions

This file is the authoritative record of agreed design choices and unresolved questions. Update it whenever the team approves a material change.

**Last reviewed:** 2026-11-05 (exploratory Math10K diversity-KL schedule approved)

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

### 2026-11-05 — Evaluate corrected full-pool diverse LoRA on configured datasets

- **Status:** Approved.
- **Decision:** Evaluate the completed `math10k-llama32-3b-diverse-kl-fullpool-02` validation-selected best adapter (step 2400) on full GSM8K test, ARC-Challenge test, ARC-Easy test, OpenBookQA test and Social IQA validation using `configs/math10k_eval_datasets.json` and the existing Math10K evaluator. First run a ten-item-per-benchmark smoke check. Preserve per-item outputs and aggregate metrics in new output directories; six-hour Slurm cap.
- **Reason:** User requested testing on all available datasets. These five are the supported pinned datasets in the Math10K evaluation configuration.
- **Consequences:** ARC-Easy is an additional evaluation alongside the previously configured Math10K retention suite; report it separately where comparators lack matching evaluation. Training and validation-based checkpoint selection are unchanged. Other target datasets and the core study's distinct evaluation harness are outside this configured suite. Existing scoring and curation-provenance limitations remain.
- **Approved by:** User, in this task.

### 2026-11-05 — Restart full-pool diverse LoRA after quota failure

- **Status:** Approved.
- **Decision:** Restart independently from the same untouched base and seed under experiment `math10k-llama32-3b-diverse-kl-fullpool-02`, keeping all optimization, data, anchor-selection, KL and six-hour settings. Retain only the best and latest checkpoints during this run, with all losses and per-step metrics preserved.
- **Reason:** Job 51398 failed while saving step 1500 because the disk quota was exhausted. Existing adapter-only checkpoints lack optimizer state; the user explicitly chose an identical-settings restart over continuation with reset AdamW moments.
- **Cleanup:** User authorized removing unnecessary old/failed-run artifacts. Removed 16 redundant/incomplete checkpoint directories totaling 4,663,492,240 bytes; exact targets recorded in `results/cleanup-2026-11-05.json`. Preserve the failed run's step-1200 adapter, the superseded part-2 run's best adapter and records, all run histories and the complete task-only baseline. Removed checkpoint weights cannot be recovered from this workspace.
- **Consequences:** Failed run remains labeled failed. New run uses a new protocol and ID. Checkpoint retention changes storage only, not training or checkpoint selection. Full optimizer/RNG checkpoint resumption remains unimplemented.
- **Approved by:** User, including explicit choice to restart with identical seed/settings.

### 2026-11-05 — Train Math10K LoRA using corrected full-pool diverse anchors

- **Status:** Approved.
- **Decision:** Launch `math10k-llama32-3b-diverse-kl-fullpool-01` from the untouched pinned Llama-3.2-3B base with the corrected 128 anchors from `full-eligible-128-v1`. Retain seed 42, KL coefficient 0.01, 255 selected response positions, 64 positions every four optimizer updates, and the existing Math10K three-epoch/3,609-update LoRA protocol. Slurm wall time is six hours.
- **Reason:** User requested LoRA training using the corrected selection.
- **Consequences:** New protocol and run directory preserve the old part-2 result. The training input contains the preselected 128 and verified continuation lengths; full 13,750-candidate ranks and selection provenance are separately linked in the protocol. Random/vulnerability comparators must use the full eligible pool, not the 128 training inputs. Original curation provenance and exploratory Math10K evaluation limitations remain unchanged.
- **Approved by:** User, in this task.

### 2026-11-05 — Length-eligible full-pool diversity selection

- **Status:** Approved by the user's request to select 128 prompts following the repository protocol.
- **Decision:** Apply the existing Llama tokenizer, Question/Answer template, 2,048-token sequence limit and 64-token continuation reservation to the complete 13,769-prompt pool before recomputing diversity. Select 128 with the existing pinned encoder, farthest-first cosine algorithm and seed 42. Cache deterministic teacher continuations for these prompts and allocate the existing 255 scored-position budget.
- **Reason:** The earlier full-pool text-only selection included an overlength prompt. Length eligibility is deterministic and can be checked before generation; only selected continuations are needed for this prompt-export task.
- **Consequences:** Save all exclusions, full ranks, selected prompt text, teacher token IDs and token allocations under `artifacts/math10k/full-eligible-128-v1`. Preserve prior artifacts. Future random/vulnerability comparators must use the identical length-eligible pool. This request does not launch training. The imported source pool's missing original curation audit remains a limitation.
- **Approved by:** User, in this task.

### 2026-11-05 — Full combined pool for Math10K diversity prompts

- **Status:** Approved.
- **Decision:** Compute seed-42 farthest-first diversity ranks over all 13,769 prompts in `data/anchors/eligible_anchors.jsonl`, using the already pinned `sentence-transformers/all-MiniLM-L6-v2` revision, and export the first 128 ranked prompts with their text and provenance. Preserve the earlier 6,880-candidate part-2 selection and its trained adapter as exploratory artifacts only.
- **Reason:** The user explicitly directed that the diverse prompts be created from the full list after identifying that the earlier run ranked only one shard.
- **Alternatives considered:** Retain the part-2 selection or concatenate independently computed part-1 and part-2 rankings. Both were rejected because neither computes diversity jointly over the full eligible list.
- **Consequences for existing experiments or documents:** The full-pool prompt export receives new artifact paths and does not overwrite prior results. It does not itself approve or launch retraining. Before KL retraining, teacher continuations must be generated for the selected full-pool anchors and a new experiment ID/configuration must be used.
- **Approved by:** User, in this task.

### 2026-11-05 — Script-free Social IQA evaluation serialization

- **Status:** Approved as a compatibility correction for the requested exploratory Math10K evaluation; benchmark content and scoring are unchanged.
- **Decision:** Load Social IQA validation from the official Parquet conversion at immutable dataset revision `6db3dcea20d5430c3dd74bf85fe9ca87268ea91b`. Retain all 1,954 validation examples and the existing context, question, three choices and label fields. Continue using the predeclared single-label conditional-likelihood scoring protocol.
- **Reason:** `datasets==4.8.5` rejects the older repository revision because it requires executing `social_i_qa.py`. The official pinned Parquet export provides the same benchmark fields without executable dataset code.
- **Alternatives considered:** Downgrade the datasets library or execute the legacy loading script. Both would change the locked environment or re-enable unsupported arbitrary dataset code.
- **Consequences for existing experiments or documents:** The failed ten-item smoke job `51306` produced no evaluation output directory. Rerun smoke before full evaluation. This remains an exploratory Math10K evaluation and does not resolve the core GSM8K/Qwen evaluation decisions.
- **Approved by:** User's request to run the configured evaluation in this task; implementation is a necessary serialization compatibility correction.

### 2026-11-05 — Exploratory Math10K diversity-KL schedule

- **Status:** Approved for an exploratory run; this does not replace the core GSM8K/Qwen comparison design.
- **Decision:** Starting independently from the pinned `meta-llama/Llama-3.2-3B` base, reuse the completed Math10K baseline's target split, LoRA configuration, optimizer, scheduler and 3,609-update/three-epoch training protocol. Select 128 diversity-ranked Dolly prompts with exactly 255 scored teacher-continuation positions. Apply teacher-to-student, temperature-1, response-token-mean KL with coefficient `0.01`, using 64 scored positions every four optimizer updates. This gives 902 KL events and nominal exposure of 57,728 scored positions. Use deterministic farthest-first cosine ranking with `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, seed 42.
- **Reason:** The user supplied the schedule from the current-run comparison and requested the Math10K training protocol. Exact prompt and token constraints preserve the stated 128-prompt/255-position selection, while chunked KL computation avoids changing its mathematical token-mean objective to fit GPU memory.
- **Alternatives considered:** A 128-token budget, 16 positions every eight updates; 128 prompts with their full continuations; and applying KL at every update. These were superseded by the supplied current-run settings.
- **Consequences for existing experiments or documents:** Preserve `math10k-llama32-3b-full-01` unchanged. Use a new protocol and experiment ID. The first run uses only the locally available part-2 anchor shard and is therefore exploratory, not a final core comparison. Any random or vulnerability comparator must draw from the same eligible pool and match both the 255-position selection budget and 57,728-position exposure. Do not generalize its results to the original GSM8K/Qwen study.
- **Approved by:** User, in this task.

### 2026-10-27 — Publish reviewed anchor curation with an optional preview

- **Status:** Implementation approved for publication by Hemanth's request to push the saved anchor-curation changes and his teammate's preview suggestion. This is not approval of numerical cleaning settings or a research launch.
- **Decision:** Provide the separate `export` / `propose` / reviewed `finalize` workflow. Approximate benchmark matches and semantic candidate duplicates require recorded review. Exact/quality exclusions remain automatic; configured evaluation inputs remain in decontamination. All anchored conditions use the same finalized pool.
- **Preview:** `propose --export-auto-included` writes `auto_included_anchors.jsonl` after full lexical/semantic screening. Exclude hard failures, both members of unresolved duplicate pairs, benchmark-overlap flags and pending spot checks. Hash and count the preview in the review manifest. The optional output does not approve a pool, alter review decisions or bypass finalization. Reject it with lexical-only screening.
- **Documentation:** Publish `docs/ANCHOR_CURATION_REVIEW.md` and an updated `docs/kl_filtering_new.md`, adapted from the supplied Markdown guide with duplication removed and current preview commands.
- **Still open:** Candidate source approval, embedding model/revision, thresholds, long-input policy, explicit target-training overlap scope, review rubric and audit sample settings must be recorded before a real curation run. The export adapter currently covers GSM8K, BoolQ, HellaSwag and ARC-Easy; extending evaluation sets requires the corresponding input renderers. No model, training, KL coefficient, primary evaluation or anchor-token budget is selected by this implementation.
- **Existing experiments:** Preserve all results, weights and training/evaluation code. No real pool, encoder download, training or KL result is claimed by publishing this workflow. Verification uses synthetic fixtures; see the review guide for reproduction commands.
- **Verification:** The latest-repository full suite passes all 85 tests, including all 22 curation tests. Preview-on/off produces identical review flags and finalized pools; both duplicate-pair members, hard failures and pending spot checks are withheld from the preview. Missing review decisions and preview tampering remain errors. The command-line help exposes the optional flag. No real encoder download or model run was used.

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
