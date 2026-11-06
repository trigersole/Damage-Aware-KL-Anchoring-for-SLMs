# Math10K full-pool diversity-KL run

Completed run: `math10k-llama32-3b-diverse-kl-fullpool-02`.
This is the exploratory Math10K/Llama extension, distinct from the core
GSM8K/Qwen comparison. No retention improvement is established without
matched base and task-only evaluations.

## Selection and budget

- Original input: `data/anchors/eligible_anchors.jsonl`, 13,769 Dolly prompts.
- Exclude 19 prompts exceeding 1,984 Llama tokens with the locked template;
  rank all remaining 13,750 jointly and select the first 128.
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; effective maximum length 256
  tokens including special tokens. Longer inputs are right-truncated for
  embedding; no chunking. Full prompt text remains in the output JSONL.
- Farthest-first cosine distance on L2-normalized embeddings, seed 42,
  deterministic hash initialization and stable-ID tie breaks.
- 255 distinct scored teacher-response positions across 128 prompts;
  64 positions every four optimizer updates, 902 KL events and 57,728 total
  scored-position exposures including reuse. No aggregate prompt-token cap.
- Teacher-to-student KL, temperature 1, response-token mean, weight 0.01.

The full eligible-pool fingerprint is
`dd317a71a64a8d2570043f890976bb717059935f44e72638f550cc9c9f87fe0b`.
Selected prompt JSONL SHA-256:
`9b99a7dfb81756abc52339a20e1e309344727783f724cd3151fd8a3dc2efbeb4`.
Generated artifacts are stored locally under
`artifacts/math10k/full-eligible-128-v1/` and excluded from Git.
The training `anchor_selection.json` describes the preselected 128-row training
input; use the generation `manifest.json` for complete pool provenance.

The imported source pool has no complete original curation audit. Absence of
exact text duplicates does not certify absence of semantic duplicates or
benchmark paraphrases. These limitations remain attached to the results.

## Training

Pinned base: `meta-llama/Llama-3.2-3B` at
`13afe5124825b4f3751f836b40dafda64c1ed062`. Seed 42; 9,619 Math10K training
examples and 300 validation examples; three epochs, 3,609 optimizer updates.
LoRA rank 128, alpha 256, dropout 0, q/k/v/o modules, bf16; AdamW at 5e-5,
linear scheduler with 361 warmup steps; batch size 1 and accumulation 8.
See the restart configuration for the full recorded settings and environment.

Job 51398 failed at checkpoint saving due to disk quota. The user chose a
fresh identical-settings restart because adapter-only checkpoints lack
optimizer/RNG state. Job 51421 completed in 02:32:40. Best and latest
checkpoints were retained; all per-step metrics remain saved.
The validation-selected checkpoint is step 2,400 with loss 0.36253294884818493.

## Full evaluation

Job 51500 completed in 02:23:53 after five ten-example smoke checks. Full
evaluation used the validation-selected best adapter, never the smoke results
for selection. GSM8K uses final-number extraction; multiple-choice datasets
use conditional likelihood of option labels with no length normalization.
Dataset revisions and scoring implementation are recorded in
`configs/math10k_eval_datasets.json` and `src/kl_anchors/evaluate_math10k.py`.

| Dataset and split | Correct | Total | Accuracy | Parse failures |
|---|---:|---:|---:|---:|
| GSM8K test | 653 | 1,319 | 49.5072% | 8 |
| ARC-Challenge test | 721 | 1,172 | 61.5188% | 0 |
| ARC-Easy test | 1,878 | 2,376 | 79.0404% | 0 |
| OpenBookQA test | 295 | 500 | 59.0000% | 0 |
| Social IQA validation | 1,205 | 1,954 | 61.6684% | 0 |

Raw results (7,321 item records) and aggregate configuration snapshots are
preserved locally under
`results/eval/math10k-diverse-fullpool-02-best-all-full-01/`.
GSM8K parse failures count as incorrect. ARC-Easy is supplemental to the
earlier four-benchmark Math10K suite. These are single-seed results under the
study-specific evaluator, not necessarily the standard benchmark harness.

## Reproduction

Run from the repository root with the locked environment. Configurations
record absolute paths on the original machine; copy/rebase the paths for
another checkout and retain the original run snapshots. Restore/download the
datasets following `data/README.md` before submitting. All jobs cap at six hours.

```bash
sbatch scripts/slurm_math10k_diversity_eligible.sbatch
# Wait for successful prompt selection and teacher-cache generation.
sbatch scripts/slurm_math10k_diverse_fullpool_restart.sbatch
# Wait for successful training.
sbatch scripts/slurm_math10k_diverse_fullpool_eval.sbatch
```

Output paths are exclusive. Use new artifact/output/experiment IDs if the
defaults already exist. The earlier `fullpool-01` and part-2-only experiment
remain historical and must not be substituted for `fullpool-02`.
