# Variant A: unrestricted damage-aware top-128 export

Approved by Hemanth on 4 November 2026: Variant A, then "will go with 128 only".
This exports membership from existing evidence; it does not start training.

## Frozen inputs

- Original friend's 300-validation MATH-10K run, discovery step600, not clean229
  or the selected step2400 model.
- Teacher: pretrained `meta-llama/Llama-3.2-3B`, revision
  `13afe5124825b4f3751f836b40dafda64c1ed062`.
- Existing pinned experiment:
  `/home/msai/hemanth005/AI6130_Srihari_Step5_20261102/experiment_256_v1`.
- Automatic source 13,769 = canonical cached/scored 13,743 + 26 original
  overlength set-asides. Original IDs, prompts, teacher tokens and scores remain.
- Descending saved full-vocabulary teacher-to-step600 response-token-mean KL;
  ascending exact example ID breaks ties. No minimum KL, response-length rule,
  topic balancing, diversity signal, or quality exclusion is added.
- Cached native EOS remains a scored response token. EOS-only, two-token and
  capped continuations can all be selected by their existing scores.

## Run on NTU

From the local Mac terminal, transfer the archive to your NTU home directory.
Then, in the NTU terminal:

```bash
cd /home/msai/hemanth005
tar -xzf AI6130_DamageA_Top128_20261104_001.tar.gz
cd AI6130_DamageA_Top128_20261104_001
mkdir -p logs
sbatch run_export.sbatch
```

The direct launcher runs the data export on an assigned compute node. The
existing environment and local tokenizer files are reused in offline mode.
GPU allocation follows the known MGPU-TC2 submission recipe, but no GPU model
inference is performed. No work is sent to the login node. The allocation limit
is ten minutes; actual duration is not yet measured.

Checks stop the export on changed manifests, code/model identities, input
checksums, incomplete membership, duplicated IDs, invalid scores, token count
errors, or a score completion receipt that fails to identify the exact cache,
score file and discovery student. The copied `source_guard.py` supplies only its
read-only guard/hash helpers; its historical model-probe functions are never
called. No remote source or existing output is patched.

## New outputs

Exclusive new directory:
`experiment_256_v1/analysis/damage_A_top128_20261104_001/`.

- `damage_A_anchors.jsonl`: authoritative 128 source prompts with exact IDs,
  ranks, existing KL values, complete saved prompt/response IDs and decoded
  cached teacher text. Selected formatted prompts are retokenized and must
  match the existing cache exactly. No text is regenerated or truncated.
- `selected_teacher_cache.jsonl`: the exact original cache records for those
  IDs; the original full-pool cache is not replaced.
- `ranked_candidates.jsonl`: complete unrestricted ranking of all 13,743 IDs.
- `damage_A_review.csv`: readable view; formula-like text receives a display
  escape only in CSV. Authoritative JSONL preserves original characters.
- `summary.json`: count, selected unique scored-response-token total, full and
  selected length histograms, EOS-only/two-token counts, KL range, source
  provenance, input/export checksums and runtime receipt.

If that directory already exists the export refuses to overwrite it. Inspect
the existing result rather than rerunning, or agree a separate new experiment ID.

## Interpretation and remaining choices

This is behavioural-vulnerability selection, not a certificate that the teacher
answers are correct or that every selected prompt represents harmful forgetting.
The diagnostic showed that raw high scores can concentrate on short instrument
responses; Variant A intentionally leaves this behaviour intact for evaluation.

The approved **128 prompts** do not establish equal anchor-token exposure across
methods. `selected_unique_response_tokens` describes this set once, including
EOS when present. Total training exposure, repetition/data order, anchor-update
frequency, KL coefficient, and the matched conditions must be agreed separately.
`training_ready` remains false, and no training protocol or launcher is created.
Future final runs initialize independently from the untouched base, not step600.

The original cache and scores are not recomputed. Full-pool logits are not
numerically audited by this data export; prior diagnostics cover selected cases.
