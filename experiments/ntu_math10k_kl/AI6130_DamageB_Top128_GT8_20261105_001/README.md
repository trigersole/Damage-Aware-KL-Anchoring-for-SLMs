# Approved Variant B selection: cached responses longer than eight tokens

This package selects anchors only. It performs **no model training, teacher
generation, new KL scoring or evaluation**. Hemanth approved this exact
selection rule on 5 November 2026; full B training is a later decision.

## Fixed procedure

1. Verify the original Step5 source commit, configuration, discovery step600
   adapter, input manifest, complete teacher cache, skipped rows and KL-score
   completion metadata. Require the existing 13,743 scored/cache IDs to match.
2. Keep only cached teacher responses with `response_token_count > 8`:
   at least nine response positions, **including EOS when generated**. This
   measures response length, not Dolly question length. Do not remove EOS,
   regenerate responses or recompute scores.
3. Rank survivors by the unchanged descending KL score, with original example
   ID ascending for exact ties. Select **128 whole prompts**. Stop if too few
   qualify; no fallback cutoff, quality filter or count change is allowed.
4. Verify the completed original A selection and compare its IDs with B.
   Preserve the original A/source files. Write B into a separate result folder.

The original source has 13,769 prompts and 26 previously excluded overlength
inputs. Those exclusions stay unchanged. B's new eligible count, selected token
total and score range can only be reported after the actual NTU export.
No benchmark labels, answers, predictions or forgetting gaps enter selection.

## Submit on NTU

Extract this separate source-only archive in `/home/msai/hemanth005`. From its
folder:

```bash
mkdir -p logs
sbatch run_export.sbatch
```

The job uses the existing pinned Step5 Python environment, one allocated GPU,
one CPU, 4G host memory and a ten-minute ceiling. It performs data processing
and loads the already cached tokenizer only. Offline flags prevent downloads.
No Llama model is loaded. Do not run computation on the head node.

If successfully completed, repeating the command verifies and reuses the
completed B export without changing it. A mismatched or corrupt existing
result stops for review. Output is committed from a temporary folder only
after all source-preservation checks pass; concurrent exports are locked out.

Inspect startup/completion, replacing `JOB_ID` with the submitted number:

```bash
squeue -j JOB_ID
tail -n 30 logs/damage-B-JOB_ID.out
tail -n 15 logs/damage-B-JOB_ID.err
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
```

## Files and interpretation

Result directory:
`results/damage-B-top128-gt8-20261105-001/`

- `damage_B_anchors.jsonl`: authoritative 128-prompt selection, original text,
  token IDs, KL/rank metadata and decoded cached teacher text.
- `selected_teacher_cache.jsonl`: byte-value-equivalent subset of the original
  cached records, preserving response IDs and EOS.
- `ranked_original_candidates.jsonl`: original 13,743 KL ranking.
- `ranked_eligible_candidates.jsonl`: complete response-length-filtered ranking.
- `excluded_short_responses.jsonl`: IDs, lengths and explicit exclusion reason.
- `eligible_not_selected.jsonl`: remaining valid candidates below rank128.
- `original_overlength.jsonl`: unchanged original 26 skipped records.
- `comparison_with_A.json`: shared/A-only/B-only IDs.
- `damage_B_review.csv`: readable review view; formula-like strings are escaped
  here only. JSONL retains the authoritative unchanged text.
- `summary.json`: actual counts, length/token distributions, KL limits, original
  checksums, provenance and `training_ready: false`.
- `completion.json`: frozen source/config identity and hashes for all outputs.

Inspect the summary with a lightweight file read:

```bash
cat results/damage-B-top128-gt8-20261105-001/summary.json
```

`training_ready: false` is intentional: this approval covers selection only.
128 prompts is not a matched token-exposure training budget. Proposed later
B training must start freshly from the untouched base and match A's settings,
including lambda0.01 and 64 scored positions every four updates. Never resume
A's already anchored optimizer/model state into B. A separate longer-anchor
hardware/recovery check and concrete training approval are still required.

## Assumptions, provenance and terms

This is an exploratory response-length ablation. The cutoff is an approved
experimental choice, not a proved optimum. Longer output is not automatically
correct or helpful. High KL is behavioural drift, not verified harmful damage.
The length idea predates A's final scores; the precise cutoff was approved
afterward. Preserve/report this chronology and all outcomes. Do not sweep
cutoffs or choose variants from final-test performance.

Standard-library validation functions were copied unchanged from the successful
Version A exporter; `snapshot_provenance.json` records parent/function hashes.
`expected_source_manifest.json` pins the same original data/model/source inputs.
Existing A, calibration, full-training and evaluation packages stay unchanged.

Original Dolly source: `databricks/databricks-dolly-15k`, under its CC BY-SA3.0
terms; retain original attribution and the upstream/source manifests. Llama3.2
tokenizer/model terms still apply, using the user's authorized cached files.
The transfer archive includes no downloaded source dataset, vector/cache,
adapter/base weights or credentials. Do not commit generated exports/data.
