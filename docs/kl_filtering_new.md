# Anchor filtering: PowerShell walkthrough

Adapted from the team's shared `kl_filtering_new.md`, with duplicate sections
removed and the optional automatic-pass preview added.

The workflow is **export → propose → human review → finalize**. Run commands from
the repository root using its configured Python environment. This guide assumes
the checkout contains `src/kl_anchors/curate_anchors.py`; no separate patch is
needed once these changes are in your branch. Existing NumPy, datasets and
sentence-transformers dependencies are sufficient.

On NTU, run data preparation and semantic computation inside an allocated Slurm
job, not on the login/head node. The commands below show the Python stages; they
do not request a GPU allocation.

## 1. Record the team's settings

Set these PowerShell environment variables before using the commands:

| Variable | Value to supply |
|---|---|
| `DATA_CONFIG` | Data configuration path with pinned source revisions and reviewed quality filters |
| `TARGET_TRAIN_OVERLAP` | Explicit `allow` or `exclude` for screening against GSM8K training questions |
| `EMBEDDING_MODEL` | Agreed sentence embedding model ID |
| `EMBEDDING_REVISION` | Its immutable 40-character commit |
| `OVERLAP_THRESHOLD` | Agreed candidate-versus-evaluation cosine threshold |
| `DEDUP_THRESHOLD` | Separate agreed candidate-versus-candidate threshold |
| `EMBEDDING_MAX_LENGTH` | Token length supported by the chosen encoder |
| `TRUNCATION_POLICY` | Explicit `reject` or `allow`; rejected inputs must be addressed rather than silently truncated |
| `DEVICE` | Available `cpu`, `mps` or CUDA device supported by the encoder |
| `AUDIT_SAMPLE_COUNT` | Agreed number of unflagged candidates to review |
| `AUDIT_SEED` | Recorded spot-check seed |

For example, assigning a path uses `$env:DATA_CONFIG = "configs/YOUR_DATA_CONFIG.json"`.
These are parameters to agree and record in `DECISIONS.md`, not numerical
recommendations. Candidate sources are selected by the data configuration; Dolly
is used only when configured there. This implementation's export adapter renders
GSM8K, BoolQ, HellaSwag and ARC-Easy inputs. Additional evaluation datasets require
their own input renderers before they can be included in a new study's audit.

`exclude` additionally screens against GSM8K training questions. `allow` leaves
that additional comparison out; GSM8K validation/test and the configured retention
evaluation sets remain in the audit in both modes. Neither mode removes whole
topics merely because they appear in a benchmark. Gold answer labels are not
exported or used for filtering.

## 2. Export the original candidate source

```powershell
$env:PYTHONPATH = "src"

python -m kl_anchors.curate_anchors export `
  --data-config "$env:DATA_CONFIG" `
  --target-train-overlap "$env:TARGET_TRAIN_OVERLAP" `
  --output-dir results/curation_inputs
```

This exports the original source, including candidates removed by an earlier
automatic pipeline, together with benchmark inputs and reproducibility metadata.

## 3. Inspect lexical flags before downloading an encoder

```powershell
python -m kl_anchors.curate_anchors propose `
  --inputs results/curation_inputs `
  --output-dir results/lexical_preview `
  --lexical-only `
  --review-sample-size "$env:AUDIT_SAMPLE_COUNT" `
  --review-sample-seed "$env:AUDIT_SEED"
```

Inspect `hard_exclusions.jsonl` and `review_flags.jsonl`. This preview cannot be
finalized and cannot use `--export-auto-included`; semantic checks have not run.
Do not reuse its review answers in a different semantic proposal.

## 4. Run semantic screening and deduplication

```powershell
python -m kl_anchors.curate_anchors propose `
  --inputs results/curation_inputs `
  --output-dir results/semantic_review `
  --model-id "$env:EMBEDDING_MODEL" `
  --model-revision "$env:EMBEDDING_REVISION" `
  --heldout-threshold "$env:OVERLAP_THRESHOLD" `
  --dedup-threshold "$env:DEDUP_THRESHOLD" `
  --max-seq-length "$env:EMBEDDING_MAX_LENGTH" `
  --truncation "$env:TRUNCATION_POLICY" `
  --device "$env:DEVICE" `
  --review-sample-size "$env:AUDIT_SAMPLE_COUNT" `
  --review-sample-seed "$env:AUDIT_SEED" `
  --export-auto-included
```

Omit the last flag if you do not need the additional preview. With it,
`auto_included_anchors.jsonl` contains only candidates without hard exclusions or
pending overlap, duplicate-pair or spot-check flags. Both members of a suspected
duplicate pair are withheld. The manifest records the preview count and hash.

This is an inspection snapshot, not a pool ready for training. It does not skip
review or change which candidates can enter the finalized pool.

## 5. Complete review decisions

```powershell
Copy-Item `
  results/semantic_review/review_answers.template.jsonl `
  results/semantic_review/review_answers.jsonl
```

Read the paired texts and evidence in `review_flags.jsonl`. In the answer file,
edit only `decision`, `reviewer` and `reason`; preserve every generated review ID
and bundle checksum.

| Flag | Decisions |
|---|---|
| Benchmark overlap | `keep` / `exclude` |
| Candidate semantic duplicate | `equivalent` / `distinct` |
| Random spot check | `keep` / `exclude` |

Resolve every flag. Keep generic wording/topic matches; exclude confirmed copied
evaluation questions or close paraphrases. Treat genuinely different requests as
distinct. A model may assist reviewers, but this workflow does not call a service
or approve its suggestions automatically. Document any assistance.

## 6. Finalize one common pool

```powershell
python -m kl_anchors.curate_anchors finalize `
  --review-dir results/semantic_review `
  --answers results/semantic_review/review_answers.jsonl `
  --output-dir results/frozen_anchors
```

Use `results/frozen_anchors/eligible_anchors.jsonl` for subsequent teacher caching
and every anchored condition. Rebuild caches and rankings if the pool changes.
Finalization still requires all decisions and verifies the preview hash when
present. It can add manually cleared candidates beyond the automatic preview.

Every output directory must be new. Preserve old inputs, review bundles and
results. If source, policy, code or thresholds change, create a fresh review.

## File inventory

| Stage | Files |
|---|---|
| Export | `candidates.jsonl`, `heldout_inputs.jsonl`, `data_config.json`, `inputs_manifest.json` |
| Propose | Copies of candidates/config/heldout inputs; `hard_exclusions.jsonl`, `review_flags.jsonl`, `semantic_summary.jsonl`, `input_provenance.json`, `review_manifest.json`, `review_answers.template.jsonl` |
| Optional propose preview | `auto_included_anchors.jsonl` |
| Human review | `review_answers.jsonl`, copied and completed by reviewers |
| Finalize | `eligible_anchors.jsonl`, `curation_audit.jsonl`, frozen `review_answers.jsonl`, `review_manifest.json`, `curation_manifest.json` |

Automatic checks and sampled review cannot prove the absence of every semantic
overlap. Similarity thresholds need calibration independently of final benchmark
performance. No training objective, model, evaluation scorer or anchor-token
budget is changed by this workflow.

## Verify the implementation without downloads

```powershell
python -B -m unittest discover -s tests -p "test_curat*.py" -v
```

For Bash commands and the detailed review rubric, see
[ANCHOR_CURATION_REVIEW.md](ANCHOR_CURATION_REVIEW.md).
