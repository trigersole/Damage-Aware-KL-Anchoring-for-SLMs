# Reviewed anchor curation

This optional workflow is available for team use; its
embedding model, thresholds, long-text policy and final pool are **not approved
experimental choices**. Record agreed choices in `DECISIONS.md` before final
anchored runs. The existing automatic preparation commands remain available for
reproducing earlier exploratory outputs.

## What the change does

- Reconstructs every candidate from the pinned source, including prompts removed
  by earlier filtering. Restoring false positives therefore does not depend on
  editing the old `eligible_anchors.jsonl` or inventing missing source text.
- Preserves exact benchmark-overlap, empty-input, exact-duplicate and configured
  quality exclusions. Approximate n-gram and cosine matches become review items.
- Keeps GSM8K validation/test and every configured retention evaluation split
  in the decontamination set. A separate required flag chooses whether to also
  screen against the remaining GSM8K training questions.
- Compares candidates with other candidates using a separate cosine threshold.
  Reviewers distinguish genuinely equivalent requests from related topics.
- Requires every proposed pair and requested spot check to have an explicit
  decision, named reviewer and reason. Multiple matches on one candidate must
  all be resolved; clearing one does not clear the others.
- Removes a semantic duplicate only if it was judged directly equivalent to an
  already-retained representative. Representatives follow sorted stable IDs,
  independently of vulnerability scores. A~B and B~C do not imply A~C.
- Freezes one common pool in the existing `example_id` / `prompt` JSONL schema.
  Downstream teacher caching, diversity ranking and vulnerability scoring can
  consume it without changes to the training or evaluation code.
- With `propose --export-auto-included`, writes an inspection preview containing
  candidates that passed automatic screening and have no pending flags. Both
  participants in duplicate pairs and candidates awaiting spot checks are omitted.

## Review rubric to agree at the meeting

For benchmark matches, use `exclude` for the same substantive question, a close
paraphrase or confirmed substantial copied evaluation content. Use `keep` for
generic wording and ordinary topic overlap. Check both instruction and context.
Do not use answer labels, benchmark predictions, correctness or vulnerability
scores to decide what stays. Similarity is evidence for review, not proof of
contamination. Freeze the rule independently of the baseline scores already seen.

For candidate pairs, use `equivalent` only when the same task is requested on
the same substantive content. Use `distinct` for different tasks or contexts,
even if their embeddings are close. The selected representative may differ from
the first item a reviewer sees; stable ID order is the recorded rule.

For `spot_check`, inspect the unflagged candidate for problems missed by the
earlier rules. Use `keep` or `exclude` with a reason. These checks sample accepted
material; they do not prove the absence of missed benchmark paraphrases. Inspect
near-threshold matches in `semantic_summary.jsonl` as an additional audit. If the
rule/model/threshold needs revision, create a new proposal and review it again.

LLMs can suggest decisions, but this draft does not call an external service or
automatically approve their suggestions. Record human reviewers and any LLM
assistance in the reasons or the team's review notes. For disputed pairs, agree
one recorded decision rather than combining contradictory answer files.

## Commands after the team agrees the settings

Run from the repository root after the draft is adopted, or from the isolated
`proposal/` directory with the existing project's Python executable. Set
`PYTHONPATH=src` so the selected copy is actually imported. Do not run `uv sync`
against the repository's CUDA lock on a Mac; use the already-working native MPS
environment. The new modules use existing NumPy / sentence-transformers / datasets
dependencies and introduce no additional packages.

The variables below are placeholders for team-approved values, not suggested
experimental choices. `PROJECT_PYTHON` is the absolute path to the working
`.venv/bin/python`; `INPUTS`, `REVIEW` and `FROZEN` must be new output directories.
`TARGET_TRAIN_OVERLAP` is `allow` or `exclude`, explicitly. `allow` preserves the
old heldout-only split scope; configured target-like filters still apply.

```bash
PYTHONPATH=src "$PROJECT_PYTHON" -m kl_anchors.curate_anchors export \
  --data-config "$DATA_CONFIG" \
  --target-train-overlap "$TARGET_TRAIN_OVERLAP" \
  --output-dir "$INPUTS"

PYTHONPATH=src "$PROJECT_PYTHON" -m kl_anchors.curate_anchors propose \
  --inputs "$INPUTS" --output-dir "$REVIEW" \
  --model-id "$EMBEDDING_MODEL" --model-revision "$EMBEDDING_REVISION" \
  --heldout-threshold "$OVERLAP_THRESHOLD" \
  --dedup-threshold "$DEDUP_THRESHOLD" \
  --max-seq-length "$EMBEDDING_MAX_LENGTH" \
  --truncation "$TRUNCATION_POLICY" --device "$DEVICE" \
  --review-sample-size "$AUDIT_SAMPLE_COUNT" --review-sample-seed "$AUDIT_SEED"
```

`DEVICE` can be `cpu`, `mps` or an appropriate CUDA device supported by the chosen
encoder. The immutable revision is required. The encoder is used symmetrically
with an empty instruction prefix; choose an encoder suitable for this use.

`TRUNCATION_POLICY=reject` stops if the encoder cannot read any complete input.
`allow` explicitly permits truncation and records every affected candidate or
benchmark ID with its original token count. Do not choose `allow` merely to get
past an error: a truncated passage could hide overlap or lose an instruction at
its end. Agree an appropriate encoder/length policy first. This draft does not
implement chunked long-document comparison. It never increases the encoder's
declared context limit automatically.

To preview lexical flags without selecting an embedding model:

```bash
PYTHONPATH=src "$PROJECT_PYTHON" -m kl_anchors.curate_anchors propose \
  --inputs "$INPUTS" --output-dir "$LEXICAL_PREVIEW" --lexical-only \
  --review-sample-size "$AUDIT_SAMPLE_COUNT" --review-sample-seed "$AUDIT_SEED"
```

A lexical-only preview cannot be finalized. Its answers cannot be copied into a
different semantic proposal because the review-bundle checksum changes.

### Optional automatic-pass preview

Add `--export-auto-included` to the full semantic `propose` command to write
`auto_included_anchors.jsonl`. Its schema is `example_id` / `prompt`, sorted by
stable ID. The review manifest records its count and checksum. Without the flag,
the existing output set is unchanged. The flag is rejected with `--lexical-only`
because semantic screening has not run.

This file is for inspection before review finishes. It is not an approved final
pool. Review flags and decisions are unchanged; `finalize` still requires every
flag to be resolved and verifies the preview checksum when present. The final
pool can also include candidates restored or cleared during review. Continue to
use only the finalized `eligible_anchors.jsonl` for teacher caching and training.

See [the PowerShell walkthrough](kl_filtering_new.md) for the stage commands and
file inventory. No numerical threshold or embedding model is selected by it.

## Review and freeze

Read `review_flags.jsonl` and copy `review_answers.template.jsonl` to a separate
`review_answers.jsonl`. Only edit `decision`, `reviewer` and `reason`. Keep the
`review_id` and `bundle_sha256` exactly as generated. Flags include both prompt
texts and the evidence; templates begin with `decision: "pending"`.

Example answer shape (identifiers here are placeholders):

```json
{
  "review_id": "COPY_FROM_TEMPLATE",
  "bundle_sha256": "COPY_FROM_TEMPLATE",
  "decision": "keep",
  "reviewer": "Team member name",
  "reason": "Only a generic question phrase overlaps; the requests and contexts differ."
}
```

Then freeze into a new directory:

```bash
PYTHONPATH=src "$PROJECT_PYTHON" -m kl_anchors.curate_anchors finalize \
  --review-dir "$REVIEW" --answers "$REVIEW/review_answers.jsonl" \
  --output-dir "$FROZEN"
```

The command refuses pending/missing/duplicate/stale answers, modified proposal
files, changed curation code and existing output directories. Exact and quality
exclusions have no override in this command; revise their configured policy and
regenerate a proposal if the team changes them. Never hand-edit generated pool
files to bypass an exclusion.

Use `$FROZEN/eligible_anchors.jsonl` for `datasets.anchors.eligible_ids_manifest`
and the anchor input to teacher caching. Build all rankings and scored pools from
that same file; retain its checksum in all protocols. Rebuild these downstream
artifacts whenever the eligible pool changes. Do not use an old teacher cache or
ranked pool that omits restored candidates or includes removed candidates.

Keep the input export, review proposal, answers and frozen directory together for
audit. The final manifest references the review directory and carries checksums;
it does not duplicate all benchmark inputs into the frozen pool. Generated data
and review text belong in ignored data/artifact storage, not in Git.

## Scope and limitations

- The workflow supports arbitrary task inputs through the pure functions in
  `curation.py`; the source-export adapter follows this project's four configured
  benchmark schemas. Another benchmark needs an input-only renderer, not a
  change to the review or deduplication algorithm.
- Existing length, target-like and unsafe regex policies are inherited exactly.
  They may themselves have false positives; their broader redesign is outside
  this draft. The old `review_completed` boolean does not approve new pair reviews.
- Whole-input embeddings can miss local overlap in long contexts. Recorded
  truncation, a sampled audit and manual pair review reduce ambiguity but cannot
  certify that every paraphrase has been detected.
- Similarity computation uses row blocks rather than a full candidate-by-candidate
  matrix. Runtime remains quadratic for candidate deduplication, and a permissive
  threshold can produce many review records. Inspect counts before assigning work.
- Semantic deduplication changes the candidate population. It is not a prerequisite
  for every valid study; use the same agreed version for every anchored condition.
- No KL formulation, training schedule, benchmark scorer, target dataset or anchor
  budget changes are included. Existing base and task-only LoRA results remain
  reusable when those settings stay matched.
- The team must still agree the encoder/revision, both thresholds, long-text
  policy, training-overlap scope, review rubric and audit sample size/seed.
  Calibrate duplicate decisions on independent candidate examples or development
  material, not by optimizing final benchmark accuracy.

## Verification

```bash
PYTHONPATH=src "$PROJECT_PYTHON" -B -m unittest discover -s tests -p 'test_curat*.py' -v
PYTHONPATH=src "$PROJECT_PYTHON" -B -m unittest discover -s tests -v
```

The tests include a generic-phrase false positive, multiple benchmark matches,
exact exclusions, non-transitive semantic deduplication, changed labels, explicit
training-overlap scope, long-input handling, checksums and a full synthetic
export/review/freeze round trip. Synthetic embeddings do not validate a real
encoder's semantic accuracy; that remains a separate data review after agreement.
