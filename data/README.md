# Data directory

Do not commit downloaded datasets or model checkpoints here.

For every dataset used, record:

- canonical dataset name and source URL;
- licence and permitted use;
- exact revision or download date;
- splits used for training, validation, anchor selection and final evaluation;
- preprocessing and prompt templates;
- excluded example IDs and the reason for exclusion;
- duplicate and semantic-overlap checks; and
- the command or script that reproduces the processed data.

The candidate-anchor pool must remain disjoint from GSM8K validation/test data and from the BoolQ, HellaSwag and ARC-Easy evaluation examples.

The proposed [reviewed curation workflow](../docs/ANCHOR_CURATION_REVIEW.md) adds
explicit decisions for approximate matches and semantic deduplication within the
candidate pool. Its source export includes raw candidates so false positives can
be restored without editing an old eligible pool. Preserve the generated input,
review and final manifests in ignored storage. All anchored conditions must use
the same frozen output. The final policy remains subject to team agreement.

## Configured sources and licences

- [GSM8K](https://huggingface.co/datasets/openai/gsm8k): MIT; use `main/train` for target training plus a fixed validation carve-out, and `main/test` only for final evaluation.
- [BoolQ](https://huggingface.co/datasets/google/boolq): CC BY-SA 3.0; use a pinned split for retention evaluation.
- [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag): MIT; its validation split contains locally auditable labels.
- [ARC-Easy](https://huggingface.co/datasets/allenai/ai2_arc): check the dataset card and original terms before the final run; pin the `ARC-Easy` configuration and split.
- [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k): CC BY-SA 3.0; a possible, **not yet approved**, general instruction pool. If selected, retain attribution and screen its prompts and contexts for benchmark overlap.

Use `python -m kl_anchors.prepare_data --config configs/data.<study>.json --output-dir data/processed/<id>` after pinning full source revisions and filters. It saves stable source IDs, GSM8K train/validation/test manifests, surface-overlap exclusions, and the first eligible anchor pool. Follow with `python -m kl_anchors.semantic_audit` using an explicitly pinned embedding model and threshold. The final eligible pool should be its output. Benchmark labels are not used to rank or select anchors.

## Study-02 target split

`configs/data.study-02.json` freezes the approved seed-42 GSM8K split with 300 validation examples and 7,173 training examples. Reproduce it from the pinned dataset revisions with:

```bash
uv run --locked python -m kl_anchors.prepare_data \
  --config configs/data.study-02.json \
  --output-dir data/processed/study-02
```

The split uses the same SHA-256 ranking rule and seed as the earlier 100-example carve-out, so those original validation IDs are a subset of the new validation set. The target-training and validation manifests have zero ID overlap. `configs/protocol.study-02.json` records the corresponding five-epoch, `2e-5` task-adaptation settings.

## NTU MATH-10K KL study inputs

The [NTU source snapshots](../experiments/ntu_math10k_kl/) use the
9,919-row MATH-10K source from [LLM-Adapters at revision fe675038](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/fe675038fb60d61b3fc03d98673d6d11d2bef4f9/ft-training_set/math_10k.json).
The source SHA-256 is
`0342d0d860ad8592b579329337c90e42eefd3d9f2898043140cbd120630418b8`.

Download it as `data/raw/math10k/math_10k.json`. From a fresh checkout, run
`PYTHONPATH=src python -m kl_anchors.prepare_math10k`. That creates the original
seed-42 9,619/300 split in `data/processed/math10k/`. For each package that
requires them, copy `train.jsonl` to `inputs/train.jsonl` and
`validation.jsonl` to `inputs/validation_original_300.jsonl`. Check both hashes
against the package's `files.sha256.json`. This reproduces the original
validation overlap.

The [upstream data license](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/DATA_LICENSE)
covers the compiled database and leaves individual item rights separate.
The snapshot packages record the model and anchor sources and their licenses.
Keep dataset files, caches and weights outside Git.
