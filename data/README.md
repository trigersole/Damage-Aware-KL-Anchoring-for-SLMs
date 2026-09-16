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

## Configured sources and licences

- [GSM8K](https://huggingface.co/datasets/openai/gsm8k): MIT; use `main/train` for target training plus a fixed validation carve-out, and `main/test` only for final evaluation.
- [BoolQ](https://huggingface.co/datasets/google/boolq): CC BY-SA 3.0; use a pinned split for retention evaluation.
- [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag): MIT; its validation split contains locally auditable labels.
- [ARC-Easy](https://huggingface.co/datasets/allenai/ai2_arc): check the dataset card and original terms before the final run; pin the `ARC-Easy` configuration and split.
- [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k): CC BY-SA 3.0; a possible, **not yet approved**, general instruction pool. If selected, retain attribution and screen its prompts and contexts for benchmark overlap.

Use `python -m kl_anchors.prepare_data --config configs/data.<study>.json --output-dir data/processed/<id>` after pinning full source revisions and filters. It saves stable source IDs, GSM8K train/validation/test manifests, surface-overlap exclusions, and the first eligible anchor pool. Follow with `python -m kl_anchors.semantic_audit` using an explicitly pinned embedding model and threshold. The final eligible pool should be its output. Benchmark labels are not used to rank or select anchors.
