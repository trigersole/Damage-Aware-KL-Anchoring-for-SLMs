# Data, source attribution and licenses

The exact user-supplied teammate MATH-10K train and original300 validation files
are included for the authorized internal study. Source:
https://github.com/AGI-Edgerunners/LLM-Adapters, `ft-training_set/math_10k.json`,
SHA2560342d0d860ad8592b579329337c90e42eefd3d9f2898043140cbd120630418b8.
Original prepared train SHA256adcbf9771c5ac3556a3b645293d132ad565c9033ef8d82e4c14031cd0f60b2eb;
validation SHA2562139a7c220259c1e1270e5123ef104465e0356d44ac2ab2f4ce930c0781c57ed.
Preserve source dataset provenance; repository-level licensing does not by
itself establish rights over every upstream record. No validation rows are
removed or added to training in this calibration.

The original MATH-10K loader, LoRA, tokenization, exact-epoch ordering and math
loss function bodies are credited to the teammate implementation previously
reviewed for this project. `legacy.py` is copied unchanged from that preserved
snapshot. KL reference code is the pinned project implementation and is kept
for numerical-equivalence checks. Source files and hashes are listed in the
bundle; new code implements only this isolated calibration/recovery/reporting.

The anchors are the already-screened Databricks Dolly15k instruction/context
prompts under CC BY-SA3.0, revisionbdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a:
https://huggingface.co/datasets/databricks/databricks-dolly-15k.
They are read from the existing job51061 NTU export. Their source references
are not training labels; KL uses the saved untouched model continuations.
Original source IDs, prompts and token sequences are preserved.

Model: https://huggingface.co/meta-llama/Llama-3.2-3B, pinned revision in protocol.
Meta's Llama3.2 license/access conditions apply. No model weights, tokenizer,
credentials, tokens or downloaded datasets beyond the authorized supplied math
split files are included. Existing authorized NTU caches are used offline.
Do not commit the archive, datasets, model outputs or checkpoints to Git.
