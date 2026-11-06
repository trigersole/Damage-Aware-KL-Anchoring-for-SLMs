# Data, attribution and limitations

This private execution archive includes only the already authorized teammate MATH-10K train/original300 split files and source/configuration snapshots. Do not commit data, the archive, model weights, cached outputs or credentials to Git. The source MATH-10K file is `AGI-Edgerunners/LLM-Adapters`, `ft-training_set/math_10k.json`, SHA256 `0342d0d860ad8592b579329337c90e42eefd3d9f2898043140cbd120630418b8`. Keep upstream record provenance; a repository license is not a blanket license for every upstream dataset.

Training file SHA256 `adcbf9771c5ac3556a3b645293d132ad565c9033ef8d82e4c14031cd0f60b2eb`; validation SHA256 `2139a7c220259c1e1270e5123ef104465e0356d44ac2ab2f4ce930c0781c57ed`. The original300 validation has29 exact and41 additional normalized training-question matches, plus one extra within-validation duplicate. Its checkpoint-selection limitation is disclosed. Do not silently substitute clean229.

B anchors are read from the completed job51284 export, not bundled or regenerated. Dolly source: https://huggingface.co/datasets/databricks/databricks-dolly-15k , CC BY-SA3.0, revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`. Source IDs and original cached teacher continuations remain unchanged. Dolly reference answers are not KL labels. Selection keeps >8 cached response positions, including native EOS, then the top128 by unchanged vulnerability score. This is a length ablation; several cached answers are wrong or unhelpful. No correctness/quality guarantee or new exclusion is implied.

Model: https://huggingface.co/meta-llama/Llama-3.2-3B , pinned revision in protocol. Meta Llama3.2 access and license conditions apply. No model weights, tokenizer files or credentials are bundled. Existing authorized NTU caches and pinned environment are used.

A's computational helpers and the reviewed teammate evaluator are copied byte-for-byte with attribution in `snapshot_provenance.json`. New code is an isolated B-specific guard/driver and reporting/reuse wrapper. `a_helpers.py` is unchanged A `calibrate.py`; only its schedule/runtime/validation/recovery functions are called. Its A-specific launch/selection functions are not called. Likewise `a_evaluate.py` contributes the unchanged scorer/backend/synthetic probes, not its A-specific launcher.

Benchmark IDs, immutable revisions, splits and counts are in `reference_evaluation.json`. Social IQa uses the official labelled validation split, not an unlabelled test. No final-test examples or outcomes select B anchors or tune the model. Surface/semantic screening does not establish absence of unknown pretraining exposure. Do not claim that distribution drift locates damaged parameters or knowledge.

Single seed42, mixed prompt/response structure, and unique-coverage differences limit inference. Equal total scored exposure does not equate prompt repetitions or EOS shares. This A/B comparison does not replace matched random/diversity conditions or multi-seed evidence.
