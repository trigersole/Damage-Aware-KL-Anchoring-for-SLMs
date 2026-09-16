# Experiment configurations

Store one versioned configuration per reproducible run or run family. Each configuration should identify the exact model revision, dataset splits, seed, LoRA settings, optimizer, training budget, anchor method, anchor-token budget, KL settings and evaluation protocol.

Do not launch final comparisons until the unresolved design choices in `../DECISIONS.md` have been approved.

Copy `data.template.json` and `protocol.template.json` to named configurations, then fill every `null` that applies to the planned condition. The data configuration pins dataset revisions, validation split and contamination rules. The shared protocol pins the model, optimization, KL, evaluation and run records. Run `python -m kl_anchors.protocol --config PATH` to inspect missing fields. Never replace a reported configuration after results exist; use a new experiment ID and a new config revision.

`protocol.template.json` records the chosen Qwen2.5 repository ID but leaves its exact commit unset. The anchor source and all open method parameters also remain unset. The `damage_diverse` selector requires an explicit `damage_weight` in `[0,1]`.
