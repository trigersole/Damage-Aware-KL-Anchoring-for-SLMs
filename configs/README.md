# Experiment configurations

Store one versioned configuration per reproducible run or run family. Each configuration should identify the exact model revision, dataset splits, seed, LoRA settings, optimizer, training budget, anchor method, anchor-token budget, KL settings and evaluation protocol.

Do not launch final comparisons until the unresolved design choices in `../DECISIONS.md` have been approved.

Copy `data.template.json` and `protocol.template.json` to named configurations, then fill every `null` that applies to the planned condition. The data configuration pins dataset revisions, validation split and contamination rules. The shared protocol pins the model, optimization, KL, evaluation and run records. Run `python -m kl_anchors.protocol --config PATH` to inspect missing fields. Never replace a reported configuration after results exist; use a new experiment ID and a new config revision.

`protocol.template.json` records the chosen Qwen2.5 repository ID but leaves its exact commit unset. The anchor source and all open method parameters also remain unset. The `damage_diverse` selector requires an explicit `damage_weight` in `[0,1]`.

## Named configurations

| File | Status | Intended use |
|---|---|---|
| `data.study-02.json` | Frozen for task-only Study-02 | Approved seed-42 GSM8K split with 300 validation and 7,173 training examples; its unused anchor-source fields do not approve the final anchored pool. |
| `protocol.study-02.json` | Completed task-only protocol | Five-epoch LoRA at `2e-5`; used by `study-02-seed0-lora`. Preserve it unchanged. |
| `data.exploratory-01.json` | Historical | Earlier 100-example validation split. Do not combine its runs with Study-02. |
| `protocol.exploratory-01.json` | Historical smoke/exploratory | Short implementation trial only. |
| `protocol.study-01.json` and `protocol.study-01-aligned.json` | Historical | Earlier one- and five-epoch task-only protocols; not interchangeable with Study-02. |

The exact protocol copied into a completed run directory is the authoritative record for that run. Do not edit a named configuration after it has produced results; create the next study/configuration ID instead.

`protocol.study-02.json` is intentionally usable only for task-only `MODE=lora`. Its anchor budget, KL coefficient and mixing fields are zero and its scored-pool paths are unset. The remaining anchored-method choices in `../DECISIONS.md` must be approved before creating a separate anchored-study protocol. Do not insert convenient placeholder values merely to make `STAGE=all` run.

Validate a protocol before submission:

```bash
uv run --locked --no-sync python -m kl_anchors.protocol \
  --config configs/protocol.study-02.json
```

See [`../scripts/README.md`](../scripts/README.md) for the Slurm variables and current submission commands.
