# Run a small local check in VS Code

Open the repository folder in VS Code, then open **Terminal → New Terminal**. The terminal should be at the folder containing `pyproject.toml`, `uv.lock`, `configs/`, `src/`, and `scripts/`.

Install `uv` once using the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/). Use Python 3.11 or 3.12, then from this repository folder run:

```bash
uv python install 3.12
uv sync --locked --python 3.12
uv run --locked python scripts/local_smoke.py
```

This takes two optimizer steps on CPU with synthetic arithmetic examples and a tiny synthetic PyTorch model. It needs no GPU, Hugging Face credentials, network model download, frozen research protocol, or NTU allocation. The first `uv sync` does need package-index access to install the locked environment.

To check every training mode flag with the same tiny inputs:

```bash
uv run --locked python scripts/local_smoke.py --all-modes
```

To check one anchored mode:

```bash
uv run --locked python scripts/local_smoke.py --mode damage_kl --steps 2
```

Every invocation creates a new folder under `results/runs/local-smoke/`. The terminal prints its absolute path. Inside that folder, `protocol.smoke.json` and `data/` are the tiny inputs. Each mode gets a `runs/toy-<mode>/` folder containing `checkpoints/step_XXXXXX/`, `best_model/`, `status.json`, `training_history.jsonl`, `validation_history.jsonl`, and `loss_vs_epoch.png`. Anchored modes also save `anchor_selection.json`.

**These are code-path checks only.** The script substitutes a tiny PyTorch model for Qwen and bypasses PEFT LoRA. The saved adapter files are stubs, not reloadable weights. Its losses, adapters, and selections cannot be used in the report. For an actual small Qwen run, first approve and fill the open choices in `DECISIONS.md` and the protocol files; follow `docs/IMPLEMENTATION.md`. Keep each reported run under a unique experiment ID.
