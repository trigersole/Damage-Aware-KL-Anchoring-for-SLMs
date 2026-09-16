"""Build auditable learning-versus-forgetting summaries from completed runs.

Reads each run's saved item records and aggregate metrics. Rejects comparisons
with different benchmark prompts, scoring rules, model revisions, or item IDs.
The resulting plots show the selected benchmark outcomes only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import benchmark_forgetting, target_gain


def _load_run(path: Path) -> tuple[dict, dict[str, dict], Path, Path]:
    if (path / "aggregate.json").is_file():
        evaluation_dir = path
        run_dir = path.parent.parent if path.parent.name == "evaluations" else path.parent
    else:
        run_dir = path
        evaluation_dir = run_dir / "evaluation"
    aggregate = json.loads((evaluation_dir / "aggregate.json").read_text(encoding="utf-8"))
    with (evaluation_dir / "items.jsonl").open(encoding="utf-8") as stream:
        items = [json.loads(line) for line in stream if line.strip()]
    by_key = {(row["benchmark"], row["item_id"]): row for row in items}
    if len(by_key) != len(items):
        raise ValueError(f"duplicate benchmark/item ID in {run_dir}")
    return aggregate, by_key, run_dir, evaluation_dir


def _validation_accuracy(run_dir: Path, evaluation_dir: Path) -> tuple[int | None, float | None]:
    status_path = run_dir / "status.json"
    history_path = run_dir / "validation_history.jsonl"
    if not status_path.is_file() or not history_path.is_file():
        return None, None
    if evaluation_dir.name.startswith("step_"):
        step = int(evaluation_dir.name.removeprefix("step_"))
    else:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        step = int(Path(status["best_checkpoint"]).name.removeprefix("step_"))
    with history_path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    for record in records:
        if record["step"] == step:
            return step, record.get("gsm8k_validation_accuracy")
    raise ValueError(f"evaluation checkpoint step {step} is absent from validation history: {run_dir}")


def compare_runs(base_dir: Path, adapted_dirs: list[Path], *, match_tolerance: float | None = None) -> dict:
    if not adapted_dirs:
        raise ValueError("at least one adapted run is required")
    if match_tolerance is not None and not 0 <= match_tolerance <= 1:
        raise ValueError("match_tolerance must be a fraction in [0, 1]")
    base, base_items, _, _ = _load_run(base_dir)
    reference = base["protocol"]
    benchmarks = {name: row["accuracy"] for name, row in base["metrics"].items()}
    retention_names = ["boolq", "hellaswag", "arc_easy"]
    if set(benchmarks) != {"gsm8k", *retention_names}:
        raise ValueError("base metrics must cover GSM8K and the three retention benchmarks")
    entries = []
    reference_study = None
    anchored_pool_hash = None
    anchored_budget = None
    anchored_cache_hash = None
    for path in adapted_dirs:
        aggregate, items, run_dir, evaluation_dir = _load_run(path)
        current = aggregate["protocol"]
        if current["benchmarks"] != reference["benchmarks"] or current["model"]["repo_id"] != reference["model"]["repo_id"] or current["model"]["revision"] != reference["model"]["revision"]:
            raise ValueError(f"evaluation protocol differs from base: {run_dir}")
        if items.keys() != base_items.keys():
            raise ValueError(f"evaluated item IDs differ from base: {run_dir}")
        study = current.get("source_study_protocol")
        if not isinstance(study, dict) or study.get("condition") != "final":
            raise ValueError(f"adapted evaluation lacks a final-run study protocol: {run_dir}")
        if reference_study is None:
            reference_study = study
        else:
            if study["lora"] != reference_study["lora"]:
                raise ValueError(f"LoRA settings differ: {run_dir}")
            for key in ("training_ids_manifest", "optimizer", "learning_rate", "scheduler", "warmup_steps", "batch_policy", "per_device_batch_size", "gradient_accumulation_steps", "max_steps"):
                if study["training"][key] != reference_study["training"][key]:
                    raise ValueError(f"target training setting {key} differs: {run_dir}")
        if study["anchor_selection"]["method"] != "none":
            selection = json.loads((run_dir / "anchor_selection.json").read_text(encoding="utf-8"))
            if anchored_pool_hash is None:
                anchored_pool_hash = selection["candidate_pool_file_sha256"]
                anchored_budget = selection["anchor_token_budget"]
                anchored_cache_hash = selection["teacher_cache_file_sha256"]
            elif selection["candidate_pool_file_sha256"] != anchored_pool_hash or selection["anchor_token_budget"] != anchored_budget or selection["teacher_cache_file_sha256"] != anchored_cache_hash:
                raise ValueError(f"anchored runs have different eligible pools, teacher caches or token budgets: {run_dir}")
        metrics = {name: row["accuracy"] for name, row in aggregate["metrics"].items()}
        checkpoint_step, validation_accuracy = _validation_accuracy(run_dir, evaluation_dir)
        forgetting = benchmark_forgetting(
            {name: benchmarks[name] for name in retention_names},
            {name: metrics[name] for name in retention_names},
        )
        transitions = {}
        for name in retention_names:
            keys = [key for key in base_items if key[0] == name]
            transitions[name] = {
                "lost": sum(base_items[key]["correct"] and not items[key]["correct"] for key in keys),
                "gained": sum(not base_items[key]["correct"] and items[key]["correct"] for key in keys),
                "same": sum(base_items[key]["correct"] == items[key]["correct"] for key in keys),
            }
        entries.append({
            "run_directory": str(run_dir.resolve()),
            "evaluation_directory": str(evaluation_dir.resolve()),
            "experiment_id": aggregate["experiment_id"],
            "method": study["anchor_selection"]["method"],
            "seed": study["seed"],
            "checkpoint_step": checkpoint_step,
            "gsm8k_validation_accuracy": validation_accuracy,
            "metrics": metrics,
            "target_gain": target_gain(benchmarks["gsm8k"], metrics["gsm8k"]),
            "forgetting": forgetting,
            "mean_retention_forgetting": sum(forgetting.values()) / len(forgetting),
            "item_transitions": transitions,
        })
    matched = []
    if match_tolerance is not None:
        task_only = [entry for entry in entries if entry["method"] == "none" and entry["gsm8k_validation_accuracy"] is not None]
        for method, seed in sorted({(entry["method"], entry["seed"]) for entry in entries if entry["method"] != "none"}):
            candidates = []
            for baseline in task_only:
                for anchored in entries:
                    if anchored["method"] != method or anchored["seed"] != seed or baseline["seed"] != seed or anchored["gsm8k_validation_accuracy"] is None:
                        continue
                    gap = abs(anchored["gsm8k_validation_accuracy"] - baseline["gsm8k_validation_accuracy"])
                    if gap <= match_tolerance:
                        candidates.append((gap, baseline["checkpoint_step"] or 0, anchored["checkpoint_step"] or 0, baseline, anchored))
            if candidates:
                gap, _, _, baseline, anchored = min(candidates, key=lambda item: item[:3])
                matched.append({
                    "method": method,
                    "seed": baseline["seed"],
                    "task_only_evaluation": baseline["evaluation_directory"],
                    "anchored_evaluation": anchored["evaluation_directory"],
                    "gsm8k_validation_accuracy_gap": gap,
                    "retention_benefit_vs_task_only": {
                        name: baseline["forgetting"][name] - anchored["forgetting"][name]
                        for name in retention_names
                    },
                })
    return {
        "base_run_directory": str(base_dir.resolve()),
        "base_experiment_id": base["experiment_id"],
        "base_metrics": benchmarks,
        "adapted": entries,
        "match_tolerance": match_tolerance,
        "matched_comparisons": matched,
    }


def write_analysis(summary: dict, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    entries = summary["adapted"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for entry in entries:
        ax.scatter(entry["metrics"]["gsm8k"] * 100, entry["mean_retention_forgetting"] * 100, s=70)
        ax.annotate(entry["experiment_id"], (entry["metrics"]["gsm8k"] * 100, entry["mean_retention_forgetting"] * 100), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("GSM8K accuracy (%)")
    ax.set_ylabel("Mean selected-benchmark forgetting (percentage points)")
    ax.set_title("Target learning versus selected-benchmark forgetting")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "target_retention.png", dpi=180)
    plt.close(fig)

    names = ["boolq", "hellaswag", "arc_easy"]
    width = 0.8 / len(entries)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, entry in enumerate(entries):
        positions = [j + (i - (len(entries) - 1) / 2) * width for j in range(len(names))]
        ax.bar(positions, [entry["forgetting"][name] * 100 for name in names], width=width, label=entry["experiment_id"])
    ax.set_xticks(range(len(names)), labels=["BoolQ", "HellaSwag", "ARC-Easy"])
    ax.set_ylabel("Base minus adapted accuracy (percentage points)")
    ax.set_title("Forgetting by retention benchmark")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "forgetting_by_benchmark.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", required=True, type=Path)
    parser.add_argument("--adapted-run", required=True, type=Path, action="append", help="Run directory or one evaluations/step_XXXXXX directory")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--match-tolerance", type=float, help="Optional maximum validation GSM8K accuracy gap for matched comparisons, as a fraction")
    args = parser.parse_args()
    write_analysis(compare_runs(args.base_run, args.adapted_run, match_tolerance=args.match_tolerance), args.output_dir)


if __name__ == "__main__":
    main()
