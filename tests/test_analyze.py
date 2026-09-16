import json
from pathlib import Path
import shutil
import tempfile
import unittest

from kl_anchors.analyze import compare_runs, write_analysis


def make_run(root, name, values, *, adapted):
    run_dir = root / name
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True)
    benchmarks = {key: {"prompt_template": key, "scoring": "locked"} for key in ("gsm8k", "boolq", "hellaswag", "arc_easy")}
    protocol = {"model": {"repo_id": "model", "revision": "a" * 40}, "benchmarks": benchmarks}
    if adapted:
        protocol["source_study_protocol"] = {
            "condition": "final", "seed": 0, "lora": {"rank": 8},
            "training": {key: 1 for key in (
                "training_ids_manifest", "optimizer", "learning_rate", "scheduler", "warmup_steps",
                "batch_policy", "per_device_batch_size", "gradient_accumulation_steps", "max_steps")},
            "anchor_selection": {"method": "none"},
        }
    aggregate = {"experiment_id": name, "protocol": protocol,
                 "metrics": {key: {"accuracy": value} for key, value in values.items()}}
    (eval_dir / "aggregate.json").write_text(json.dumps(aggregate))
    with (eval_dir / "items.jsonl").open("w") as stream:
        for key, value in values.items():
            stream.write(json.dumps({"benchmark": key, "item_id": key + ":1", "correct": value > 0.5}) + "\n")
    return run_dir


class AnalysisTests(unittest.TestCase):
    def test_forgetting_and_plots_from_saved_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = make_run(root, "base", {"gsm8k": 0.2, "boolq": 0.8, "hellaswag": 0.6, "arc_easy": 0.7}, adapted=False)
            adapted = make_run(root, "lora", {"gsm8k": 0.5, "boolq": 0.7, "hellaswag": 0.7, "arc_easy": 0.6}, adapted=True)
            summary = compare_runs(base, [adapted])
            self.assertAlmostEqual(summary["adapted"][0]["target_gain"], 0.3)
            self.assertAlmostEqual(summary["adapted"][0]["forgetting"]["boolq"], 0.1)
            self.assertAlmostEqual(summary["adapted"][0]["forgetting"]["hellaswag"], -0.1)
            output = root / "analysis"
            write_analysis(summary, output)
            self.assertTrue((output / "target_retention.png").exists())
            self.assertTrue((output / "forgetting_by_benchmark.png").exists())
            checkpoint_eval = adapted / "evaluations" / "step_000001"
            shutil.copytree(adapted / "evaluation", checkpoint_eval)
            checkpoint_summary = compare_runs(base, [checkpoint_eval])
            self.assertEqual(checkpoint_summary["adapted"][0]["evaluation_directory"], str(checkpoint_eval.resolve()))


if __name__ == "__main__":
    unittest.main()
