"""Evaluate the frozen base or a saved LoRA adapter on the four study tasks.

The shared study protocol supplies prompts, splits, revisions and scoring.
GSM8K test rows come from its frozen manifest. All item predictions and raw
choice scores are saved in a new result directory.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .anchor_pipeline import read_jsonl
from .evaluate import (
    HuggingFaceBackend, aggregate_records, evaluate_item, runtime_info,
    validate_protocol as validate_evaluation_protocol, write_outputs,
)
from .hf_runtime import load_tokenizer, render_chat_prompt, set_seed
from .protocol import load_protocol, reserve_run, validate_protocol
from .provenance import evaluation_code_digest


TASKS = {"gsm8k": "gsm8k", "boolq": "boolq", "hellaswag": "index_mc", "arc_easy": "key_mc"}


def evaluation_from_study_protocol(study: dict, tokenizer, *, adapter_path: Path | None = None) -> dict:
    """Convert one common study protocol into the evaluation backend schema."""
    errors = validate_protocol(study)
    if errors:
        raise ValueError("incomplete study protocol:\n- " + "\n- ".join(errors))
    evaluation = study["evaluation"]
    if evaluation["harness"] != "internal_kl_anchors_v1":
        raise ValueError("this entry point requires evaluation.harness=internal_kl_anchors_v1")
    if evaluation["harness_revision"] != evaluation_code_digest():
        raise ValueError("evaluation.harness_revision does not match current evaluator source")
    dtype = evaluation["model_dtype"]
    dtype_map = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
    if dtype not in dtype_map:
        raise ValueError("evaluation.model_dtype must be fp32, fp16 or bf16")
    device_map = "auto" if study["runtime"]["device"].startswith("cuda") else study["runtime"]["device"]
    model = {
        "repo_id": study["model"]["repository_id"],
        "revision": study["model"]["revision"],
        "dtype": dtype_map[dtype],
        "quantization": evaluation["model_quantization"],
        "device_map": device_map,
        "trust_remote_code": evaluation["trust_remote_code"],
    }
    if adapter_path is not None:
        model.update(adapter_path=str(adapter_path.resolve()), adapter_revision="local")
    benchmarks = {}
    for name, task in TASKS.items():
        dataset = study["datasets"][name]
        template = evaluation["prompt_templates"][name]
        if name == "gsm8k":
            template = template.replace("{question}", "{prompt}")
        rendered_template = render_chat_prompt(tokenizer, template, study["runtime"]["target_system_prompt"])
        benchmarks[name] = {
            "task": task,
            "dataset": {
                "repo_id": dataset["repository_id"],
                "revision": dataset["revision"],
                "config": dataset["configuration"],
                "split": dataset["split"],
            },
            "prompt_template": rendered_template,
            "fields": ({"answer": "response", "id": "example_id"} if name == "gsm8k" else evaluation["fields"][name]),
            "scoring": evaluation["scoring"][name],
            "limit": evaluation["limits"][name],
        }
    protocol = {
        "experiment_id": study["experiment_id"],
        "seed": study["seed"],
        "model": model,
        "tokenizer": evaluation["tokenizer"],
        "generation": evaluation["generation"],
        "benchmarks": benchmarks,
        "source_study_protocol": study,
    }
    validate_evaluation_protocol(protocol)
    return protocol


def evaluate_study(study: dict, output_dir: Path, *, adapter_path: Path | None = None) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    set_seed(study["seed"])
    tokenizer = load_tokenizer(study["model"])
    protocol = evaluation_from_study_protocol(study, tokenizer, adapter_path=adapter_path)
    backend = HuggingFaceBackend(protocol)
    records = []
    from datasets import load_dataset

    for name, spec in protocol["benchmarks"].items():
        if name == "gsm8k":
            rows = read_jsonl(Path(study["datasets"]["gsm8k"]["test_ids_manifest"]))
        else:
            source = spec["dataset"]
            ds = load_dataset(source["repo_id"], name=source["config"], revision=source["revision"], split=source["split"])
            rows = [ds[i] for i in range(len(ds))]
        if spec["limit"] is not None:
            rows = rows[:spec["limit"]]
        for i, row in enumerate(rows):
            records.append(evaluate_item(backend, name, spec, row, i, max_new_tokens=protocol["generation"]["max_new_tokens"]))
    aggregate = aggregate_records(protocol, records)
    aggregate["runtime"] = runtime_info()
    write_outputs(output_dir, records, aggregate)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--base-protocol", type=Path, help="Completed shared config for a new untouched-base evaluation")
    group.add_argument("--run-dir", type=Path, help="Existing final training run; evaluate its best_model adapter")
    parser.add_argument("--experiment-id", help="Required for --base-protocol")
    parser.add_argument("--checkpoint", help="Optional saved step directory name, e.g. step_000100; evaluates that checkpoint instead of best_model")
    args = parser.parse_args()
    if args.base_protocol:
        if not args.experiment_id:
            parser.error("--experiment-id is required for base evaluation")
        study = copy.deepcopy(load_protocol(args.base_protocol))
        study["experiment_id"] = args.experiment_id
        study["condition"] = "base"
        study["evaluation"]["checkpoint_selection_rule"] = "untouched"
        run_dir = reserve_run(study)
        adapter = None
    else:
        run_dir = args.run_dir
        study = load_protocol(run_dir / "protocol.json")
        if args.checkpoint and ("/" in args.checkpoint or ".." in args.checkpoint):
            parser.error("--checkpoint must be a saved step directory name")
        adapter = run_dir / "checkpoints" / args.checkpoint if args.checkpoint else run_dir / "best_model"
        if study["condition"] not in ("final", "discovery") or not adapter.is_dir():
            raise ValueError("run directory lacks a completed LoRA best_model")
    output_dir = run_dir / "evaluations" / args.checkpoint if args.checkpoint else run_dir / "evaluation"
    try:
        aggregate = evaluate_study(study, output_dir, adapter_path=adapter)
    except Exception as exc:
        (run_dir / "evaluation_status.json").write_text(json.dumps({"status": "failed", "error": str(exc)}, indent=2) + "\n", encoding="utf-8")
        raise
    (run_dir / "evaluation_status.json").write_text(json.dumps({"status": "complete", "output_dir": str(output_dir)}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate["metrics"], indent=2))


if __name__ == "__main__":
    main()
