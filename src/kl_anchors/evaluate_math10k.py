"""Isolated MATH-10K transfer/retention evaluation.

Usage: python -m kl_anchors.evaluate_math10k --training-protocol configs/protocol.math10k-llama32-3b.json --datasets-config configs/math10k_eval_datasets.json --model base --benchmarks gsm8k --output-dir results/eval/base-gsm8k

Dataset config must pin immutable revisions (commit hashes), not branches.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .evaluate import HuggingFaceBackend, score_multiple_choice, runtime_info, set_evaluation_seed
from .metrics import aggregate_accuracy, extract_gsm8k_final_answer

ALL = ("gsm8k", "arc_challenge", "arc_easy", "openbookqa", "social_iqa")
DEFAULT_DATASETS = {
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test"),
    "openbookqa": ("allenai/openbookqa", "main", "test"),
    "social_iqa": ("allenai/social_i_qa", None, "validation"),
}

def choice_data(benchmark, row):
    if benchmark in {"arc_challenge", "arc_easy", "openbookqa"}:
        options = row["choices"]
        labels, texts = list(options["label"]), list(options["text"])

        gold = str(row["answerKey"]).strip()

        if gold not in labels:
            raise ValueError(f"gold key {gold!r} not among {labels!r}")

        if benchmark in {"arc_challenge", "arc_easy"}:
            question = row["question"]
        else:
            question = row["question_stem"]

        return question, labels, texts, labels.index(gold)

    if benchmark == "social_iqa":
        labels = ["A", "B", "C"]
        texts = [row["answerA"], row["answerB"], row["answerC"]]

        gold = int(row["label"]) - 1

        if gold not in (0, 1, 2):
            raise ValueError(f"invalid Social IQA label: {row['label']!r}")

        question = f"{row['context']}\n{row['question']}"

        return question, labels, texts, gold

    raise ValueError(benchmark)

def eval_row(backend, benchmark, row, index, max_new_tokens):
    item_id = str(row.get("id", f"{benchmark}:{index}"))
    if benchmark == "gsm8k":
        prompt = f"Question: {row['question']}\nAnswer:"
        response = backend.generate(prompt, max_new_tokens=max_new_tokens)
        gold = extract_gsm8k_final_answer(str(row["answer"]))
        pred = extract_gsm8k_final_answer(response)
        if gold is None:
            raise ValueError(f"unparseable gold answer at {item_id}")
        return {"benchmark": benchmark, "item_id": item_id, "prompt": prompt, "response": response,
                "gold_answer": gold, "predicted_answer": pred, "parse_failure": pred is None,
                "correct": pred is not None and pred == gold}
    question, labels, texts, gold_index = choice_data(benchmark, row)
    if not 2 <= len(labels) <= 5 or len(set(labels)) != len(labels):
        raise ValueError(f"invalid choice labels: {labels!r}")
    if any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("empty choice text")
    # Score single-token labels rather than variable-length option text.
    # This is a study-specific protocol choice, not necessarily the paper's method.
    #prompt = f"Question: {question}\nOptions:\n" + "\n".join(
    #    f"{label}. {text}" for label, text in zip(labels, texts)
    #) + "\nAnswer:"
    prompt = (
        f"Question: {question}\n"
        "Options:\n"
        + "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
        + "\n\nSelect the correct answer option.\n"
        "Respond with only the option letter.\n"
        "Answer:"
    )
    continuations = [" " + label for label in labels]
    scored = score_multiple_choice(backend, prompt, continuations, length_normalization=False)
    return {"benchmark": benchmark, "item_id": item_id, "prompt": prompt,
            "labels": labels, "choices": texts, "gold_index": gold_index,
            "predicted_index": scored["prediction_index"], "correct": scored["prediction_index"] == gold_index,
            "parse_failure": False, **scored}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-protocol", type=Path, required=True)
    p.add_argument("--datasets-config", type=Path, required=True,
                   help="JSON mapping benchmark -> {repo_id, config, revision, split}; revision must be immutable")
    p.add_argument("--model", choices=("base", "best", "final"), required=True)
    p.add_argument("--run-dir", type=Path, help="Training run directory; required for best/final")
    p.add_argument("--benchmarks", nargs="+", choices=ALL, default=list(ALL))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None, help="Smoke test only; do not report as final")
    args = p.parse_args()
    if args.output_dir.exists():
        p.error("output directory already exists")
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")
    if args.model != "base" and args.run_dir is None:
        p.error("--run-dir required for LoRA models")
    training = json.loads(args.training_protocol.read_text())
    datasets_cfg = json.loads(args.datasets_config.read_text())
    for b in args.benchmarks:
        if b not in datasets_cfg:
            p.error(f"missing dataset configuration for {b}")
        for k in ("repo_id", "revision", "split"):
            if not datasets_cfg[b].get(k):
                p.error(f"{b}: missing {k}")
        rev = datasets_cfg[b]["revision"]
        if not (len(rev) >= 12 and all(c in "0123456789abcdef" for c in rev.lower())):
            p.error(f"{b}: revision must be an immutable commit hash, not {rev!r}")
    model_cfg = training["model"]
    eval_cfg = training["evaluation"]
    spec = {
        "model": {"repo_id": model_cfg["repository_id"], "revision": model_cfg["revision"],
                  "dtype": "bfloat16", "device_map": training["runtime"]["device"],
                  "trust_remote_code": False, "quantization": "none"},
        "tokenizer": eval_cfg["tokenizer"],
        "generation": eval_cfg["generation"],
    }
    if args.model != "base":
        adapter = args.run_dir / ("best_model" if args.model == "best" else "checkpoints/step_003609")
        if not (adapter / "adapter_config.json").exists():
            p.error(f"adapter not found: {adapter}")
        spec["model"]["adapter_path"] = str(adapter.resolve())
    set_evaluation_seed(int(training["seed"]))
    from datasets import load_dataset
    backend = HuggingFaceBackend(spec)
    results = []
    for benchmark in dict.fromkeys(args.benchmarks):
        ds = datasets_cfg[benchmark]
        dataset = load_dataset(ds["repo_id"], name=ds.get("config"), revision=ds["revision"], split=ds["split"])
        n = len(dataset) if args.limit is None else min(len(dataset), args.limit)
        print(f"Evaluating {benchmark}: {n} examples", flush=True)
        for i in range(n):
            results.append(eval_row(backend, benchmark, dataset[i], i, int(eval_cfg["generation"]["max_new_tokens"])))
            if (i + 1) % 100 == 0:
                print(f"  {benchmark}: {i+1}/{n}", flush=True)
    metrics = {}
    for benchmark in dict.fromkeys(args.benchmarks):
        subset = [r for r in results if r["benchmark"] == benchmark]
        a = aggregate_accuracy(r["correct"] for r in subset)
        metrics[benchmark] = {"correct": a.correct, "total": a.total, "accuracy": a.accuracy,
                              "parse_failures": sum(r["parse_failure"] for r in subset)}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "items.jsonl").open("x") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "aggregate.json").write_text(json.dumps({
        "model_condition": args.model, "training_protocol": training,
        "dataset_config": {b: datasets_cfg[b] for b in args.benchmarks},
        "scoring": {"gsm8k": "gsm8k_final_number_v1", "multiple_choice": "single-label conditional log likelihood, no length normalization, separate tokenization"},
        "limit": args.limit, "metrics": metrics, "runtime": runtime_info()
    }, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)

if __name__ == "__main__":
    main()
