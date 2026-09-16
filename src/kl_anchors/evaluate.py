"""Low-level evaluation for GSM8K and three retention task shapes.

This module chooses no checkpoint, dataset revision, split, prompt, or scoring
variant. A JSON protocol supplies all of them. The reusable evaluators depend
only on a backend with ``generate`` and ``token_logprobs`` methods; the CLI
loads Hugging Face dependencies lazily.

Use ``python -m kl_anchors.evaluate_study`` for the shared Team G4 protocol.
This module's standalone CLI accepts its lower-level backend schema.
Run ``python -m kl_anchors.evaluate --protocol PROTOCOL.json --output-dir DIR``
with ``src`` on PYTHONPATH. The output directory must not exist. It receives
``items.jsonl`` and ``aggregate.json``. The latter embeds the full protocol.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import random
from typing import Any, Mapping, Protocol, Sequence

from .metrics import aggregate_accuracy, extract_gsm8k_final_answer


class EvaluationBackend(Protocol):
    def generate(self, prompt: str, *, max_new_tokens: int) -> str: ...

    def token_logprobs(self, prompt: str, continuation: str) -> Sequence[float]: ...


def _require(mapping: Mapping[str, Any], key: str, expected: type) -> Any:
    value = mapping.get(key)
    if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
        raise ValueError(f"{key!r} must be {expected.__name__}")
    if expected is str and not value:
        raise ValueError(f"{key!r} must be nonempty")
    return value


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    """Reject absent design choices before any model or dataset is loaded."""

    _require(protocol, "experiment_id", str)
    seed = _require(protocol, "seed", int)
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    model = _require(protocol, "model", dict)
    for key in ("repo_id", "revision", "dtype", "device_map"):
        _require(model, key, str)
    _require(model, "trust_remote_code", bool)
    tokenizer = _require(protocol, "tokenizer", dict)
    for key in ("repo_id", "revision"):
        _require(tokenizer, key, str)
    _require(tokenizer, "use_fast", bool)
    _require(tokenizer, "add_special_tokens", bool)
    if "adapter_path" in model:
        _require(model, "adapter_path", str)
        if not Path(model["adapter_path"]).exists():
            _require(model, "adapter_revision", str)
    generation = _require(protocol, "generation", dict)
    if _require(generation, "max_new_tokens", int) <= 0:
        raise ValueError("generation.max_new_tokens must be positive")
    if _require(generation, "do_sample", bool):
        raise ValueError("GSM8K generation must have do_sample=false")
    if _require(generation, "num_beams", int) != 1:
        raise ValueError("generation.num_beams must be 1 for greedy decoding")
    _require(generation, "skip_special_tokens", bool)
    benchmarks = _require(protocol, "benchmarks", dict)
    if not benchmarks:
        raise ValueError("benchmarks must be nonempty")
    for name, spec in benchmarks.items():
        if not isinstance(name, str) or not name or not isinstance(spec, dict):
            raise ValueError("each benchmark must have a nonempty name and object spec")
        task = _require(spec, "task", str)
        if task not in {"gsm8k", "boolq", "index_mc", "key_mc"}:
            raise ValueError(f"unsupported task for {name}: {task}")
        dataset = _require(spec, "dataset", dict)
        for key in ("repo_id", "revision", "split"):
            _require(dataset, key, str)
        # A null configuration is explicit and distinct from omitting it.
        if "config" not in dataset or not (
            dataset["config"] is None or isinstance(dataset["config"], str)
        ):
            raise ValueError(f"{name}: dataset.config must be a string or null")
        _require(spec, "prompt_template", str)
        fields = _require(spec, "fields", dict)
        required = {
            "gsm8k": ("answer",),
            "boolq": ("answer",),
            "index_mc": ("choices", "answer"),
            "key_mc": ("choices", "choice_text", "choice_key", "answer"),
        }[task]
        for key in required:
            _require(fields, key, str)
        scoring = _require(spec, "scoring", dict)
        if task == "gsm8k":
            if _require(scoring, "answer_extraction", str) != "gsm8k_final_number_v1":
                raise ValueError(f"{name}: unsupported GSM8K answer extraction")
        else:
            if _require(scoring, "method", str) != "conditional_log_likelihood":
                raise ValueError(f"{name}: unsupported multiple-choice scoring method")
            if _require(scoring, "tokenization", str) != "separate_no_special_tokens":
                raise ValueError(f"{name}: unsupported continuation tokenization")
            _require(scoring, "length_normalization", bool)
            if not isinstance(scoring.get("choice_prefix"), str):
                raise ValueError(f"{name}: choice_prefix must be a string (empty is allowed)")
        if task == "boolq":
            choices = _require(scoring, "choice_texts", list)
            if len(choices) != 2 or any(not isinstance(x, str) or not x for x in choices):
                raise ValueError(f"{name}: choice_texts must contain false and true text")
        if "limit" not in spec or (
            spec["limit"] is not None
            and (isinstance(spec["limit"], bool) or not isinstance(spec["limit"], int) or spec["limit"] <= 0)
        ):
            raise ValueError(f"{name}: limit must be a positive integer or null")


def score_choice(token_logprobs: Sequence[float], *, length_normalization: bool) -> float:
    """Sum continuation-token log probabilities, optionally dividing by length."""

    if not token_logprobs:
        raise ValueError("a choice must tokenize to at least one continuation token")
    if any(not math.isfinite(float(value)) for value in token_logprobs):
        raise ValueError("token log probabilities must be finite")
    total = sum(float(value) for value in token_logprobs)
    return total / len(token_logprobs) if length_normalization else total


def _legacy_evaluation_from_study_protocol(
    study_protocol: Mapping[str, Any],
    *,
    adapter_path: str | Path | None = None,
    split_overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Translate the shared study protocol into this evaluator's explicit schema.

    Use ``split_overrides`` to evaluate a predeclared validation split. For a
    prepared local GSM8K validation JSONL, read its rows and call
    :func:`evaluate_item` with the returned ``benchmarks['gsm8k']`` spec.
    Neither this adapter nor the evaluator chooses a held-out set.
    """

    def object_at(*path: str) -> Mapping[str, Any]:
        current: Any = study_protocol
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                raise ValueError(f"study protocol lacks {'.'.join(path)}")
            current = current[key]
        if not isinstance(current, Mapping):
            raise ValueError(f"study protocol {'.'.join(path)} must be an object")
        return current

    def value_at(*path: str) -> Any:
        current: Any = study_protocol
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                raise ValueError(f"study protocol lacks {'.'.join(path)}")
            current = current[key]
        if current is None:
            raise ValueError(f"study protocol {'.'.join(path)} remains unresolved")
        return current

    condition = value_at("condition")
    if condition not in {"base", "discovery", "final"}:
        raise ValueError("study condition must be base, discovery, or final")
    if condition != "base" and adapter_path is None:
        raise ValueError("discovery/final evaluation requires an explicit adapter_path")
    evaluation = object_at("evaluation")
    lora = object_at("lora")
    chosen_dtype = value_at("evaluation", "model_dtype")
    precision_map = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
    if chosen_dtype not in precision_map:
        raise ValueError("evaluation.model_dtype must be fp32, fp16, or bf16")
    training_precision = lora.get("precision")
    if condition != "base" and training_precision not in precision_map:
        raise ValueError("lora.precision must be specified for adapted evaluation")
    if condition != "base" and training_precision != chosen_dtype:
        raise ValueError("evaluation.model_dtype and lora.precision differ; record and resolve this explicitly")

    model = object_at("model")
    tokenizer = object_at("evaluation", "tokenizer")
    generation = object_at("evaluation", "generation")
    overrides = dict(split_overrides or {})
    benchmark_tasks = {
        "gsm8k": "gsm8k",
        "boolq": "boolq",
        "hellaswag": "index_mc",
        "arc_easy": "key_mc",
    }
    unknown = set(overrides) - set(benchmark_tasks)
    if unknown:
        raise ValueError(f"unknown split override(s): {sorted(unknown)}")
    benchmarks: dict[str, dict[str, Any]] = {}
    for name, task in benchmark_tasks.items():
        dataset = object_at("datasets", name)
        split = overrides.get(name, value_at("datasets", name, "split"))
        if not isinstance(split, str) or not split:
            raise ValueError(f"{name} split override must be a nonempty string")
        benchmarks[name] = {
            "task": task,
            "dataset": {
                "repo_id": value_at("datasets", name, "repository_id"),
                "revision": value_at("datasets", name, "revision"),
                "config": value_at("datasets", name, "configuration"),
                "split": split,
            },
            "prompt_template": value_at("evaluation", "prompt_templates", name),
            "fields": dict(object_at("evaluation", "fields", name)),
            "scoring": dict(object_at("evaluation", "scoring", name)),
            "limit": value_at("evaluation", "limits", name),
        }
    eval_protocol: dict[str, Any] = {
        "experiment_id": value_at("experiment_id"),
        "seed": value_at("seed"),
        "model": {
            "repo_id": value_at("model", "repository_id"),
            "revision": value_at("model", "revision"),
            "dtype": precision_map[chosen_dtype],
            "quantization": value_at("evaluation", "model_quantization"),
            "device_map": value_at("runtime", "device"),
            "trust_remote_code": value_at("evaluation", "trust_remote_code"),
            "training_precision": training_precision,
        },
        "tokenizer": {
            "repo_id": value_at("evaluation", "tokenizer", "repo_id"),
            "revision": value_at("evaluation", "tokenizer", "revision"),
            "use_fast": value_at("evaluation", "tokenizer", "use_fast"),
            "add_special_tokens": value_at("evaluation", "tokenizer", "add_special_tokens"),
        },
        "generation": {
            "max_new_tokens": value_at("evaluation", "generation", "max_new_tokens"),
            "do_sample": value_at("evaluation", "generation", "do_sample"),
            "num_beams": value_at("evaluation", "generation", "num_beams"),
            "skip_special_tokens": value_at("evaluation", "generation", "skip_special_tokens"),
        },
        "benchmarks": benchmarks,
        "source_study_protocol": study_protocol,
    }
    if adapter_path is not None:
        eval_protocol["model"]["adapter_path"] = str(adapter_path)
        if not Path(adapter_path).exists():
            eval_protocol["model"]["adapter_revision"] = value_at("evaluation", "adapter_revision")
    validate_protocol(eval_protocol)
    return eval_protocol


def score_multiple_choice(
    backend: EvaluationBackend,
    prompt: str,
    choices: Sequence[str],
    *,
    length_normalization: bool,
) -> dict[str, Any]:
    """Teacher-force every choice against the same prompt; ties use first choice."""

    if len(choices) < 2 or any(not isinstance(choice, str) or not choice for choice in choices):
        raise ValueError("multiple choice requires at least two nonempty strings")
    logprobs = [list(backend.token_logprobs(prompt, choice)) for choice in choices]
    scores = [score_choice(values, length_normalization=length_normalization) for values in logprobs]
    prediction_index = max(range(len(scores)), key=scores.__getitem__)
    return {
        "prediction_index": prediction_index,
        "choice_scores": scores,
        "choice_token_counts": [len(values) for values in logprobs],
        "choice_logprob_sums": [sum(values) for values in logprobs],
        "choice_token_logprobs": logprobs,
    }


def _answer_index(task: str, row: Mapping[str, Any], spec: Mapping[str, Any], choices: Sequence[str]) -> int:
    fields = spec["fields"]
    answer = row[fields["answer"]]
    if task == "boolq":
        if not isinstance(answer, bool):
            raise ValueError("BoolQ answer must be boolean")
        return int(answer)
    if task == "index_mc":
        try:
            index = int(answer)
        except (ValueError, TypeError) as exc:
            raise ValueError("index_mc answer must be an integer index") from exc
        if not 0 <= index < len(choices):
            raise ValueError("index_mc answer index is out of range")
        return index
    labels = row[fields["choices"]][fields["choice_key"]]
    if len(labels) != len(choices) or labels.count(answer) != 1:
        raise ValueError("key_mc answer key must match exactly one choice")
    return labels.index(answer)


def evaluate_item(
    backend: EvaluationBackend,
    benchmark: str,
    spec: Mapping[str, Any],
    row: Mapping[str, Any],
    index: int,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Evaluate one source row and retain raw information for an audit."""

    prompt = spec["prompt_template"].format_map(row)
    fields = spec["fields"]
    item_id = str(row[fields["id"]]) if "id" in fields else f"{benchmark}:{index}"
    record: dict[str, Any] = {"benchmark": benchmark, "item_id": item_id, "prompt": prompt}
    task = spec["task"]
    if task == "gsm8k":
        gold_text = str(row[fields["answer"]])
        gold = extract_gsm8k_final_answer(gold_text)
        if gold is None:
            raise ValueError(f"{benchmark}/{item_id}: reference answer did not parse")
        response = backend.generate(prompt, max_new_tokens=max_new_tokens)
        predicted = extract_gsm8k_final_answer(response)
        record.update(
            response=response,
            gold_text=gold_text,
            gold_answer=gold,
            predicted_answer=predicted,
            parse_failure=predicted is None,
            correct=predicted == gold,
        )
        return record

    scoring = spec["scoring"]
    prefix = scoring["choice_prefix"]
    if task == "boolq":
        choice_texts = list(scoring["choice_texts"])
    elif task == "index_mc":
        choice_texts = list(row[fields["choices"]])
    else:
        choice_texts = list(row[fields["choices"]][fields["choice_text"]])
    if any(not isinstance(choice, str) or not choice.strip() for choice in choice_texts):
        raise ValueError("all choice texts must be nonempty strings")
    continuations = [prefix + choice for choice in choice_texts]
    gold_index = _answer_index(task, row, spec, continuations)
    result = score_multiple_choice(
        backend,
        prompt,
        continuations,
        length_normalization=scoring["length_normalization"],
    )
    record.update(
        choices=continuations,
        gold_index=gold_index,
        correct=result["prediction_index"] == gold_index,
        parse_failure=False,
        length_normalization=scoring["length_normalization"],
        **result,
    )
    return record


def aggregate_records(protocol: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Calculate audit-friendly accuracy and parse-failure counts per benchmark."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["benchmark"])].append(record)
    expected = set(protocol["benchmarks"])
    if set(grouped) != expected:
        raise ValueError("records must contain every configured benchmark and no others")
    metrics = {}
    for name, rows in grouped.items():
        summary = aggregate_accuracy(row["correct"] for row in rows)
        metrics[name] = {
            "correct": summary.correct,
            "total": summary.total,
            "accuracy": summary.accuracy,
            "parse_failures": sum(bool(row["parse_failure"]) for row in rows),
        }
    return {"experiment_id": protocol["experiment_id"], "protocol": protocol, "metrics": metrics}


def write_outputs(output_dir: Path, records: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Any]) -> None:
    """Create a new result directory; existing results are never overwritten."""

    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "items.jsonl").open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    with (output_dir / "aggregate.json").open("x", encoding="utf-8") as stream:
        json.dump(aggregate, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


class HuggingFaceBackend:
    """Minimal causal-model adapter, imported only by the CLI."""

    def __init__(self, protocol: Mapping[str, Any]) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install torch and transformers to use the evaluation CLI") from exc

        self.torch = torch
        model_spec = protocol["model"]
        tokenizer_spec = protocol["tokenizer"]
        dtype_name = model_spec["dtype"]
        dtypes = {"auto": "auto", "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        if dtype_name not in dtypes:
            raise ValueError("model.dtype must be auto, float32, float16, or bfloat16")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_spec["repo_id"],
            revision=tokenizer_spec["revision"],
            use_fast=tokenizer_spec["use_fast"],
            trust_remote_code=model_spec["trust_remote_code"],
        )
        self.add_special_tokens = tokenizer_spec["add_special_tokens"]
        self.skip_special_tokens = protocol["generation"]["skip_special_tokens"]
        model_kwargs = {
            "revision": model_spec["revision"],
            "dtype": dtypes[dtype_name],
            "device_map": model_spec["device_map"],
            "trust_remote_code": model_spec["trust_remote_code"],
        }
        quantization = model_spec.get("quantization", "none")
        if quantization != "none":
            from transformers import BitsAndBytesConfig
            if quantization == "4bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtypes[dtype_name])
            elif quantization == "8bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise ValueError("model.quantization must be none, 4bit or 8bit")
        self.model = AutoModelForCausalLM.from_pretrained(model_spec["repo_id"], **model_kwargs)
        if "adapter_path" in model_spec:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("install peft to evaluate a LoRA adapter") from exc
            adapter_kwargs = {}
            if "adapter_revision" in model_spec and model_spec["adapter_revision"] != "local":
                adapter_kwargs["revision"] = model_spec["adapter_revision"]
            self.model = PeftModel.from_pretrained(
                self.model, model_spec["adapter_path"], **adapter_kwargs
            )
        self.model.eval()

    def _input_device(self):
        return next(self.model.parameters()).device

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=self.add_special_tokens
        )
        inputs = {key: value.to(self._input_device()) for key, value in inputs.items()}
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_id,
            )
        continuation = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(continuation, skip_special_tokens=self.skip_special_tokens)

    def token_logprobs(self, prompt: str, continuation: str) -> list[float]:
        # Separate encoding fixes the prompt/continuation boundary. The protocol
        # records this choice because joint BPE tokenization can merge across it.
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=self.add_special_tokens)
        choice_ids = self.tokenizer.encode(continuation, add_special_tokens=False)
        if not prompt_ids or not choice_ids:
            raise ValueError("prompt and continuation must both tokenize to at least one token")
        ids = self.torch.tensor([prompt_ids + choice_ids], device=self._input_device())
        with self.torch.inference_mode():
            logits = self.model(input_ids=ids, attention_mask=self.torch.ones_like(ids)).logits[0]
        first = len(prompt_ids) - 1
        selected = logits[first : first + len(choice_ids)].float().log_softmax(dim=-1)
        tokens = self.torch.tensor(choice_ids, device=selected.device)
        return selected.gather(1, tokens[:, None]).squeeze(1).tolist()


def run_protocol(protocol: Mapping[str, Any], backend: EvaluationBackend | None = None) -> list[dict[str, Any]]:
    """Load configured datasets and evaluate each item; no output is written."""

    validate_protocol(protocol)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install datasets to use the evaluation CLI") from exc
    if backend is None:
        backend = HuggingFaceBackend(protocol)
    records: list[dict[str, Any]] = []
    for name, spec in protocol["benchmarks"].items():
        dataset_spec = spec["dataset"]
        dataset = load_dataset(
            dataset_spec["repo_id"],
            name=dataset_spec["config"],
            revision=dataset_spec["revision"],
            split=dataset_spec["split"],
        )
        limit = len(dataset) if spec["limit"] is None else min(len(dataset), spec["limit"])
        for index in range(limit):
            records.append(
                evaluate_item(
                    backend,
                    name,
                    spec,
                    dataset[index],
                    index,
                    max_new_tokens=protocol["generation"]["max_new_tokens"],
                )
            )
    return records


def runtime_info() -> dict[str, Any]:
    """Record software and available accelerator details for a reported run."""

    versions: dict[str, str | None] = {}
    for package in ("torch", "transformers", "datasets", "peft", "accelerate"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }
    try:
        import torch
    except ImportError:
        return info
    info["cuda_available"] = torch.cuda.is_available()
    if info["cuda_available"]:
        info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    info["mps_available"] = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    return info


def set_evaluation_seed(seed: int) -> None:
    """Seed installed random generators before loading a model or evaluating."""

    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    set_evaluation_seed(protocol["seed"])
    records = run_protocol(protocol)
    aggregate = aggregate_records(protocol, records)
    aggregate["runtime"] = runtime_info()
    write_outputs(args.output_dir, records, aggregate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
