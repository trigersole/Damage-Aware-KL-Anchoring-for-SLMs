"""Validate and reserve immutable experiment records before a run starts.

The JSON protocol deliberately keeps unresolved research choices as ``null``.
``reserve_run`` accepts only a fully specified protocol and atomically creates
one new experiment directory. It never reuses an existing experiment ID.
``evaluation.prompt_templates`` values are literal user-message format strings,
not filenames. The GSM8K template must contain ``{question}``; the target
system message is recorded separately in ``runtime.target_system_prompt``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Mapping


_REVISION = re.compile(r"[0-9a-fA-F]{40}")
_EXPERIMENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_METHODS = {"none", "random", "diversity", "vulnerability", "damage_diverse"}
_BENCHMARKS = ("gsm8k", "boolq", "hellaswag", "arc_easy")


def load_protocol(path: str | Path) -> dict[str, Any]:
    """Load a JSON protocol without filling or approving missing choices."""
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("protocol root must be a JSON object")
    return value


def validate_protocol(protocol: Mapping[str, Any]) -> list[str]:
    """Return every missing or invalid field that prevents a recorded run.

    Model and dataset revisions must be full 40-character Git commit SHAs,
    not mutable branch names. The internal evaluator uses a 40-character
    source digest in the harness revision field. Dataset source IDs and splits are explicit so the saved
    protocol remains auditable if a hosted dataset changes later.
    """
    if not isinstance(protocol, Mapping):
        return ["protocol must be an object"]
    errors: list[str] = []

    def at(path: str) -> Any:
        current: Any = protocol
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return None
            current = current[part]
        return current

    def nonempty(path: str) -> None:
        value = at(path)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{path} must be a nonempty string")

    def integer(path: str, *, minimum: int = 0) -> None:
        value = at(path)
        if type(value) is not int or value < minimum:
            errors.append(f"{path} must be an integer >= {minimum}")

    def number(path: str, *, positive: bool = False) -> None:
        value = at(path)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{path} must be a finite number")
            return
        import math

        if not math.isfinite(value) or (positive and value <= 0) or (not positive and value < 0):
            errors.append(f"{path} must be a finite {'positive' if positive else 'nonnegative'} number")

    def revision(path: str) -> None:
        value = at(path)
        if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
            errors.append(f"{path} must be a full 40-character commit SHA")

    if type(at("schema_version")) is not int or at("schema_version") != 1:
        errors.append("schema_version must be 1")
    condition = at("condition")
    if not isinstance(condition, str) or condition not in {"base", "discovery", "final"}:
        errors.append("condition must be base, discovery, or final")
    experiment_id = at("experiment_id")
    if not isinstance(experiment_id, str) or _EXPERIMENT_ID.fullmatch(experiment_id) is None or experiment_id in {".", ".."}:
        errors.append("experiment_id must be a safe, nonempty directory name")
    output_directory = at("output_directory")
    if not isinstance(output_directory, str) or not Path(output_directory).is_absolute():
        errors.append("output_directory must be an absolute path")
    nonempty("model.repository_id")
    revision("model.revision")

    for name in _BENCHMARKS:
        prefix = f"datasets.{name}"
        nonempty(f"{prefix}.repository_id")
        revision(f"{prefix}.revision")
        nonempty(f"{prefix}.configuration")
        nonempty(f"{prefix}.split")
        nonempty(f"{prefix}.preprocessing")
    for path in (
        "datasets.gsm8k.validation_ids_manifest",
        "datasets.gsm8k.test_ids_manifest",
    ):
        nonempty(path)

    integer("seed")
    nonempty("runtime.device")
    integer("runtime.max_sequence_tokens", minimum=1)
    integer("runtime.validation_max_new_tokens", minimum=1)
    for path in ("runtime.target_system_prompt", "runtime.anchor_system_prompt"):
        if not isinstance(at(path), str):
            errors.append(f"{path} must be a string (empty is allowed)")
    if condition in {"discovery", "final"}:
        nonempty("training.training_ids_manifest")
        method = at("anchor_selection.method")
        needs_anchor_pool = condition == "discovery" or (
            isinstance(method, str) and method in _METHODS - {"none"}
        )
        if needs_anchor_pool:
            for path in ("datasets.anchors.eligible_ids_manifest", "datasets.anchors.exclusions_manifest"):
                nonempty(path)
            for field in ("repository_id", "configuration", "split", "preprocessing"):
                nonempty(f"datasets.anchors.{field}")
            revision("datasets.anchors.revision")

        integer("lora.rank", minimum=1)
        number("lora.alpha", positive=True)
        number("lora.dropout")
        dropout = at("lora.dropout")
        if isinstance(dropout, (int, float)) and not isinstance(dropout, bool) and dropout > 1:
            errors.append("lora.dropout must be <= 1")
        targets = at("lora.target_modules")
        if not isinstance(targets, list) or not targets or any(not isinstance(x, str) or not x.strip() for x in targets):
            errors.append("lora.target_modules must be a nonempty list of module names")
        if at("lora.precision") not in ("fp32", "fp16", "bf16"):
            errors.append("lora.precision must be fp32, fp16, or bf16")
        if at("lora.quantization") not in ("none", "4bit", "8bit"):
            errors.append("lora.quantization must be none, 4bit, or 8bit")

        for path in ("training.optimizer", "training.scheduler", "training.batch_policy"):
            nonempty(path)
        number("training.learning_rate", positive=True)
        integer("training.per_device_batch_size", minimum=1)
        integer("training.gradient_accumulation_steps", minimum=1)
        integer("training.max_steps", minimum=1)
        integer("training.warmup_steps")
        integer("training.target_examples", minimum=1)
        integer("training.target_tokens", minimum=1)
        number("training.epochs", positive=True)
        integer("training.anchor_interleave_every_steps")
        if at("training.scheduler") not in ("constant", "linear", "cosine"):
            errors.append("training.scheduler must be constant, linear or cosine")
        if at("training.optimizer") != "adamw":
            errors.append("training.optimizer must be adamw for this runner")
        if at("training.batch_policy") != "shuffled_repeat":
            errors.append("training.batch_policy must be shuffled_repeat for this runner")
        if isinstance(at("training.warmup_steps"), int) and isinstance(at("training.max_steps"), int) and at("training.warmup_steps") >= at("training.max_steps"):
            errors.append("training.warmup_steps must be smaller than max_steps")

        if needs_anchor_pool:
            integer("runtime.teacher_max_new_tokens", minimum=1)
        integer("runtime.anchor_tokens_per_update")
        integer("runtime.checkpoint_every_steps", minimum=1)
        number("runtime.max_grad_norm")
        for path in ("runtime.scale_kl_by_temperature_squared", "runtime.target_answer_append_eos"):
            if type(at(path)) is not bool:
                errors.append(f"{path} must be a boolean")

        if not isinstance(method, str) or method not in _METHODS:
            errors.append("anchor_selection.method must be none, random, diversity, vulnerability, or damage_diverse")
        integer("anchor_selection.token_budget")
        if needs_anchor_pool:
            nonempty("anchor_selection.candidate_pool_manifest")
        if condition == "discovery" and method != "none":
            errors.append("discovery condition requires method none")
        if isinstance(method, str) and method in _METHODS - {"none"}:
            nonempty("anchor_selection.selected_ids_manifest")
            if at("anchor_selection.token_budget") == 0:
                errors.append("anchored conditions require a positive anchor token budget")
        elif method == "none" and at("anchor_selection.token_budget") != 0:
            errors.append("method none requires zero anchor tokens")
        if method == "damage_diverse":
            number("anchor_selection.damage_weight")
            weight = at("anchor_selection.damage_weight")
            if isinstance(weight, (int, float)) and not isinstance(weight, bool) and not 0 <= weight <= 1:
                errors.append("anchor_selection.damage_weight must be in [0, 1]")
        if method == "none":
            if at("training.anchor_interleave_every_steps") != 0:
                errors.append("method none requires zero anchor interleave frequency")
            if at("runtime.anchor_tokens_per_update") != 0:
                errors.append("method none requires zero anchor tokens per update")
        elif isinstance(method, str) and method in _METHODS:
            if at("training.anchor_interleave_every_steps") == 0:
                errors.append("anchored conditions require a positive anchor interleave frequency")
            if at("runtime.anchor_tokens_per_update") == 0:
                errors.append("anchored conditions require positive anchor tokens per update")

        number("kl.coefficient")
        if method != "none" or condition == "discovery":
            direction = at("kl.direction")
            if direction not in ("teacher_to_student", "student_to_teacher"):
                errors.append("kl.direction must be teacher_to_student or student_to_teacher")
            number("kl.temperature", positive=True)
            nonempty("kl.token_mask")
            nonempty("kl.normalization")
            if at("kl.token_mask") != "response_tokens":
                errors.append("kl.token_mask must be response_tokens for this runner")
            if at("kl.normalization") not in ("token_mean", "sequence_mean"):
                errors.append("kl.normalization must be token_mean or sequence_mean")
            nonempty("kl.teacher_continuation")
        if method == "none" and at("kl.coefficient") != 0:
            errors.append("method none requires zero KL coefficient")
        if isinstance(method, str) and method in _METHODS - {"none"} and at("kl.coefficient") == 0:
            errors.append("anchored conditions require a positive KL coefficient")

    nonempty("evaluation.harness")
    revision("evaluation.harness_revision")
    nonempty("evaluation.chat_template")
    if at("evaluation.model_dtype") not in ("fp32", "fp16", "bf16"):
        errors.append("evaluation.model_dtype must be fp32, fp16, or bf16")
    if at("evaluation.model_quantization") not in ("none", "4bit", "8bit"):
        errors.append("evaluation.model_quantization must be none, 4bit, or 8bit")
    if condition in {"discovery", "final"} and at("evaluation.model_dtype") != at("lora.precision"):
        errors.append("evaluation.model_dtype must match lora.precision")
    if condition in {"discovery", "final"} and at("evaluation.model_quantization") != at("lora.quantization"):
        errors.append("evaluation.model_quantization must match lora.quantization")
    if type(at("evaluation.trust_remote_code")) is not bool:
        errors.append("evaluation.trust_remote_code must be boolean")
    for path in ("evaluation.tokenizer.repo_id", "evaluation.tokenizer.revision"):
        if path.endswith("revision"):
            revision(path)
        else:
            nonempty(path)
    if at("evaluation.tokenizer.repo_id") != at("model.repository_id") or at("evaluation.tokenizer.revision") != at("model.revision"):
        errors.append("evaluation tokenizer must match pinned model ID and revision")
    for path in ("evaluation.tokenizer.use_fast", "evaluation.tokenizer.add_special_tokens", "evaluation.generation.do_sample", "evaluation.generation.skip_special_tokens"):
        if type(at(path)) is not bool:
            errors.append(f"{path} must be boolean")
    integer("evaluation.generation.max_new_tokens", minimum=1)
    integer("evaluation.generation.num_beams", minimum=1)
    if at("evaluation.generation.do_sample") is not False or at("evaluation.generation.num_beams") != 1:
        errors.append("evaluation generation must use greedy decoding")
    selection_rule = at("evaluation.checkpoint_selection_rule")
    if condition == "base":
        if selection_rule != "untouched":
            errors.append("base checkpoint_selection_rule must be untouched")
    elif selection_rule not in ("min_validation_loss", "max_gsm8k_exact_match"):
        errors.append("checkpoint_selection_rule must be min_validation_loss or max_gsm8k_exact_match")
    integer("evaluation.few_shot")
    if at("evaluation.few_shot") != 0:
        errors.append("current evaluator supports only zero-shot prompts")
    for name in _BENCHMARKS:
        nonempty(f"evaluation.prompt_templates.{name}")
        score = at(f"evaluation.scoring.{name}")
        if not isinstance(score, Mapping):
            errors.append(f"evaluation.scoring.{name} must be an object")
        fields = at(f"evaluation.fields.{name}")
        if not isinstance(fields, Mapping):
            errors.append(f"evaluation.fields.{name} must be an object")
        limit = at(f"evaluation.limits.{name}")
        if limit is not None and (type(limit) is not int or limit <= 0):
            errors.append(f"evaluation.limits.{name} must be null or a positive integer")
    gsm8k_template = at("evaluation.prompt_templates.gsm8k")
    if isinstance(gsm8k_template, str) and "{question}" not in gsm8k_template:
        errors.append("evaluation.prompt_templates.gsm8k must contain {question}")

    nonempty("environment.python_version")
    nonempty("environment.software_versions")
    nonempty("environment.hardware")
    return errors


def reserve_run(protocol: Mapping[str, Any]) -> Path:
    """Create an experiment directory and its exact protocol snapshot once.

    The directory creation is exclusive, so even concurrent callers cannot
    overwrite a reported or partly completed experiment with the same ID.
    A newly reserved run is not evidence that training or evaluation finished.
    """
    errors = validate_protocol(protocol)
    if errors:
        raise ValueError("final protocol is incomplete or invalid:\n- " + "\n- ".join(errors))
    serialized = json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    root = Path(protocol["output_directory"])
    root.mkdir(parents=True, exist_ok=True)
    run_directory = root / protocol["experiment_id"]
    run_directory.mkdir(exist_ok=False)
    with (run_directory / "protocol.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
    return run_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a completed study protocol before downloads or runs")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    errors = validate_protocol(load_protocol(args.config))
    if errors:
        print("Protocol is incomplete:\n- " + "\n- ".join(errors))
        raise SystemExit(2)
    print("Protocol is complete and structurally valid. Research choices still require team approval.")


if __name__ == "__main__":
    main()
