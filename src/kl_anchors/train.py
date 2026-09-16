"""Run a fresh LoRA condition from a pinned base checkpoint.

The only method switch is ``--mode``: lora, random_kl, diverse_kl,
damage_kl, or damage_diverse_kl. The latter requires ``--damage-weight``.
Inputs are frozen JSONL manifests and an explicit study protocol. Discovery
runs use ``--condition discovery --mode lora`` and a separate experiment ID.

Every run saves step checkpoints, validation metrics, the best validation
checkpoint as ``best_model/``, item-level GSM8K validation predictions when
used, a training log, and a loss-versus-epoch plot. Test examples are never
used for checkpoint selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import shutil
import time

import torch

from .anchor_pipeline import file_sha256, read_jsonl
from .hf_runtime import attach_new_lora, load_base, load_tokenizer, render_chat_prompt, set_seed
from .metrics import gsm8k_exact_match
from .protocol import load_protocol, reserve_run
from .provenance import evaluation_code_digest
from .selection import Candidate, select_anchors
from .training_core import (
    AnchorTokenScheduler, TokenizedExample, collate, target_causal_loss,
    tokenize_completion, tokenize_saved_continuation, training_update,
)


MODE_TO_METHOD = {
    "lora": "none",
    "random_kl": "random",
    "diverse_kl": "diversity",
    "damage_kl": "vulnerability",
    "damage_diverse_kl": "damage_diverse",
}


class TargetStream:
    """Repeat a deterministic shuffled order independently in each run."""

    def __init__(self, examples: list[TokenizedExample], seed: int):
        if not examples:
            raise ValueError("training manifest contains no examples")
        self.examples = examples
        self.rng = random.Random(seed)
        self.order: list[int] = []
        self.cursor = 0
        self.completed_epochs = 0

    def take(self, count: int) -> list[TokenizedExample]:
        result = []
        for _ in range(count):
            if self.cursor == len(self.order):
                self.order = list(range(len(self.examples)))
                self.rng.shuffle(self.order)
                self.cursor = 0
                self.completed_epochs += 1
            result.append(self.examples[self.order[self.cursor]])
            self.cursor += 1
        return result


def _target_examples(tokenizer, rows: list[dict], protocol: dict) -> list[TokenizedExample]:
    runtime = protocol["runtime"]
    return [tokenize_completion(
        tokenizer,
        example_id=row["example_id"],
        rendered_prompt=render_chat_prompt(tokenizer, row["prompt"], runtime["target_system_prompt"]),
        response=row["response"],
        max_sequence_tokens=runtime["max_sequence_tokens"],
        append_eos=runtime["target_answer_append_eos"],
    ) for row in rows]


def _validation_loss(student, examples: list[TokenizedExample], *, batch_size: int, pad_token_id: int, device: str) -> float:
    student.eval()
    loss_sum, token_count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = collate(examples[start:start + batch_size], pad_token_id=pad_token_id, device=device)
            logits = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            count = int(batch["response_mask"].sum().item())
            loss_sum += float(target_causal_loss(logits, batch["input_ids"], batch["response_mask"]).cpu()) * count
            token_count += count
    return loss_sum / token_count


def _validation_gsm8k(student, tokenizer, rows: list[dict], protocol: dict, output: Path) -> float:
    student.eval()
    records = []
    runtime = protocol["runtime"]
    for row in rows:
        user_content = protocol["evaluation"]["prompt_templates"]["gsm8k"].format(question=row["prompt"])
        prompt = render_chat_prompt(tokenizer, user_content, runtime["target_system_prompt"])
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        ids = torch.tensor([input_ids], dtype=torch.long, device=runtime["device"])
        with torch.no_grad():
            output_ids = student.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=runtime["validation_max_new_tokens"],
                do_sample=False, num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prediction = tokenizer.decode(output_ids[0, len(input_ids):], skip_special_tokens=True)
        correct = gsm8k_exact_match(prediction, row["response"])
        records.append({"example_id": row["example_id"], "prediction": prediction, "gold": row["response"], "correct": correct})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return sum(row["correct"] for row in records) / len(records)


def _make_scheduler(optimizer, protocol: dict):
    name = protocol["training"]["scheduler"]
    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if name not in {"linear", "cosine"}:
        raise ValueError("scheduler must be constant, linear or cosine")
    warmup = protocol["training"]["warmup_steps"]
    max_steps = protocol["training"]["max_steps"]
    if warmup >= max_steps:
        raise ValueError("warmup_steps must be less than max_steps")
    def scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, max_steps - warmup)
        return max(0.0, 1.0 - progress) if name == "linear" else 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _load_anchor_schedule(protocol: dict, tokenizer, selection: dict) -> AnchorTokenScheduler:
    cache_path = Path(protocol["kl"]["teacher_continuation"])
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
    if not meta_path.is_file():
        raise ValueError("teacher cache provenance metadata is missing")
    cache_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if cache_meta.get("model") != protocol["model"] or cache_meta.get("cache_sha256") != file_sha256(cache_path):
        raise ValueError("teacher cache does not match the pinned model or file digest")
    eligible_path = Path(protocol["datasets"]["anchors"]["eligible_ids_manifest"])
    if not eligible_path.is_file() or cache_meta.get("source_sha256") != file_sha256(eligible_path):
        raise ValueError("teacher cache does not match the frozen eligible-anchor pool")
    cache_rows = read_jsonl(cache_path)
    cache = {row["example_id"]: row for row in cache_rows}
    if {row["example_id"] for row in selection["selected"]} - set(cache):
        raise ValueError("selected anchors are absent from the teacher cache")
    selected_ids = {row["example_id"] for row in selection["selected"]}
    examples = {row["example_id"]: tokenize_saved_continuation(
        example_id=row["example_id"],
        prompt_token_ids=row["prompt_token_ids"],
        continuation_token_ids=row["continuation_token_ids"],
        max_sequence_tokens=protocol["runtime"]["max_sequence_tokens"],
    ) for row in cache_rows if row["example_id"] in selected_ids}
    return AnchorTokenScheduler(examples, selection["selected"], protocol["runtime"]["anchor_tokens_per_update"])


def run(protocol_path: Path, *, mode: str, experiment_id: str, condition: str, damage_weight: float | None,
        best_metric: str | None) -> Path:
    if mode not in MODE_TO_METHOD:
        raise ValueError(f"unsupported mode: {mode}")
    if condition == "discovery" and mode != "lora":
        raise ValueError("discovery adaptation must be task-only LoRA")
    protocol = copy.deepcopy(load_protocol(protocol_path))
    protocol["experiment_id"] = experiment_id
    protocol["condition"] = condition
    method = MODE_TO_METHOD[mode]
    protocol["anchor_selection"]["method"] = method
    if best_metric is not None:
        protocol["evaluation"]["checkpoint_selection_rule"] = best_metric
    if method == "none":
        protocol["anchor_selection"]["token_budget"] = 0
        protocol["anchor_selection"]["selected_ids_manifest"] = None
        protocol["anchor_selection"]["damage_weight"] = None
        protocol["kl"]["coefficient"] = 0
        protocol["training"]["anchor_interleave_every_steps"] = 0
        protocol["runtime"]["anchor_tokens_per_update"] = 0
    else:
        protocol["anchor_selection"]["selected_ids_manifest"] = str(Path(protocol["output_directory"]) / experiment_id / "anchor_selection.json")
        if method == "damage_diverse":
            damage_weight = damage_weight if damage_weight is not None else protocol["anchor_selection"].get("damage_weight")
        protocol["anchor_selection"]["damage_weight"] = damage_weight if method == "damage_diverse" else None
    rule = protocol["evaluation"]["checkpoint_selection_rule"]
    if protocol["evaluation"]["harness"] != "internal_kl_anchors_v1" or protocol["evaluation"]["harness_revision"] != evaluation_code_digest():
        raise ValueError("training requires the current internal evaluator source fingerprint")
    if rule not in ("min_validation_loss", "max_gsm8k_exact_match"):
        raise ValueError("best metric must be min_validation_loss or max_gsm8k_exact_match")
    if protocol["training"]["optimizer"] != "adamw":
        raise ValueError("current runner supports training.optimizer=adamw")
    if protocol["training"]["batch_policy"] != "shuffled_repeat":
        raise ValueError("current runner supports training.batch_policy=shuffled_repeat")
    from .evaluate_study import evaluation_from_study_protocol
    class _TemplateProbeTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "\n".join(message["content"] for message in messages) + "\n<assistant>"
    evaluation_from_study_protocol(protocol, _TemplateProbeTokenizer())
    run_dir = reserve_run(protocol)
    status_path = run_dir / "status.json"
    status_path.write_text(json.dumps({"status": "running", "mode": mode}) + "\n", encoding="utf-8")
    started = time.time()
    try:
        set_seed(protocol["seed"])
        tokenizer = load_tokenizer(protocol["model"])
        train_rows = read_jsonl(Path(protocol["training"]["training_ids_manifest"]))
        validation_rows = read_jsonl(Path(protocol["datasets"]["gsm8k"]["validation_ids_manifest"]))
        target_train = _target_examples(tokenizer, train_rows, protocol)
        target_validation = _target_examples(tokenizer, validation_rows, protocol)
        stream = TargetStream(target_train, protocol["seed"])
        lora, runtime = protocol["lora"], protocol["runtime"]
        student = attach_new_lora(load_base(protocol["model"], precision=lora["precision"], quantization=lora["quantization"], device=runtime["device"]), lora)
        teacher = None
        anchor_schedule = None
        if method != "none":
            pool_path = Path(protocol["anchor_selection"]["candidate_pool_manifest"])
            pool_meta_path = pool_path.with_suffix(pool_path.suffix + ".meta.json")
            if not pool_meta_path.is_file():
                raise ValueError("scored pool provenance metadata is missing")
            pool_meta = json.loads(pool_meta_path.read_text(encoding="utf-8"))
            if pool_meta.get("pool_sha256") != file_sha256(pool_path) or pool_meta.get("cache_sha256") != file_sha256(Path(protocol["kl"]["teacher_continuation"])):
                raise ValueError("scored pool does not match its source cache or digest")
            pool_rows = read_jsonl(pool_path)
            candidates = [Candidate(
                example_id=row["example_id"],
                response_token_count=row["response_token_count"],
                vulnerability_score=row.get("vulnerability_score"),
                diversity_rank=row.get("diversity_rank"),
            ) for row in pool_rows]
            selection = select_anchors(
                candidates, method=method,
                anchor_token_budget=protocol["anchor_selection"]["token_budget"],
                seed=protocol["seed"], damage_weight=damage_weight,
            )
            selection = {
                **selection,
                "candidate_pool_file_sha256": file_sha256(Path(protocol["anchor_selection"]["candidate_pool_manifest"])),
                "teacher_cache_file_sha256": file_sha256(Path(protocol["kl"]["teacher_continuation"])),
            }
            (run_dir / "anchor_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
            anchor_schedule = _load_anchor_schedule(protocol, tokenizer, selection)
            teacher = load_base(protocol["model"], precision=lora["precision"], quantization=lora["quantization"], device=runtime["device"])
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=protocol["training"]["learning_rate"])
        scheduler = _make_scheduler(optimizer, protocol)
        batch_size = protocol["training"]["per_device_batch_size"]
        grad_accum = protocol["training"]["gradient_accumulation_steps"]
        max_steps = protocol["training"]["max_steps"]
        checkpoint_every = runtime["checkpoint_every_steps"]
        history = []
        best_value = None
        best_path = None
        best_tiebreak_loss = float("inf")
        target_tokens_seen = 0
        anchor_tokens_seen = 0
        anchor_updates = 0
        for step in range(1, max_steps + 1):
            microbatches = [collate(stream.take(batch_size), pad_token_id=tokenizer.pad_token_id, device=runtime["device"]) for _ in range(grad_accum)]
            anchor_batch = None
            if anchor_schedule is not None and step % protocol["training"]["anchor_interleave_every_steps"] == 0:
                anchor_batch = collate(anchor_schedule.next_batch(), pad_token_id=tokenizer.pad_token_id, device=runtime["device"])
                anchor_updates += 1
            values = training_update(
                student=student, teacher=teacher, target_batches=microbatches,
                anchor_batch=anchor_batch, optimizer=optimizer,
                kl_weight=protocol["kl"]["coefficient"] if anchor_batch is not None else None,
                kl_direction=protocol["kl"]["direction"] if anchor_batch is not None else None,
                kl_temperature=protocol["kl"]["temperature"] if anchor_batch is not None else None,
                kl_reduction=protocol["kl"]["normalization"] if anchor_batch is not None else None,
                scale_by_temperature_squared=runtime["scale_kl_by_temperature_squared"] if anchor_batch is not None else None,
                scheduler=scheduler, max_grad_norm=runtime["max_grad_norm"] or None,
            )
            target_tokens_seen += values["target_tokens"]
            anchor_tokens_seen += values["anchor_tokens"]
            values.update(step=step, epoch=(step * batch_size * grad_accum / len(target_train)), learning_rate=scheduler.get_last_lr()[0])
            history.append(values)
            if step % checkpoint_every != 0 and step != max_steps:
                continue
            checkpoint = run_dir / "checkpoints" / f"step_{step:06d}"
            checkpoint.mkdir(parents=True, exist_ok=False)
            student.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            val_loss = _validation_loss(student, target_validation, batch_size=batch_size, pad_token_id=tokenizer.pad_token_id, device=runtime["device"])
            record = {"step": step, "epoch": values["epoch"], "validation_loss": val_loss}
            if rule == "max_gsm8k_exact_match":
                record["gsm8k_validation_accuracy"] = _validation_gsm8k(
                    student, tokenizer, validation_rows, protocol,
                    run_dir / "validation_predictions" / f"step_{step:06d}.jsonl",
                )
                criterion = record["gsm8k_validation_accuracy"]
                improved = best_value is None or criterion > best_value or (criterion == best_value and val_loss < best_tiebreak_loss)
            else:
                criterion = val_loss
                improved = best_value is None or criterion < best_value
            if improved:
                best_value = criterion
                best_path = checkpoint
                best_tiebreak_loss = val_loss
                record["is_best"] = True
            else:
                record["is_best"] = False
            with (run_dir / "validation_history.jsonl").open("a", encoding="utf-8") as stream_out:
                stream_out.write(json.dumps(record) + "\n")
        if best_path is None:
            raise AssertionError("training ended without a checkpoint")
        with (run_dir / "training_history.jsonl").open("x", encoding="utf-8") as stream_out:
            for record in history:
                stream_out.write(json.dumps(record) + "\n")
        shutil.copytree(best_path, run_dir / "best_model")
        summary = {
            "status": "complete", "mode": mode, "condition": condition,
            "best_checkpoint": str(best_path), "best_value": best_value,
            "best_metric": rule, "training_steps": max_steps,
            "target_examples_seen": max_steps * batch_size * grad_accum,
            "target_tokens_seen": target_tokens_seen,
            "anchor_tokens_seen": anchor_tokens_seen,
            "anchor_updates": anchor_updates,
            "per_anchor_tokens_seen": anchor_schedule.per_anchor_emitted if anchor_schedule else {},
            "target_epochs_completed": max_steps * batch_size * grad_accum / len(target_train),
            "selection_token_budget": anchor_schedule.selection_token_budget if anchor_schedule else 0,
            "elapsed_seconds": time.time() - started,
        }
        from .evaluate import runtime_info
        summary["runtime"] = runtime_info()
        status_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        from .plot_loss import plot_loss
        plot_loss(run_dir / "training_history.jsonl", run_dir / "loss_vs_epoch.png")
        return run_dir
    except Exception as exc:
        status_path.write_text(json.dumps({"status": "failed", "mode": mode, "error": str(exc), "elapsed_seconds": time.time() - started}, indent=2) + "\n", encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=tuple(MODE_TO_METHOD))
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--condition", choices=("final", "discovery"), default="final")
    parser.add_argument("--damage-weight", type=float)
    parser.add_argument("--best-metric", choices=("min_validation_loss", "max_gsm8k_exact_match"))
    args = parser.parse_args()
    path = run(args.protocol, mode=args.mode, experiment_id=args.experiment_id,
               condition=args.condition, damage_weight=args.damage_weight,
               best_metric=args.best_metric)
    print(f"Run complete: {path}")


if __name__ == "__main__":
    main()
