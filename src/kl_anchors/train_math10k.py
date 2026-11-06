"""Exact-epoch MATH-10K LoRA and diversity-KL runner.

The task-only mode preserves the completed baseline behavior. ``diverse_kl``
starts a fresh LoRA adapter from the same pinned base and applies token-aligned
teacher-to-student KL using a pre-ranked anchor pool.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch

from .hf_runtime import attach_new_lora, load_base, load_tokenizer, set_seed
from .anchor_pipeline import file_sha256
from .selection import Candidate, select_anchors
from .training_core import (
    AnchorTokenScheduler, collate, tokenize_completion,
    tokenize_saved_continuation, target_causal_loss, training_update,
)


def read_rows(path: str):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    ids = [r["example_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate example IDs in {path}")
    return rows


def render_prompt(template: str, question: str) -> str:
    if template.count("{question}") != 1:
        raise ValueError("Prompt template must contain exactly one {question}")
    return template.replace("{question}", question)


def make_examples(tokenizer, rows, protocol):
    template = protocol["evaluation"]["prompt_format"].replace("\\n", "\n")
    limit = int(protocol["runtime"]["max_sequence_tokens"])
    return [tokenize_completion(
        tokenizer,
        example_id=row["example_id"],
        rendered_prompt=render_prompt(template, row["prompt"]),
        response=row["response"],
        max_sequence_tokens=limit,
        append_eos=bool(protocol["runtime"]["target_answer_append_eos"]),
    ) for row in rows]


def epoch_batches(examples, seed, epochs, effective_batch_size):
    """Shuffle separately each epoch; never repeat examples across partial batches."""
    for epoch in range(1, epochs + 1):
        order = list(range(len(examples)))
        random.Random(seed + epoch - 1).shuffle(order)
        for start in range(0, len(order), effective_batch_size):
            yield epoch, [examples[i] for i in order[start:start + effective_batch_size]]


def validation_loss(model, examples, batch_size, pad_token_id, device):
    model.eval()
    weighted, tokens = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = collate(examples[start:start + batch_size], pad_token_id=pad_token_id, device=device)
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            count = int(batch["response_mask"].sum().item())
            weighted += float(target_causal_loss(logits, batch["input_ids"], batch["response_mask"]).item()) * count
            tokens += count
    return weighted / tokens


def prune_checkpoints(run_dir, best_path, keep_last):
    """Apply optional disk retention after a checkpoint is saved and evaluated."""
    if type(keep_last) is not int or keep_last < 0:
        raise ValueError("keep_last_checkpoints must be a nonnegative integer")
    if keep_last == 0:
        return []
    root = Path(run_dir) / "checkpoints"
    checkpoints = sorted(
        (p for p in root.iterdir() if p.is_dir() and not p.is_symlink()
         and p.name.startswith("step_") and p.name[5:].isdigit()),
        key=lambda p: int(p.name[5:]),
    )
    protected = set(checkpoints[-keep_last:]) | {Path(best_path)}
    removed = []
    for checkpoint in checkpoints:
        if checkpoint not in protected:
            shutil.rmtree(checkpoint)
            removed.append(str(checkpoint))
    if removed:
        with (Path(run_dir) / "checkpoint_cleanup.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"removed": removed, "preserved_best": str(best_path),
                                     "keep_last": keep_last}) + "\n")
    return removed


def validate_mode(protocol, mode):
    """Fail closed on the task-only or screenshot-approved diverse-KL protocol."""
    if mode not in {"lora", "diverse_kl"}:
        raise ValueError("mode must be lora or diverse_kl")
    selection = protocol["anchor_selection"]
    training = protocol["training"]
    runtime = protocol["runtime"]
    kl = protocol["kl"]
    if mode == "lora":
        if selection["method"] != "none" or selection["token_budget"] != 0:
            raise ValueError("task-only mode requires method none and zero anchor budget")
        if training["anchor_interleave_every_steps"] != 0 or runtime["anchor_tokens_per_update"] != 0:
            raise ValueError("task-only mode requires zero KL exposure")
        if kl["coefficient"] != 0:
            raise ValueError("task-only mode requires zero KL coefficient")
        return
    if selection["method"] != "diversity":
        raise ValueError("diverse_kl mode requires anchor_selection.method=diversity")
    if type(selection.get("prompt_budget")) is not int or selection["prompt_budget"] < 1:
        raise ValueError("diverse_kl requires a positive prompt_budget")
    if type(selection.get("token_budget")) is not int or selection["token_budget"] < selection["prompt_budget"]:
        raise ValueError("token_budget must allow at least one position per prompt")
    if type(training["anchor_interleave_every_steps"]) is not int or training["anchor_interleave_every_steps"] < 1:
        raise ValueError("diverse_kl requires a positive KL frequency")
    if type(runtime["anchor_tokens_per_update"]) is not int or not 1 <= runtime["anchor_tokens_per_update"] <= selection["token_budget"]:
        raise ValueError("anchor_tokens_per_update must be within the selected token budget")
    if type(runtime.get("anchor_microbatch_size")) is not int or runtime["anchor_microbatch_size"] < 1:
        raise ValueError("diverse_kl requires a positive anchor_microbatch_size")
    if not isinstance(kl["coefficient"], (int, float)) or isinstance(kl["coefficient"], bool) or kl["coefficient"] <= 0:
        raise ValueError("diverse_kl requires a positive KL coefficient")
    if kl["direction"] != "teacher_to_student" or kl["temperature"] <= 0:
        raise ValueError("diverse_kl requires positive-temperature teacher_to_student KL")
    if kl["token_mask"] != "response_tokens" or kl["normalization"] != "token_mean":
        raise ValueError("diverse_kl requires response-token masking and token-mean normalization")
    for path in (selection["candidate_pool_manifest"], kl["teacher_continuation"]):
        if not isinstance(path, str) or not Path(path).is_file():
            raise ValueError(f"required anchor artifact is missing: {path}")


def prepare_diverse_anchors(protocol, tokenizer):
    """Validate provenance, select exact prompt/token budgets, and build a scheduler."""
    selection_config = protocol["anchor_selection"]
    pool_path = Path(selection_config["candidate_pool_manifest"])
    cache_path = Path(protocol["kl"]["teacher_continuation"])
    pool_rows = read_rows(pool_path)
    cache_rows = read_rows(cache_path)
    if {row["example_id"] for row in pool_rows} != {row["example_id"] for row in cache_rows}:
        raise ValueError("diversity ranks and teacher cache must cover exactly the same anchors")
    diversity_config = selection_config["diversity"]
    expected_metadata = {
        "model_id": diversity_config["model_id"],
        "model_revision": diversity_config["model_revision"],
        "algorithm": diversity_config["algorithm"],
        "seed": diversity_config["seed"],
    }
    for row in pool_rows:
        metadata = row.get("diversity_metadata", {})
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError("diversity-rank metadata does not match the locked protocol")
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
    if not meta_path.is_file():
        raise ValueError("teacher-cache provenance metadata is missing")
    cache_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    eligible_path = Path(protocol["datasets"]["anchors"]["eligible_ids_manifest"])
    if cache_meta.get("model") != protocol["model"]:
        raise ValueError("teacher cache was produced by a different model")
    if cache_meta.get("cache_sha256") != file_sha256(cache_path):
        raise ValueError("teacher-cache digest does not match its metadata")
    if cache_meta.get("source_sha256") != file_sha256(eligible_path):
        raise ValueError("teacher cache does not match the locked eligible-anchor manifest")
    candidates = [Candidate(
        row["example_id"], int(row["response_token_count"]),
        diversity_rank=int(row["diversity_rank"]),
    ) for row in pool_rows]
    selection = select_anchors(
        candidates, method="diversity",
        anchor_token_budget=int(selection_config["token_budget"]),
        anchor_prompt_budget=int(selection_config["prompt_budget"]),
        seed=int(protocol["seed"]),
    )
    selection = {
        **selection,
        "candidate_pool_file_sha256": file_sha256(pool_path),
        "teacher_cache_file_sha256": file_sha256(cache_path),
    }
    selected_ids = {row["example_id"] for row in selection["selected"]}
    examples = {row["example_id"]: tokenize_saved_continuation(
        example_id=row["example_id"],
        prompt_token_ids=row["prompt_token_ids"],
        continuation_token_ids=row["continuation_token_ids"],
        max_sequence_tokens=int(protocol["runtime"]["max_sequence_tokens"]),
    ) for row in cache_rows if row["example_id"] in selected_ids}
    schedule = AnchorTokenScheduler(
        examples, selection["selected"],
        int(protocol["runtime"]["anchor_tokens_per_update"]),
    )
    return selection, schedule


def run(args):
    protocol_path = Path(args.protocol)
    p = json.loads(protocol_path.read_text(encoding="utf-8"))
    if p["model"]["repository_id"] != "meta-llama/Llama-3.2-3B":
        raise ValueError("This runner is scoped to the pinned Llama-3.2-3B base study")
    if p["lora"]["quantization"] != "none" or p["training"]["optimizer"] != "adamw":
        raise ValueError("This baseline requires unquantized LoRA and standard AdamW")
    if p["evaluation"]["checkpoint_selection_rule"] != "min_validation_loss":
        raise ValueError("Checkpoint selection must use validation loss")
    p["condition"] = "final"
    p["experiment_id"] = args.experiment_id
    validate_mode(p, args.mode)
    required_manifests = (
        Path(p["training"]["training_ids_manifest"]),
        Path(p["datasets"]["math10k"]["validation_ids_manifest"]),
    )
    missing = [str(path) for path in required_manifests if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing prepared MATH-10K manifests: " + ", ".join(missing)
            + ". Run python -m kl_anchors.prepare_math10k as documented in data/README.md."
        )
    set_seed(int(p["seed"]))
    tokenizer = load_tokenizer(p["model"])
    train_rows = read_rows(p["training"]["training_ids_manifest"])
    val_rows = read_rows(p["datasets"]["math10k"]["validation_ids_manifest"])
    if len(train_rows) != 9619 or len(val_rows) != 300:
        raise ValueError("Unexpected MATH-10K split sizes")
    if set(r["example_id"] for r in train_rows) & set(r["example_id"] for r in val_rows):
        raise ValueError("Train and validation IDs overlap")
    train = make_examples(tokenizer, train_rows, p)
    val = make_examples(tokenizer, val_rows, p)
    print(f"Tokenization OK: train={len(train)}, validation={len(val)}, longest={max(map(lambda x: len(x.input_ids), train + val))}", flush=True)
    anchor_selection = anchor_schedule = None
    if args.mode == "diverse_kl":
        anchor_selection, anchor_schedule = prepare_diverse_anchors(p, tokenizer)
    if args.preflight:
        if anchor_selection is not None:
            print(
                f"Anchor preflight OK: prompts={anchor_selection['selected_prompt_count']} "
                f"positions={anchor_selection['anchor_token_budget']}", flush=True,
            )
        return

    t = p["training"]
    runtime = p["runtime"]
    keep_last = runtime.get("keep_last_checkpoints", 0)
    if type(keep_last) is not int or keep_last < 0:
        raise ValueError("keep_last_checkpoints must be a nonnegative integer")
    device = runtime["device"]
    micro = int(t["per_device_batch_size"])
    accum = int(t["gradient_accumulation_steps"])
    effective = micro * accum
    epochs = int(t["epochs"])
    steps_per_epoch = math.ceil(len(train) / effective)
    total_steps = steps_per_epoch * epochs
    if int(t["max_steps"]) != total_steps:
        raise ValueError(f"Protocol max_steps={t['max_steps']} but exact epochs require {total_steps}")
    if micro < 1 or accum < 1 or epochs < 1:
        raise ValueError("Batch size, accumulation and epochs must be positive")
    run_dir = Path(p["output_directory"]) / args.experiment_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "protocol.json").write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
    if anchor_selection is not None:
        (run_dir / "anchor_selection.json").write_text(
            json.dumps(anchor_selection, indent=2) + "\n", encoding="utf-8"
        )
    (run_dir / "status.json").write_text('{"status":"running"}\n', encoding="utf-8")
    try:
        model = attach_new_lora(load_base(p["model"], precision=p["lora"]["precision"], quantization="none", device=device), p["lora"])
        teacher = None
        if args.mode == "diverse_kl":
            teacher = load_base(
                p["model"], precision=p["lora"]["precision"],
                quantization="none", device=device,
            )
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=float(t["learning_rate"]), weight_decay=0.01)
        warmup = int(t["warmup_steps"])
        scheduler_name = t["scheduler"]
        if scheduler_name not in {"linear", "constant"} or warmup >= total_steps:
            raise ValueError("Unsupported scheduler or warmup")
        def lr_factor(step):
            if scheduler_name == "constant":
                return 1.0
            if step < warmup:
                return (step + 1) / max(1, warmup)
            return max(0.0, (total_steps - step) / max(1, total_steps - warmup))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        checkpoint_every = int(runtime["checkpoint_every_steps"])
        best_loss = float("inf")
        best_path = None
        examples_seen = 0
        anchor_tokens_seen = 0
        anchor_updates = 0
        start_time = time.monotonic()
        with (run_dir / "training_history.jsonl").open("x", encoding="utf-8") as history:
            for step, (epoch, group) in enumerate(epoch_batches(train, int(p["seed"]), epochs, effective), 1):
                microbatches = [collate(group[i:i + micro], pad_token_id=tokenizer.pad_token_id, device=device) for i in range(0, len(group), micro)]
                anchor_batches = None
                if anchor_schedule is not None and step % int(t["anchor_interleave_every_steps"]) == 0:
                    anchor_examples = anchor_schedule.next_batch()
                    anchor_micro = int(runtime["anchor_microbatch_size"])
                    anchor_batches = [collate(
                        anchor_examples[i:i + anchor_micro],
                        pad_token_id=tokenizer.pad_token_id, device=device,
                    ) for i in range(0, len(anchor_examples), anchor_micro)]
                    anchor_updates += 1
                metrics = training_update(
                    student=model, teacher=teacher, target_batches=microbatches,
                    anchor_batch=anchor_batches, optimizer=optimizer,
                    kl_weight=float(p["kl"]["coefficient"]) if anchor_batches else None,
                    kl_direction=p["kl"]["direction"] if anchor_batches else None,
                    kl_temperature=float(p["kl"]["temperature"]) if anchor_batches else None,
                    kl_reduction=p["kl"]["normalization"] if anchor_batches else None,
                    scale_by_temperature_squared=runtime["scale_kl_by_temperature_squared"] if anchor_batches else None,
                    scheduler=scheduler,
                    max_grad_norm=runtime["max_grad_norm"],
                )
                examples_seen += len(group)
                anchor_tokens_seen += metrics["anchor_tokens"]
                record = {"step": step, "epoch": epoch, "examples_seen": examples_seen, "batch_examples": len(group), **metrics}
                history.write(json.dumps(record) + "\n")
                if step % 25 == 0 or step == total_steps:
                    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
                    print(f"step={step}/{total_steps} epoch={epoch} loss={metrics['target_loss']:.4f} peak_gpu_gib={peak:.2f} elapsed_min={(time.monotonic()-start_time)/60:.1f}", flush=True)
                if step % checkpoint_every != 0 and step != total_steps:
                    continue
                checkpoint = run_dir / "checkpoints" / f"step_{step:06d}"
                checkpoint.mkdir(parents=True, exist_ok=False)
                model.save_pretrained(checkpoint)
                tokenizer.save_pretrained(checkpoint)
                val_loss = validation_loss(model, val, micro, tokenizer.pad_token_id, device)
                with (run_dir / "validation_history.jsonl").open("a", encoding="utf-8") as out:
                    out.write(json.dumps({"step": step, "validation_loss": val_loss}) + "\n")
                if val_loss < best_loss:
                    best_loss, best_path = val_loss, checkpoint
                prune_checkpoints(run_dir, best_path, keep_last)
                print(f"checkpoint={step} validation_loss={val_loss:.5f}", flush=True)
                if args.smoke and step >= args.smoke:
                    break
        if best_path is None:
            raise RuntimeError("No checkpoint saved; for smoke set checkpoint_every_steps <= smoke steps")
        shutil.copytree(best_path, run_dir / "best_model")
        summary = {
            "status": "complete", "mode": args.mode,
            "steps_completed": step, "examples_seen": examples_seen,
            "best_validation_loss": best_loss, "best_checkpoint": str(best_path),
            "smoke": bool(args.smoke), "anchor_updates": anchor_updates,
            "anchor_tokens_seen": anchor_tokens_seen,
            "selection_prompt_budget": anchor_selection["selected_prompt_count"] if anchor_selection else 0,
            "selection_token_budget": anchor_selection["anchor_token_budget"] if anchor_selection else 0,
            "per_anchor_tokens_seen": anchor_schedule.per_anchor_emitted if anchor_schedule else {},
        }
        if not args.smoke and examples_seen != len(train) * epochs:
            raise AssertionError("Not exactly three complete epochs")
        if not args.smoke and anchor_schedule is not None:
            expected = (total_steps // int(t["anchor_interleave_every_steps"])) * int(runtime["anchor_tokens_per_update"])
            if anchor_tokens_seen != expected:
                raise AssertionError(f"expected {expected} anchor positions, observed {anchor_tokens_seen}")
        (run_dir / "status.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    except Exception as e:
        (run_dir / "status.json").write_text(json.dumps({"status": "failed", "error": str(e)}, indent=2) + "\n", encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--mode", choices=("lora", "diverse_kl"), default="lora")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", type=int, default=0, help="Stop after N updates (requires checkpoint interval <= N)")
    run(parser.parse_args())

if __name__ == "__main__":
    main()
