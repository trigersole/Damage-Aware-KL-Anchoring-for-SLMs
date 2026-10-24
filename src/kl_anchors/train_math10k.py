"""Exact-epoch MATH-10K LoRA baseline; separate from the existing GSM8K runner.

Run: python -m kl_anchors.train_math10k --protocol configs/protocol.math10k-llama32-3b.json --experiment-id UNIQUE_ID
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
from .training_core import collate, tokenize_completion, target_causal_loss, training_update


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


def run(args):
    protocol_path = Path(args.protocol)
    p = json.loads(protocol_path.read_text(encoding="utf-8"))
    if p["model"]["repository_id"] != "meta-llama/Llama-3.2-3B":
        raise ValueError("This runner is scoped to the pinned Llama-3.2-3B base study")
    if p["lora"]["quantization"] != "none" or p["training"]["optimizer"] != "adamw":
        raise ValueError("This baseline requires unquantized LoRA and standard AdamW")
    if p["evaluation"]["checkpoint_selection_rule"] != "min_validation_loss":
        raise ValueError("Checkpoint selection must use validation loss")
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
    if args.preflight:
        return

    t = p["training"]
    runtime = p["runtime"]
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
    (run_dir / "status.json").write_text('{"status":"running"}\n', encoding="utf-8")
    try:
        model = attach_new_lora(load_base(p["model"], precision=p["lora"]["precision"], quantization="none", device=device), p["lora"])
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
        start_time = time.monotonic()
        with (run_dir / "training_history.jsonl").open("x", encoding="utf-8") as history:
            for step, (epoch, group) in enumerate(epoch_batches(train, int(p["seed"]), epochs, effective), 1):
                microbatches = [collate(group[i:i + micro], pad_token_id=tokenizer.pad_token_id, device=device) for i in range(0, len(group), micro)]
                metrics = training_update(
                    student=model, teacher=None, target_batches=microbatches,
                    anchor_batch=None, optimizer=optimizer, kl_weight=None,
                    kl_direction=None, kl_temperature=None, kl_reduction=None,
                    scale_by_temperature_squared=None, scheduler=scheduler,
                    max_grad_norm=runtime["max_grad_norm"],
                )
                examples_seen += len(group)
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
                print(f"checkpoint={step} validation_loss={val_loss:.5f}", flush=True)
                if args.smoke and step >= args.smoke:
                    break
        if best_path is None:
            raise RuntimeError("No checkpoint saved; for smoke set checkpoint_every_steps <= smoke steps")
        shutil.copytree(best_path, run_dir / "best_model")
        summary = {"status": "complete", "steps_completed": step, "examples_seen": examples_seen, "best_validation_loss": best_loss, "best_checkpoint": str(best_path), "smoke": bool(args.smoke)}
        if not args.smoke and examples_seen != len(train) * epochs:
            raise AssertionError("Not exactly three complete epochs")
        (run_dir / "status.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    except Exception as e:
        (run_dir / "status.json").write_text(json.dumps({"status": "failed", "error": str(e)}, indent=2) + "\n", encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", type=int, default=0, help="Stop after N updates (requires checkpoint interval <= N)")
    run(parser.parse_args())

if __name__ == "__main__":
    main()
