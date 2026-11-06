"""Cache teacher continuations, score discovery drift, and merge anchor signals.

Examples::

    python -m kl_anchors.anchor_pipeline cache --protocol protocol.json --anchors eligible_anchors.jsonl --output teacher_cache.jsonl
    python -m kl_anchors.anchor_pipeline score --protocol protocol.json --cache teacher_cache.jsonl --discovery-adapter ... --output scores.jsonl
    python -m kl_anchors.anchor_pipeline merge --cache teacher_cache.jsonl --scores scores.jsonl --ranks ranks.jsonl --output scored_pool.jsonl

The cache is the common continuation sequence for all selection methods.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Mapping

import torch

from .hf_runtime import attach_saved_lora, load_base, load_tokenizer, render_chat_prompt, set_seed
from .kl import aligned_response_kl
from .protocol import load_protocol
from .training_core import collate, tokenize_saved_continuation


_REVISION = re.compile(r"[0-9a-fA-F]{40}")


def validate_anchor_protocol(protocol: Mapping[str, object]) -> list[str]:
    """Validate fields that affect teacher caching and vulnerability scoring."""
    errors: list[str] = []

    def at(path: str):
        value = protocol
        for part in path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                return None
            value = value[part]
        return value

    def nonempty(path: str) -> None:
        value = at(path)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{path} must be a nonempty string")

    nonempty("model.repository_id")
    revision = at("model.revision")
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        errors.append("model.revision must be a full 40-character commit SHA")
    if type(at("seed")) is not int:
        errors.append("seed must be an integer")
    if at("lora.precision") not in ("fp32", "fp16", "bf16"):
        errors.append("lora.precision must be fp32, fp16, or bf16")
    if at("lora.quantization") not in ("none", "4bit", "8bit"):
        errors.append("lora.quantization must be none, 4bit, or 8bit")
    nonempty("runtime.device")
    for path in ("runtime.max_sequence_tokens", "runtime.teacher_max_new_tokens"):
        value = at(path)
        if type(value) is not int or value < 1:
            errors.append(f"{path} must be a positive integer")
    if not isinstance(at("runtime.anchor_system_prompt"), str):
        errors.append("runtime.anchor_system_prompt must be a string")
    if type(at("runtime.scale_kl_by_temperature_squared")) is not bool:
        errors.append("runtime.scale_kl_by_temperature_squared must be boolean")
    nonempty("datasets.anchors.eligible_ids_manifest")
    if at("kl.direction") != "teacher_to_student":
        errors.append("kl.direction must be teacher_to_student for vulnerability scoring")
    temperature = at("kl.temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature <= 0:
        errors.append("kl.temperature must be a positive number")
    return errors


def require_anchor_protocol(protocol: Mapping[str, object]) -> None:
    errors = validate_anchor_protocol(protocol)
    if errors:
        raise ValueError("invalid anchor-pipeline protocol:\n- " + "\n- ".join(errors))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = [row["example_id"] for row in rows]
    if len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError(f"duplicate or empty example ID in {path}")
    return rows


def write_jsonl_exclusive(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_teacher(protocol: dict, anchors_path: Path, output: Path, limit: int | None = None) -> None:
    require_anchor_protocol(protocol)
    if output.exists():
        raise FileExistsError(output)
    manifest = protocol.get("datasets", {}).get("anchors", {}).get("eligible_ids_manifest")
    if manifest and anchors_path.resolve() != Path(manifest).resolve():
        raise ValueError("teacher cache input must match the protocol's frozen eligible-anchor manifest")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    set_seed(int(protocol["seed"]))
    tokenizer = load_tokenizer(protocol["model"])
    runtime, lora = protocol["runtime"], protocol["lora"]
    teacher = load_base(protocol["model"], precision=lora["precision"], quantization=lora["quantization"], device=runtime["device"])
    teacher.eval()
    all_anchors = read_jsonl(anchors_path)
    if limit is not None and limit < len(all_anchors):
        # Deterministic random pilot, not simply the first 200 Dolly records.
        rng = random.Random(int(protocol["seed"]))
        indices = sorted(rng.sample(range(len(all_anchors)), limit))
        all_anchors = [all_anchors[i] for i in indices]
    rows = []
    skipped = []
    started = time.perf_counter()
    for index, item in enumerate(all_anchors, 1):
        template = protocol["evaluation"]["prompt_format"].replace("\\n", "\n")
        rendered = template.replace("{question}", item["prompt"])
        prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
        if not prompt_ids:
            raise ValueError(f"empty tokenized anchor prompt: {item['example_id']}")
        if len(prompt_ids) + runtime["teacher_max_new_tokens"] > runtime["max_sequence_tokens"]:
            skipped.append({
                "example_id": item["example_id"],
                "reason": "prompt_exceeds_token_limit",
                "prompt_token_count": len(prompt_ids),
                "maximum_prompt_tokens": runtime["max_sequence_tokens"] - runtime["teacher_max_new_tokens"],
            })
            print(f"SKIP overlength: {item['example_id']} ({len(prompt_ids)} tokens)", flush=True)
            if index % 25 == 0 or index == len(all_anchors):
                print(f"cache: {index}/{len(all_anchors)} processed, {len(rows)} cached, "
                      f"{len(skipped)} skipped, elapsed={time.perf_counter()-started:.1f}s", flush=True)
            continue
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=runtime["device"])
        with torch.no_grad():
            generated = teacher.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                do_sample=False,
                max_new_tokens=runtime["teacher_max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        continuation = generated[0, len(prompt_ids):].tolist()
        if not continuation:
            raise ValueError(f"teacher generated no continuation: {item['example_id']}")
        rows.append({
            "example_id": item["example_id"],
            "prompt_token_ids": prompt_ids,
            "continuation_token_ids": continuation,
            "response_token_count": len(continuation),
        })
        if index % 25 == 0 or index == len(all_anchors):
            print(f"cache: {index}/{len(all_anchors)} anchors, elapsed={time.perf_counter()-started:.1f}s", flush=True)
    if not rows:
        raise ValueError("no eligible anchors remained after length filtering")
    skip_path = output.with_suffix(output.suffix + ".skipped.jsonl")
    write_jsonl_exclusive(skip_path, skipped)
    write_jsonl_exclusive(output, rows)
    metadata = {
        "model": protocol["model"],
        "source_anchor_manifest": str(anchors_path.resolve()),
        "source_sha256": file_sha256(anchors_path),
        "seed": protocol["seed"],
        "teacher_max_new_tokens": runtime["teacher_max_new_tokens"],
        "cached_anchor_count": len(rows),
        "input_anchor_count": len(all_anchors),
        "skipped_anchor_count": len(skipped),
        "skipped_anchors_file": str(skip_path.resolve()),
        "skipped_anchors_sha256": file_sha256(skip_path),
        "cache_elapsed_seconds": time.perf_counter() - started,
        "anchor_system_prompt": runtime["anchor_system_prompt"],
        "cache_sha256": file_sha256(output),
    }
    with output.with_suffix(output.suffix + ".meta.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(metadata, indent=2) + "\n")


def score_discovery(
    protocol: dict, cache_path: Path, adapter_path: Path, output: Path,
    *, batch_size: int = 1, allow_final_checkpoint: bool = False,
) -> None:
    """Score each cached anchor individually; report throughput and GPU peak memory."""
    require_anchor_protocol(protocol)
    if output.exists():
        raise FileExistsError(output)
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if not adapter_path.is_dir() or adapter_path.parent.name != "checkpoints" or not adapter_path.name.startswith("step_"):
        raise ValueError("adapter must be an explicit checkpoints/step_* directory")
    discovery_protocol_path = adapter_path.parent.parent / "protocol.json"
    if not discovery_protocol_path.is_file():
        raise ValueError("checkpoint lacks its original run protocol.json")
    checkpoint_protocol = load_protocol(discovery_protocol_path)
    condition = checkpoint_protocol.get("condition")
    allowed = {"discovery", "final"} if allow_final_checkpoint else {"discovery"}
    if condition not in allowed:
        raise ValueError(
            f"checkpoint condition={condition!r}; use --allow-final-checkpoint "
            "to score a step checkpoint from an existing final training run"
        )
    if checkpoint_protocol.get("model") != protocol["model"]:
        raise ValueError("checkpoint and scoring reference must use the same pinned base model")
    cache_meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
    if not cache_meta_path.is_file():
        raise ValueError("teacher cache lacks provenance metadata")
    cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
    if cache_meta.get("model") != protocol["model"] or cache_meta.get("cache_sha256") != file_sha256(cache_path):
        raise ValueError("teacher cache does not match pinned scoring reference")
    source_path = Path(cache_meta["source_anchor_manifest"])
    if not source_path.is_file() or cache_meta.get("source_sha256") != file_sha256(source_path):
        raise ValueError("original eligible-anchor pool differs from cached source")
    manifest = protocol.get("datasets", {}).get("anchors", {}).get("eligible_ids_manifest")
    if manifest and (Path(manifest).resolve() != source_path.resolve()):
        raise ValueError("scoring protocol anchor manifest differs from teacher cache source")
    runtime, lora = protocol["runtime"], protocol["lora"]
    model_args = dict(precision=lora["precision"], quantization=lora["quantization"], device=runtime["device"])
    if protocol["kl"]["direction"] != "teacher_to_student":
        raise ValueError("damage scoring requires teacher_to_student KL")
    teacher = load_base(protocol["model"], **model_args)
    student = attach_saved_lora(load_base(protocol["model"], **model_args), str(adapter_path))
    teacher.eval()
    student.eval()
    cache_rows = read_jsonl(cache_path)
    if not cache_rows:
        raise ValueError("teacher cache is empty")
    rows = []
    device = runtime["device"]
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for offset in range(0, len(cache_rows), batch_size):
        chunk = cache_rows[offset:offset + batch_size]
        examples = [tokenize_saved_continuation(
            example_id=item["example_id"],
            prompt_token_ids=item["prompt_token_ids"],
            continuation_token_ids=item["continuation_token_ids"],
            max_sequence_tokens=runtime["max_sequence_tokens"],
        ) for item in chunk]
        batch = collate(examples, pad_token_id=0, device=device)
        with torch.inference_mode():
            teacher_logits = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            student_logits = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            per_token = aligned_response_kl(
                teacher_logits, student_logits, batch["response_mask"],
                direction="teacher_to_student", temperature=protocol["kl"]["temperature"],
                reduction="none", scale_by_temperature_squared=runtime["scale_kl_by_temperature_squared"],
            )
            mask = batch["response_mask"][:, 1:]
            scores = (per_token.sum(dim=1) / mask.sum(dim=1)).cpu().tolist()
        rows.extend({"example_id": item["example_id"], "vulnerability_score": float(score)}
                    for item, score in zip(chunk, scores))
        del batch, teacher_logits, student_logits, per_token, mask
        done = offset + len(chunk)
        if done % 25 < batch_size or done == len(cache_rows):
            print(f"score: {done}/{len(cache_rows)} anchors, elapsed={time.perf_counter()-started:.1f}s", flush=True)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
    else:
        peak_bytes = peak_reserved_bytes = None
    elapsed = time.perf_counter() - started
    write_jsonl_exclusive(output, rows)
    metadata = {
        "teacher_model": protocol["model"],
        "checkpoint_condition": condition,
        "discovery_adapter": str(adapter_path.resolve()),
        "teacher_cache": str(cache_path.resolve()),
        "teacher_cache_sha256": file_sha256(cache_path),
        "direction": "teacher_to_student",
        "temperature": protocol["kl"]["temperature"],
        "normalization": "mean_over_response_tokens",
        "batch_size": batch_size,
        "anchor_count": len(rows),
        "scoring_elapsed_seconds": elapsed,
        "anchors_per_second": len(rows) / elapsed,
        "estimated_full_13722_seconds": 13722 * elapsed / len(rows),
        "peak_cuda_allocated_bytes": peak_bytes,
        "peak_cuda_reserved_bytes": peak_reserved_bytes,
        "scores_sha256": file_sha256(output),
    }
    with output.with_suffix(output.suffix + ".meta.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: metadata[k] for k in (
        "batch_size", "anchor_count", "scoring_elapsed_seconds", "anchors_per_second",
        "estimated_full_13722_seconds", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes")}, indent=2), flush=True)


def merge_signals(cache_path: Path, scores_path: Path, ranks_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    score_meta_path = scores_path.with_suffix(scores_path.suffix + ".meta.json")
    if not score_meta_path.is_file():
        raise ValueError("vulnerability scores lack provenance metadata")
    score_meta = json.loads(score_meta_path.read_text(encoding="utf-8"))
    if score_meta.get("teacher_cache_sha256") != file_sha256(cache_path) or score_meta.get("scores_sha256") != file_sha256(scores_path):
        raise ValueError("vulnerability scores do not match the teacher cache or file digest")
    cache = read_jsonl(cache_path)
    scores = {r["example_id"]: r["vulnerability_score"] for r in read_jsonl(scores_path)}
    ranks = {r["example_id"]: r["diversity_rank"] for r in read_jsonl(ranks_path)}
    ids = {r["example_id"] for r in cache}
    if ids != set(scores) or ids != set(ranks):
        raise ValueError("cache, scores and diversity ranks must cover exactly the same eligible pool")
    rows = [{
        "example_id": row["example_id"],
        "response_token_count": row["response_token_count"],
        "vulnerability_score": scores[row["example_id"]],
        "diversity_rank": ranks[row["example_id"]],
    } for row in cache]
    write_jsonl_exclusive(output, rows)
    meta = {
        "cache_sha256": file_sha256(cache_path),
        "scores_sha256": file_sha256(scores_path),
        "ranks_sha256": file_sha256(ranks_path),
        "pool_sha256": file_sha256(output),
    }
    with output.with_suffix(output.suffix + ".meta.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("cache")
    cache.add_argument("--protocol", type=Path, required=True)
    cache.add_argument("--anchors", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--limit", type=int, default=None, help="deterministic random pilot sample size")
    score = sub.add_parser("score")
    score.add_argument("--protocol", type=Path, required=True)
    score.add_argument("--cache", type=Path, required=True)
    score.add_argument("--discovery-adapter", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--batch-size", type=int, default=1)
    score.add_argument("--allow-final-checkpoint", action="store_true", help="allow step checkpoint from a final training run")
    merge = sub.add_parser("merge")
    merge.add_argument("--cache", type=Path, required=True)
    merge.add_argument("--scores", type=Path, required=True)
    merge.add_argument("--ranks", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "cache":
        cache_teacher(load_protocol(args.protocol), args.anchors, args.output, limit=args.limit)
    elif args.command == "score":
        score_discovery(load_protocol(args.protocol), args.cache, args.discovery_adapter, args.output,
                        batch_size=args.batch_size, allow_final_checkpoint=args.allow_final_checkpoint)
    else:
        merge_signals(args.cache, args.scores, args.ranks, args.output)


if __name__ == "__main__":
    main()
