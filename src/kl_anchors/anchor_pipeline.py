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
from pathlib import Path

import torch

from .hf_runtime import attach_saved_lora, load_base, load_tokenizer, render_chat_prompt, set_seed
from .kl import aligned_response_kl
from .protocol import load_protocol
from .training_core import collate, tokenize_saved_continuation


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


def cache_teacher(protocol: dict, anchors_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    recorded_path = Path(protocol["datasets"]["anchors"]["eligible_ids_manifest"])
    if anchors_path.resolve() != recorded_path.resolve():
        raise ValueError("teacher cache input must match the protocol's frozen eligible-anchor manifest")
    set_seed(int(protocol["seed"]))
    tokenizer = load_tokenizer(protocol["model"])
    runtime, lora = protocol["runtime"], protocol["lora"]
    teacher = load_base(protocol["model"], precision=lora["precision"], quantization=lora["quantization"], device=runtime["device"])
    teacher.eval()
    rows = []
    for item in read_jsonl(anchors_path):
        rendered = render_chat_prompt(tokenizer, item["prompt"], runtime["anchor_system_prompt"])
        prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
        if not prompt_ids:
            raise ValueError(f"empty tokenized anchor prompt: {item['example_id']}")
        if len(prompt_ids) + runtime["teacher_max_new_tokens"] > runtime["max_sequence_tokens"]:
            raise ValueError(f"anchor too long for fixed continuation length: {item['example_id']}")
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
    write_jsonl_exclusive(output, rows)
    metadata = {
        "model": protocol["model"],
        "source_anchor_manifest": str(anchors_path.resolve()),
        "source_sha256": file_sha256(anchors_path),
        "seed": protocol["seed"],
        "teacher_max_new_tokens": runtime["teacher_max_new_tokens"],
        "anchor_system_prompt": runtime["anchor_system_prompt"],
        "cache_sha256": file_sha256(output),
    }
    with output.with_suffix(output.suffix + ".meta.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(metadata, indent=2) + "\n")


def score_discovery(protocol: dict, cache_path: Path, adapter_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    if not adapter_path.is_dir() or adapter_path.parent.name != "checkpoints" or not adapter_path.name.startswith("step_"):
        raise ValueError("discovery adapter must be an explicit saved step checkpoint")
    discovery_protocol_path = adapter_path.parent.parent / "protocol.json"
    if not discovery_protocol_path.is_file():
        raise ValueError("discovery checkpoint lacks its immutable protocol")
    discovery_protocol = load_protocol(discovery_protocol_path)
    if discovery_protocol.get("condition") != "discovery" or discovery_protocol.get("model") != protocol["model"]:
        raise ValueError("discovery checkpoint and scoring reference must use the same pinned base model")
    cache_meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
    if not cache_meta_path.is_file():
        raise ValueError("teacher cache lacks provenance metadata")
    cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
    if cache_meta.get("model") != protocol["model"] or cache_meta.get("cache_sha256") != file_sha256(cache_path):
        raise ValueError("teacher cache does not match the pinned scoring reference")
    eligible_path = Path(protocol["datasets"]["anchors"]["eligible_ids_manifest"])
    if not eligible_path.is_file() or cache_meta.get("source_sha256") != file_sha256(eligible_path):
        raise ValueError("teacher cache does not match the frozen eligible-anchor pool")
    runtime, lora = protocol["runtime"], protocol["lora"]
    model_args = dict(precision=lora["precision"], quantization=lora["quantization"], device=runtime["device"])
    teacher = load_base(protocol["model"], **model_args)
    student = attach_saved_lora(load_base(protocol["model"], **model_args), str(adapter_path))
    teacher.eval()
    student.eval()
    # The current documented vulnerability definition is teacher-to-adapted KL.
    # Its temperature and normalization still need the team's final protocol.
    if protocol["kl"]["direction"] != "teacher_to_student":
        raise ValueError("discovery vulnerability score requires teacher_to_student KL")
    rows = []
    for item in read_jsonl(cache_path):
        ex = tokenize_saved_continuation(
            example_id=item["example_id"],
            prompt_token_ids=item["prompt_token_ids"],
            continuation_token_ids=item["continuation_token_ids"],
            max_sequence_tokens=runtime["max_sequence_tokens"],
        )
        batch = collate([ex], pad_token_id=0, device=runtime["device"])
        with torch.no_grad():
            teacher_logits = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            student_logits = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            score = aligned_response_kl(
                teacher_logits, student_logits, batch["response_mask"],
                direction="teacher_to_student", temperature=protocol["kl"]["temperature"],
                reduction="token_mean", scale_by_temperature_squared=runtime["scale_kl_by_temperature_squared"],
            )
        rows.append({"example_id": item["example_id"], "vulnerability_score": float(score.cpu())})
    write_jsonl_exclusive(output, rows)
    metadata = {
        "teacher_model": protocol["model"],
        "discovery_adapter": str(adapter_path.resolve()),
        "teacher_cache": str(cache_path.resolve()),
        "teacher_cache_sha256": file_sha256(cache_path),
        "direction": "teacher_to_student",
        "temperature": protocol["kl"]["temperature"],
        "normalization": "mean_over_response_tokens",
        "scores_sha256": file_sha256(output),
    }
    with output.with_suffix(output.suffix + ".meta.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(metadata, indent=2) + "\n")


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
    score = sub.add_parser("score")
    score.add_argument("--protocol", type=Path, required=True)
    score.add_argument("--cache", type=Path, required=True)
    score.add_argument("--discovery-adapter", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--cache", type=Path, required=True)
    merge.add_argument("--scores", type=Path, required=True)
    merge.add_argument("--ranks", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "cache":
        cache_teacher(load_protocol(args.protocol), args.anchors, args.output)
    elif args.command == "score":
        score_discovery(load_protocol(args.protocol), args.cache, args.discovery_adapter, args.output)
    else:
        merge_signals(args.cache, args.scores, args.ranks, args.output)


if __name__ == "__main__":
    main()
