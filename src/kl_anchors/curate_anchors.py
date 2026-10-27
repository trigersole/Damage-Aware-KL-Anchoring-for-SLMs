"""Export inputs, propose anchor reviews, and freeze a reviewed common pool.

This is an opt-in replacement for the automatic surface/semantic exclusion path.
Legacy commands remain available. ``propose --lexical-only`` produces a preview
that cannot be finalized. Real semantic runs require explicit model, revision,
thresholds and text-length policy; no research values are chosen by default.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import re

import numpy as np

from .curation import apply_reviews, assemble_flags, checked_inputs, digest, semantic_pairs, surface_proposal
from .prepare_data import (
    ROW_KEYS, _specs, build_candidates, build_heldout, load_hf_rows, split_gsm8k_train,
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _jsonl(path: Path, values: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in values:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def file_sha(path: Path) -> str:
    result = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def code_hashes() -> dict:
    directory = Path(__file__).resolve().parent
    return {name: file_sha(directory / name) for name in (
        "curation.py", "curate_anchors.py", "data_audit.py", "prepare_data.py",
    )}


def software() -> dict:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "torch", "sentence-transformers", "transformers", "datasets"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def input_rows(config: dict, rows_by_key: dict, *, target_train_overlap: str) -> tuple[list, list]:
    """Rebuild the full raw pool, including candidates the old pipeline excluded."""
    if target_train_overlap not in {"allow", "exclude"}:
        raise ValueError("target_train_overlap must be explicitly allow or exclude")
    specs, gsm = _specs(config), config["gsm8k"]
    train, validation = split_gsm8k_train(
        rows_by_key["gsm8k_train"], specs["gsm8k_train"],
        question_column=gsm["question_column"], answer_column=gsm["answer_column"],
        validation_count=gsm["validation_count"], split_seed=gsm["split_seed"],
    )
    anchors = config["anchors"]
    candidates = build_candidates(
        rows_by_key["anchors"], specs["anchors"], prompt_column=anchors["prompt_column"],
        context_column=anchors.get("context_column"), context_template=anchors.get("context_template"),
    )
    heldout = [{"dataset": "gsm8k_validation", "example_id": x["example_id"], "text": x["prompt"]}
               for x in validation]
    if target_train_overlap == "exclude":
        heldout.extend({"dataset": "gsm8k_train", "example_id": x["example_id"], "text": x["prompt"]}
                       for x in train)
    for kind in ("gsm8k", "boolq", "hellaswag", "arc_easy"):
        key = "gsm8k_test" if kind == "gsm8k" else kind
        columns = {"question": gsm["question_column"]} if kind == "gsm8k" else config["retention"][kind]["columns"]
        heldout.extend({"dataset": x.dataset, "example_id": x.id, "text": x.text}
                       for x in build_heldout(kind, rows_by_key[key], specs[key], columns, dataset_tag=key))
    return checked_inputs([{"example_id": x.id, "prompt": x.text} for x in candidates], heldout)


def export_inputs(config_path: Path, output: Path, *, target_train_overlap: str, loader=load_hf_rows) -> dict:
    if output.exists():
        raise FileExistsError(output)
    config = _read_json(config_path)
    specs = _specs(config)
    for key, spec in specs.items():
        if not re.fullmatch(r"[0-9a-fA-F]{40}", spec.revision):
            raise ValueError(f"{key}: pin a 40-character source revision")
    rows = {key: loader(specs[key]) for key in ROW_KEYS}
    candidates, heldout = input_rows(config, rows, target_train_overlap=target_train_overlap)
    # Validate the inherited quality policy before creating files. Its old
    # review_completed flag is not accepted as evidence of pair review here.
    surface_proposal(candidates, heldout, overlap=config["overlap"],
                     candidate_filters=config["candidate_filters"])
    output.mkdir(parents=True, exist_ok=False)
    _jsonl(output / "candidates.jsonl", candidates)
    _jsonl(output / "heldout_inputs.jsonl", heldout)
    _json(output / "data_config.json", config)
    manifest = {
        "schema_version": 1, "stage": "inputs", "target_train_overlap": target_train_overlap,
        "candidate_count": len(candidates), "heldout_count": len(heldout),
        "sources": {key: asdict(spec) for key, spec in specs.items()},
        "source_counts": {key: len(value) for key, value in rows.items()},
        "files": {name: file_sha(output / name) for name in (
            "candidates.jsonl", "heldout_inputs.jsonl", "data_config.json")},
        "code_sha256": code_hashes(), "software": software(),
    }
    _json(output / "inputs_manifest.json", manifest)
    return manifest


def checked_manifest(directory: Path, name: str, expected_files: set[str]) -> dict:
    manifest = _read_json(directory / name)
    if manifest.get("schema_version") != 1 or set(manifest.get("files", {})) != expected_files:
        raise ValueError("invalid manifest schema or file set")
    for filename, expected in manifest["files"].items():
        if file_sha(directory / filename) != expected:
            raise ValueError(f"input checksum mismatch: {filename}")
    return manifest


def encode_inputs(candidates: list[dict], heldout: list[dict], *, model_id: str,
                  model_revision: str, max_seq_length: int, truncation: str,
                  device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Encode full rendered inputs; reject unapproved silent text truncation."""
    if not model_id or not re.fullmatch(r"[0-9a-fA-F]{40}", model_revision or ""):
        raise ValueError("a model ID and pinned 40-character embedding revision are required")
    if type(max_seq_length) is not int or max_seq_length <= 0 or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("max_seq_length and batch_size must be positive")
    if truncation not in {"reject", "allow"} or not device:
        raise ValueError("explicit truncation policy and device are required")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, revision=model_revision, device=device, trust_remote_code=False)
    if max_seq_length > model.max_seq_length:
        raise ValueError(f"requested length exceeds encoder's configured limit {model.max_seq_length}")
    model.max_seq_length = max_seq_length
    model.eval()
    texts = [x["prompt"] for x in candidates] + [x["text"] for x in heldout]
    identities = ([{"candidate_id": x["example_id"]} for x in candidates] +
                  [{"dataset": x["dataset"], "heldout_id": x["example_id"]} for x in heldout])
    overlong = []
    for start in range(0, len(texts), batch_size):
        encoded = model.tokenizer(texts[start:start + batch_size], truncation=False,
                                  padding=False, add_special_tokens=True)
        for i, ids in enumerate(encoded["input_ids"]):
            if len(ids) > max_seq_length:
                overlong.append({**identities[start + i], "tokens": len(ids)})
    if overlong and truncation == "reject":
        raise ValueError(f"{len(overlong)} inputs exceed the encoder limit; no embeddings produced. "
                         "Agree a long-text policy or a suitable encoder before proceeding.")
    # An empty prompt disables any model-specific default instruction prefix;
    # therefore the token-length check describes the exact input being encoded.
    vectors = np.asarray(model.encode(texts, batch_size=batch_size, prompt="",
                                     convert_to_numpy=True, normalize_embeddings=True,
                                     show_progress_bar=True))
    metadata = {"model_id": model_id, "model_revision": model_revision,
                "max_seq_length": max_seq_length, "truncation": truncation,
                "truncated_inputs": overlong, "device": device, "batch_size": batch_size,
                "encoder_prompt": "", "vectors_sha256": sha256(vectors.tobytes()).hexdigest(),
                "vectors_shape": list(vectors.shape), "vectors_dtype": str(vectors.dtype)}
    return vectors[:len(candidates)], vectors[len(candidates):], metadata


def propose(inputs: Path, output: Path, *, lexical_only: bool, sample_size: int,
            sample_seed: int, heldout_threshold: float | None = None,
            dedup_threshold: float | None = None, chunk_size: int = 128,
            embedding_options: dict | None = None, encoder=encode_inputs,
            export_auto_included: bool = False) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if export_auto_included and lexical_only:
        raise ValueError("auto-included preview requires complete semantic screening")
    if type(sample_size) is not int or sample_size < 0 or type(sample_seed) is not int:
        raise ValueError("sample size must be nonnegative and sample seed an integer")
    source = checked_manifest(inputs, "inputs_manifest.json", {
        "candidates.jsonl", "heldout_inputs.jsonl", "data_config.json"})
    if source.get("stage") != "inputs":
        raise ValueError("expected exported inputs")
    candidates, heldout = checked_inputs(_read_rows(inputs / "candidates.jsonl"),
                                        _read_rows(inputs / "heldout_inputs.jsonl"))
    config = _read_json(inputs / "data_config.json")
    hard, lexical = surface_proposal(candidates, heldout, overlap=config["overlap"],
                                    candidate_filters=config["candidate_filters"])
    active_ids = {x["example_id"] for x in hard}
    active = [x for x in candidates if x["example_id"] not in active_ids]
    if not active:
        raise ValueError("quality/exact filters removed every candidate")
    semantic, embedding = None, None
    if lexical_only:
        if heldout_threshold is not None or dedup_threshold is not None or embedding_options:
            raise ValueError("lexical-only preview must not supply semantic settings")
    else:
        # Validate thresholds and sampling before any model download.
        semantic_pairs(np.ones((1, 1)), np.ones((1, 1)), heldout_threshold=heldout_threshold,
                       dedup_threshold=dedup_threshold, chunk_size=chunk_size)
        a, b, embedding = encoder(active, heldout, **(embedding_options or {}))
        if len(a) != len(active) or len(b) != len(heldout):
            raise ValueError("encoder outputs are not aligned with input rows")
        semantic = semantic_pairs(a, b, heldout_threshold=heldout_threshold,
                                  dedup_threshold=dedup_threshold, chunk_size=chunk_size)
    flags, summary = assemble_flags(candidates, heldout, hard, lexical, semantic=semantic,
                                   sample_size=sample_size, sample_seed=sample_seed)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("candidates.jsonl", candidates), ("heldout_inputs.jsonl", heldout),
                       ("hard_exclusions.jsonl", hard), ("review_flags.jsonl", flags),
                       ("semantic_summary.jsonl", summary)):
        _jsonl(output / name, rows)
    _json(output / "input_provenance.json", source)
    _json(output / "data_config.json", config)
    filenames = {"candidates.jsonl", "heldout_inputs.jsonl", "hard_exclusions.jsonl", "review_flags.jsonl",
                 "semantic_summary.jsonl", "input_provenance.json", "data_config.json"}
    if export_auto_included:
        # Both participants in a duplicate pair and pending spot checks require
        # review. This inspection output does not approve or freeze a pool.
        pending_ids = {flag["candidate_id"] for flag in flags}
        pending_ids.update(flag["other_candidate_id"] for flag in flags
                           if flag["kind"] == "semantic_duplicate")
        preview = [row for row in active if row["example_id"] not in pending_ids]
        _jsonl(output / "auto_included_anchors.jsonl", preview)
        filenames.add("auto_included_anchors.jsonl")
    manifest = {
        "schema_version": 1, "stage": "review", "semantic_screen_complete": not lexical_only,
        "candidate_count": len(candidates), "hard_excluded_count": len(hard),
        "review_count": len(flags), "target_train_overlap": source["target_train_overlap"],
        "review_sample": {"requested": sample_size, "seed": sample_seed,
                          "actual": sum(x["kind"] == "spot_check" for x in flags)},
        "embedding": embedding, "heldout_threshold": heldout_threshold,
        "dedup_threshold": dedup_threshold, "chunk_size": chunk_size,
        "deduplication": "reviewed_direct_retained_representative_in_sorted_id_order",
        "files": {name: file_sha(output / name) for name in sorted(filenames)},
        "code_sha256": code_hashes(), "software": software(),
    }
    if export_auto_included:
        manifest["auto_included_preview"] = {
            "file": "auto_included_anchors.jsonl", "count": len(preview),
            "purpose": "inspection only; use finalized eligible_anchors.jsonl for training",
        }
    manifest["bundle_sha256"] = digest(manifest)
    _json(output / "review_manifest.json", manifest)
    _jsonl(output / "review_answers.template.jsonl", [
        {"review_id": flag["review_id"], "bundle_sha256": manifest["bundle_sha256"],
         "decision": "pending", "reviewer": "", "reason": ""} for flag in flags])
    return manifest


def finalize(review: Path, answers_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    expected_files = {
        "candidates.jsonl", "heldout_inputs.jsonl", "hard_exclusions.jsonl", "review_flags.jsonl",
        "semantic_summary.jsonl", "input_provenance.json", "data_config.json"}
    recorded = _read_json(review / "review_manifest.json")
    if "auto_included_anchors.jsonl" in recorded.get("files", {}):
        expected_files.add("auto_included_anchors.jsonl")
    manifest = checked_manifest(review, "review_manifest.json", expected_files)
    if manifest.get("stage") != "review" or not manifest.get("semantic_screen_complete"):
        raise ValueError("lexical-only previews cannot be finalized; run the semantic proposal first")
    expected = manifest.get("bundle_sha256")
    if digest({k: v for k, v in manifest.items() if k != "bundle_sha256"}) != expected:
        raise ValueError("review manifest checksum mismatch")
    if manifest.get("code_sha256") != code_hashes():
        raise ValueError("curation source changed since proposal; create a new review bundle")
    candidates, _ = checked_inputs(_read_rows(review / "candidates.jsonl"),
                                   _read_rows(review / "heldout_inputs.jsonl"))
    answers = _read_rows(answers_path)
    eligible, audit = apply_reviews(candidates, _read_rows(review / "hard_exclusions.jsonl"),
                                    _read_rows(review / "review_flags.jsonl"), answers,
                                    bundle_sha256=expected)
    output.mkdir(parents=True, exist_ok=False)
    _jsonl(output / "eligible_anchors.jsonl", eligible)
    _jsonl(output / "curation_audit.jsonl", audit)
    _jsonl(output / "review_answers.jsonl", answers)
    _json(output / "review_manifest.json", manifest)
    result = {"schema_version": 1, "stage": "frozen", "review_bundle_sha256": expected,
              "candidate_count": len(candidates), "eligible_count": len(eligible),
              "excluded_count": len(candidates) - len(eligible),
              "eligible_sha256": file_sha(output / "eligible_anchors.jsonl"),
              "source_answers_sha256": file_sha(answers_path),
              "files": {name: file_sha(output / name) for name in (
                  "eligible_anchors.jsonl", "curation_audit.jsonl", "review_answers.jsonl", "review_manifest.json")},
              "software": software(), "code_sha256": code_hashes(),
              "review_source": str(review.resolve()),
              "note": "Use this exact eligible pool for every anchored condition. Keep the review directory for audit."}
    _json(output / "curation_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="export raw candidate prompts and evaluation inputs")
    export.add_argument("--data-config", type=Path, required=True)
    export.add_argument("--target-train-overlap", choices=("allow", "exclude"), required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    review = commands.add_parser("propose", help="write approximate-match reviews; do not select anchors")
    review.add_argument("--inputs", type=Path, required=True)
    review.add_argument("--output-dir", type=Path, required=True)
    review.add_argument("--lexical-only", action="store_true", help="preview only; cannot be finalized")
    review.add_argument("--export-auto-included", action="store_true",
                        help="export unflagged candidates for inspection after semantic screening; not a final pool")
    review.add_argument("--review-sample-size", type=int, required=True)
    review.add_argument("--review-sample-seed", type=int, required=True)
    review.add_argument("--model-id")
    review.add_argument("--model-revision")
    review.add_argument("--heldout-threshold", type=float)
    review.add_argument("--dedup-threshold", type=float)
    review.add_argument("--max-seq-length", type=int)
    review.add_argument("--truncation", choices=("reject", "allow"))
    review.add_argument("--device")
    review.add_argument("--batch-size", type=int, default=16)
    review.add_argument("--chunk-size", type=int, default=128)
    freeze = commands.add_parser("finalize", help="freeze a common pool after all reviews are resolved")
    freeze.add_argument("--review-dir", type=Path, required=True)
    freeze.add_argument("--answers", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        result = export_inputs(args.data_config, args.output_dir, target_train_overlap=args.target_train_overlap)
    elif args.command == "propose":
        fields = ("model_id", "model_revision", "max_seq_length", "truncation", "device")
        values = {name: getattr(args, name) for name in fields}
        if not args.lexical_only and any(v is None for v in values.values()):
            parser.error("semantic proposals require model ID/revision, max sequence length, truncation policy and device")
        if args.lexical_only and any(v is not None for v in values.values()):
            parser.error("do not supply embedding settings to a lexical-only preview")
        result = propose(args.inputs, args.output_dir, lexical_only=args.lexical_only,
                         sample_size=args.review_sample_size, sample_seed=args.review_sample_seed,
                         heldout_threshold=args.heldout_threshold, dedup_threshold=args.dedup_threshold,
                         chunk_size=args.chunk_size,
                         embedding_options=None if args.lexical_only else {**values, "batch_size": args.batch_size},
                         export_auto_included=args.export_auto_included)
    else:
        result = finalize(args.review_dir, args.answers, args.output_dir)
    print(json.dumps({k: v for k, v in result.items() if k in {
        "stage", "candidate_count", "heldout_count", "hard_excluded_count", "review_count",
        "eligible_count", "excluded_count", "semantic_screen_complete", "bundle_sha256"}}, indent=2))


if __name__ == "__main__":
    main()
