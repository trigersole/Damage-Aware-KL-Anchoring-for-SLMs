"""Screen candidate anchors for semantic overlap with held-out inputs.

Use after surface-form decontamination and before teacher caching. The model,
revision and cosine threshold are mandatory protocol choices. Only held-out
questions, passages and answer options are embedded; labels are never used.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from .anchor_pipeline import file_sha256, read_jsonl, write_jsonl_exclusive
from .data_audit import HeldoutExample
from .prepare_data import _specs, build_heldout, load_hf_rows


def screen_embeddings(
    candidate_ids: list[str], candidate_vectors: np.ndarray,
    heldout: list[HeldoutExample], heldout_vectors: np.ndarray,
    *, threshold: float, chunk_size: int,
) -> list[dict]:
    """Return best cosine match and an explicit exclude decision per candidate."""
    if not 0 < threshold <= 1 or not math.isfinite(threshold):
        raise ValueError("threshold must be finite in (0, 1]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    a = np.asarray(candidate_vectors, dtype=np.float32)
    b = np.asarray(heldout_vectors, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != len(candidate_ids) or b.shape[0] != len(heldout) or a.shape[1] != b.shape[1] or b.shape[0] == 0:
        raise ValueError("embedding matrices and IDs must be aligned and nonempty")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("embeddings must be finite")
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    if np.any(a_norm == 0) or np.any(b_norm == 0):
        raise ValueError("zero embeddings are invalid")
    a = a / a_norm[:, None]
    b = b / b_norm[:, None]
    decisions = []
    for start in range(0, len(candidate_ids), chunk_size):
        similarities = a[start:start + chunk_size] @ b.T
        indices = np.argmax(similarities, axis=1)
        for local, best_index in enumerate(indices):
            i = start + local
            match = heldout[int(best_index)]
            score = float(similarities[local, best_index])
            decisions.append({
                "example_id": candidate_ids[i],
                "max_cosine": score,
                "matched_dataset": match.dataset,
                "matched_example_id": match.id,
                "excluded": score >= threshold,
            })
    return decisions


def _heldout_inputs(data_config: dict, data_dir: Path) -> list[HeldoutExample]:
    specs = _specs(data_config)
    validation = read_jsonl(data_dir / "gsm8k_validation.jsonl")
    heldout = [HeldoutExample("gsm8k_validation", row["example_id"], row["prompt"]) for row in validation]
    gsm8k_cfg = data_config["gsm8k"]
    heldout.extend(build_heldout(
        "gsm8k", load_hf_rows(specs["gsm8k_test"]), specs["gsm8k_test"],
        {"question": gsm8k_cfg["question_column"]}, dataset_tag="gsm8k_test",
    ))
    for kind in ("boolq", "hellaswag", "arc_easy"):
        heldout.extend(build_heldout(
            kind, load_hf_rows(specs[kind]), specs[kind],
            data_config["retention"][kind]["columns"],
        ))
    return heldout


def run(
    *, data_config_path: Path, data_dir: Path, candidate_path: Path,
    model_id: str, model_revision: str, threshold: float,
    batch_size: int, chunk_size: int, device: str,
    output_path: Path, exclusions_path: Path,
) -> None:
    if output_path.exists() or exclusions_path.exists():
        raise FileExistsError("semantic audit output already exists")
    if len(model_revision) != 40 or any(ch not in "0123456789abcdefABCDEF" for ch in model_revision):
        raise ValueError("embedding model revision must be a full 40-character commit")
    with data_config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    prepared_manifest = json.loads((data_dir / "data_manifest.json").read_text(encoding="utf-8"))
    expected_sources = {key: asdict(value) for key, value in _specs(config).items()}
    if prepared_manifest.get("sources") != expected_sources:
        raise ValueError("semantic audit data configuration differs from frozen prepared sources")
    candidates = read_jsonl(candidate_path)
    heldout = _heldout_inputs(config, data_dir)
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model_id, revision=model_revision, device=device)
    a = encoder.encode([row["prompt"] for row in candidates], batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
    b = encoder.encode([row.text for row in heldout], batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
    decisions = screen_embeddings(
        [row["example_id"] for row in candidates], a, heldout, b,
        threshold=threshold, chunk_size=chunk_size,
    )
    excluded = {row["example_id"] for row in decisions if row["excluded"]}
    if len(excluded) == len(candidates):
        raise ValueError("semantic screen removed every candidate prompt")
    write_jsonl_exclusive(output_path, [row for row in candidates if row["example_id"] not in excluded])
    write_jsonl_exclusive(exclusions_path, decisions)
    meta = {
        "source_candidates": str(candidate_path.resolve()),
        "source_sha256": file_sha256(candidate_path),
        "heldout_source_config": str(data_config_path.resolve()),
        "embedding_model_id": model_id,
        "embedding_model_revision": model_revision,
        "threshold": threshold,
        "candidate_count": len(candidates),
        "excluded_count": len(excluded),
        "eligible_count": len(candidates) - len(excluded),
        "eligible_sha256": file_sha256(output_path),
    }
    with output_path.with_suffix(output_path.suffix + ".meta.json").open("x", encoding="utf-8") as stream:
        json.dump(meta, stream, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    args = parser.parse_args()
    run(
        data_config_path=args.data_config, data_dir=args.data_dir,
        candidate_path=args.candidates,
        model_id=args.model_id, model_revision=args.model_revision,
        threshold=args.threshold, batch_size=args.batch_size,
        chunk_size=args.chunk_size, device=args.device,
        output_path=args.output, exclusions_path=args.exclusions,
    )


if __name__ == "__main__":
    main()
