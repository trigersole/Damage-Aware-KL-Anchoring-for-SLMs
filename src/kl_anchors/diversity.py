"""Deterministic semantic-diversity ranks for a frozen eligible anchor pool.

Run ``PYTHONPATH=src python -m kl_anchors.diversity --pool POOL.jsonl
--output RANKED.jsonl --model-id ID --model-revision COMMIT --algorithm
farthest_first --seed SEED``. The model revision must be a 40-character commit
hash, so a moving branch cannot silently change embeddings. The output keeps
all input row fields and adds ``diversity_rank`` plus provenance metadata.

The farthest-first algorithm L2-normalizes embeddings, uses cosine distance,
starts from the minimum SHA-256 rank of ``seed`` and stable example ID, then
chooses the candidate with the greatest distance to its nearest selected
candidate. Equal distances resolve by stable example ID.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np


_COMMIT = re.compile(r"[0-9a-fA-F]{40}\Z")


@dataclass(frozen=True)
class DiversityConfig:
    model_id: str
    model_revision: str
    algorithm: str
    seed: int
    batch_size: int = 32
    device: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a nonempty string")
        if not isinstance(self.model_revision, str) or not _COMMIT.fullmatch(self.model_revision):
            raise ValueError("model_revision must be a 40-character commit hash")
        if self.algorithm != "farthest_first":
            raise ValueError("algorithm must be explicitly set to farthest_first")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if self.device is not None and (not isinstance(self.device, str) or not self.device.strip()):
            raise ValueError("device must be a nonempty string when supplied")


def _normalized(ids: Sequence[str], embeddings: Sequence[Sequence[float]]) -> tuple[list[str], np.ndarray]:
    if not ids or any(not isinstance(item_id, str) or not item_id for item_id in ids):
        raise ValueError("candidate IDs must be nonempty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("candidate IDs must be unique")
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(ids) or matrix.shape[1] < 1:
        raise ValueError("embeddings must be a nonempty two-dimensional matrix aligned with IDs")
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings must be finite")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("embeddings must have finite, nonzero norms")
    ordering = sorted(range(len(ids)), key=lambda index: ids[index])
    return [ids[index] for index in ordering], matrix[ordering] / norms[ordering, None]


def farthest_first_ranks(
    ids: Sequence[str], embeddings: Sequence[Sequence[float]], *, seed: int
) -> dict[str, int]:
    """Return zero-based ranks, independent of input row order and vector scale."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    ordered_ids, vectors = _normalized(ids, embeddings)
    count = len(ordered_ids)
    first = min(
        range(count),
        key=lambda index: (
            sha256(f"{seed}\0{ordered_ids[index]}".encode("utf-8")).digest(),
            ordered_ids[index],
        ),
    )
    selected = np.zeros(count, dtype=bool)
    nearest_distance = np.full(count, np.inf, dtype=np.float64)
    ranks: dict[str, int] = {}
    for rank in range(count):
        if rank == 0:
            chosen = first
        else:
            best = float(np.max(nearest_distance[~selected]))
            ties = np.flatnonzero((~selected) & np.isclose(
                nearest_distance, best, rtol=0.0, atol=1e-12
            ))
            chosen = int(ties[0])  # IDs are sorted, so this is the stable tie break.
        selected[chosen] = True
        ranks[ordered_ids[chosen]] = rank
        distances = 1.0 - np.clip(vectors @ vectors[chosen], -1.0, 1.0)
        np.minimum(nearest_distance, distances, out=nearest_distance)
    return ranks


def pool_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash exact IDs and prompts in stable order, excluding later scores/ranks."""
    payload = [(row["example_id"], row["prompt"]) for row in sorted(rows, key=lambda row: row["example_id"])]
    return sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def read_pool(path: str | Path) -> list[dict[str, Any]]:
    """Read a frozen eligible JSONL pool and validate stable IDs and prompts."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                item_id, prompt = row["example_id"], row["prompt"]
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                raise ValueError(f"invalid pool row on line {line_number}") from exc
            if not isinstance(row, dict) or not isinstance(item_id, str) or not item_id:
                raise ValueError(f"invalid example_id on line {line_number}")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"empty prompt on line {line_number}")
            if "diversity_rank" in row or "diversity_metadata" in row:
                raise ValueError(f"pool row on line {line_number} is already ranked")
            rows.append(row)
    if not rows or len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError("pool must contain unique candidate IDs")
    return sorted(rows, key=lambda row: row["example_id"])


def embed_prompts(prompts: Sequence[str], config: DiversityConfig) -> np.ndarray:
    """Load the pinned SentenceTransformer lazily and encode the supplied prompts."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("install sentence-transformers to compute semantic ranks") from exc
    model = SentenceTransformer(
        config.model_id,
        revision=config.model_revision,
        device=config.device,
        trust_remote_code=False,
    )
    model.eval()
    return np.asarray(model.encode(
        list(prompts),
        batch_size=config.batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ))


def rank_pool_rows(
    rows: Sequence[Mapping[str, Any]],
    embeddings: Sequence[Sequence[float]],
    config: DiversityConfig,
) -> list[dict[str, Any]]:
    """Attach full-pool diversity ranks to source rows; no I/O or model load."""
    if not rows:
        raise ValueError("pool is empty")
    ids = [row["example_id"] for row in rows]
    ranks = farthest_first_ranks(ids, embeddings, seed=config.seed)
    metadata = {
        **asdict(config),
        "distance": "cosine_on_l2_normalized_embeddings",
        "initialization": "minimum_sha256_of_seed_and_example_id",
        "tie_break": "lexicographic_example_id_with_1e-12_distance_tolerance",
        "eligible_pool_sha256": pool_fingerprint(rows),
        "embedding_dimensions": int(np.asarray(embeddings).shape[1]),
    }
    return [
        {**row, "diversity_rank": ranks[row["example_id"]], "diversity_metadata": metadata}
        for row in sorted(rows, key=lambda row: row["example_id"])
    ]


def write_ranked_rows(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    """Write a new ranked JSONL file and fail if it already exists."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, type=Path, help="Frozen eligible prompt JSONL")
    parser.add_argument("--output", required=True, type=Path, help="New ranked JSONL path")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True, help="Full 40-character model commit hash")
    parser.add_argument("--algorithm", required=True, choices=("farthest_first",))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", help="Explicit inference device, e.g. cpu or cuda:0")
    args = parser.parse_args()
    config = DiversityConfig(
        model_id=args.model_id,
        model_revision=args.model_revision,
        algorithm=args.algorithm,
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
    )
    rows = read_pool(args.pool)
    embeddings = embed_prompts([row["prompt"] for row in rows], config)
    ranked = rank_pool_rows(rows, embeddings, config)
    write_ranked_rows(ranked, args.output)
    print(f"wrote {len(ranked)} diversity ranks to {args.output}")


if __name__ == "__main__":
    main()
