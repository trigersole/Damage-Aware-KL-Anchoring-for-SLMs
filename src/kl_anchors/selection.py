"""Deterministic, equal-response-token anchor selection.

This module does not choose a candidate pool, vulnerability definition, or
diversity algorithm. Those remain protocol decisions. It consumes a *frozen*
candidate manifest with externally computed scores/ranks and selects exactly
the requested number of scored teacher-continuation tokens. The final prompt
may be truncated to its first ``selected_response_tokens`` continuation tokens.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

Method = Literal["random", "diversity", "vulnerability", "damage_diverse"]


@dataclass(frozen=True)
class Candidate:
    example_id: str
    response_token_count: int
    vulnerability_score: float | None = None
    diversity_rank: int | None = None


def _validate(candidates: Sequence[Candidate], budget: int, method: Method) -> None:
    if not candidates:
        raise ValueError("candidate pool is empty")
    if budget <= 0:
        raise ValueError("anchor-token budget must be positive")
    ids = [c.example_id for c in candidates]
    if not all(isinstance(x, str) and x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("candidate IDs must be nonempty and unique")
    if any(c.response_token_count <= 0 for c in candidates):
        raise ValueError("all candidates must contain scored response tokens")
    if sum(c.response_token_count for c in candidates) < budget:
        raise ValueError("candidate pool has fewer response tokens than budget")
    if method in ("vulnerability", "damage_diverse") and any(
        c.vulnerability_score is None or not math.isfinite(c.vulnerability_score)
        for c in candidates
    ):
        raise ValueError("vulnerability scores must be finite for the entire pool")
    if method in ("diversity", "damage_diverse"):
        ranks = [c.diversity_rank for c in candidates]
        if any(r is None or r < 0 for r in ranks) or len(set(ranks)) != len(ranks):
            raise ValueError("diversity ranks must be nonnegative and unique for the entire pool")
    if method not in ("random", "diversity", "vulnerability", "damage_diverse"):
        raise ValueError("unsupported selection method")


def pool_fingerprint(candidates: Sequence[Candidate]) -> str:
    """Hash the same eligible IDs and continuation lengths across conditions."""
    payload = [(c.example_id, c.response_token_count) for c in sorted(candidates, key=lambda c: c.example_id)]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def select_anchors(
    candidates: Sequence[Candidate],
    *,
    method: Method,
    anchor_token_budget: int,
    seed: int,
    damage_weight: float | None = None,
) -> Mapping[str, object]:
    """Choose anchors from one eligible pool and spend the exact token budget.

    A random ordering is uniform over prompts. Vulnerability uses descending
    scores. Diversity uses a precomputed order supplied by a separately locked
    protocol. Ties are resolved by stable example ID. The final continuation
    can be shortened, so the *scored response-token* count is equal exactly.
    """
    _validate(candidates, anchor_token_budget, method)
    if method == "damage_diverse":
        if damage_weight is None or not math.isfinite(damage_weight) or not 0 <= damage_weight <= 1:
            raise ValueError("damage_diverse requires an explicit damage_weight in [0, 1]")
    elif damage_weight is not None:
        raise ValueError("damage_weight is only used by damage_diverse")
    if method == "random":
        ordered = sorted(candidates, key=lambda c: c.example_id)
        random.Random(seed).shuffle(ordered)
    elif method == "vulnerability":
        ordered = sorted(candidates, key=lambda c: (-float(c.vulnerability_score), c.example_id))
    elif method == "damage_diverse":
        # Rank fusion preserves each signal's ordering without assuming their
        # raw units are comparable. The weight is a locked protocol parameter.
        damage_order = sorted(candidates, key=lambda c: (-float(c.vulnerability_score), c.example_id))
        damage_rank = {c.example_id: i for i, c in enumerate(damage_order)}
        denominator = max(1, len(candidates) - 1)
        ordered = sorted(candidates, key=lambda c: (
            damage_weight * damage_rank[c.example_id] / denominator
            + (1 - damage_weight) * int(c.diversity_rank) / denominator,
            c.example_id,
        ))
    else:
        ordered = sorted(candidates, key=lambda c: (int(c.diversity_rank), c.example_id))

    selected: list[dict[str, object]] = []
    remaining = anchor_token_budget
    for rank, candidate in enumerate(ordered, start=1):
        if remaining == 0:
            break
        used = min(candidate.response_token_count, remaining)
        selected.append({
            "example_id": candidate.example_id,
            "selection_rank": rank,
            "available_response_tokens": candidate.response_token_count,
            "selected_response_tokens": used,
        })
        remaining -= used
    if remaining:
        raise AssertionError("validated pool could not fill budget")
    return {
        "method": method,
        "seed": seed,
        "damage_weight": damage_weight,
        "eligible_pool_sha256": pool_fingerprint(candidates),
        "eligible_prompt_count": len(candidates),
        "anchor_token_budget": anchor_token_budget,
        "selected_prompt_count": len(selected),
        "selected": selected,
    }
