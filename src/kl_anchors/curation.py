"""Input-only anchor curation with explicit, auditable human review.

Approximate overlap proposes a review, never an automatic deletion. Confirmed
semantic duplicates are removed only when directly equivalent to a retained
representative; equivalence is deliberately not propagated through a graph.
This module has no model loading, dataset downloads, or training dependencies.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math
import re
from typing import Any

import numpy as np

from .data_audit import (
    AnchorCandidate, HeldoutExample, OverlapConfig, audit_anchor_pool, normalize_text,
)


DECISIONS = {
    "benchmark_overlap": ("keep", "exclude"),
    "semantic_duplicate": ("distinct", "equivalent"),
    "spot_check": ("keep", "exclude"),
}


def digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def checked_inputs(candidates: list[dict], heldout: list[dict]) -> tuple[list[dict], list[dict]]:
    """Require input-only schemas: labels, answers and scores cannot pass through."""
    if not candidates or not heldout:
        raise ValueError("candidates and heldout inputs must be nonempty")
    for row in candidates:
        if set(row) != {"example_id", "prompt"}:
            raise ValueError("candidate rows must contain only example_id and prompt")
        if not isinstance(row["example_id"], str) or not row["example_id"].strip():
            raise ValueError("candidate IDs must be nonempty strings")
        if not isinstance(row["prompt"], str):
            raise ValueError("candidate prompt must be a string")
    for row in heldout:
        if set(row) != {"dataset", "example_id", "text"}:
            raise ValueError("heldout rows must contain only dataset, example_id and text")
        if any(not isinstance(v, str) or not v.strip() for v in row.values()):
            raise ValueError("heldout fields must be nonempty strings")
    if len({x["example_id"] for x in candidates}) != len(candidates):
        raise ValueError("duplicate candidate IDs")
    if len({(x["dataset"], x["example_id"]) for x in heldout}) != len(heldout):
        raise ValueError("duplicate heldout IDs")
    return (sorted(candidates, key=lambda x: x["example_id"]),
            sorted(heldout, key=lambda x: (x["dataset"], x["example_id"])))


def _filters(filters: dict, ids: set[str]) -> tuple[int, dict, dict]:
    size = filters.get("max_characters")
    if type(size) is not int or size <= 0:
        raise ValueError("candidate_filters.max_characters must be positive")
    patterns = {}
    for field in ("target_like_patterns", "unsafe_patterns"):
        values = filters.get(field)
        if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
            raise ValueError(f"candidate_filters.{field} must be a list of strings")
        patterns[field] = [re.compile(x, re.I) for x in values]
    manual = filters.get("manual_exclusions")
    if not isinstance(manual, dict) or any(
        k not in ids or not isinstance(v, str) or not v.strip() for k, v in manual.items()
    ):
        raise ValueError("manual exclusions must use existing IDs and nonempty reasons")
    return size, patterns, manual


def surface_proposal(candidates: list[dict], heldout: list[dict], *,
                     overlap: dict, candidate_filters: dict) -> tuple[list[dict], list[dict]]:
    """Preserve quality/exact exclusions, convert lexical near matches to reviews."""
    candidates, heldout = checked_inputs(candidates, heldout)
    limit, patterns, manual = _filters(candidate_filters, {x["example_id"] for x in candidates})
    result = audit_anchor_pool(
        [AnchorCandidate(x["example_id"], x["prompt"]) for x in candidates],
        [HeldoutExample(x["dataset"], x["example_id"], x["text"]) for x in heldout],
        config=OverlapConfig(**overlap),
    )
    audit = {x.candidate_id: x for x in result.manifest}
    hard, near, representatives = [], [], {}
    for row in candidates:
        item_id, prompt = row["example_id"], row["prompt"]
        reasons = [asdict(x) for x in audit[item_id].reasons
                   if x.code not in {"near_heldout", "duplicate_candidate"}]
        if len(prompt) > limit:
            reasons.append({"code": "excessive_length"})
        for field, code in (("target_like_patterns", "target_like_pattern"),
                            ("unsafe_patterns", "unsafe_pattern")):
            if any(p.search(prompt) for p in patterns[field]):
                reasons.append({"code": code})
        if item_id in manual:
            reasons.append({"code": "manual_review", "reason": manual[item_id]})
        # Choose the first otherwise-eligible exact representative. A manual
        # exclusion of one source ID must not eliminate every equivalent row.
        if not reasons:
            normalized = normalize_text(prompt)
            if normalized in representatives:
                reasons.append({"code": "duplicate_candidate",
                                "representative_id": representatives[normalized]})
            else:
                representatives[normalized] = item_id
        if reasons:
            hard.append({"example_id": item_id, "reasons": reasons})
        else:
            for reason in audit[item_id].reasons:
                if reason.code == "near_heldout":
                    near.append({"candidate_id": item_id, "dataset": reason.dataset,
                                 "heldout_id": reason.example_id,
                                 "evidence": {"method": "ngram", "shared_ngrams": reason.shared_ngrams,
                                              "containment": reason.containment}})
    return hard, near


def _vectors(values: np.ndarray, rows: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != rows or matrix.shape[1] == 0:
        raise ValueError("embedding dimensions must align with input rows")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(matrix).all() or not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("embeddings must be finite and nonzero")
    return matrix / norms[:, None]


def semantic_pairs(candidate_vectors: np.ndarray, heldout_vectors: np.ndarray, *,
                   heldout_threshold: float, dedup_threshold: float,
                   chunk_size: int = 128) -> tuple[list[tuple], list[tuple], list[tuple]]:
    """Return all flagged pairs and nearest heldout matches, with bounded matrices.

Row indices refer to the caller's ordered inputs. There is no top-k cap that
could hide another above-threshold match after a false positive is cleared.
"""
    for threshold in (heldout_threshold, dedup_threshold):
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("semantic thresholds must be finite in (0, 1]")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    a = _vectors(candidate_vectors, len(candidate_vectors))
    b = _vectors(heldout_vectors, len(heldout_vectors))
    if len(a) == 0 or len(b) == 0 or a.shape[1] != b.shape[1]:
        raise ValueError("embedding sets must be nonempty and share dimensions")
    overlaps, duplicates, nearest = [], [], []
    for start in range(0, len(a), chunk_size):
        stop = min(start + chunk_size, len(a))
        scores = np.clip(a[start:stop] @ b.T, -1, 1)
        best = np.argmax(scores, axis=1)
        nearest.extend((start + i, int(j), float(scores[i, j])) for i, j in enumerate(best))
        ii, jj = np.nonzero(scores >= heldout_threshold)
        overlaps.extend((start + int(i), int(j), float(scores[i, j])) for i, j in zip(ii, jj))
        scores = np.clip(a[start:stop] @ a.T, -1, 1)
        for local, i in enumerate(range(start, stop)):
            for j in np.flatnonzero(scores[local, i + 1:] >= dedup_threshold) + i + 1:
                duplicates.append((i, int(j), float(scores[local, j])))
    return overlaps, duplicates, nearest


def _flag(kind: str, identities: list, **fields: Any) -> dict:
    return {"review_id": digest([kind, identities]), "kind": kind,
            "allowed_decisions": list(DECISIONS[kind]), **fields}


def assemble_flags(candidates: list[dict], heldout: list[dict], hard: list[dict],
                   lexical: list[dict], *, semantic: tuple | None,
                   sample_size: int, sample_seed: int) -> tuple[list[dict], list[dict]]:
    """Combine evidence for a pair and draw reproducible unflagged spot checks."""
    if type(sample_size) is not int or sample_size < 0 or type(sample_seed) is not int:
        raise ValueError("sample size must be nonnegative and sample seed an integer")
    candidates, heldout = checked_inputs(candidates, heldout)
    blocked = {x["example_id"] for x in hard}
    active = [x for x in candidates if x["example_id"] not in blocked]
    by_id = {x["example_id"]: x for x in candidates}
    by_heldout = {(x["dataset"], x["example_id"]): x for x in heldout}
    matches: dict[tuple, list] = {}
    for row in lexical:
        key = (row["candidate_id"], row["dataset"], row["heldout_id"])
        matches.setdefault(key, []).append(row["evidence"])
    duplicates, summary = [], []
    if semantic is not None:
        overlaps, pairs, nearest = semantic
        for i, j, score in overlaps:
            ref = heldout[j]
            key = (active[i]["example_id"], ref["dataset"], ref["example_id"])
            matches.setdefault(key, []).append({"method": "cosine", "score": score})
        for i, j, score in pairs:
            left, right = active[i], active[j]
            duplicates.append(_flag(
                "semantic_duplicate", [left["example_id"], right["example_id"]],
                candidate_id=left["example_id"], other_candidate_id=right["example_id"],
                candidate_prompt=left["prompt"], other_prompt=right["prompt"], cosine=score,
            ))
        for i, j, score in nearest:
            summary.append({"example_id": active[i]["example_id"], "max_heldout_cosine": score,
                            "dataset": heldout[j]["dataset"], "heldout_id": heldout[j]["example_id"]})
    flags = []
    for (item_id, dataset, ref_id), evidence in sorted(matches.items()):
        flags.append(_flag(
            "benchmark_overlap", [item_id, dataset, ref_id], candidate_id=item_id,
            candidate_prompt=by_id[item_id]["prompt"], dataset=dataset, heldout_id=ref_id,
            heldout_text=by_heldout[(dataset, ref_id)]["text"], evidence=evidence,
        ))
    flags.extend(duplicates)
    already_flagged = {f["candidate_id"] for f in flags} | {
        f["other_candidate_id"] for f in duplicates}
    available = [x for x in active if x["example_id"] not in already_flagged]
    available.sort(key=lambda x: (digest([sample_seed, x["example_id"]]), x["example_id"]))
    for row in available[:sample_size]:
        flags.append(_flag("spot_check", [row["example_id"]], candidate_id=row["example_id"],
                           candidate_prompt=row["prompt"]))
    return sorted(flags, key=lambda f: (f["kind"], f["review_id"])), summary


def apply_reviews(candidates: list[dict], hard: list[dict], flags: list[dict],
                  answers: list[dict], *, bundle_sha256: str) -> tuple[list[dict], list[dict]]:
    """Resolve every flagged pair, then deduplicate through retained representatives.

Exact/quality exclusions are non-overridable here. Restoring a near match still
requires clearing every other match on that candidate. Missing, stale or
duplicate reviews fail rather than silently turning into keep decisions.
"""
    decisions = {}
    flag_by_id = {x["review_id"]: x for x in flags}
    if len(flag_by_id) != len(flags):
        raise ValueError("duplicate review IDs")
    for row in answers:
        if set(row) != {"review_id", "bundle_sha256", "decision", "reviewer", "reason"}:
            raise ValueError("invalid review answer fields")
        key = row["review_id"]
        if key not in flag_by_id or key in decisions:
            raise ValueError("unknown or duplicate review ID")
        if row["bundle_sha256"] != bundle_sha256:
            raise ValueError("stale review: bundle checksum differs")
        flag = flag_by_id[key]
        if row["decision"] not in DECISIONS[flag["kind"]]:
            raise ValueError(f"pending or invalid decision for {key}")
        if any(not isinstance(row[k], str) or not row[k].strip() for k in ("reviewer", "reason")):
            raise ValueError("reviewer and reason are required for every decision")
        decisions[key] = row
    if decisions.keys() != flag_by_id.keys():
        raise ValueError("all flagged pairs and spot checks must be reviewed")
    excluded = {x["example_id"]: list(x["reasons"]) for x in hard}
    equivalent: dict[str, set[str]] = {}
    associated: dict[str, list[str]] = {}
    for flag in flags:
        key, left = flag["review_id"], flag["candidate_id"]
        answer = decisions[key]
        associated.setdefault(left, []).append(key)
        if flag["kind"] == "semantic_duplicate":
            right = flag["other_candidate_id"]
            associated.setdefault(right, []).append(key)
            if answer["decision"] == "equivalent":
                equivalent.setdefault(left, set()).add(right)
                equivalent.setdefault(right, set()).add(left)
        elif answer["decision"] == "exclude":
            excluded.setdefault(left, []).append({"code": "reviewed_exclusion", "review_id": key})
    eligible, audit, retained = [], [], set()
    for row in sorted(candidates, key=lambda r: r["example_id"]):
        item_id = row["example_id"]
        reasons = list(excluded.get(item_id, []))
        if not reasons:
            matches = sorted(retained & equivalent.get(item_id, set()))
            if matches:
                reasons.append({"code": "reviewed_semantic_duplicate", "representative_id": matches[0]})
            else:
                retained.add(item_id)
                eligible.append({"example_id": item_id, "prompt": row["prompt"]})
        audit.append({"example_id": item_id, "excluded": bool(reasons), "reasons": reasons,
                      "review_ids": sorted(associated.get(item_id, [])),
                      "prompt_sha256": sha256(row["prompt"].encode()).hexdigest()})
    if not eligible:
        raise ValueError("review removed every candidate")
    return eligible, audit
