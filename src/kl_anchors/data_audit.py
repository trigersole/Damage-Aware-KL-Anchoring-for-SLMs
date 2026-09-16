"""Deterministic, dataset-agnostic decontamination of candidate anchor prompts.

Callers supply stable item IDs and *question/prompt text only* for held-out sets.
In particular, this module does not load benchmark data or inspect answer labels.
The overlap policy is deliberately explicit: pass an :class:`OverlapConfig` chosen
and recorded in the experiment protocol before final evaluation.

Near overlap uses distinct Unicode word n-grams. Its containment score is the
number of shared n-grams divided by the candidate's n-gram count. Texts shorter
than ``ngram_size`` receive exact-match screening only. This surface-form check
does not detect all paraphrases, so a separate semantic review may be needed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Iterable
import unicodedata


_WORDS = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class AnchorCandidate:
    id: str
    text: str


@dataclass(frozen=True)
class HeldoutExample:
    dataset: str
    id: str
    text: str


@dataclass(frozen=True)
class OverlapConfig:
    ngram_size: int
    min_shared_ngrams: int
    min_containment: float

    def __post_init__(self) -> None:
        if type(self.ngram_size) is not int or self.ngram_size < 1:
            raise ValueError("ngram_size must be a positive integer")
        if type(self.min_shared_ngrams) is not int or self.min_shared_ngrams < 1:
            raise ValueError("min_shared_ngrams must be a positive integer")
        if not isinstance(self.min_containment, (int, float)) or isinstance(self.min_containment, bool):
            raise ValueError("min_containment must be a finite number in (0, 1]")
        if not math.isfinite(self.min_containment) or not 0 < self.min_containment <= 1:
            raise ValueError("min_containment must be a finite number in (0, 1]")


@dataclass(frozen=True)
class ExclusionReason:
    code: str
    candidate_id: str | None = None
    dataset: str | None = None
    example_id: str | None = None
    shared_ngrams: int | None = None
    containment: float | None = None


@dataclass(frozen=True)
class AuditEntry:
    candidate_id: str
    normalized_sha256: str
    excluded: bool
    reasons: tuple[ExclusionReason, ...]


@dataclass(frozen=True)
class AuditResult:
    config: OverlapConfig
    eligible_ids: tuple[str, ...]
    manifest: tuple[AuditEntry, ...]

    def to_dict(self) -> dict:
        """Return a JSON-ready manifest with stable ordering and no source text."""
        return {
            "config": asdict(self.config),
            "eligible_ids": list(self.eligible_ids),
            "entries": [asdict(entry) for entry in self.manifest],
        }

    def write_manifest(self, path: str | Path) -> None:
        """Write a deterministic UTF-8 JSON manifest at an explicit path."""
        with Path(path).open("w", encoding="utf-8", newline="\n") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, sort_keys=True, indent=2)
            file.write("\n")


def normalize_text(text: str) -> str:
    """Apply Unicode NFKC, case folding, and whitespace collapsing."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _ngrams(normalized: str, size: int) -> set[tuple[str, ...]]:
    words = _WORDS.findall(normalized)
    return {tuple(words[index : index + size]) for index in range(len(words) - size + 1)}


def _stable_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def audit_anchor_pool(
    candidates: Iterable[AnchorCandidate],
    heldout: Iterable[HeldoutExample],
    *,
    config: OverlapConfig,
) -> AuditResult:
    """Exclude empty, duplicate, exact-held-out, and near-held-out candidates.

    Candidate and held-out IDs must be unique within their respective namespaces.
    Results do not depend on input iteration order. Among duplicate candidates,
    the lexicographically smallest ID is retained unless it has another exclusion.
    Every held-out overlap satisfying the policy is included in the manifest.
    """
    if not isinstance(config, OverlapConfig):
        raise TypeError("config must be an OverlapConfig")

    candidate_list = list(candidates)
    heldout_list = list(heldout)
    for item in candidate_list:
        if not isinstance(item, AnchorCandidate):
            raise TypeError("candidates must contain AnchorCandidate items")
        _stable_id(item.id, "candidate ID")
        if not isinstance(item.text, str):
            raise TypeError("candidate text must be a string")
    for item in heldout_list:
        if not isinstance(item, HeldoutExample):
            raise TypeError("heldout must contain HeldoutExample items")
        _stable_id(item.dataset, "held-out dataset")
        _stable_id(item.id, "held-out example ID")
        if not isinstance(item.text, str):
            raise TypeError("held-out text must be a string")

    candidate_list.sort(key=lambda item: item.id)
    heldout_list.sort(key=lambda item: (item.dataset, item.id))
    if len({item.id for item in candidate_list}) != len(candidate_list):
        raise ValueError("candidate IDs must be unique")
    if len({(item.dataset, item.id) for item in heldout_list}) != len(heldout_list):
        raise ValueError("held-out (dataset, ID) pairs must be unique")

    exact_heldout: dict[str, list[HeldoutExample]] = defaultdict(list)
    ngram_index: dict[tuple[str, ...], list[tuple[str, str]]] = defaultdict(list)
    for item in heldout_list:
        normalized = normalize_text(item.text)
        if not normalized:
            raise ValueError(f"held-out text is empty for {item.dataset}/{item.id}")
        exact_heldout[normalized].append(item)
        for gram in _ngrams(normalized, config.ngram_size):
            ngram_index[gram].append((item.dataset, item.id))

    first_candidate_for_text: dict[str, str] = {}
    entries: list[AuditEntry] = []
    eligible_ids: list[str] = []
    for item in candidate_list:
        normalized = normalize_text(item.text)
        reasons: list[ExclusionReason] = []
        if not normalized:
            reasons.append(ExclusionReason("empty_candidate"))
        else:
            first_id = first_candidate_for_text.setdefault(normalized, item.id)
            if first_id != item.id:
                reasons.append(ExclusionReason("duplicate_candidate", candidate_id=first_id))
            for match in exact_heldout.get(normalized, ()):
                reasons.append(
                    ExclusionReason("exact_heldout", dataset=match.dataset, example_id=match.id)
                )
            grams = _ngrams(normalized, config.ngram_size)
            if grams:
                counts: Counter[tuple[str, str]] = Counter()
                for gram in grams:
                    counts.update(ngram_index.get(gram, ()))
                exact_keys = {(match.dataset, match.id) for match in exact_heldout.get(normalized, ())}
                for (dataset, example_id), shared in sorted(counts.items()):
                    containment = shared / len(grams)
                    if (dataset, example_id) not in exact_keys and (
                        shared >= config.min_shared_ngrams
                        and containment >= config.min_containment
                    ):
                        reasons.append(
                            ExclusionReason(
                                "near_heldout",
                                dataset=dataset,
                                example_id=example_id,
                                shared_ngrams=shared,
                                containment=containment,
                            )
                        )
        excluded = bool(reasons)
        if not excluded:
            eligible_ids.append(item.id)
        entries.append(
            AuditEntry(
                candidate_id=item.id,
                normalized_sha256=sha256(normalized.encode("utf-8")).hexdigest(),
                excluded=excluded,
                reasons=tuple(reasons),
            )
        )
    return AuditResult(config, tuple(eligible_ids), tuple(entries))
