"""Small, harness-independent helpers for the project's reported metrics.

Scores are fractions in [0, 1], so differences are fractions too. Multiply a
difference by 100 when reporting percentage points. GSM8K extraction is a
documented heuristic; the final evaluation protocol must still lock its prompt
format and answer parser before reported runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import math
import re
from typing import Iterable, Mapping


_NUMBER = re.compile(
    r"(?<![\w.,])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w,]|\.\d)"
)
_ANSWER_PHRASE = re.compile(r"\b(?:final\s+answer|answer)\s*(?:is|:|=)\s*", re.I)


def normalize_gsm8k_answer(value: str) -> str | None:
    """Canonicalize a numeric answer, or return ``None`` if it is not numeric.

    Thousands separators and insignificant decimal zeroes are removed. Units,
    percentages, fractions, and expressions are not interpreted.
    """

    if not isinstance(value, str):
        raise TypeError("answer must be a string")
    raw = value.strip()
    if _NUMBER.fullmatch(raw) is None:
        return None
    try:
        number = Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    if number == 0:
        return "0"
    return format(number.normalize(), "f")


def extract_gsm8k_final_answer(text: str) -> str | None:
    """Extract a final number from GSM8K gold text or a model response.

    Preference order: first number after the last ``####`` marker; first number
    after the last explicit ``answer is``/``answer:``/``final answer`` phrase;
    otherwise the last number in the response. An explicit answer cue without
    a following number returns ``None`` instead of falling back to scratchwork.
    """

    if not isinstance(text, str):
        raise TypeError("response must be a string")
    if "####" in text:
        tail = text.rsplit("####", 1)[1]
        match = _NUMBER.search(tail)
        return normalize_gsm8k_answer(match.group()) if match else None

    phrases = list(_ANSWER_PHRASE.finditer(text))
    if phrases:
        match = _NUMBER.search(text[phrases[-1].end() :])
        return normalize_gsm8k_answer(match.group()) if match else None

    numbers = list(_NUMBER.finditer(text))
    return normalize_gsm8k_answer(numbers[-1].group()) if numbers else None


def gsm8k_exact_match(prediction: str, reference: str) -> bool:
    """Compare extracted final numeric answers; two parse failures never match."""

    predicted = extract_gsm8k_final_answer(prediction)
    expected = extract_gsm8k_final_answer(reference)
    return predicted is not None and expected is not None and predicted == expected


@dataclass(frozen=True)
class AccuracySummary:
    correct: int
    total: int
    accuracy: float


def aggregate_accuracy(correctness: Iterable[bool]) -> AccuracySummary:
    """Aggregate item-level correctness, rejecting an empty or malformed set."""

    total = 0
    correct = 0
    for item in correctness:
        if not isinstance(item, bool):
            raise TypeError("each item must be a bool")
        total += 1
        correct += item
    if total == 0:
        raise ValueError("cannot calculate accuracy from zero items")
    return AccuracySummary(correct=correct, total=total, accuracy=correct / total)


def _validated_score(score: float, name: str) -> float:
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TypeError(f"{name} must be a numeric score")
    score = float(score)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"{name} must be a finite fraction in [0, 1]")
    return score


def target_gain(base_score: float, adapted_score: float) -> float:
    """Return adapted target accuracy minus untouched-base target accuracy."""

    return _validated_score(adapted_score, "adapted_score") - _validated_score(
        base_score, "base_score"
    )


def benchmark_forgetting(
    base_scores: Mapping[str, float], adapted_scores: Mapping[str, float]
) -> dict[str, float]:
    """Return base minus adapted score for each identically named benchmark.

    Negative values are preserved: they mean the adapted model improved.
    """

    if base_scores.keys() != adapted_scores.keys():
        raise ValueError("base and adapted benchmark names must match exactly")
    return {
        name: _validated_score(base_score, f"base[{name}]")
        - _validated_score(adapted_scores[name], f"adapted[{name}]")
        for name, base_score in base_scores.items()
    }
