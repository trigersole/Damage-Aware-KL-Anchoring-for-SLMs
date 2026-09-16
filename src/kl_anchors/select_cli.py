"""Select a token-budgeted anchor manifest from a frozen JSONL candidate pool.

Run ``python -m kl_anchors.select_cli --help`` with ``PYTHONPATH=src``.
This does not fetch datasets or run a model. Output files are created only if
they do not already exist, so reported selections cannot be overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .selection import Candidate, select_anchors


def _load(path: Path) -> list[Candidate]:
    candidates = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                candidates.append(Candidate(
                    example_id=row["example_id"],
                    response_token_count=row["response_token_count"],
                    vulnerability_score=row.get("vulnerability_score"),
                    diversity_rank=row.get("diversity_rank"),
                ))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid candidate on line {line_number}: {exc}") from exc
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, type=Path, help="Frozen eligible candidate JSONL")
    parser.add_argument("--method", required=True, choices=("random", "diversity", "vulnerability", "damage_diverse"))
    parser.add_argument("--damage-weight", type=float, help="Required only for damage_diverse; fixed rank-fusion weight")
    parser.add_argument("--anchor-token-budget", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    selection = select_anchors(
        _load(args.pool),
        method=args.method,
        anchor_token_budget=args.anchor_token_budget,
        seed=args.seed,
        damage_weight=args.damage_weight,
    )
    output = {
        "schema_version": 1,
        "source_pool": str(args.pool.resolve()),
        **selection,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(f"selected {output['selected_prompt_count']} prompts, {output['anchor_token_budget']} scored tokens")


if __name__ == "__main__":
    main()
