"""Prepare frozen, decontaminated JSONL inputs from explicitly configured datasets.

Run with ``PYTHONPATH=src python -m kl_anchors.prepare_data --config CONFIG.json
--output-dir DIRECTORY``. ``datasets`` is imported only when downloading/loading
Hugging Face datasets. The config must name every dataset path, revision, split,
column, GSM8K validation count and seed, and overlap threshold. No source or
cleaning threshold is selected by this module.

Output schema:
* ``gsm8k_train.jsonl``, ``gsm8k_validation.jsonl`` and ``gsm8k_test.jsonl``: ``example_id``,
  ``prompt`` (the raw question), ``response`` (the raw gold answer).
* ``eligible_anchors.jsonl``: ``example_id``, ``prompt`` (rendered candidate).
* ``excluded_anchors.jsonl``: ``example_id``, normalized-text hash and reasons.
* ``pool_audit.json``: full inclusion/exclusion audit, without benchmark text.
* ``data_manifest.json``: source specs, IDs, counts, rendering and split policy.

The module excludes empty, exact-duplicate and surface-overlapping candidates.
It does not implement unresolved unsafe, length, target-like or semantic filters.
Those require a separately approved protocol before a final pool is frozen.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from string import Formatter
from typing import Any, Callable, Mapping, Sequence

from .data_audit import AnchorCandidate, HeldoutExample, OverlapConfig, audit_anchor_pool, normalize_text


ROW_KEYS = ("gsm8k_train", "gsm8k_test", "boolq", "hellaswag", "arc_easy", "anchors")


@dataclass(frozen=True)
class DatasetSource:
    path: str
    revision: str
    split: str
    name: str | None = None
    id_column: str | None = None

    def __post_init__(self) -> None:
        for field in ("path", "revision", "split"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise ValueError(f"source {field} must be a nonempty string")
        for field in ("name", "id_column"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"source {field} must be a nonempty string when supplied")


@dataclass(frozen=True)
class PreparedData:
    gsm8k_train: tuple[dict[str, str], ...]
    gsm8k_validation: tuple[dict[str, str], ...]
    gsm8k_test: tuple[dict[str, str], ...]
    eligible_anchors: tuple[dict[str, str], ...]
    excluded_anchors: tuple[dict[str, Any], ...]
    pool_audit: dict[str, Any]
    data_manifest: dict[str, Any]


def _source(spec: Mapping[str, Any]) -> DatasetSource:
    if not isinstance(spec, Mapping):
        raise ValueError("source must be an object")
    return DatasetSource(**spec)


def _text(row: Mapping[str, Any], column: str) -> str:
    if not isinstance(column, str) or not column:
        raise ValueError("column name must be a nonempty string")
    value = row[column]
    if not isinstance(value, str):
        raise ValueError(f"column {column!r} must contain text")
    return value.strip()


def source_id(source: DatasetSource, row: Mapping[str, Any], index: int) -> str:
    """Return an unambiguous ID tied to source revision, split and row identity."""
    if type(index) is not int or index < 0:
        raise ValueError("index must be a nonnegative integer")
    if source.id_column is None:
        row_key: str | int = index
    else:
        row_key = row[source.id_column]
        if type(row_key) not in (str, int) or row_key == "":
            raise ValueError(f"source ID column {source.id_column!r} must contain a string or integer")
    return json.dumps(
        [source.path, source.name, source.revision, source.split, row_key],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _identified_rows(rows: Sequence[Mapping[str, Any]], source: DatasetSource) -> list[tuple[str, Mapping[str, Any]]]:
    identified = [(source_id(source, row, index), row) for index, row in enumerate(rows)]
    ids = [item_id for item_id, _ in identified]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate source IDs in {source.path}/{source.split}")
    return identified


def split_gsm8k_train(
    rows: Sequence[Mapping[str, Any]],
    source: DatasetSource,
    *,
    question_column: str,
    answer_column: str,
    validation_count: int,
    split_seed: int,
) -> tuple[tuple[dict[str, str], ...], tuple[dict[str, str], ...]]:
    """Assign validation examples from GSM8K training IDs by stable SHA-256 rank."""
    if type(validation_count) is not int or not 0 < validation_count < len(rows):
        raise ValueError("validation_count must be between 1 and train size - 1")
    if type(split_seed) is not int:
        raise ValueError("split_seed must be an integer")
    identified = _identified_rows(rows, source)
    ranked_ids = sorted(
        (item_id for item_id, _ in identified),
        key=lambda item_id: (sha256(f"{split_seed}\0{item_id}".encode("utf-8")).digest(), item_id),
    )
    validation_ids = set(ranked_ids[:validation_count])
    train: list[dict[str, str]] = []
    validation: list[dict[str, str]] = []
    for item_id, row in identified:
        record = {
            "example_id": item_id,
            "prompt": _text(row, question_column),
            "response": _text(row, answer_column),
        }
        if not record["prompt"] or not record["response"]:
            raise ValueError(f"empty GSM8K question or answer for {item_id}")
        (validation if item_id in validation_ids else train).append(record)
    return tuple(train), tuple(validation)


def _choices(value: Any, column: str) -> list[str]:
    if isinstance(value, Mapping):
        value = value["text"]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"column {column!r} must contain a nonempty choice list")
    if any(not isinstance(choice, str) or not choice.strip() for choice in value):
        raise ValueError(f"column {column!r} has an empty or non-text choice")
    return [choice.strip() for choice in value]


def render_heldout_text(kind: str, row: Mapping[str, Any], columns: Mapping[str, str]) -> str:
    """Render only evaluation inputs, never the gold answer/label field."""
    if kind == "gsm8k":
        return _text(row, columns["question"])
    if kind == "boolq":
        return f"{_text(row, columns['question'])}\n\n{_text(row, columns['passage'])}"
    if kind in ("hellaswag", "arc_easy"):
        lead = _text(row, columns["context" if kind == "hellaswag" else "question"])
        choices_column = columns["choices"]
        return "\n".join([lead, *_choices(row[choices_column], choices_column)])
    raise ValueError(f"unsupported held-out kind: {kind}")


def build_heldout(
    kind: str,
    rows: Sequence[Mapping[str, Any]],
    source: DatasetSource,
    columns: Mapping[str, str],
    *,
    dataset_tag: str | None = None,
) -> tuple[HeldoutExample, ...]:
    tag = dataset_tag or kind
    return tuple(
        HeldoutExample(tag, item_id, render_heldout_text(kind, row, columns))
        for item_id, row in _identified_rows(rows, source)
    )


def build_candidates(
    rows: Sequence[Mapping[str, Any]],
    source: DatasetSource,
    *,
    prompt_column: str,
    context_column: str | None = None,
    context_template: str | None = None,
) -> tuple[AnchorCandidate, ...]:
    """Render configured prompt and optional context fields without other columns."""
    if context_column is not None and context_template is None:
        raise ValueError("context_template is required when context_column is set")
    if context_column is None and context_template is not None:
        raise ValueError("context_template requires context_column")
    if context_template is not None:
        try:
            fields = {
                field for _, field, _, _ in Formatter().parse(context_template)
                if field is not None
            }
            context_template.format(prompt="p", context="c")
        except (KeyError, ValueError, IndexError) as exc:
            raise ValueError("context_template must use {prompt} and {context}") from exc
        if fields != {"prompt", "context"}:
            raise ValueError("context_template must use {prompt} and {context}")
    candidates: list[AnchorCandidate] = []
    for item_id, row in _identified_rows(rows, source):
        prompt = _text(row, prompt_column)
        if context_column is not None:
            context = _text(row, context_column)
            if context:
                prompt = context_template.format(prompt=prompt, context=context)
        candidates.append(AnchorCandidate(item_id, prompt))
    return tuple(candidates)


def _specs(config: Mapping[str, Any]) -> dict[str, DatasetSource]:
    gsm8k = config["gsm8k"]
    retention = config["retention"]
    return {
        "gsm8k_train": _source(gsm8k["train_source"]),
        "gsm8k_test": _source(gsm8k["test_source"]),
        "boolq": _source(retention["boolq"]["source"]),
        "hellaswag": _source(retention["hellaswag"]["source"]),
        "arc_easy": _source(retention["arc_easy"]["source"]),
        "anchors": _source(config["anchors"]["source"]),
    }


def prepare_from_rows(
    config: Mapping[str, Any],
    rows_by_key: Mapping[str, Sequence[Mapping[str, Any]]],
) -> PreparedData:
    """Pure preparation path for supplied rows; suitable for synthetic tests."""
    specs = _specs(config)
    missing = set(ROW_KEYS) - rows_by_key.keys()
    if missing:
        raise ValueError(f"missing datasets: {', '.join(sorted(missing))}")
    gsm8k_cfg = config["gsm8k"]
    train, validation = split_gsm8k_train(
        rows_by_key["gsm8k_train"],
        specs["gsm8k_train"],
        question_column=gsm8k_cfg["question_column"],
        answer_column=gsm8k_cfg["answer_column"],
        validation_count=gsm8k_cfg["validation_count"],
        split_seed=gsm8k_cfg["split_seed"],
    )
    test = tuple({
        "example_id": item_id,
        "prompt": _text(row, gsm8k_cfg["question_column"]),
        "response": _text(row, gsm8k_cfg["answer_column"]),
    } for item_id, row in _identified_rows(rows_by_key["gsm8k_test"], specs["gsm8k_test"]))
    validation_heldout = tuple(
        HeldoutExample("gsm8k_validation", record["example_id"], record["prompt"])
        for record in validation
    )
    heldout = [*validation_heldout]
    heldout.extend(
        build_heldout(
            "gsm8k",
            rows_by_key["gsm8k_test"],
            specs["gsm8k_test"],
            {"question": gsm8k_cfg["question_column"]},
            dataset_tag="gsm8k_test",
        )
    )
    for kind in ("boolq", "hellaswag", "arc_easy"):
        benchmark = config["retention"][kind]
        heldout.extend(
            build_heldout(kind, rows_by_key[kind], specs[kind], benchmark["columns"])
        )
    anchors_cfg = config["anchors"]
    candidates = build_candidates(
        rows_by_key["anchors"],
        specs["anchors"],
        prompt_column=anchors_cfg["prompt_column"],
        context_column=anchors_cfg.get("context_column"),
        context_template=anchors_cfg.get("context_template"),
    )
    audit = audit_anchor_pool(candidates, heldout, config=OverlapConfig(**config["overlap"]))
    eligible_set = set(audit.eligible_ids)
    eligible_initial = tuple(
        {"example_id": item.id, "prompt": item.text}
        for item in sorted(candidates, key=lambda item: item.id)
        if item.id in eligible_set
    )
    excluded_initial = tuple(
        {
            "example_id": entry.candidate_id,
            "normalized_sha256": entry.normalized_sha256,
            "reasons": [asdict(reason) for reason in entry.reasons],
        }
        for entry in audit.manifest
        if entry.excluded
    )
    filters = config.get("candidate_filters")
    if filters is None:
        eligible = eligible_initial
        excluded = excluded_initial
        extra_filter_names: list[str] = []
    else:
        max_characters = filters["max_characters"]
        if type(max_characters) is not int or max_characters <= 0:
            raise ValueError("candidate_filters.max_characters must be positive")
        for key in ("target_like_patterns", "unsafe_patterns"):
            if not isinstance(filters.get(key), list) or any(not isinstance(pattern, str) for pattern in filters[key]):
                raise ValueError(f"candidate_filters.{key} must be a list of regex strings")
        target_patterns = [re.compile(pattern, re.I) for pattern in filters["target_like_patterns"]]
        unsafe_patterns = [re.compile(pattern, re.I) for pattern in filters["unsafe_patterns"]]
        manual = filters["manual_exclusions"]
        if not isinstance(manual, Mapping) or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in manual.items()):
            raise ValueError("candidate_filters.manual_exclusions must map IDs to nonempty reasons")
        eligible_rows = []
        extra_excluded = []
        for row in eligible_initial:
            reasons = []
            if len(row["prompt"]) > max_characters:
                reasons.append("excessive_length")
            if any(pattern.search(row["prompt"]) for pattern in target_patterns):
                reasons.append("target_like_pattern")
            if any(pattern.search(row["prompt"]) for pattern in unsafe_patterns):
                reasons.append("unsafe_pattern")
            if row["example_id"] in manual:
                reasons.append("manual_review: " + manual[row["example_id"]])
            if reasons:
                extra_excluded.append({
                    "example_id": row["example_id"],
                    "normalized_sha256": sha256(normalize_text(row["prompt"]).encode()).hexdigest(),
                    "reasons": [{"code": reason.split(": ", 1)[0]} for reason in reasons],
                    "manual_review_note": manual.get(row["example_id"]),
                })
            else:
                eligible_rows.append(row)
        eligible = tuple(eligible_rows)
        excluded = tuple([*excluded_initial, *extra_excluded])
        extra_filter_names = ["excessive_length", "target_like_pattern", "unsafe_pattern", "manual_review"]
    if not eligible:
        raise ValueError("no eligible anchor prompts remain after decontamination")
    heldout_ids: dict[str, list[str]] = {}
    for item in heldout:
        heldout_ids.setdefault(item.dataset, []).append(item.id)
    manifest = {
        "schema_version": 1,
        "sources": {key: asdict(specs[key]) for key in ROW_KEYS},
        "source_counts": {key: len(rows_by_key[key]) for key in ROW_KEYS},
        "gsm8k_split": {
            "method": "sha256_rank_of_seed_and_source_id",
            "split_seed": gsm8k_cfg["split_seed"],
            "validation_count": gsm8k_cfg["validation_count"],
            "train_ids": [item["example_id"] for item in train],
            "validation_ids": [item["example_id"] for item in validation],
        },
        "heldout_ids": {key: sorted(value) for key, value in sorted(heldout_ids.items())},
        "anchor_pool": {
            "candidate_ids": [item.id for item in sorted(candidates, key=lambda item: item.id)],
            "eligible_ids": [row["example_id"] for row in eligible],
            "excluded_ids": sorted(row["example_id"] for row in excluded),
        },
        "rendering": {
            "gsm8k_prompt": "raw question column",
            "candidate_prompt_column": anchors_cfg["prompt_column"],
            "candidate_context_column": anchors_cfg.get("context_column"),
            "candidate_context_template": anchors_cfg.get("context_template"),
            "retention_columns": {
                kind: config["retention"][kind]["columns"]
                for kind in ("boolq", "hellaswag", "arc_easy")
            },
        },
        "overlap": asdict(audit.config),
        "filters_applied": ["empty_candidate", "duplicate_candidate", "exact_heldout", "near_heldout", *extra_filter_names],
        "candidate_filters": filters,
    }
    return PreparedData(train, validation, test, eligible, excluded, audit.to_dict(), manifest)


def load_hf_rows(source: DatasetSource) -> Sequence[Mapping[str, Any]]:
    """Load one named Hugging Face split, importing ``datasets`` lazily."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the Hugging Face 'datasets' package to load configured data") from exc
    return load_dataset(source.path, name=source.name, split=source.split, revision=source.revision)


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def write_prepared_data(prepared: PreparedData, output_dir: str | Path) -> None:
    """Freeze outputs in a new directory; never replace an existing run."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    _write_jsonl(directory / "gsm8k_train.jsonl", prepared.gsm8k_train)
    _write_jsonl(directory / "gsm8k_validation.jsonl", prepared.gsm8k_validation)
    _write_jsonl(directory / "gsm8k_test.jsonl", prepared.gsm8k_test)
    _write_jsonl(directory / "eligible_anchors.jsonl", prepared.eligible_anchors)
    _write_jsonl(directory / "excluded_anchors.jsonl", prepared.excluded_anchors)
    _write_json(directory / "pool_audit.json", prepared.pool_audit)
    _write_json(directory / "data_manifest.json", prepared.data_manifest)


def run_config(
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    loader: Callable[[DatasetSource], Sequence[Mapping[str, Any]]] = load_hf_rows,
) -> PreparedData:
    """Load configured sources, prepare records, and write a new frozen run."""
    specs = _specs(config)
    rows = {key: loader(specs[key]) for key in ROW_KEYS}
    prepared = prepare_from_rows(config, rows)
    write_prepared_data(prepared, output_dir)
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Explicit dataset and audit JSON config")
    parser.add_argument("--output-dir", required=True, type=Path, help="New, never-overwritten run directory")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config.get("candidate_filters"), Mapping) or config["candidate_filters"].get("review_completed") is not True:
        parser.error("candidate_filters.review_completed must be true after source/pattern/manual review")
    for key, source in _specs(config).items():
        if re.fullmatch(r"[0-9a-fA-F]{40}", source.revision) is None:
            parser.error(f"{key} source revision must be a full 40-character commit SHA")
    prepared = run_config(config, args.output_dir)
    print(
        f"prepared {len(prepared.gsm8k_train)} GSM8K train, "
        f"{len(prepared.gsm8k_validation)} validation and "
        f"{len(prepared.eligible_anchors)} eligible anchor rows in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
