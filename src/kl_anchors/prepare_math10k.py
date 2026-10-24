
"""Prepare deterministic MATH-10K training and validation manifests."""

import hashlib
import json
import random
from pathlib import Path

SOURCE = Path("data/raw/math10k/math_10k.json")
DEST = Path("data/processed/math10k")
EXPECTED_SHA256 = "0342d0d860ad8592b579329337c90e42eefd3d9f2898043140cbd120630418b8"

SEED = 42
VALIDATION_SIZE = 300


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Unexpected dataset SHA-256: {digest}")

    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    if len(data) != 9919:
        raise ValueError(f"Expected 9919 examples, got {len(data)}")

    prepared = []
    for i, row in enumerate(data):
        for field in ("instruction", "input", "output", "answer"):
            if not isinstance(row.get(field), str):
                raise ValueError(f"Row {i}: invalid {field}")

        question = row["instruction"].strip()
        context = row["input"].strip()
        response = row["output"].strip()

        if context:
            question += "\n\n" + context

        if not question or not response:
            raise ValueError(f"Row {i}: empty question or response")

        prepared.append({
            "example_id": f"math10k:{i:05d}",
            "prompt": question,
            "response": response,
            "answer": row["answer"].strip(),
            "source_index": i,
        })

    indices = list(range(len(prepared)))
    random.Random(SEED).shuffle(indices)

    validation_indices = set(indices[:VALIDATION_SIZE])
    train = [row for i, row in enumerate(prepared)
             if i not in validation_indices]
    validation = [row for i, row in enumerate(prepared)
                  if i in validation_indices]

    DEST.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "validation.jsonl"):
        if (DEST / name).exists():
            raise FileExistsError(f"{DEST / name} already exists")

    write_jsonl(DEST / "train.jsonl", train)
    write_jsonl(DEST / "validation.jsonl", validation)

    metadata = {
        "source_sha256": digest,
        "source_examples": len(data),
        "seed": SEED,
        "validation_size": len(validation),
        "training_size": len(train),
        "split_method": "shuffled source indices, seed 42",
    }
    (DEST / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
