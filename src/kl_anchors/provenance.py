"""Print the internal evaluator's immutable source fingerprint and environment."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
from pathlib import Path
import platform


EVALUATION_FILES = ("evaluate.py", "evaluate_study.py", "metrics.py", "hf_runtime.py", "train.py", "training_core.py", "provenance.py")


def evaluation_code_digest() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha1()
    for name in EVALUATION_FILES:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update((root / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digest-only", action="store_true")
    args = parser.parse_args()
    print(evaluation_code_digest())
    if args.digest_only:
        return
    print("Python", platform.python_version())
    for package in ("torch", "transformers", "datasets", "peft", "accelerate", "sentence-transformers", "matplotlib"):
        try:
            print(package, metadata.version(package))
        except metadata.PackageNotFoundError:
            print(package, "not installed")


if __name__ == "__main__":
    main()
