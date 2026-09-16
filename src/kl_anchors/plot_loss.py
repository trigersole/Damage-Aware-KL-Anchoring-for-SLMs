"""Plot the saved training objective and target loss against completed epochs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot_loss(history_path: Path, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output_path.exists():
        raise FileExistsError(output_path)
    with history_path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError("training history is empty")
    x = [row["epoch"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(x, [row["target_loss"] for row in rows], label="Target training loss", linewidth=1.5)
    if any(row["anchor_tokens"] > 0 for row in rows):
        ax.plot(x, [row["total_loss"] for row in rows], label="Total training objective", linewidth=1.2)
    ax.set_xlabel("Target-data epochs (examples seen / training examples)")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss by epoch")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_loss(args.history, args.output)


if __name__ == "__main__":
    main()
