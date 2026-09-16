"""Synthetic embedding tests; no model download is needed."""

import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kl_anchors.diversity import (  # noqa: E402
    DiversityConfig,
    farthest_first_ranks,
    rank_pool_rows,
    read_pool,
    write_ranked_rows,
)


COMMIT = "a" * 40
CONFIG = DiversityConfig("example/embedding-model", COMMIT, "farthest_first", 17)


class DiversityTests(unittest.TestCase):
    def test_farthest_first_is_order_and_scale_independent(self):
        ids = ["a", "b", "c", "d"]
        vectors = [[1, 0], [-2, 0], [0, 3], [0, -4]]
        ranks = farthest_first_ranks(ids, vectors, seed=17)
        reordered = farthest_first_ranks(
            [ids[index] for index in (3, 1, 0, 2)],
            [vectors[index] for index in (3, 1, 0, 2)],
            seed=17,
        )
        self.assertEqual(ranks, reordered)
        self.assertEqual(sorted(ranks.values()), [0, 1, 2, 3])
        first = min(ranks, key=ranks.get)
        opposite = {"a": "b", "b": "a", "c": "d", "d": "c"}[first]
        self.assertEqual(ranks[opposite], 1)

    def test_stable_tie_break_on_same_distance(self):
        ranks = farthest_first_ranks(["z", "b", "a"], [[1, 0], [1, 0], [1, 0]], seed=3)
        first = min(ranks, key=ranks.get)
        rest = sorted(set(ranks) - {first})
        self.assertEqual([ranks[item_id] for item_id in rest], [1, 2])

    def test_reject_bad_vectors_and_ids(self):
        with self.assertRaisesRegex(ValueError, "nonzero norms"):
            farthest_first_ranks(["a", "b"], [[1, 0], [0, 0]], seed=0)
        with self.assertRaisesRegex(ValueError, "unique"):
            farthest_first_ranks(["a", "a"], [[1, 0], [0, 1]], seed=0)
        with self.assertRaisesRegex(ValueError, "finite"):
            farthest_first_ranks(["a"], [[float("nan"), 1]], seed=0)

    def test_ranked_rows_have_metadata_and_preserve_other_fields(self):
        rows = [
            {"example_id": "b", "prompt": "B", "response_token_count": 4},
            {"example_id": "a", "prompt": "A", "response_token_count": 6},
        ]
        output = rank_pool_rows(rows, [[0, 2], [3, 0]], CONFIG)
        self.assertEqual([row["example_id"] for row in output], ["a", "b"])
        self.assertEqual({row["diversity_rank"] for row in output}, {0, 1})
        self.assertEqual(output[0]["response_token_count"], 6)
        self.assertEqual(output[0]["diversity_metadata"]["model_revision"], COMMIT)
        self.assertEqual(output[0]["diversity_metadata"]["embedding_dimensions"], 2)

    def test_jsonl_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = Path(directory) / "eligible.jsonl"
            pool.write_text(json.dumps({"example_id": "a", "prompt": "A"}) + "\n", encoding="utf-8")
            rows = read_pool(pool)
            ranked = rank_pool_rows(rows, [[1, 0]], CONFIG)
            output = Path(directory) / "ranked.jsonl"
            write_ranked_rows(ranked, output)
            self.assertEqual(json.loads(output.read_text().splitlines()[0])["diversity_rank"], 0)
            with self.assertRaises(FileExistsError):
                write_ranked_rows(ranked, output)

    def test_pinned_revision_and_algorithm_required(self):
        with self.assertRaisesRegex(ValueError, "commit hash"):
            DiversityConfig("model", "main", "farthest_first", 0)
        with self.assertRaisesRegex(ValueError, "farthest_first"):
            DiversityConfig("model", COMMIT, "kmeans", 0)


if __name__ == "__main__":
    unittest.main()
