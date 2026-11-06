"""Synthetic embedding tests; no model download is needed."""

import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kl_anchors.diversity import (  # noqa: E402
    align_pool_to_teacher_cache,
    DiversityConfig,
    farthest_first_ranks,
    rank_pool_rows,
    read_pool,
    select_top_ranked_rows,
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

    def test_select_top_ranked_rows_returns_selection_order(self):
        rows = [
            {"example_id": "last", "prompt": "L", "diversity_rank": 2},
            {"example_id": "first", "prompt": "F", "diversity_rank": 0},
            {"example_id": "middle", "prompt": "M", "diversity_rank": 1},
        ]
        selected = select_top_ranked_rows(rows, 2)
        self.assertEqual([row["example_id"] for row in selected], ["first", "middle"])
        self.assertEqual([row["diversity_rank"] for row in selected], [0, 1])

    def test_select_top_ranked_rows_rejects_invalid_count_or_ranks(self):
        rows = [{"example_id": "a", "prompt": "A", "diversity_rank": 0}]
        with self.assertRaisesRegex(ValueError, "positive"):
            select_top_ranked_rows(rows, 0)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            select_top_ranked_rows(rows, 2)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            select_top_ranked_rows(
                [{"example_id": "a", "prompt": "A", "diversity_rank": 1}], 1
            )

    def test_align_pool_to_teacher_cache_filters_skips_and_adds_lengths(self):
        rows = [
            {"example_id": "a", "prompt": "A"},
            {"example_id": "b", "prompt": "B"},
            {"example_id": "skipped", "prompt": "too long"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.jsonl"
            cache.write_text(
                json.dumps({"example_id": "b", "response_token_count": 2}) + "\n"
                + json.dumps({"example_id": "a", "response_token_count": 1}) + "\n",
                encoding="utf-8",
            )
            aligned = align_pool_to_teacher_cache(rows, cache)
        self.assertEqual([row["example_id"] for row in aligned], ["a", "b"])
        self.assertEqual([row["response_token_count"] for row in aligned], [1, 2])


if __name__ == "__main__":
    unittest.main()
