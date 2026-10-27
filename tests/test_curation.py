"""Regression checks for review errors that would change the scientific pool."""
import copy
import unittest

import numpy as np

from kl_anchors.curation import (
    apply_reviews, assemble_flags, checked_inputs, semantic_pairs, surface_proposal,
)


FILTERS = {"max_characters": 4096, "target_like_patterns": [], "unsafe_patterns": [], "manual_exclusions": {}}
OVERLAP = {"ngram_size": 5, "min_shared_ngrams": 2, "min_containment": 0.5}


def answers(flags, decisions=None):
    return [{"review_id": x["review_id"], "bundle_sha256": "bundle",
             "decision": (decisions or {}).get(x["review_id"], "equivalent" if x["kind"] == "semantic_duplicate" else "keep"),
             "reviewer": "synthetic reviewer", "reason": "Synthetic test decision; not a real data review."}
            for x in flags]


class CurationTests(unittest.TestCase):
    def test_generic_phrase_false_positive_is_reviewable_and_restorable(self):
        candidates = [{"example_id": "relax", "prompt": "What is the best way to relax?"}]
        heldout = [{"dataset": "arc_easy", "example_id": "salt", "text":
                    "In an experiment, water and salt are mixed together. What is the best way to separate the salt from the water?"}]
        hard, lexical = surface_proposal(candidates, heldout, overlap=OVERLAP, candidate_filters=FILTERS)
        self.assertEqual(hard, [])
        self.assertEqual(len(lexical), 1)
        flags, _ = assemble_flags(candidates, heldout, hard, lexical, semantic=None, sample_size=0, sample_seed=42)
        eligible, audit = apply_reviews(candidates, hard, flags, answers(flags), bundle_sha256="bundle")
        self.assertEqual(eligible, candidates)
        self.assertFalse(audit[0]["excluded"])

    def test_exact_evaluation_overlap_and_quality_exclusions_remain_blocked(self):
        candidates = [{"example_id": "a", "prompt": "evaluation item"},
                      {"example_id": "b", "prompt": "keep me"},
                      {"example_id": "c", "prompt": "unsafe example"}]
        heldout = [{"dataset": "boolq", "example_id": "h", "text": "evaluation item"}]
        filters = {**FILTERS, "unsafe_patterns": ["unsafe"]}
        hard, lexical = surface_proposal(candidates, heldout, overlap=OVERLAP, candidate_filters=filters)
        self.assertEqual({x["example_id"] for x in hard}, {"a", "c"})
        flags, _ = assemble_flags(candidates, heldout, hard, lexical, semantic=None, sample_size=1, sample_seed=0)
        eligible, _ = apply_reviews(candidates, hard, flags, answers(flags), bundle_sha256="bundle")
        self.assertEqual([x["example_id"] for x in eligible], ["b"])

    def test_exact_representative_is_not_an_already_excluded_id(self):
        candidates = [{"example_id": k, "prompt": "same prompt"} for k in ("a", "b", "c")]
        heldout = [{"dataset": "test", "example_id": "h", "text": "unrelated"}]
        filters = {**FILTERS, "manual_exclusions": {"a": "source-specific exclusion"}}
        hard, _ = surface_proposal(candidates, heldout, overlap=OVERLAP, candidate_filters=filters)
        self.assertEqual([x["example_id"] for x in hard], ["a", "c"])
        self.assertEqual(hard[1]["reasons"][0]["representative_id"], "b")

    def test_all_semantic_matches_are_reviewed_not_just_the_top_match(self):
        candidates = [{"example_id": "a", "prompt": "candidate"}, {"example_id": "b", "prompt": "safe"}]
        heldout = [{"dataset": "test", "example_id": k, "text": k} for k in ("x", "y")]
        pairs = semantic_pairs(np.array([[1, 0], [-1, 0]]), np.array([[1, 0], [.99, .1]]),
                               heldout_threshold=.9, dedup_threshold=.95, chunk_size=1)
        self.assertEqual(len(pairs[0]), 2)
        flags, _ = assemble_flags(candidates, heldout, [], [], semantic=pairs, sample_size=0, sample_seed=0)
        review = answers(flags)
        review[0]["decision"] = "exclude"
        eligible, _ = apply_reviews(candidates, [], flags, review, bundle_sha256="bundle")
        self.assertEqual([x["example_id"] for x in eligible], ["b"])

    def test_no_transitive_deduplication_through_removed_candidate(self):
        candidates = [{"example_id": k, "prompt": f"request {k}"} for k in ("a", "b", "c")]
        heldout = [{"dataset": "test", "example_id": "h", "text": "heldout"}]
        radians = np.deg2rad([0, 30, 60])
        vectors = np.column_stack([np.cos(radians), np.sin(radians)])
        pairs = semantic_pairs(vectors, np.array([[0, -1]]), heldout_threshold=.9, dedup_threshold=.85, chunk_size=1)
        self.assertEqual([(i, j) for i, j, _ in pairs[1]], [(0, 1), (1, 2)])
        flags, _ = assemble_flags(candidates, heldout, [], [], semantic=pairs, sample_size=0, sample_seed=0)
        eligible, audit = apply_reviews(candidates, [], flags, answers(flags), bundle_sha256="bundle")
        self.assertEqual([x["example_id"] for x in eligible], ["a", "c"])
        self.assertEqual(audit[1]["reasons"][0]["representative_id"], "a")
        review = answers(flags)
        for x in review:
            x["decision"] = "distinct"
        self.assertEqual(apply_reviews(candidates, [], flags, review, bundle_sha256="bundle")[0], candidates)

    def test_reviews_cannot_be_missing_stale_duplicate_or_unattributed(self):
        candidates = [{"example_id": "a", "prompt": "safe"}]
        heldout = [{"dataset": "test", "example_id": "h", "text": "heldout"}]
        flags, _ = assemble_flags(candidates, heldout, [], [], semantic=None, sample_size=1, sample_seed=0)
        valid = answers(flags)
        invalid = [[], valid + valid]
        for field, value in (("bundle_sha256", "old"), ("reviewer", ""), ("reason", ""),
                             ("decision", "pending"), ("review_id", "unknown")):
            modified = copy.deepcopy(valid)
            modified[0][field] = value
            invalid.append(modified)
        for review in invalid:
            with self.subTest(review=review), self.assertRaises(ValueError):
                apply_reviews(candidates, [], flags, review, bundle_sha256="bundle")

    def test_deterministic_input_order_and_spot_checks(self):
        candidates = [{"example_id": str(i), "prompt": f"unique request {i}"} for i in range(10)]
        heldout = [{"dataset": "test", "example_id": "h", "text": "heldout"}]
        first = assemble_flags(candidates, heldout, [], [], semantic=None, sample_size=3, sample_seed=18)
        other = assemble_flags(candidates[::-1], heldout, [], [], semantic=None, sample_size=3, sample_seed=18)
        self.assertEqual(first, other)
        self.assertEqual(len(first[0]), 3)

    def test_rejects_labels_duplicate_ids_and_invalid_embeddings(self):
        c = [{"example_id": "a", "prompt": "safe"}]
        h = [{"dataset": "test", "example_id": "h", "text": "heldout"}]
        for candidates, heldout in ((c + c, h), ([{**c[0], "damage_score": 1}], h),
                                    (c, [{**h[0], "answer": "secret"}])):
            with self.assertRaises(ValueError):
                checked_inputs(candidates, heldout)
        for value in (np.array([[0, 0]]), np.array([[np.nan, 1]]), np.array([[np.inf, 1]])):
            with self.assertRaises(ValueError):
                semantic_pairs(value, np.array([[1, 0]]), heldout_threshold=.9, dedup_threshold=.9)
        for threshold in (None, 0, 2, float("nan"), True):
            with self.assertRaises(ValueError):
                semantic_pairs(np.array([[1, 0]]), np.array([[1, 0]]), heldout_threshold=threshold, dedup_threshold=.9)


if __name__ == "__main__":
    unittest.main()
