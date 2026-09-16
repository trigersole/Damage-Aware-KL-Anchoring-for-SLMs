"""Tests for deterministic candidate-anchor contamination screening."""

from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kl_anchors.data_audit import (  # noqa: E402
    AnchorCandidate,
    HeldoutExample,
    OverlapConfig,
    audit_anchor_pool,
    normalize_text,
)


CONFIG = OverlapConfig(ngram_size=2, min_shared_ngrams=2, min_containment=0.75)


class DataAuditTests(unittest.TestCase):
    def test_normalization_and_exact_heldout_exclusion(self):
        self.assertEqual(normalize_text("  ＦＯＯ\u00a0\nBAR  "), "foo bar")
        result = audit_anchor_pool(
            [AnchorCandidate("a", " ＦＯＯ\nBAR ")],
            [HeldoutExample("gsm8k_validation", "q1", "foo bar")],
            config=CONFIG,
        )
        self.assertEqual(result.eligible_ids, ())
        self.assertEqual(result.manifest[0].reasons[0].code, "exact_heldout")
        self.assertEqual(result.manifest[0].reasons[0].dataset, "gsm8k_validation")

    def test_duplicate_survivor_and_order_are_deterministic(self):
        candidates = [
            AnchorCandidate("z", "Same prompt"),
            AnchorCandidate("b", "Different prompt"),
            AnchorCandidate("a", " same  PROMPT "),
        ]
        first = audit_anchor_pool(candidates, [], config=CONFIG)
        second = audit_anchor_pool(reversed(candidates), [], config=CONFIG)
        self.assertEqual(first, second)
        self.assertEqual(first.eligible_ids, ("a", "b"))
        duplicate = first.manifest[2].reasons[0]
        self.assertEqual((duplicate.code, duplicate.candidate_id), ("duplicate_candidate", "a"))

    def test_near_overlap_across_supplied_benchmarks(self):
        datasets = (
            "gsm8k_validation",
            "gsm8k_test",
            "boolq",
            "hellaswag",
            "arc_easy",
        )
        heldout = [
            HeldoutExample(dataset, "q1", "prefix alpha beta gamma delta suffix")
            for dataset in datasets
        ]
        result = audit_anchor_pool(
            [AnchorCandidate("candidate", "alpha beta gamma delta")],
            heldout,
            config=CONFIG,
        )
        reasons = result.manifest[0].reasons
        self.assertEqual({reason.dataset for reason in reasons}, set(datasets))
        self.assertTrue(all(reason.code == "near_heldout" for reason in reasons))
        self.assertTrue(all(reason.shared_ngrams == 3 for reason in reasons))
        self.assertTrue(all(reason.containment == 1.0 for reason in reasons))

    def test_thresholds_are_explicit_and_boundary_is_inclusive(self):
        candidate = [AnchorCandidate("a", "alpha beta gamma delta")]
        heldout = [HeldoutExample("boolq", "q", "alpha beta gamma other")]
        boundary = audit_anchor_pool(
            candidate,
            heldout,
            config=OverlapConfig(2, 2, 2 / 3),
        )
        stricter = audit_anchor_pool(
            candidate,
            heldout,
            config=OverlapConfig(2, 3, 2 / 3),
        )
        self.assertEqual(boundary.eligible_ids, ())
        self.assertEqual(stricter.eligible_ids, ("a",))

    def test_empty_short_text_and_duplicate_ids(self):
        result = audit_anchor_pool(
            [AnchorCandidate("empty", "  \n"), AnchorCandidate("short", "one")],
            [HeldoutExample("arc_easy", "q", "one extra words")],
            config=CONFIG,
        )
        self.assertEqual(result.eligible_ids, ("short",))
        self.assertEqual(result.manifest[0].reasons[0].code, "empty_candidate")
        with self.assertRaisesRegex(ValueError, "candidate IDs must be unique"):
            audit_anchor_pool([AnchorCandidate("x", "one"), AnchorCandidate("x", "two")], [], config=CONFIG)

    def test_manifest_json_is_order_independent(self):
        candidates = [AnchorCandidate("b", "alpha beta gamma"), AnchorCandidate("a", "unique prompt")]
        heldout = [HeldoutExample("boolq", "q", "alpha beta gamma suffix")]
        first = audit_anchor_pool(candidates, heldout, config=CONFIG)
        second = audit_anchor_pool(reversed(candidates), reversed(heldout), config=CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            one = Path(directory) / "one.json"
            two = Path(directory) / "two.json"
            first.write_manifest(one)
            second.write_manifest(two)
            self.assertEqual(one.read_bytes(), two.read_bytes())
            self.assertNotIn(b"alpha beta gamma", one.read_bytes())

    def test_invalid_thresholds(self):
        for args in [(0, 1, 0.5), (2, 0, 0.5), (2, 1, 0), (2, 1, 1.1), (2, 1, float("nan"))]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                OverlapConfig(*args)


if __name__ == "__main__":
    unittest.main()
