"""Tests for metric conventions, independent of a model or evaluation harness."""

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kl_anchors.metrics import (  # noqa: E402
    AccuracySummary,
    aggregate_accuracy,
    benchmark_forgetting,
    extract_gsm8k_final_answer,
    gsm8k_exact_match,
    normalize_gsm8k_answer,
    target_gain,
)


class GSM8KMetricTests(unittest.TestCase):
    def test_gold_marker_and_normalization(self) -> None:
        self.assertEqual(extract_gsm8k_final_answer("work: 8 + 2\n#### 1,200.00"), "1200")
        self.assertEqual(normalize_gsm8k_answer("-0002.50"), "-2.5")
        self.assertEqual(normalize_gsm8k_answer("-0.00"), "0")

    def test_explicit_answer_takes_priority_over_scratchwork(self) -> None:
        self.assertEqual(extract_gsm8k_final_answer("3 plus 4 is 7. Final answer: $7."), "7")
        self.assertEqual(extract_gsm8k_final_answer("There are 3 apples. Answer is 12 of 20."), "12")

    def test_fallback_and_parse_failures(self) -> None:
        self.assertEqual(extract_gsm8k_final_answer("Compute 2 + 5 = 7."), "7")
        self.assertIsNone(extract_gsm8k_final_answer("Answer: unknown"))
        self.assertIsNone(extract_gsm8k_final_answer("#### no number"))
        self.assertIsNone(normalize_gsm8k_answer("12 kg"))
        self.assertFalse(gsm8k_exact_match("Answer: unknown", "#### no number"))

    def test_exact_match_uses_numeric_value(self) -> None:
        self.assertTrue(gsm8k_exact_match("The answer is 1,200.0", "#### 1200"))
        self.assertFalse(gsm8k_exact_match("The answer is 1201", "#### 1200"))


class AggregateMetricTests(unittest.TestCase):
    def test_accuracy_summary(self) -> None:
        self.assertEqual(
            aggregate_accuracy([True, False, True, True]),
            AccuracySummary(correct=3, total=4, accuracy=0.75),
        )

    def test_accuracy_rejects_empty_and_non_boolean_items(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_accuracy([])
        with self.assertRaises(TypeError):
            aggregate_accuracy([True, 1])

    def test_forgetting_preserves_improvements(self) -> None:
        result = benchmark_forgetting(
            {"BoolQ": 0.6, "ARC-Easy": 0.4},
            {"BoolQ": 0.5, "ARC-Easy": 0.45},
        )
        self.assertAlmostEqual(result["BoolQ"], 0.1)
        self.assertAlmostEqual(result["ARC-Easy"], -0.05)

    def test_forgetting_requires_matching_benchmarks_and_valid_scores(self) -> None:
        with self.assertRaises(ValueError):
            benchmark_forgetting({"BoolQ": 0.6}, {"ARC-Easy": 0.5})
        with self.assertRaises(ValueError):
            benchmark_forgetting({"BoolQ": float("nan")}, {"BoolQ": 0.5})

    def test_target_gain(self) -> None:
        self.assertAlmostEqual(target_gain(0.2, 0.55), 0.35)
        with self.assertRaises(ValueError):
            target_gain(0.2, 1.1)


if __name__ == "__main__":
    unittest.main()
