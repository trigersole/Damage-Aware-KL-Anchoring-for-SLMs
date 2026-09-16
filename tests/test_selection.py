import unittest

from kl_anchors.selection import Candidate, select_anchors


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.pool = [
            Candidate("a", 4, vulnerability_score=0.3, diversity_rank=2),
            Candidate("b", 3, vulnerability_score=0.8, diversity_rank=0),
            Candidate("c", 5, vulnerability_score=0.5, diversity_rank=1),
        ]

    def test_same_budget_and_pool_across_methods(self):
        outputs = [select_anchors(self.pool, method=m, anchor_token_budget=6, seed=7)
                   for m in ("random", "diversity", "vulnerability")]
        self.assertEqual(len({o["eligible_pool_sha256"] for o in outputs}), 1)
        for output in outputs:
            self.assertEqual(sum(x["selected_response_tokens"] for x in output["selected"]), 6)
            self.assertEqual(output["anchor_token_budget"], 6)

    def test_vulnerability_and_diversity_orders(self):
        vulnerability = select_anchors(self.pool, method="vulnerability", anchor_token_budget=6, seed=7)
        diversity = select_anchors(self.pool, method="diversity", anchor_token_budget=6, seed=7)
        self.assertEqual([x["example_id"] for x in vulnerability["selected"]], ["b", "c"])
        self.assertEqual([x["example_id"] for x in diversity["selected"]], ["b", "c"])
        self.assertEqual(vulnerability["selected"][1]["selected_response_tokens"], 3)

    def test_random_is_deterministic(self):
        a = select_anchors(self.pool, method="random", anchor_token_budget=6, seed=42)
        b = select_anchors(list(reversed(self.pool)), method="random", anchor_token_budget=6, seed=42)
        self.assertEqual(a, b)

    def test_invalid_pool_fails_closed(self):
        with self.assertRaises(ValueError):
            select_anchors(self.pool[:1], method="random", anchor_token_budget=5, seed=1)
        with self.assertRaises(ValueError):
            select_anchors([Candidate("a", 1), Candidate("a", 2)], method="random", anchor_token_budget=2, seed=1)

    def test_combined_requires_explicit_weight_and_same_budget(self):
        with self.assertRaises(ValueError):
            select_anchors(self.pool, method="damage_diverse", anchor_token_budget=6, seed=1)
        result = select_anchors(self.pool, method="damage_diverse", anchor_token_budget=6, seed=1, damage_weight=0.5)
        self.assertEqual(sum(x["selected_response_tokens"] for x in result["selected"]), 6)
        self.assertEqual(result["damage_weight"], 0.5)


if __name__ == "__main__":
    unittest.main()
