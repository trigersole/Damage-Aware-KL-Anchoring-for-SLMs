import unittest

import numpy as np

from kl_anchors.data_audit import HeldoutExample
from kl_anchors.semantic_audit import screen_embeddings


class SemanticAuditTests(unittest.TestCase):
    def test_excludes_high_similarity_without_labels(self):
        heldout = [HeldoutExample("boolq", "b1", "question"), HeldoutExample("arc_easy", "a1", "question")]
        result = screen_embeddings(
            ["x", "y"], np.array([[1, 0], [0, 1]]), heldout,
            np.array([[1, 0], [1, 1]]), threshold=0.9, chunk_size=1,
        )
        self.assertTrue(result[0]["excluded"])
        self.assertFalse(result[1]["excluded"])
        self.assertEqual(result[0]["matched_dataset"], "boolq")

    def test_rejects_unaligned_or_zero_embeddings(self):
        heldout = [HeldoutExample("gsm8k_test", "t1", "q")]
        with self.assertRaises(ValueError):
            screen_embeddings(["x"], np.array([[0, 0]]), heldout, np.array([[1, 0]]), threshold=0.8, chunk_size=1)


if __name__ == "__main__":
    unittest.main()
