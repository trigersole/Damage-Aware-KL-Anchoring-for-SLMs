"""Check embedding adapter length policies without downloading a real model."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from kl_anchors.curate_anchors import encode_inputs


class FakeEncoder:
    max_seq_length = 8

    def __init__(self):
        self.encoded = False
        self.options = None

    def eval(self):
        return self

    def tokenizer(self, texts, **kwargs):
        return {"input_ids": [[0] * (len(x.split()) + 2) for x in texts]}

    def encode(self, texts, **kwargs):
        self.encoded = True
        self.options = kwargs
        return np.ones((len(texts), 3))


class EncoderPolicyTests(unittest.TestCase):
    def run_encoder(self, fake, prompt, truncation="reject", max_length=8):
        module = SimpleNamespace(SentenceTransformer=lambda *args, **kwargs: fake)
        with patch.dict("sys.modules", {"sentence_transformers": module}):
            return encode_inputs(
                [{"example_id": "candidate", "prompt": prompt}],
                [{"dataset": "boolq", "example_id": "evaluation", "text": "heldout input"}],
                model_id="synthetic/encoder", model_revision="a" * 40,
                max_seq_length=max_length, truncation=truncation, device="cpu", batch_size=2,
            )

    def test_rejects_truncation_before_encoding(self):
        fake = FakeEncoder()
        with self.assertRaisesRegex(ValueError, "exceed the encoder limit"):
            self.run_encoder(fake, "one two three four five six seven")
        self.assertFalse(fake.encoded)

    def test_explicit_truncation_records_affected_id_and_original_length(self):
        fake = FakeEncoder()
        a, b, meta = self.run_encoder(fake, "one two three four five six seven", "allow")
        self.assertEqual(meta["truncated_inputs"], [{"candidate_id": "candidate", "tokens": 9}])
        self.assertEqual(a.shape, (1, 3))
        self.assertEqual(b.shape, (1, 3))
        self.assertEqual(fake.options["prompt"], "")

    def test_does_not_claim_to_increase_encoder_context_capacity(self):
        with self.assertRaisesRegex(ValueError, "configured limit"):
            self.run_encoder(FakeEncoder(), "short", max_length=1000)


if __name__ == "__main__":
    unittest.main()
