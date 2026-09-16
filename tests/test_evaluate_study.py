import tempfile
import unittest

from kl_anchors.evaluate_study import evaluation_from_study_protocol
from test_protocol import complete_protocol


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<chat>" + "|".join(item["content"] for item in messages) + "<assistant>"


class EvaluationAdapterTests(unittest.TestCase):
    def test_common_protocol_renders_locked_chat_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            study = complete_protocol(directory)
            study["evaluation"]["prompt_templates"]["boolq"] = "Passage: {passage}\nQuestion: {question}"
            study["evaluation"]["prompt_templates"]["hellaswag"] = "{ctx}"
            study["evaluation"]["prompt_templates"]["arc_easy"] = "{question}"
            protocol = evaluation_from_study_protocol(study, FakeTokenizer())
            self.assertIn("{prompt}", protocol["benchmarks"]["gsm8k"]["prompt_template"])
            self.assertIn("<chat>Passage: {passage}", protocol["benchmarks"]["boolq"]["prompt_template"])
            self.assertEqual(protocol["model"]["dtype"], "bfloat16")
            self.assertEqual(protocol["model"]["revision"], study["model"]["revision"])


if __name__ == "__main__":
    unittest.main()
