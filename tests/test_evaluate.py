"""Pure evaluation and persistence tests with an in-memory toy backend."""

from pathlib import Path
import importlib.util
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kl_anchors.evaluate import (  # noqa: E402
    aggregate_records,
    evaluate_item,
    HuggingFaceBackend,
    run_protocol,
    score_choice,
    score_multiple_choice,
    validate_protocol,
    write_outputs,
)


class ToyTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.strip().split()


class ToyModel:
    def __init__(self, token_scores: dict[str, list[float]], generation: str = "") -> None:
        self.token_scores = token_scores
        self.generation = generation

    def continuation_scores(self, token_ids: list[str]) -> list[float]:
        return self.token_scores[" ".join(token_ids)]


class ToyBackend:
    def __init__(self, token_scores: dict[str, list[float]], generation: str = "") -> None:
        self.tokenizer = ToyTokenizer()
        self.model = ToyModel(token_scores, generation)

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        assert max_new_tokens > 0
        return self.model.generation

    def token_logprobs(self, prompt: str, continuation: str) -> list[float]:
        assert prompt
        return self.model.continuation_scores(self.tokenizer.encode(continuation))


def protocol_fixture() -> dict:
    return {
        "experiment_id": "toy-validation",
        "seed": 17,
        "model": {
            "repo_id": "example/model",
            "revision": "fixed-revision",
            "dtype": "float32",
            "device_map": "cpu",
            "trust_remote_code": False,
        },
        "tokenizer": {
            "repo_id": "example/model",
            "revision": "fixed-revision",
            "use_fast": True,
            "add_special_tokens": True,
        },
        "generation": {
            "max_new_tokens": 16,
            "do_sample": False,
            "num_beams": 1,
            "skip_special_tokens": True,
        },
        "benchmarks": {
            "gsm8k": {
                "task": "gsm8k",
                "dataset": {"repo_id": "example/gsm8k", "config": "main", "revision": "fixed-data", "split": "validation"},
                "limit": 1,
                "prompt_template": "Question: {question}\nAnswer:",
                "fields": {"answer": "answer"},
                "scoring": {"answer_extraction": "gsm8k_final_number_v1"},
            },
            "boolq": {
                "task": "boolq",
                "dataset": {"repo_id": "example/boolq", "config": None, "revision": "fixed-data", "split": "validation"},
                "limit": 1,
                "prompt_template": "Passage: {passage}\nQuestion: {question}\nAnswer:",
                "fields": {"answer": "answer"},
                "scoring": {
                    "method": "conditional_log_likelihood",
                    "tokenization": "separate_no_special_tokens",
                    "length_normalization": False,
                    "choice_prefix": " ",
                    "choice_texts": ["No", "Yes"],
                },
            },
        },
    }


class ScoringTests(unittest.TestCase):
    def test_length_normalization_can_change_prediction(self) -> None:
        backend = ToyBackend({"short": [-0.6], "two words": [-0.4, -0.4]})
        raw = score_multiple_choice(backend, "prompt", ["short", "two words"], length_normalization=False)
        normalized = score_multiple_choice(backend, "prompt", ["short", "two words"], length_normalization=True)
        self.assertEqual(raw["prediction_index"], 0)
        self.assertEqual(normalized["prediction_index"], 1)
        self.assertEqual(normalized["choice_token_counts"], [1, 2])

    def test_empty_or_nonfinite_choice_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            score_choice([], length_normalization=False)
        with self.assertRaises(ValueError):
            score_choice([float("nan")], length_normalization=False)

    def test_gsm8k_validation_item_and_parse_failure(self) -> None:
        spec = protocol_fixture()["benchmarks"]["gsm8k"]
        backend = ToyBackend({}, "The answer is 1,200.0")
        item = evaluate_item(backend, "gsm8k", spec, {"question": "What?", "answer": "steps\n#### 1200"}, 0, max_new_tokens=16)
        self.assertEqual(item["predicted_answer"], "1200")
        self.assertTrue(item["correct"])
        backend.model.generation = "No numeric answer"
        failed = evaluate_item(backend, "gsm8k", spec, {"question": "What?", "answer": "#### 1200"}, 0, max_new_tokens=16)
        self.assertTrue(failed["parse_failure"])
        self.assertFalse(failed["correct"])

    def test_boolq_index_and_key_choices(self) -> None:
        backend = ToyBackend({"No": [-1.0], "Yes": [-0.1], "A": [-0.1], "B": [-1.0]})
        boolq = protocol_fixture()["benchmarks"]["boolq"]
        record = evaluate_item(backend, "boolq", boolq, {"passage": "P", "question": "Q", "answer": True}, 0, max_new_tokens=16)
        self.assertTrue(record["correct"])
        self.assertEqual(record["prediction_index"], 1)
        index_spec = {**boolq, "task": "index_mc", "fields": {"choices": "endings", "answer": "label"}}
        index_record = evaluate_item(backend, "hellaswag", index_spec, {"passage": "P", "question": "Q", "endings": ["A", "B"], "label": "0"}, 2, max_new_tokens=16)
        self.assertTrue(index_record["correct"])
        key_spec = {**boolq, "task": "key_mc", "fields": {"choices": "choices", "choice_text": "text", "choice_key": "label", "answer": "answerKey"}}
        key_record = evaluate_item(backend, "arc_easy", key_spec, {"passage": "P", "question": "Q", "choices": {"text": ["A", "B"], "label": ["A", "B"]}, "answerKey": "A"}, 1, max_new_tokens=16)
        self.assertTrue(key_record["correct"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
    def test_teacher_forced_positions_align_with_continuation(self) -> None:
        import torch

        class TinyTokenizer:
            def encode(self, text, *, add_special_tokens):
                return [1] if add_special_tokens else [2, 3]

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))

            def forward(self, *, input_ids, attention_mask):
                logits = torch.zeros((1, input_ids.shape[1], 5))
                logits[0, 0, 2] = 5
                logits[0, 1, 3] = 5
                return types.SimpleNamespace(logits=logits)

        backend = HuggingFaceBackend.__new__(HuggingFaceBackend)
        backend.torch = torch
        backend.tokenizer = TinyTokenizer()
        backend.model = TinyModel()
        backend.add_special_tokens = True
        scores = backend.token_logprobs("prompt", "choice")
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(score > -0.1 for score in scores))


class ProtocolAndOutputTests(unittest.TestCase):
    def test_protocol_requires_explicit_split_scoring_and_revision(self) -> None:
        protocol = protocol_fixture()
        validate_protocol(protocol)
        del protocol["benchmarks"]["boolq"]["dataset"]["revision"]
        with self.assertRaises(ValueError):
            validate_protocol(protocol)
        protocol = protocol_fixture()
        del protocol["benchmarks"]["boolq"]["scoring"]["length_normalization"]
        with self.assertRaises(ValueError):
            validate_protocol(protocol)

    def test_aggregate_and_no_overwrite(self) -> None:
        protocol = protocol_fixture()
        records = [
            {"benchmark": "gsm8k", "item_id": "gsm8k:0", "correct": True, "parse_failure": False},
            {"benchmark": "boolq", "item_id": "boolq:0", "correct": False, "parse_failure": False},
        ]
        aggregate = aggregate_records(protocol, records)
        self.assertEqual(aggregate["metrics"]["gsm8k"]["accuracy"], 1.0)
        self.assertEqual(aggregate["metrics"]["boolq"]["accuracy"], 0.0)
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "run-1"
            write_outputs(output, records, aggregate)
            self.assertEqual(len((output / "items.jsonl").read_text().splitlines()), 2)
            self.assertIn('"split": "validation"', (output / "aggregate.json").read_text())
            with self.assertRaises(FileExistsError):
                write_outputs(output, records, aggregate)

    def test_run_protocol_uses_configured_validation_split(self) -> None:
        protocol = protocol_fixture()
        backend = ToyBackend({"No": [-1.0], "Yes": [-0.1]}, "Answer: 42")
        calls = []

        def fake_load_dataset(repo_id, *, name, revision, split):
            calls.append((repo_id, name, revision, split))
            if repo_id.endswith("gsm8k"):
                return [{"question": "What is 40 + 2?", "answer": "#### 42"}]
            return [{"passage": "P", "question": "Q", "answer": True}]

        fake_datasets = types.SimpleNamespace(load_dataset=fake_load_dataset)
        with patch.dict(sys.modules, {"datasets": fake_datasets}):
            records = run_protocol(protocol, backend)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(call[-1] == "validation" for call in calls))
        self.assertTrue(all(record["correct"] for record in records))


if __name__ == "__main__":
    unittest.main()
