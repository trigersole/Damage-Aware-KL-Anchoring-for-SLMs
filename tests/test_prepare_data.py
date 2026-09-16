"""Synthetic, network-free checks for the dataset preparation pipeline."""

import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kl_anchors.prepare_data import (  # noqa: E402
    DatasetSource,
    build_candidates,
    prepare_from_rows,
    render_heldout_text,
    run_config,
    source_id,
    split_gsm8k_train,
    write_prepared_data,
)


def source(path, split):
    return {"path": path, "name": "fixture", "revision": "commit-abc", "split": split}


def fixture_config():
    return {
        "gsm8k": {
            "train_source": source("gsm8k", "train"),
            "test_source": source("gsm8k", "test"),
            "question_column": "question",
            "answer_column": "answer",
            "validation_count": 1,
            "split_seed": 7,
        },
        "retention": {
            "boolq": {
                "source": source("boolq", "validation"),
                "columns": {"question": "question", "passage": "passage"},
            },
            "hellaswag": {
                "source": source("hellaswag", "validation"),
                "columns": {"context": "ctx", "choices": "endings"},
            },
            "arc_easy": {
                "source": source("arc_easy", "validation"),
                "columns": {"question": "question", "choices": "choices"},
            },
        },
        "anchors": {
            "source": source("general_instructions", "train"),
            "prompt_column": "instruction",
            "context_column": "context",
            "context_template": "{prompt}\n\nContext:\n{context}",
        },
        "overlap": {"ngram_size": 2, "min_shared_ngrams": 2, "min_containment": 0.5},
    }


def fixture_rows():
    return {
        "gsm8k_train": [
            {"question": "train question one", "answer": "reasoning one #### 1"},
            {"question": "train question two", "answer": "reasoning two #### 2"},
            {"question": "train question three", "answer": "reasoning three #### 3"},
        ],
        "gsm8k_test": [{"question": "test only question", "answer": "secret test answer"}],
        "boolq": [{"question": "is sky blue", "passage": "sky is blue", "answer": True}],
        "hellaswag": [{"ctx": "a person runs", "endings": ["then rests", "then jumps"], "label": "0"}],
        "arc_easy": [{"question": "what is water", "choices": {"label": ["A", "B"], "text": ["liquid", "solid"]}, "answerKey": "A"}],
        "anchors": [
            {"instruction": "test only question", "context": "", "response": "ignored"},
            {"instruction": "is sky blue", "context": "sky is blue", "response": "ignored"},
            {"instruction": "write a novel poem", "context": "about clouds", "response": "ignored"},
            {"instruction": "another harmless prompt", "context": "", "response": "ignored"},
        ],
    }


class PrepareDataTests(unittest.TestCase):
    def test_source_ids_include_revision_split_and_row_id(self):
        first = DatasetSource("repo", "rev-1", "train", id_column="uid")
        second = DatasetSource("repo", "rev-2", "train", id_column="uid")
        self.assertEqual(source_id(first, {"uid": "row-a"}, 0), source_id(first, {"uid": "row-a"}, 5))
        self.assertNotEqual(source_id(first, {"uid": "row-a"}, 0), source_id(second, {"uid": "row-a"}, 0))
        with self.assertRaisesRegex(ValueError, "duplicate source IDs"):
            split_gsm8k_train(
                [{"uid": "same", "q": "one", "a": "1"}, {"uid": "same", "q": "two", "a": "2"}],
                first,
                question_column="q",
                answer_column="a",
                validation_count=1,
                split_seed=0,
            )

    def test_train_validation_split_is_deterministic_and_train_only(self):
        config = fixture_config()
        rows = fixture_rows()
        source_spec = DatasetSource(**config["gsm8k"]["train_source"])
        first = split_gsm8k_train(
            rows["gsm8k_train"], source_spec,
            question_column="question", answer_column="answer",
            validation_count=1, split_seed=7,
        )
        second = split_gsm8k_train(
            rows["gsm8k_train"], source_spec,
            question_column="question", answer_column="answer",
            validation_count=1, split_seed=7,
        )
        self.assertEqual(first, second)
        self.assertEqual((len(first[0]), len(first[1])), (2, 1))
        self.assertEqual(
            {row["example_id"] for row in first[0]} & {row["example_id"] for row in first[1]},
            set(),
        )
        self.assertTrue(all("test" not in row["example_id"] for group in first for row in group))

    def test_heldout_renderers_ignore_gold_labels(self):
        rows = fixture_rows()
        self.assertEqual(render_heldout_text("gsm8k", rows["gsm8k_test"][0], {"question": "question"}), "test only question")
        boolq = render_heldout_text("boolq", rows["boolq"][0], {"question": "question", "passage": "passage"})
        hellaswag = render_heldout_text("hellaswag", rows["hellaswag"][0], {"context": "ctx", "choices": "endings"})
        arc = render_heldout_text("arc_easy", rows["arc_easy"][0], {"question": "question", "choices": "choices"})
        self.assertEqual(boolq, "is sky blue\n\nsky is blue")
        self.assertEqual(hellaswag, "a person runs\nthen rests\nthen jumps")
        self.assertEqual(arc, "what is water\nliquid\nsolid")
        self.assertNotIn("secret test answer", " ".join((boolq, hellaswag, arc)))

    def test_candidate_context_template_and_decontamination(self):
        config = fixture_config()
        rows = fixture_rows()
        candidates = build_candidates(
            rows["anchors"], DatasetSource(**config["anchors"]["source"]),
            prompt_column="instruction", context_column="context",
            context_template=config["anchors"]["context_template"],
        )
        self.assertEqual(candidates[2].text, "write a novel poem\n\nContext:\nabout clouds")
        prepared = prepare_from_rows(config, rows)
        self.assertEqual(len(prepared.eligible_anchors), 2)
        self.assertEqual(len(prepared.excluded_anchors), 2)
        codes = {reason["code"] for row in prepared.excluded_anchors for reason in row["reasons"]}
        self.assertIn("exact_heldout", codes)
        self.assertIn("near_heldout", codes)
        self.assertEqual(set(prepared.data_manifest["heldout_ids"]), {"gsm8k_validation", "gsm8k_test", "boolq", "hellaswag", "arc_easy"})
        self.assertEqual(prepared.gsm8k_train[0].keys(), {"example_id", "prompt", "response"})
        self.assertEqual(prepared.eligible_anchors[0].keys(), {"example_id", "prompt"})

    def test_frozen_outputs_refuse_overwrite(self):
        prepared = prepare_from_rows(fixture_config(), fixture_rows())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run-1"
            write_prepared_data(prepared, output)
            train = [json.loads(line) for line in (output / "gsm8k_train.jsonl").read_text().splitlines()]
            self.assertEqual(len(train), 2)
            self.assertTrue((output / "data_manifest.json").exists())
            with self.assertRaises(FileExistsError):
                write_prepared_data(prepared, output)

    def test_loader_is_injected_without_importing_datasets(self):
        config = fixture_config()
        rows = fixture_rows()
        by_path_split = {
            (spec["path"], spec["split"]): rows[key]
            for key, spec in (
                ("gsm8k_train", config["gsm8k"]["train_source"]),
                ("gsm8k_test", config["gsm8k"]["test_source"]),
                *[(kind, config["retention"][kind]["source"]) for kind in ("boolq", "hellaswag", "arc_easy")],
                ("anchors", config["anchors"]["source"]),
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            prepared = run_config(
                config, Path(directory) / "run",
                loader=lambda spec: by_path_split[(spec.path, spec.split)],
            )
            self.assertEqual(len(prepared.gsm8k_validation), 1)


if __name__ == "__main__":
    unittest.main()
