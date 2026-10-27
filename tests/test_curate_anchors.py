"""Network-free curation round trip, source integrity and label isolation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from kl_anchors.curate_anchors import export_inputs, finalize, input_rows, propose
from test_prepare_data import fixture_config, fixture_rows


class CurationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = fixture_config()
        for spec in [self.config["gsm8k"]["train_source"], self.config["gsm8k"]["test_source"],
                     self.config["anchors"]["source"], *[x["source"] for x in self.config["retention"].values()]]:
            spec["revision"] = "a" * 40
        self.config["candidate_filters"] = {"max_characters": 4096, "target_like_patterns": [],
                                             "unsafe_patterns": [], "manual_exclusions": {}}
        self.rows = fixture_rows()
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config))
        self.inputs = self.root / "inputs"
        self.review = self.root / "review"

    def loader(self, source):
        key = "anchors" if source.path == "general_instructions" else source.path
        if key == "gsm8k":
            key += "_" + source.split
        return self.rows[key]

    def export(self):
        return export_inputs(self.config_path, self.inputs, target_train_overlap="allow", loader=self.loader)

    @staticmethod
    def fake_encoder(candidates, heldout, **kwargs):
        # Same embeddings create reviewed duplicate candidates. Heldout vectors
        # are orthogonal. No real model or threshold recommendation is implied.
        return (np.tile([1., 0.], (len(candidates), 1)),
                np.tile([0., 1.], (len(heldout), 1)), {"synthetic_test": True})

    def proposal(self, lexical=False):
        return propose(self.inputs, self.review, lexical_only=lexical, sample_size=2, sample_seed=5,
                       **({} if lexical else {"heldout_threshold": .9, "dedup_threshold": .95,
                                              "embedding_options": {}, "encoder": self.fake_encoder}))

    def complete_answers(self):
        flags = [json.loads(x) for x in (self.review / "review_flags.jsonl").read_text().splitlines()]
        answer = [json.loads(x) for x in (self.review / "review_answers.template.jsonl").read_text().splitlines()]
        types = {x["review_id"]: x["kind"] for x in flags}
        for row in answer:
            row.update(decision="equivalent" if types[row["review_id"]] == "semantic_duplicate" else "keep",
                       reviewer="test reviewer", reason="Synthetic fixture decision only")
        path = self.root / "answers.jsonl"
        path.write_text("".join(json.dumps(x) + "\n" for x in answer))
        return path

    def test_full_round_trip_and_no_overwrite(self):
        self.export()
        self.proposal()
        answer = self.complete_answers()
        result = finalize(self.review, answer, self.root / "frozen")
        self.assertEqual(result["stage"], "frozen")
        self.assertGreater(result["eligible_count"], 0)
        self.assertLess(result["eligible_count"], result["candidate_count"])
        pool = [json.loads(x) for x in (self.root / "frozen/eligible_anchors.jsonl").read_text().splitlines()]
        self.assertTrue(all(set(x) == {"example_id", "prompt"} for x in pool))
        for operation in (self.export, self.proposal, lambda: finalize(self.review, answer, self.root / "frozen")):
            with self.assertRaises(FileExistsError):
                operation()

    @staticmethod
    def preview_encoder(candidates, heldout, **kwargs):
        vectors = np.zeros((len(candidates), len(candidates) + 2))
        for index, row in enumerate(candidates):
            dimension = (1 if row["prompt"].startswith("equivalent request") else
                         0 if row["prompt"] == "reference paraphrase candidate" else index + 2)
            vectors[index, dimension] = 1
        references = np.zeros((len(heldout), vectors.shape[1]))
        references[:, 0] = 1
        return vectors, references, {"synthetic_test": True}

    def export_preview_fixture(self):
        self.rows["anchors"].extend({"instruction": prompt, "context": "", "response": "ignored"}
                                   for prompt in ("equivalent request left", "equivalent request right",
                                                  "reference paraphrase candidate", "independent writing task"))
        self.export()

    def preview_proposal(self, enabled=True):
        return propose(self.inputs, self.review, lexical_only=False, sample_size=1, sample_seed=5,
                       heldout_threshold=.9, dedup_threshold=.95, encoder=self.preview_encoder,
                       export_auto_included=enabled)

    def test_preview_omits_hard_overlap_both_duplicate_members_and_spot_checks(self):
        self.export_preview_fixture()
        manifest = self.preview_proposal()
        with (self.review / "auto_included_anchors.jsonl").open() as stream:
            preview = [json.loads(line) for line in stream]
        with (self.review / "review_flags.jsonl").open() as stream:
            flags = [json.loads(line) for line in stream]
        self.assertEqual({row["kind"] for row in flags},
                         {"benchmark_overlap", "semantic_duplicate", "spot_check"})
        self.assertGreater(len(preview), 0)
        self.assertEqual(len(preview), manifest["auto_included_preview"]["count"])
        prompts = {row["prompt"] for row in preview}
        for excluded in ("test only question", "equivalent request left", "equivalent request right",
                         "reference paraphrase candidate"):
            self.assertNotIn(excluded, prompts)
        self.assertNotIn(next(row["candidate_prompt"] for row in flags if row["kind"] == "spot_check"),
                         prompts)
        self.assertTrue(all(set(row) == {"example_id", "prompt"} for row in preview))
        with self.assertRaisesRegex(ValueError, "pending"):
            finalize(self.review, self.review / "review_answers.template.jsonl", self.root / "premature")

    def test_optional_preview_does_not_change_review_flags_or_final_pool(self):
        self.export_preview_fixture()
        pools, flags = [], []
        for enabled in (False, True):
            self.review = self.root / f"review-{enabled}"
            self.preview_proposal(enabled)
            self.assertEqual((self.review / "auto_included_anchors.jsonl").exists(), enabled)
            flags.append((self.review / "review_flags.jsonl").read_bytes())
            target = self.root / f"frozen-{enabled}"
            finalize(self.review, self.complete_answers(), target)
            pools.append((target / "eligible_anchors.jsonl").read_bytes())
        self.assertEqual(flags[0], flags[1])
        self.assertEqual(pools[0], pools[1])

    def test_preview_checksum_tampering_is_rejected(self):
        self.export_preview_fixture()
        self.preview_proposal()
        answers = self.complete_answers()
        with (self.review / "auto_included_anchors.jsonl").open("a") as stream:
            stream.write('{"example_id":"new","prompt":"tampered"}\n')
        with self.assertRaisesRegex(ValueError, "checksum"):
            finalize(self.review, answers, self.root / "frozen")

    def test_lexical_only_cannot_export_auto_included_preview(self):
        self.export()
        with self.assertRaisesRegex(ValueError, "complete semantic screening"):
            propose(self.inputs, self.review, lexical_only=True, sample_size=0, sample_seed=5,
                    export_auto_included=True)
        self.assertFalse(self.review.exists())

    def test_lexical_preview_cannot_be_used_as_frozen_pool(self):
        self.export()
        self.proposal(lexical=True)
        answer = self.complete_answers()
        with self.assertRaisesRegex(ValueError, "lexical-only"):
            finalize(self.review, answer, self.root / "frozen")
        self.assertFalse((self.root / "frozen").exists())

    def test_input_tampering_is_rejected_before_encoder(self):
        self.export()
        with (self.inputs / "candidates.jsonl").open("a") as out:
            out.write('{"example_id":"new","prompt":"new"}\n')
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.proposal()

    def test_flags_and_manifest_tampering_are_rejected(self):
        self.export()
        self.proposal()
        answer = self.complete_answers()
        flags = self.review / "review_flags.jsonl"
        old = flags.read_bytes()
        flags.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "checksum"):
            finalize(self.review, answer, self.root / "frozen")
        flags.write_bytes(old)
        manifest_path = self.review / "review_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["heldout_threshold"] = .1
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "checksum"):
            finalize(self.review, answer, self.root / "frozen")

    def test_no_gold_labels_exported_or_used_to_change_inputs(self):
        first = input_rows(self.config, self.rows, target_train_overlap="allow")
        altered = copy.deepcopy(self.rows)
        altered["gsm8k_test"][0]["answer"] = "DIFFERENT_SECRET"
        altered["boolq"][0]["answer"] = False
        altered["hellaswag"][0]["label"] = "1"
        altered["arc_easy"][0]["answerKey"] = "B"
        second = input_rows(self.config, altered, target_train_overlap="allow")
        self.assertEqual(first, second)
        self.assertNotIn("secret test answer", json.dumps(first))
        self.assertNotIn("ignored", json.dumps(first))

    def test_target_training_overlap_is_an_explicit_separate_choice(self):
        _, allow = input_rows(self.config, self.rows, target_train_overlap="allow")
        _, exclude = input_rows(self.config, self.rows, target_train_overlap="exclude")
        self.assertEqual(len(exclude) - len(allow), 2)
        self.assertNotIn("gsm8k_train", {x["dataset"] for x in allow})
        self.assertIn("gsm8k_train", {x["dataset"] for x in exclude})
        for tag in ("gsm8k_test", "gsm8k_validation", "boolq", "hellaswag", "arc_easy"):
            self.assertIn(tag, {x["dataset"] for x in allow})

    def test_pending_answers_and_stale_answers_are_rejected(self):
        self.export()
        self.proposal()
        with self.assertRaisesRegex(ValueError, "pending"):
            finalize(self.review, self.review / "review_answers.template.jsonl", self.root / "frozen")
        answer = self.complete_answers()
        rows = [json.loads(x) for x in answer.read_text().splitlines()]
        rows[0]["bundle_sha256"] = "old"
        answer.write_text("".join(json.dumps(x) + "\n" for x in rows))
        with self.assertRaisesRegex(ValueError, "stale"):
            finalize(self.review, answer, self.root / "frozen")


if __name__ == "__main__":
    unittest.main()
