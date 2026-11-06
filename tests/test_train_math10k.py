"""CPU-only checks for the Math10K diversity-KL protocol wiring."""

import json
from pathlib import Path
import tempfile
import unittest

from kl_anchors.anchor_pipeline import file_sha256
from kl_anchors.train_math10k import prepare_diverse_anchors, prune_checkpoints, validate_mode


class Math10KDiverseKLTests(unittest.TestCase):
    def test_retention_preserves_best_latest_and_noncheckpoint_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['step_000300', 'step_000600', 'step_000900', 'notes']:
                folder = root / 'checkpoints' / name
                folder.mkdir(parents=True)
                (folder / 'marker').write_text(name)
            best = root / 'checkpoints' / 'step_000300'
            removed = prune_checkpoints(root, best, 1)
            self.assertEqual(removed, [str(root / 'checkpoints' / 'step_000600')])
            self.assertTrue((best / 'marker').exists())
            self.assertTrue((root / 'checkpoints' / 'step_000900' / 'marker').exists())
            self.assertTrue((root / 'checkpoints' / 'notes' / 'marker').exists())
            self.assertTrue((root / 'checkpoint_cleanup.jsonl').is_file())

    def test_retention_disabled_leaves_checkpoints_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'checkpoints' / 'step_000300'
            checkpoint.mkdir(parents=True)
            self.assertEqual(prune_checkpoints(root, checkpoint, 0), [])
            self.assertTrue(checkpoint.exists())

    def make_protocol(self, root: Path):
        eligible = root / "eligible.jsonl"
        cache = root / "cache.jsonl"
        ranks = root / "ranks.jsonl"
        ids = ("a", "b", "c")
        eligible.write_text("".join(
            json.dumps({"example_id": item_id, "prompt": item_id.upper()}) + "\n"
            for item_id in ids
        ), encoding="utf-8")
        cache.write_text("".join(json.dumps({
            "example_id": item_id,
            "prompt_token_ids": [1, 2],
            "continuation_token_ids": [3, 4],
            "response_token_count": 2,
        }) + "\n" for item_id in ids), encoding="utf-8")
        model = {"repository_id": "example/model", "revision": "a" * 40}
        cache.with_suffix(".jsonl.meta.json").write_text(json.dumps({
            "model": model,
            "source_sha256": file_sha256(eligible),
            "cache_sha256": file_sha256(cache),
        }), encoding="utf-8")
        metadata = {
            "model_id": "example/embed", "model_revision": "b" * 40,
            "algorithm": "farthest_first", "seed": 42,
        }
        ranks.write_text("".join(json.dumps({
            "example_id": item_id, "prompt": item_id.upper(),
            "response_token_count": 2, "diversity_rank": rank,
            "diversity_metadata": metadata,
        }) + "\n" for rank, item_id in enumerate(ids)), encoding="utf-8")
        return {
            "model": model,
            "datasets": {"anchors": {"eligible_ids_manifest": str(eligible)}},
            "seed": 42,
            "training": {"anchor_interleave_every_steps": 4},
            "runtime": {
                "max_sequence_tokens": 16, "anchor_tokens_per_update": 2,
                "anchor_microbatch_size": 1,
            },
            "anchor_selection": {
                "method": "diversity", "prompt_budget": 2, "token_budget": 3,
                "candidate_pool_manifest": str(ranks),
                "diversity": metadata,
            },
            "kl": {
                "teacher_continuation": str(cache), "coefficient": 0.01,
                "direction": "teacher_to_student", "temperature": 1,
                "token_mask": "response_tokens", "normalization": "token_mean",
            },
        }

    def test_exact_prompt_and_position_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = self.make_protocol(Path(directory))
            validate_mode(protocol, "diverse_kl")
            selection, schedule = prepare_diverse_anchors(protocol, tokenizer=None)
        self.assertEqual(selection["selected_prompt_count"], 2)
        self.assertEqual(selection["anchor_token_budget"], 3)
        self.assertEqual(schedule.selection_token_budget, 3)
        self.assertEqual(sum(sum(row.response_mask) for row in schedule.next_batch()), 2)

    def test_rejects_zero_kl_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = self.make_protocol(Path(directory))
            protocol["kl"]["coefficient"] = 0
            with self.assertRaisesRegex(ValueError, "positive KL"):
                validate_mode(protocol, "diverse_kl")


if __name__ == "__main__":
    unittest.main()
