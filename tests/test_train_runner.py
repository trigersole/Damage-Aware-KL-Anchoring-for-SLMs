"""CPU smoke for the real runner using a tiny model and local manifests."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from kl_anchors.anchor_pipeline import file_sha256
from kl_anchors.train import run
from test_protocol import complete_protocol


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(item["content"] for item in messages) + " Answer:"

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(char) % 30) for char in text]

    def decode(self, ids, skip_special_tokens=True):
        return "1"

    def save_pretrained(self, path):
        (Path(path) / "tokenizer.json").write_text("{}")


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32, 8)
        self.head = torch.nn.Linear(8, 32)

    def forward(self, input_ids, attention_mask):
        from types import SimpleNamespace
        return SimpleNamespace(logits=self.head(self.emb(input_ids)))

    def save_pretrained(self, path):
        (Path(path) / "adapter_config.json").write_text("{}")


class RunnerTests(unittest.TestCase):
    def _base_config(self, root):
        protocol = complete_protocol(str(root / "runs"))
        protocol["training"].update(max_steps=1, warmup_steps=0, per_device_batch_size=1,
                                     gradient_accumulation_steps=1, scheduler="constant")
        protocol["runtime"].update(device="cpu", max_sequence_tokens=256, checkpoint_every_steps=1)
        train_path = root / "train.jsonl"
        val_path = root / "val.jsonl"
        train_path.write_text(json.dumps({"example_id": "t1", "prompt": "1+1?", "response": "#### 2"}) + "\n")
        val_path.write_text(json.dumps({"example_id": "v1", "prompt": "2+2?", "response": "#### 4"}) + "\n")
        protocol["training"]["training_ids_manifest"] = str(train_path)
        protocol["datasets"]["gsm8k"]["validation_ids_manifest"] = str(val_path)
        return protocol

    def test_task_only_saves_checkpoints_best_plot_and_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = complete_protocol(str(root / "runs"))
            protocol["training"].update(max_steps=2, warmup_steps=0, per_device_batch_size=1,
                                         gradient_accumulation_steps=1, scheduler="constant")
            protocol["runtime"].update(device="cpu", max_sequence_tokens=256, checkpoint_every_steps=1)
            train_path = root / "train.jsonl"
            val_path = root / "val.jsonl"
            train_path.write_text(json.dumps({"example_id": "t1", "prompt": "1+1?", "response": "#### 2"}) + "\n")
            val_path.write_text(json.dumps({"example_id": "v1", "prompt": "2+2?", "response": "#### 4"}) + "\n")
            protocol["training"]["training_ids_manifest"] = str(train_path)
            protocol["datasets"]["gsm8k"]["validation_ids_manifest"] = str(val_path)
            config = root / "protocol.json"
            config.write_text(json.dumps(protocol))
            with patch("kl_anchors.train.load_tokenizer", return_value=TinyTokenizer()), \
                 patch("kl_anchors.train.load_base", side_effect=lambda *args, **kwargs: TinyModel()), \
                 patch("kl_anchors.train.attach_new_lora", side_effect=lambda model, config: model):
                run_dir = run(config, mode="lora", experiment_id="toy-lora", condition="final",
                              damage_weight=None, best_metric="min_validation_loss")
            self.assertTrue((run_dir / "checkpoints" / "step_000001" / "adapter_config.json").exists())
            self.assertTrue((run_dir / "checkpoints" / "step_000002" / "adapter_config.json").exists())
            self.assertTrue((run_dir / "best_model" / "adapter_config.json").exists())
            self.assertTrue((run_dir / "loss_vs_epoch.png").exists())
            self.assertEqual(json.loads((run_dir / "status.json").read_text())["status"], "complete")

    def test_random_anchor_mode_uses_teacher_and_exact_token_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = self._base_config(root)
            protocol["anchor_selection"]["token_budget"] = 2
            protocol["runtime"]["anchor_tokens_per_update"] = 1
            cache = root / "cache.jsonl"
            pool = root / "pool.jsonl"
            eligible = root / "eligible.jsonl"
            eligible.write_text(json.dumps({"example_id": "a", "prompt": "general prompt"}) + "\n")
            protocol["datasets"]["anchors"]["eligible_ids_manifest"] = str(eligible)
            cache.write_text(json.dumps({"example_id": "a", "prompt_token_ids": [2, 3],
                                         "continuation_token_ids": [4, 5], "response_token_count": 2}) + "\n")
            pool.write_text(json.dumps({"example_id": "a", "response_token_count": 2}) + "\n")
            cache.with_suffix(cache.suffix + ".meta.json").write_text(json.dumps({
                "model": protocol["model"], "cache_sha256": file_sha256(cache),
                "source_sha256": file_sha256(eligible),
            }))
            pool.with_suffix(pool.suffix + ".meta.json").write_text(json.dumps({
                "pool_sha256": file_sha256(pool), "cache_sha256": file_sha256(cache),
            }))
            protocol["kl"]["teacher_continuation"] = str(cache)
            protocol["anchor_selection"]["candidate_pool_manifest"] = str(pool)
            config = root / "protocol.json"
            config.write_text(json.dumps(protocol))
            with patch("kl_anchors.train.load_tokenizer", return_value=TinyTokenizer()), \
                 patch("kl_anchors.train.load_base", side_effect=lambda *args, **kwargs: TinyModel()), \
                 patch("kl_anchors.train.attach_new_lora", side_effect=lambda model, config: model):
                run_dir = run(config, mode="random_kl", experiment_id="toy-random", condition="final",
                              damage_weight=None, best_metric="min_validation_loss")
            status = json.loads((run_dir / "status.json").read_text())
            self.assertEqual(status["anchor_tokens_seen"], 1)
            self.assertEqual(status["selection_token_budget"], 2)
            self.assertEqual(json.loads((run_dir / "anchor_selection.json").read_text())["method"], "random")


if __name__ == "__main__":
    unittest.main()
