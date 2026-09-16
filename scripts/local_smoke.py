"""Run a tiny, offline training smoke test and keep its artifacts on disk.

This script deliberately substitutes a synthetic PyTorch model and tokenizer
for Qwen and PEFT. It checks the real training runner, method switch, anchor
selection, checkpoint writing, best-model selection, and loss plot on CPU.
Its outputs are demonstrations, never research results.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
from uuid import uuid4

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kl_anchors.anchor_pipeline import file_sha256  # noqa: E402
from kl_anchors.provenance import evaluation_code_digest  # noqa: E402
from kl_anchors.train import MODE_TO_METHOD, run  # noqa: E402


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(item["content"] for item in messages) + " Answer:"

    def encode(self, content, add_special_tokens=False):
        return [2 + (ord(char) % 30) for char in content]

    def save_pretrained(self, path):
        (Path(path) / "tokenizer.json").write_text('{"smoke_only": true}\n', encoding="utf-8")


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32, 8)
        self.head = torch.nn.Linear(8, 32)

    def forward(self, input_ids, attention_mask):
        from types import SimpleNamespace

        return SimpleNamespace(logits=self.head(self.emb(input_ids)))

    def save_pretrained(self, path):
        (Path(path) / "adapter_config.json").write_text(
            '{"smoke_only": true, "not_a_real_lora_adapter": true}\n', encoding="utf-8"
        )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _make_inputs(session: Path, steps: int) -> Path:
    data_dir = session / "data"
    data_dir.mkdir(parents=True)
    train = data_dir / "train.jsonl"
    validation = data_dir / "validation.jsonl"
    eligible = data_dir / "eligible_anchors.jsonl"
    cache = data_dir / "teacher_cache.jsonl"
    pool = data_dir / "scored_pool.jsonl"
    exclusions = data_dir / "exclusions.jsonl"

    _write_jsonl(train, [
        {"example_id": "toy-train-1", "prompt": "1 + 1?", "response": "#### 2"},
        {"example_id": "toy-train-2", "prompt": "2 + 3?", "response": "#### 5"},
    ])
    _write_jsonl(validation, [
        {"example_id": "toy-validation-1", "prompt": "3 + 4?", "response": "#### 7"},
    ])
    anchors = [
        {"example_id": "toy-anchor-a", "prompt": "Write a greeting."},
        {"example_id": "toy-anchor-b", "prompt": "Name a color."},
        {"example_id": "toy-anchor-c", "prompt": "Describe a tree."},
    ]
    _write_jsonl(eligible, anchors)
    _write_jsonl(exclusions, [])
    _write_jsonl(cache, [
        {"example_id": row["example_id"], "prompt_token_ids": [2, 3],
         "continuation_token_ids": [4 + i, 7 + i], "response_token_count": 2}
        for i, row in enumerate(anchors)
    ])
    _write_jsonl(pool, [
        {"example_id": row["example_id"], "response_token_count": 2,
         "vulnerability_score": float(3 - i), "diversity_rank": i}
        for i, row in enumerate(anchors)
    ])

    protocol = json.loads((PROJECT_ROOT / "configs" / "protocol.template.json").read_text(encoding="utf-8"))
    protocol.update(condition="final", experiment_id="smoke-placeholder",
                    output_directory=str(session / "runs"), seed=0)
    protocol["model"].update(repository_id="smoke/TinyModel", revision="0" * 40)
    for name, dataset in protocol["datasets"].items():
        dataset.update(repository_id=f"smoke/{name}", revision="0" * 40,
                       configuration="synthetic", split="synthetic",
                       preprocessing="synthetic local smoke inputs; no hosted dataset")
    protocol["datasets"]["anchors"].update(
        repository_id="smoke/local-anchors", configuration="synthetic", split="synthetic",
        eligible_ids_manifest=str(eligible), exclusions_manifest=str(exclusions),
    )
    protocol["datasets"]["gsm8k"].update(
        repository_id="smoke/local-target", configuration="synthetic", split="synthetic",
        validation_ids_manifest=str(validation), test_ids_manifest=str(validation),
    )
    protocol["lora"].update(rank=1, alpha=1, dropout=0, target_modules=["head"],
                            precision="fp32", quantization="none")
    protocol["training"].update(
        training_ids_manifest=str(train), optimizer="adamw", learning_rate=0.001,
        scheduler="constant", warmup_steps=0, batch_policy="shuffled_repeat",
        per_device_batch_size=1, gradient_accumulation_steps=1, max_steps=steps,
        epochs=steps / 2, target_examples=2, target_tokens=20,
        anchor_interleave_every_steps=1,
    )
    protocol["runtime"].update(
        device="cpu", max_sequence_tokens=128, validation_max_new_tokens=4,
        target_system_prompt="", anchor_system_prompt="", teacher_max_new_tokens=2,
        anchor_tokens_per_update=1, checkpoint_every_steps=1, max_grad_norm=1.0,
        scale_kl_by_temperature_squared=True, target_answer_append_eos=True,
    )
    protocol["anchor_selection"].update(
        method="random", token_budget=4, damage_weight=None,
        candidate_pool_manifest=str(pool), selected_ids_manifest="set by runner",
    )
    protocol["kl"].update(
        direction="teacher_to_student", temperature=1.0, coefficient=0.1,
        token_mask="response_tokens", normalization="token_mean", teacher_continuation=str(cache),
    )
    evaluation = protocol["evaluation"]
    evaluation.update(
        harness="internal_kl_anchors_v1", harness_revision=evaluation_code_digest(),
        chat_template="synthetic TinyTokenizer", model_dtype="fp32", model_quantization="none",
        trust_remote_code=False, few_shot=0, checkpoint_selection_rule="min_validation_loss",
    )
    evaluation["tokenizer"].update(repo_id="smoke/TinyModel", revision="0" * 40,
                                   use_fast=True, add_special_tokens=False)
    evaluation["generation"].update(max_new_tokens=4, do_sample=False, num_beams=1,
                                    skip_special_tokens=True)
    for benchmark in ("gsm8k", "boolq", "hellaswag", "arc_easy"):
        evaluation["prompt_templates"][benchmark] = (
            "Question: {question}\nAnswer:" if benchmark == "gsm8k"
            else f"synthetic {benchmark} prompt"
        )
        if benchmark != "gsm8k":
            evaluation["scoring"][benchmark].update(length_normalization=False, choice_prefix=" ")
    evaluation["scoring"]["boolq"]["choice_texts"] = ["No", "Yes"]
    protocol["environment"].update(
        python_version=sys.version.split()[0], software_versions=f"torch={torch.__version__}",
        hardware="local CPU; synthetic smoke only",
    )
    cache.with_suffix(cache.suffix + ".meta.json").write_text(json.dumps({
        "model": protocol["model"], "cache_sha256": file_sha256(cache),
        "source_sha256": file_sha256(eligible),
    }, indent=2) + "\n", encoding="utf-8")
    pool.with_suffix(pool.suffix + ".meta.json").write_text(json.dumps({
        "pool_sha256": file_sha256(pool), "cache_sha256": file_sha256(cache),
    }, indent=2) + "\n", encoding="utf-8")
    path = session / "protocol.smoke.json"
    path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    (session / "SMOKE_ONLY.md").write_text(
        "# Synthetic local smoke test\n\n"
        "These inputs and model are synthetic. The script substitutes a tiny PyTorch model "
        "for Qwen and does not exercise PEFT LoRA or a Hugging Face model. "
        "The saved adapter files are stubs, not reloadable model weights. "
        "Its losses and selected anchors are not research results.\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(MODE_TO_METHOD), default="lora")
    parser.add_argument("--all-modes", action="store_true", help="Run the five method flags in one smoke session")
    parser.add_argument("--steps", type=int, default=2, help="Tiny optimizer steps per mode (default: 2)")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "runs" / "local-smoke")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")

    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    session = args.output_dir.expanduser().resolve() / session_id
    session.mkdir(parents=True, exist_ok=False)
    # Keep Matplotlib and Fontconfig caches inside this ignored smoke session.
    for variable, name in (("MPLCONFIGDIR", "matplotlib-cache"), ("XDG_CACHE_HOME", "xdg-cache")):
        cache_dir = session / name
        cache_dir.mkdir()
        os.environ.setdefault(variable, str(cache_dir))
    protocol_path = _make_inputs(session, args.steps)
    torch.manual_seed(0)
    base_state = {name: tensor.detach().clone() for name, tensor in TinyModel().state_dict().items()}

    def load_tiny_base(*_args, **_kwargs):
        model = TinyModel()
        model.load_state_dict(base_state)
        return model

    modes = tuple(MODE_TO_METHOD) if args.all_modes else (args.mode,)
    print("SYNTHETIC SMOKE ONLY: no Qwen weights, real LoRA adapter, GPU, or benchmark data.")
    print(f"Session: {session}")
    with patch("kl_anchors.train.load_tokenizer", return_value=TinyTokenizer()), \
         patch("kl_anchors.train.load_base", side_effect=load_tiny_base), \
         patch("kl_anchors.train.attach_new_lora", side_effect=lambda model, config: model):
        for mode in modes:
            run_dir = run(
                protocol_path, mode=mode, experiment_id=f"toy-{mode}", condition="final",
                damage_weight=0.5 if mode == "damage_diverse_kl" else None,
                best_metric="min_validation_loss",
            )
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            if status["status"] != "complete" or not (run_dir / "best_model").is_dir() \
                    or not (run_dir / "loss_vs_epoch.png").is_file():
                raise RuntimeError(f"smoke artifacts missing for {mode}: {run_dir}")
            print(f"{mode}: {run_dir}")
            print(f"  steps={status['training_steps']} checkpoints={len(list((run_dir / 'checkpoints').iterdir()))} "
                  f"best={Path(status['best_checkpoint']).name} plot=loss_vs_epoch.png")


if __name__ == "__main__":
    main()
