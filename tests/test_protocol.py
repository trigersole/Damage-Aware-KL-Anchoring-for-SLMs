"""Tests for final-run completeness and exclusive experiment reservation."""

from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kl_anchors.protocol import load_protocol, reserve_run, validate_protocol  # noqa: E402
from kl_anchors.provenance import evaluation_code_digest  # noqa: E402


TEMPLATE = Path(__file__).resolve().parents[1] / "configs" / "protocol.template.json"


def complete_protocol(output_directory: str) -> dict:
    protocol = load_protocol(TEMPLATE)
    protocol["condition"] = "final"
    protocol["experiment_id"] = "seed0-vulnerability"
    protocol["output_directory"] = output_directory
    protocol["model"]["revision"] = "a" * 40
    for dataset in protocol["datasets"].values():
        dataset["revision"] = "b" * 40
        dataset["preprocessing"] = "frozen preprocessing specification v1"
    protocol["datasets"]["anchors"].update(
        repository_id="example/instruction-prompts", configuration="default", split="train"
    )
    protocol["datasets"]["gsm8k"]["validation_ids_manifest"] = "manifests/gsm8k-validation.json"
    protocol["datasets"]["gsm8k"]["test_ids_manifest"] = "manifests/gsm8k-test.json"
    protocol["datasets"]["anchors"]["eligible_ids_manifest"] = "manifests/eligible.json"
    protocol["datasets"]["anchors"]["exclusions_manifest"] = "manifests/exclusions.json"
    protocol["seed"] = 0
    protocol["lora"].update(rank=8, alpha=16, dropout=0.05, target_modules=["q_proj", "v_proj"], precision="bf16", quantization="none")
    protocol["training"].update(
        training_ids_manifest="manifests/gsm8k-training.json",
        optimizer="adamw", learning_rate=2e-4, scheduler="linear", warmup_steps=0, batch_policy="shuffled_repeat",
        per_device_batch_size=2, gradient_accumulation_steps=8, max_steps=100,
        epochs=1.0, target_examples=1000, target_tokens=50000,
        anchor_interleave_every_steps=1,
    )
    protocol["runtime"].update(
        device="cuda:0", max_sequence_tokens=1024,
        validation_max_new_tokens=256,
        target_system_prompt="", anchor_system_prompt="",
        teacher_max_new_tokens=32, anchor_tokens_per_update=128,
        checkpoint_every_steps=20, max_grad_norm=1.0,
        scale_kl_by_temperature_squared=True, target_answer_append_eos=True,
    )
    protocol["anchor_selection"].update(
        method="vulnerability", token_budget=4096,
        candidate_pool_manifest="manifests/candidates.json",
        selected_ids_manifest="manifests/selected.json",
    )
    protocol["kl"].update(
        direction="teacher_to_student", temperature=1.0, coefficient=0.1,
        token_mask="response_tokens", normalization="token_mean",
        teacher_continuation="greedy, 32 tokens, fixed EOS policy",
    )
    protocol["evaluation"].update(
        harness="internal_kl_anchors_v1", harness_revision=evaluation_code_digest(),
        chat_template="model tokenizer chat template at pinned revision",
        model_dtype="bf16", model_quantization="none", trust_remote_code=False,
        few_shot=0, checkpoint_selection_rule="max_gsm8k_exact_match",
    )
    protocol["evaluation"]["tokenizer"].update(repo_id=protocol["model"]["repository_id"], revision=protocol["model"]["revision"], use_fast=True, add_special_tokens=False)
    protocol["evaluation"]["generation"].update(max_new_tokens=256, do_sample=False, num_beams=1, skip_special_tokens=True)
    for benchmark in ("gsm8k", "boolq", "hellaswag", "arc_easy"):
        protocol["evaluation"]["prompt_templates"][benchmark] = (
            "Question: {question}\nAnswer:" if benchmark == "gsm8k" else f"{benchmark} prompt format"
        )
        if benchmark != "gsm8k":
            protocol["evaluation"]["scoring"][benchmark].update(length_normalization=False, choice_prefix=" ")
    protocol["evaluation"]["scoring"]["boolq"]["choice_texts"] = ["No", "Yes"]
    protocol["environment"].update(
        python_version="3.11.8", software_versions="torch=2.5.1; transformers=4.46.0",
        hardware="NVIDIA A100 40GB; CUDA 12.4",
    )
    return protocol


class ProtocolTests(unittest.TestCase):
    def test_template_keeps_open_choices_unset_and_blocks_final_run(self):
        template = load_protocol(TEMPLATE)
        self.assertEqual(template["model"]["repository_id"], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertIsNone(template["model"]["revision"])
        self.assertIsNone(template["datasets"]["anchors"]["repository_id"])
        errors = validate_protocol(template)
        self.assertTrue(any("model.revision" in error for error in errors))
        self.assertTrue(any("condition" in error for error in errors))
        with self.assertRaisesRegex(ValueError, "final protocol is incomplete"):
            reserve_run(template)

    def test_complete_protocol_reserves_snapshot_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = complete_protocol(directory)
            self.assertEqual(validate_protocol(protocol), [])
            run_dir = reserve_run(protocol)
            self.assertEqual(load_protocol(run_dir / "protocol.json"), protocol)
            with self.assertRaises(FileExistsError):
                reserve_run(protocol)
            self.assertEqual(load_protocol(run_dir / "protocol.json"), protocol)

    def test_mutable_revision_and_unsafe_experiment_id_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = complete_protocol(directory)
            protocol["model"]["revision"] = "main"
            protocol["experiment_id"] = "../old-run"
            errors = validate_protocol(protocol)
            self.assertTrue(any("model.revision" in error for error in errors))
            self.assertTrue(any("experiment_id" in error for error in errors))

    def test_no_anchor_baseline_records_zero_budget_and_coefficient(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = complete_protocol(directory)
            protocol["anchor_selection"].update(method="none", token_budget=0, selected_ids_manifest=None)
            protocol["kl"]["coefficient"] = 0.0
            protocol["training"]["anchor_interleave_every_steps"] = 0
            protocol["runtime"]["anchor_tokens_per_update"] = 0
            self.assertEqual(validate_protocol(protocol), [])
            protocol["anchor_selection"]["token_budget"] = 1
            self.assertTrue(any("method none requires zero anchor tokens" in error for error in validate_protocol(protocol)))

    def test_combined_damage_diverse_method_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = complete_protocol(directory)
            protocol["anchor_selection"]["method"] = "damage_diverse"
            protocol["anchor_selection"]["damage_weight"] = 0.5
            self.assertEqual(validate_protocol(protocol), [])
            protocol["anchor_selection"]["damage_weight"] = None
            self.assertTrue(any("damage_weight" in error for error in validate_protocol(protocol)))

    def test_base_record_needs_benchmark_protocol_but_no_training(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = complete_protocol(directory)
            protocol["condition"] = "base"
            protocol["evaluation"]["checkpoint_selection_rule"] = "untouched"
            protocol["lora"] = {}
            protocol["training"] = {}
            protocol["kl"] = {}
            protocol["anchor_selection"] = {}
            protocol["datasets"]["anchors"] = {}
            self.assertEqual(validate_protocol(protocol), [])
            self.assertTrue((reserve_run(protocol) / "protocol.json").exists())

    def test_discovery_is_distinct_and_task_only(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = complete_protocol(directory)
            protocol["condition"] = "discovery"
            self.assertTrue(any("discovery condition requires method none" in error for error in validate_protocol(protocol)))
            protocol["anchor_selection"].update(method="none", token_budget=0, selected_ids_manifest=None)
            protocol["training"]["anchor_interleave_every_steps"] = 0
            protocol["runtime"]["anchor_tokens_per_update"] = 0
            protocol["kl"]["coefficient"] = 0
            self.assertEqual(validate_protocol(protocol), [])


if __name__ == "__main__":
    unittest.main()
