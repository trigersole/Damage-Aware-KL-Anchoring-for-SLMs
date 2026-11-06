import copy
import unittest

from kl_anchors.anchor_pipeline import require_anchor_protocol, validate_anchor_protocol


def complete_anchor_protocol():
    return {
        "model": {"repository_id": "example/model", "revision": "a" * 40},
        "datasets": {"anchors": {"eligible_ids_manifest": "anchors.jsonl"}},
        "seed": 42,
        "lora": {"precision": "bf16", "quantization": "none"},
        "runtime": {
            "device": "cuda:0",
            "max_sequence_tokens": 2048,
            "anchor_system_prompt": "",
            "teacher_max_new_tokens": 64,
            "scale_kl_by_temperature_squared": True,
        },
        "kl": {"direction": "teacher_to_student", "temperature": 1.0},
    }


class AnchorProtocolTests(unittest.TestCase):
    def test_complete_anchor_protocol_is_valid(self):
        protocol = complete_anchor_protocol()
        self.assertEqual(validate_anchor_protocol(protocol), [])
        require_anchor_protocol(protocol)

    def test_scoring_choices_are_explicit(self):
        protocol = copy.deepcopy(complete_anchor_protocol())
        protocol["model"]["revision"] = "main"
        protocol["kl"]["direction"] = "student_to_teacher"
        protocol["kl"]["temperature"] = 0
        errors = validate_anchor_protocol(protocol)
        self.assertIn("model.revision must be a full 40-character commit SHA", errors)
        self.assertIn(
            "kl.direction must be teacher_to_student for vulnerability scoring",
            errors,
        )
        self.assertIn("kl.temperature must be a positive number", errors)


if __name__ == "__main__":
    unittest.main()
