import unittest
from types import SimpleNamespace

import torch

from kl_anchors.training_core import (
    AnchorTokenScheduler, TokenizedExample, collate, target_causal_loss,
    tokenize_completion, training_step,
)


class DummyTokenizer:
    eos_token_id = 9

    def encode(self, text, add_special_tokens=False):
        return [ord(x) % 9 for x in text]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(10, 8)
        self.head = torch.nn.Linear(8, 10)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.head(self.emb(input_ids)))


class TrainingCoreTests(unittest.TestCase):
    def test_tokenize_refuses_silent_truncation(self):
        with self.assertRaises(ValueError):
            tokenize_completion(DummyTokenizer(), example_id="x", rendered_prompt="abc", response="def", max_sequence_tokens=5, append_eos=False)

    def test_exact_anchor_exposure(self):
        examples = {
            "a": TokenizedExample("a", (1, 2, 3, 4, 5), (False, False, True, True, True)),
            "b": TokenizedExample("b", (1, 2, 3, 4), (False, False, True, True)),
        }
        schedule = AnchorTokenScheduler(examples, [
            {"example_id": "a", "selected_response_tokens": 3},
            {"example_id": "b", "selected_response_tokens": 2},
        ], tokens_per_update=2)
        for _ in range(7):
            batch = schedule.next_batch()
            self.assertEqual(sum(sum(ex.response_mask) for ex in batch), 2)
        self.assertEqual(schedule.selection_token_budget, 5)
        self.assertEqual(schedule.total_emitted, 14)

    def test_one_step_updates_student_only(self):
        torch.manual_seed(3)
        student, teacher = TinyModel(), TinyModel()
        for p in teacher.parameters():
            p.requires_grad_(False)
        target = collate([TokenizedExample("g", (1, 2, 3), (False, True, True))], pad_token_id=0, device="cpu")
        anchor = collate([TokenizedExample("a", (1, 4, 5), (False, True, True))], pad_token_id=0, device="cpu")
        before_student = student.head.weight.detach().clone()
        before_teacher = teacher.head.weight.detach().clone()
        output = training_step(
            student=student, teacher=teacher, target_batch=target,
            anchor_batch=anchor, optimizer=torch.optim.AdamW(student.parameters(), lr=0.01),
            kl_weight=0.5, kl_direction="teacher_to_student", kl_temperature=1.0,
            kl_reduction="token_mean", scale_by_temperature_squared=False,
        )
        self.assertFalse(torch.equal(before_student, student.head.weight))
        self.assertTrue(torch.equal(before_teacher, teacher.head.weight))
        self.assertEqual(output["target_tokens"], 2)
        self.assertEqual(output["anchor_tokens"], 2)

    def test_target_loss_masks_prompt(self):
        logits = torch.zeros(1, 3, 10)
        logits[0, 0, 2] = 20
        logits[0, 1, 3] = 20
        ids = torch.tensor([[1, 2, 3]])
        mask = torch.tensor([[False, False, True]])
        loss = target_causal_loss(logits, ids, mask)
        self.assertLess(loss.item(), 0.01)


if __name__ == "__main__":
    unittest.main()
