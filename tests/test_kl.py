import unittest

import torch

from kl_anchors.kl import aligned_response_kl


class KLTests(unittest.TestCase):
    def test_mask_shifts_to_next_token_and_ignores_prompt(self):
        teacher = torch.tensor([[[0., 0.], [2., 0.], [0., 2.], [0., 0.]]])
        student = teacher.clone()
        student[:, 0, :] = torch.tensor([10., -10.])  # predicts a prompt token
        student[:, 2, :] = torch.tensor([2., 0.])  # predicts response token at index 3
        mask = torch.tensor([[False, False, False, True]])
        result = aligned_response_kl(
            teacher, student, mask,
            direction="teacher_to_student", temperature=1,
            reduction="token_mean", scale_by_temperature_squared=False,
        )
        expected = torch.sum(torch.softmax(teacher[0, 2], -1) * (
            torch.log_softmax(teacher[0, 2], -1) - torch.log_softmax(student[0, 2], -1)))
        self.assertTrue(torch.allclose(result, expected))

    def test_gradient_only_through_student(self):
        teacher = torch.randn(1, 3, 4, requires_grad=True)
        student = torch.randn(1, 3, 4, requires_grad=True)
        mask = torch.tensor([[False, True, True]])
        loss = aligned_response_kl(
            teacher, student, mask,
            direction="teacher_to_student", temperature=2,
            reduction="sequence_mean", scale_by_temperature_squared=True,
        )
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(student.grad)
        self.assertGreater(student.grad.abs().sum().item(), 0)

    def test_rejects_missing_response(self):
        logits = torch.zeros(1, 3, 4)
        with self.assertRaises(ValueError):
            aligned_response_kl(
                logits, logits, torch.zeros(1, 3, dtype=torch.bool),
                direction="teacher_to_student", temperature=1,
                reduction="token_mean", scale_by_temperature_squared=False,
            )


if __name__ == "__main__":
    unittest.main()
