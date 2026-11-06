"""KL on aligned next-token distributions.

The caller must feed the teacher and student identical token IDs. For a causal
language model, logits at position i predict token i+1. ``response_mask`` marks
tokens belonging to the saved teacher continuation, including its first token.
The function shifts that mask so only predictions of continuation tokens count.
No generated-string comparison or prompt-token KL is performed here.
"""

from __future__ import annotations

from typing import Literal

import torch

Direction = Literal["teacher_to_student", "student_to_teacher"]
Reduction = Literal["token_mean", "sequence_mean", "none"]


def aligned_response_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    direction: Direction,
    temperature: float,
    reduction: Reduction,
    scale_by_temperature_squared: bool,
) -> torch.Tensor:
    """Return KL over valid continuation predictions.

    Args:
        teacher_logits, student_logits: ``[batch, sequence, vocabulary]``
            outputs obtained with the same ``input_ids`` and attention mask.
        response_mask: Boolean ``[batch, sequence]`` mask on *token* positions.
            The initial prompt and padding positions are False.
        direction: KL direction, always chosen explicitly by the experiment.
        temperature: Positive probability-softening temperature.
        reduction: ``token_mean`` weights all scored tokens equally;
            ``sequence_mean`` weights examples equally; ``none`` returns
            per-token KL with zero at masked positions.
        scale_by_temperature_squared: Whether to multiply by T squared, a
            common distillation convention. Must be recorded in the protocol.
    """
    if teacher_logits.ndim != 3 or teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher and student logits must have identical [B, L, V] shapes")
    if response_mask.shape != teacher_logits.shape[:2] or response_mask.dtype != torch.bool:
        raise ValueError("response_mask must be boolean with shape [B, L]")
    if teacher_logits.shape[1] < 2:
        raise ValueError("at least two token positions are needed for causal scoring")
    if direction not in ("teacher_to_student", "student_to_teacher"):
        raise ValueError("direction must be explicit and supported")
    if reduction not in ("token_mean", "sequence_mean", "none"):
        raise ValueError("unsupported reduction")
    if not 0 < temperature < float("inf"):
        raise ValueError("temperature must be finite and positive")

    valid = response_mask[:, 1:]
    counts = valid.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("each example needs at least one response token")

    teacher_logp = torch.log_softmax(teacher_logits[:, :-1, :].detach().float() / temperature, dim=-1)
    student_logp = torch.log_softmax(student_logits[:, :-1, :].float() / temperature, dim=-1)
    if direction == "teacher_to_student":
        source, target = teacher_logp, student_logp
    else:
        source, target = student_logp, teacher_logp
    per_token = (source.exp() * (source - target)).sum(dim=-1)
    if scale_by_temperature_squared:
        per_token = per_token * (temperature * temperature)
    per_token = torch.where(valid, per_token, torch.zeros_like(per_token))

    if reduction == "none":
        return per_token
    if reduction == "token_mean":
        return per_token.sum() / counts.sum()
    return (per_token.sum(dim=1) / counts).mean()
