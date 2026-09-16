"""Model-independent training primitives for GSM8K and anchor KL.

Inputs are already-rendered prompts and saved teacher continuations. All final
conditions must build a fresh student from the same pinned base checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from .kl import aligned_response_kl


@dataclass(frozen=True)
class TokenizedExample:
    example_id: str
    input_ids: tuple[int, ...]
    response_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.example_id or len(self.input_ids) < 2:
            raise ValueError("example needs an ID and at least two tokens")
        if len(self.input_ids) != len(self.response_mask):
            raise ValueError("input IDs and response mask lengths differ")
        if not any(self.response_mask[1:]):
            raise ValueError("example has no scoreable response token")
        if any(self.response_mask[:1]):
            raise ValueError("first token cannot be scored by a causal model")


def tokenize_completion(
    tokenizer,
    *,
    example_id: str,
    rendered_prompt: str,
    response: str,
    max_sequence_tokens: int,
    append_eos: bool,
) -> TokenizedExample:
    """Tokenize a prompt and response separately, without silent truncation."""
    prompt_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    if append_eos:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has no EOS token")
        response_ids.append(tokenizer.eos_token_id)
    if not prompt_ids or not response_ids:
        raise ValueError("empty tokenized prompt or response")
    if len(prompt_ids) + len(response_ids) > max_sequence_tokens:
        raise ValueError(f"example {example_id} exceeds max_sequence_tokens")
    return TokenizedExample(
        example_id,
        tuple(prompt_ids + response_ids),
        tuple([False] * len(prompt_ids) + [True] * len(response_ids)),
    )


def tokenize_saved_continuation(
    *, example_id: str, prompt_token_ids: Sequence[int],
    continuation_token_ids: Sequence[int], max_sequence_tokens: int,
) -> TokenizedExample:
    """Join the *saved token IDs* used by teacher, discovery and final runs."""
    if not prompt_token_ids or not continuation_token_ids:
        raise ValueError("empty prompt or teacher continuation")
    if len(prompt_token_ids) + len(continuation_token_ids) > max_sequence_tokens:
        raise ValueError(f"anchor {example_id} exceeds max_sequence_tokens")
    return TokenizedExample(
        example_id,
        tuple(prompt_token_ids) + tuple(continuation_token_ids),
        tuple([False] * len(prompt_token_ids) + [True] * len(continuation_token_ids)),
    )


def collate(examples: Sequence[TokenizedExample], *, pad_token_id: int, device: str | torch.device) -> dict[str, torch.Tensor]:
    """Right-pad examples; labels/masks are independent of padding."""
    if not examples:
        raise ValueError("cannot collate an empty batch")
    length = max(len(ex.input_ids) for ex in examples)
    ids = torch.full((len(examples), length), pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(examples), length), dtype=torch.long)
    response = torch.zeros((len(examples), length), dtype=torch.bool)
    for row, ex in enumerate(examples):
        n = len(ex.input_ids)
        ids[row, :n] = torch.tensor(ex.input_ids, dtype=torch.long)
        attention[row, :n] = 1
        response[row, :n] = torch.tensor(ex.response_mask, dtype=torch.bool)
    return {
        "input_ids": ids.to(device),
        "attention_mask": attention.to(device),
        "response_mask": response.to(device),
    }


def target_causal_loss(logits: torch.Tensor, input_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Cross entropy on target response tokens, excluding prompt and padding."""
    if logits.shape[:2] != input_ids.shape or response_mask.shape != input_ids.shape:
        raise ValueError("misaligned target logits, IDs or response mask")
    labels = input_ids[:, 1:].masked_fill(~response_mask[:, 1:], -100)
    if not torch.any(labels != -100):
        raise ValueError("target batch contains no response tokens")
    return F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)


class AnchorTokenScheduler:
    """Cycle selected response positions at an exact per-update token rate.

    For each anchor update, ``next_batch`` returns masks covering exactly
    ``tokens_per_update`` positions. The same rate and number of anchor updates
    yields the same scored-token exposure across selection methods. Input
    sequence lengths may still differ; record wall time and teacher cost.
    """

    def __init__(self, examples: Mapping[str, TokenizedExample], selected: Sequence[Mapping[str, object]], tokens_per_update: int):
        if tokens_per_update <= 0:
            raise ValueError("tokens_per_update must be positive")
        positions: list[tuple[str, int]] = []
        for row in selected:
            example_id = str(row["example_id"])
            allowed = int(row["selected_response_tokens"])
            ex = examples[example_id]
            valid = [i for i, keep in enumerate(ex.response_mask) if keep]
            if allowed <= 0 or allowed > len(valid):
                raise ValueError(f"invalid token allocation for {example_id}")
            positions.extend((example_id, i) for i in valid[:allowed])
        if not positions:
            raise ValueError("anchor selection has no tokens")
        if tokens_per_update > len(positions):
            raise ValueError("tokens_per_update cannot exceed the selected token budget")
        self._examples = examples
        self._positions = positions
        self.tokens_per_update = tokens_per_update
        self._cursor = 0
        self.total_emitted = 0
        self.per_anchor_emitted = {example_id: 0 for example_id in examples}

    @property
    def selection_token_budget(self) -> int:
        return len(self._positions)

    def next_batch(self) -> list[TokenizedExample]:
        active: dict[str, list[bool]] = {}
        for _ in range(self.tokens_per_update):
            example_id, token_index = self._positions[self._cursor]
            self._cursor = (self._cursor + 1) % len(self._positions)
            if example_id not in active:
                active[example_id] = [False] * len(self._examples[example_id].input_ids)
            active[example_id][token_index] = True
            self.per_anchor_emitted[example_id] += 1
        if sum(sum(mask) for mask in active.values()) != self.tokens_per_update:
            raise AssertionError("scheduler emitted duplicate positions")
        self.total_emitted += self.tokens_per_update
        return [TokenizedExample(id_, self._examples[id_].input_ids, tuple(mask)) for id_, mask in active.items()]


def training_step(
    *, student, teacher, target_batch: Mapping[str, torch.Tensor],
    anchor_batch: Mapping[str, torch.Tensor] | None, optimizer,
    kl_weight: float | None, kl_direction: str | None,
    kl_temperature: float | None, kl_reduction: str | None,
    scale_by_temperature_squared: bool | None, scheduler=None,
) -> dict[str, float]:
    """One microbatch optimizer update, useful for small smoke tests."""
    return training_update(
        student=student, teacher=teacher, target_batches=[target_batch],
        anchor_batch=anchor_batch, optimizer=optimizer, kl_weight=kl_weight,
        kl_direction=kl_direction, kl_temperature=kl_temperature,
        kl_reduction=kl_reduction,
        scale_by_temperature_squared=scale_by_temperature_squared,
        scheduler=scheduler,
    )


def training_update(
    *, student, teacher, target_batches: Sequence[Mapping[str, torch.Tensor]],
    anchor_batch: Mapping[str, torch.Tensor] | None, optimizer,
    kl_weight: float | None, kl_direction: str | None,
    kl_temperature: float | None, kl_reduction: str | None,
    scale_by_temperature_squared: bool | None, scheduler=None,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    """One optimizer update with explicit target microbatch accumulation."""
    if not target_batches:
        raise ValueError("at least one target microbatch is required")
    student.train()
    optimizer.zero_grad(set_to_none=True)
    counts = [int(batch["response_mask"].sum().item()) for batch in target_batches]
    target_tokens = sum(counts)
    if target_tokens <= 0:
        raise ValueError("target update contains no response tokens")
    weighted_loss_sum = 0.0
    for target_batch, count in zip(target_batches, counts):
        target_logits = student(input_ids=target_batch["input_ids"], attention_mask=target_batch["attention_mask"]).logits
        target_loss = target_causal_loss(target_logits, target_batch["input_ids"], target_batch["response_mask"])
        (target_loss * count / target_tokens).backward()
        weighted_loss_sum += float(target_loss.detach().cpu()) * count
    target_mean = weighted_loss_sum / target_tokens
    kl_value = None
    if anchor_batch is not None:
        if teacher is None or kl_weight is None or kl_weight < 0 or kl_direction is None or kl_temperature is None or kl_reduction is None or scale_by_temperature_squared is None:
            raise ValueError("anchored step requires an explicit KL protocol and teacher")
        with torch.no_grad():
            teacher_logits = teacher(input_ids=anchor_batch["input_ids"], attention_mask=anchor_batch["attention_mask"]).logits
        # Score the student's inference-time distributions, with LoRA dropout
        # disabled. eval() does not block gradients through trainable adapters.
        student.eval()
        student_logits = student(input_ids=anchor_batch["input_ids"], attention_mask=anchor_batch["attention_mask"]).logits
        kl_value = aligned_response_kl(
            teacher_logits, student_logits, anchor_batch["response_mask"],
            direction=kl_direction, temperature=kl_temperature,
            reduction=kl_reduction, scale_by_temperature_squared=scale_by_temperature_squared,
        )
        (kl_weight * kl_value).backward()
        student.train()
    if max_grad_norm is not None:
        if max_grad_norm < 0:
            raise ValueError("max_grad_norm must be nonnegative")
        torch.nn.utils.clip_grad_norm_((p for p in student.parameters() if p.requires_grad), max_grad_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return {
        "target_loss": target_mean,
        "anchor_kl": float(kl_value.detach().cpu()) if kl_value is not None else 0.0,
        "total_loss": target_mean + (kl_weight * float(kl_value.detach().cpu()) if kl_value is not None else 0.0),
        "target_tokens": target_tokens,
        "anchor_tokens": int(anchor_batch["response_mask"].sum().item()) if anchor_batch is not None else 0,
    }
