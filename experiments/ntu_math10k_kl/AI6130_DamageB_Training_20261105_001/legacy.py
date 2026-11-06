from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
import random
import re
import numpy as np
import torch
import torch.nn.functional as F

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

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_tokenizer(model: Mapping[str, str]):
    from transformers import AutoTokenizer

    if re.fullmatch(r"[0-9a-fA-F]{40}", model["revision"]) is None:
        raise ValueError("model revision must be a full immutable commit SHA")

    tokenizer = AutoTokenizer.from_pretrained(
        model["repository_id"], revision=model["revision"], use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_base(model: Mapping[str, str], *, precision: str, quantization: str, device: str):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if re.fullmatch(r"[0-9a-fA-F]{40}", model["revision"]) is None:
        raise ValueError("model revision must be a full immutable commit SHA")

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    kwargs = {"revision": model["revision"], "dtype": dtype, "use_safetensors": True}
    if quantization != "none":
        if not device.startswith("cuda"):
            raise ValueError("bitsandbytes quantization requires CUDA")
        if quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = {"": device}
    base = AutoModelForCausalLM.from_pretrained(model["repository_id"], **kwargs)
    if quantization == "none":
        base.to(device)
    return base

def attach_new_lora(base, lora: Mapping[str, object]):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if lora["quantization"] != "none":
        base = prepare_model_for_kbit_training(base)
    config = LoraConfig(
        r=int(lora["rank"]),
        lora_alpha=float(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    student = get_peft_model(base, config)
    if not any(p.requires_grad for p in student.parameters()):
        raise RuntimeError("LoRA model has no trainable parameters")
    return student

def render_prompt(template: str, question: str) -> str:
    if template.count("{question}") != 1:
        raise ValueError("Prompt template must contain exactly one {question}")
    return template.replace("{question}", question)

def make_examples(tokenizer, rows, protocol):
    template = protocol["evaluation"]["prompt_format"].replace("\\n", "\n")
    limit = int(protocol["runtime"]["max_sequence_tokens"])
    return [tokenize_completion(
        tokenizer,
        example_id=row["example_id"],
        rendered_prompt=render_prompt(template, row["prompt"]),
        response=row["response"],
        max_sequence_tokens=limit,
        append_eos=bool(protocol["runtime"]["target_answer_append_eos"]),
    ) for row in rows]

def epoch_batches(examples, seed, epochs, effective_batch_size):
    """Shuffle separately each epoch; never repeat examples across partial batches."""
    for epoch in range(1, epochs + 1):
        order = list(range(len(examples)))
        random.Random(seed + epoch - 1).shuffle(order)
        for start in range(0, len(order), effective_batch_size):
            yield epoch, [examples[i] for i in order[start:start + effective_batch_size]]

