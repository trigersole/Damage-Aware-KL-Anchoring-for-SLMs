"""Lazy Hugging Face loading shared by every study condition.

No pretrained model is loaded on import. Every call requires an immutable model
revision. Base, discovery and each final run call ``load_base`` independently.
"""

from __future__ import annotations

import random
import re
from typing import Mapping

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def render_chat_prompt(tokenizer, content: str, system_prompt: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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


def attach_saved_lora(base, adapter_path: str):
    from peft import PeftModel

    return PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
