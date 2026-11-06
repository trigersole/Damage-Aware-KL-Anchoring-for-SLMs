"""Small-microbatch math + globally token-mean forward KL; no design defaults."""
from __future__ import annotations
from collections import defaultdict
import math


def anchor_groups(cache, cursor, count):
    positions = [(r['example_id'], i) for r in cache for i in range(r['response_token_count'])]
    if not positions or not 0 < count <= len(positions) or not 0 <= cursor < len(positions):
        raise ValueError('Invalid anchor cursor or per-update token allocation')
    selected = defaultdict(list)
    for offset in range(count):
        key, index = positions[(cursor + offset) % len(positions)]
        selected[key].append(index)
    return list(selected.items()), (cursor + count) % len(positions)


def compact_kl_sum(teacher_logits, student_logits, response_mask):
    """Same full-vocabulary forward KL, selecting scored rows before softmax.

    This avoids materializing FP32 probabilities on unscored prompt rows.
    The caller divides the sum by all64 scored positions across microbatches.
    """
    import torch
    if teacher_logits.shape != student_logits.shape or teacher_logits.ndim != 3:
        raise ValueError('Teacher/student logits do not align')
    if response_mask.shape != teacher_logits.shape[:2] or response_mask.dtype != torch.bool:
        raise ValueError('Response mask does not align')
    valid = response_mask[:, 1:]
    if not valid.any():
        raise ValueError('No scored causal response position')
    t = teacher_logits[:, :-1, :][valid].detach().float().log_softmax(-1)
    s = student_logits[:, :-1, :][valid].float().log_softmax(-1)
    return (t.exp() * (t - s)).sum(), int(valid.sum().item())


def norm(tensors):
    import torch
    values = [t.detach().float().norm(2) for t in tensors if t is not None]
    return float(torch.stack(values).norm(2).item()) if values else 0.0


def update(student, teacher, group, optimizer, scheduler, p, tokenizer, cache, cursor, coefficient, step):
    import torch
    import legacy
    student.train(); optimizer.zero_grad(set_to_none=True)
    params = [x for x in student.parameters() if x.requires_grad]
    target_tokens = sum(sum(ex.response_mask[1:]) for ex in group)
    if target_tokens <= 0:
        raise ValueError('No target response tokens')
    target_sum = 0.0
    for ex in group:
        batch = legacy.collate([ex], pad_token_id=tokenizer.pad_token_id, device=p['runtime']['device'])
        logits = student(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
        loss = legacy.target_causal_loss(logits, batch['input_ids'], batch['response_mask'])
        count = sum(ex.response_mask[1:])
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite target loss')
        (loss * count / target_tokens).backward()
        target_sum += float(loss.detach().item()) * count
        del logits, loss, batch
    target_norm = norm([x.grad for x in params])
    anchor_tokens = 0; kl_sum = 0.0; weighted_grad_norm = None; anchor_ids = []
    anchored = coefficient > 0 and step % p['training']['anchor_interleave_every_steps'] == 0
    if anchored:
        if teacher is None:
            raise ValueError('Anchored update needs the frozen teacher')
        target_grads = [x.grad.detach().clone() if x.grad is not None else None for x in params]
        rate = p['runtime']['anchor_tokens_per_update']
        groups, cursor = anchor_groups(cache, cursor, rate)
        lookup = {r['example_id']: r for r in cache}
        student.eval(); teacher.eval()
        for key, indices in groups:
            row = lookup[key]; prompt, response = row['prompt_token_ids'], row['continuation_token_ids']
            mask = [False] * (len(prompt) + len(response))
            for j in indices:
                mask[len(prompt) + j] = True
            ex = legacy.TokenizedExample(key, tuple(prompt + response), tuple(mask))
            batch = legacy.collate([ex], pad_token_id=tokenizer.pad_token_id, device=p['runtime']['device'])
            with torch.no_grad():
                tlog = teacher(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
            slog = student(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
            value, count = compact_kl_sum(tlog, slog, batch['response_mask'])
            if not torch.isfinite(value):
                raise ValueError('Nonfinite anchor KL')
            (coefficient * value / rate).backward()
            kl_sum += float(value.detach().item()); anchor_tokens += count; anchor_ids.append(key)
            del tlog, slog, value, batch
        if anchor_tokens != rate:
            raise ValueError('Anchor microbatches did not score the exact approved token rate')
        weighted_grad_norm = norm([x.grad - old if old is not None else x.grad
                                   for x, old in zip(params, target_grads)])
        del target_grads
        student.train()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, p['runtime']['max_grad_norm'],
                                                   error_if_nonfinite=True).item())
    lr_used = optimizer.param_groups[0]['lr']
    optimizer.step(); scheduler.step()
    raw_kl = kl_sum / anchor_tokens if anchor_tokens else None
    weighted_kl = coefficient * raw_kl if raw_kl is not None else 0.0
    target_loss = target_sum / target_tokens
    metrics = dict(step=step, target_loss=target_loss, target_tokens=target_tokens,
                   anchor_kl=raw_kl, weighted_anchor_kl=weighted_kl,
                   total_loss=target_loss + weighted_kl, anchor_tokens=anchor_tokens,
                   is_anchor_update=anchored, anchor_ids=anchor_ids,
                   target_gradient_norm=target_norm, weighted_anchor_gradient_norm=weighted_grad_norm,
                   combined_gradient_norm_before_clip=grad_norm, learning_rate_used=lr_used,
                   learning_rate_next=optimizer.param_groups[0]['lr'])
    if any(not math.isfinite(v) for v in metrics.values() if type(v) is float):
        raise ValueError('Nonfinite update metric')
    return metrics, cursor
