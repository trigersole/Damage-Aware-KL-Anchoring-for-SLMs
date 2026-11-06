"""Preparation-only whole-prompt adapter for the existing normalized-rank selector.

No CLI, encoder inference, actual research export, training or protocol approval.
Raw prompt/embedding/cache provenance must be verified by the future driver.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def _by_id(rows: Sequence[dict], name: str) -> dict:
    if not rows:
        raise ValueError(f'{name} is empty')
    out = {}
    for row in rows:
        key = row.get('example_id')
        if not isinstance(key, str) or not key or key in out:
            raise ValueError(f'{name} IDs must be nonempty and unique')
        out[key] = row
    return out


def combined_order(scores: Sequence[dict], ranks: Sequence[dict], *, damage_weight: float) -> list[dict]:
    if isinstance(damage_weight, bool) or not isinstance(damage_weight, (float, int)) or not math.isfinite(damage_weight) or not 0 <= damage_weight <= 1:
        raise ValueError('Explicit finite damage weight in [0,1] required')
    scored, ranked = _by_id(scores, 'scores'), _by_id(ranks, 'ranks')
    if scored.keys() != ranked.keys():
        raise ValueError('Complete common-pool ranks required; top128 alone is insufficient')
    for row in scored.values():
        value, tokens = row.get('vulnerability_score'), row.get('response_token_count')
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError('All saved vulnerability scores must be finite')
        if type(tokens) is not int or tokens <= 0:
            raise ValueError('Positive integer full response counts required')
    values = [row.get('diversity_rank') for row in ranked.values()]
    if any(type(value) is not int for value in values) or sorted(values) != list(range(len(scored))):
        raise ValueError('Common-pool diversity ranks must be contiguous0..N-1')
    damage_order = sorted(scored, key=lambda key: (-float(scored[key]['vulnerability_score']), key))
    damage_rank = {key: i for i, key in enumerate(damage_order)}
    denominator = max(1, len(scored) - 1)
    combined = []
    for key, row in scored.items():
        d, v = damage_rank[key], ranked[key]['diversity_rank']
        cost = damage_weight * d / denominator + (1 - damage_weight) * v / denominator
        combined.append({'example_id': key, 'vulnerability_score': row['vulnerability_score'],
                         'response_token_count': row['response_token_count'],
                         'damage_rank': d, 'diversity_rank': v, 'combined_rank_cost': cost})
    return sorted(combined, key=lambda row: (row['combined_rank_cost'], row['example_id']))


def select_whole_prompts(scores: Sequence[dict], ranks: Sequence[dict], *, count: int, damage_weight: float) -> dict:
    ordered = combined_order(scores, ranks, damage_weight=damage_weight)
    if type(count) is not int or not 1 <= count <= len(ordered):
        raise ValueError('Whole-prompt count must be between1 and pool size')
    selected = []
    for i, row in enumerate(ordered[:count], 1):
        selected.append({**row, 'selection_rank': i,
                         'selected_response_tokens': row['response_token_count']})
    return {'method': 'damage_diverse_normalized_rank_fusion',
            'damage_weight': damage_weight, 'diversity_weight': 1 - damage_weight,
            'eligible_prompt_count': len(ordered), 'selected_prompt_count': count,
            'selected_unique_response_positions': sum(row['response_token_count'] for row in selected),
            'partial_continuations': 0, 'selected': selected}


def align_selected_cache(selection: dict, cache: Sequence[dict], *, sequence_limit: int) -> list[dict]:
    cached = _by_id(cache, 'cache')
    selected = selection['selected']
    if len({row['example_id'] for row in selected}) != len(selected):
        raise ValueError('Selected IDs repeated')
    out = []
    for row in selected:
        key = row['example_id']
        if key not in cached:
            raise ValueError('Selected teacher cache row missing')
        item = cached[key]
        prompt, response = item.get('prompt_token_ids'), item.get('continuation_token_ids')
        if not isinstance(prompt, list) or not prompt or not isinstance(response, list) or not response:
            raise ValueError('Nonempty aligned prompt/continuation IDs required')
        if any(type(token) is not int or token < 0 for token in prompt + response):
            raise ValueError('Token IDs must be nonnegative integers')
        if len(response) != item.get('response_token_count') or len(response) != row['response_token_count'] or row['selected_response_tokens'] != len(response):
            raise ValueError('No partial or mismatched teacher continuations permitted')
        if len(prompt) + len(response) > sequence_limit:
            raise ValueError('Selected total sequence exceeds fixed limit; never truncate silently')
        out.append(dict(item))
    return out
