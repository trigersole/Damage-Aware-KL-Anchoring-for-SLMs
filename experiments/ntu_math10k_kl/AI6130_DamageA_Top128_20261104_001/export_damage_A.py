#!/usr/bin/env python3
"""Export approved unrestricted top-128 membership from existing Step5 evidence.

No language-model loads, generation, KL recomputation or training. The only
model-related object constructed is the already-cached tokenizer for readable
teacher text and exact selected-prompt token verification.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import source_guard as g

ROOT = Path(__file__).resolve().parent
OUTPUT_NAME = 'damage_A_top128_20261104_001'


def index_rows(rows, label):
    indexed = {}
    for row in rows:
        key = row.get('example_id')
        if not isinstance(key, str) or not key or key in indexed:
            raise ValueError(f'Invalid or duplicate {label} example_id: {key!r}')
        indexed[key] = row
    return indexed


def valid_ids(value):
    return isinstance(value, list) and bool(value) and all(type(x) is int and x >= 0 for x in value)


def rank_candidates(anchors, cache, scores, skipped, *, count=128,
                    expected_input=13769, expected_cached=13743):
    a = index_rows(anchors, 'source')
    c = index_rows(cache, 'cache')
    s = index_rows(scores, 'score')
    k = index_rows(skipped, 'skip')
    if len(a) != expected_input or len(c) != expected_cached:
        raise ValueError('Frozen input/cache membership count mismatch')
    if set(c) != set(s) or set(c) & set(k) or set(c) | set(k) != set(a):
        raise ValueError('Cache/score/skip coverage does not exactly partition the source')
    if type(count) is not int or not 1 <= count <= len(c):
        raise ValueError('Invalid selection count')
    for key, row in a.items():
        if not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
            raise ValueError(f'Empty or invalid original prompt: {key}')
    for key, row in c.items():
        pids, rids = row.get('prompt_token_ids'), row.get('continuation_token_ids')
        if not valid_ids(pids) or not valid_ids(rids):
            raise ValueError(f'Invalid saved token IDs: {key}')
        if type(row.get('response_token_count')) is not int or row['response_token_count'] != len(rids):
            raise ValueError(f'Cached response-token count differs: {key}')
        if len(pids) > 1792 or len(rids) > 256 or len(pids) + len(rids) > 2048:
            raise ValueError(f'Cached sequence violates frozen length policy: {key}')
        if 128001 in rids[:-1]:
            raise ValueError(f'Native EOS inside a saved continuation: {key}')
        score = s[key].get('vulnerability_score')
        if type(score) not in (int, float) or not math.isfinite(score) or score < 0:
            raise ValueError(f'Invalid stored KL score: {key}')
    for key, row in k.items():
        if (row.get('reason') != 'prompt_exceeds_token_limit'
                or type(row.get('prompt_token_count')) is not int
                or row['prompt_token_count'] <= 1792
                or row.get('maximum_prompt_tokens') != 1792):
            raise ValueError(f'Wrong frozen overlength reason: {key}')
    ordered = sorted(c, key=lambda key: (-s[key]['vulnerability_score'], key))
    ranked = [dict(example_id=key, selection_rank=i,
                   vulnerability_score=s[key]['vulnerability_score'],
                   response_token_count=len(c[key]['continuation_token_ids']),
                   ended_with_eos=c[key]['continuation_token_ids'][-1] == 128001)
              for i, key in enumerate(ordered, 1)]
    # Every positive-length cached response remains eligible: no KL minimum,
    # response-length cutoff, topic cap, MMR reranking, or human quality filter.
    return ranked, ranked[:count], a, c


def write_jsonl(path, rows):
    with path.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def histogram(rows):
    return dict(sorted(Counter(str(r['response_token_count']) for r in rows).items(),
                       key=lambda pair: int(pair[0])))


def verify_score_metadata(metadata, *, score_sha, cache_sha, manifest):
    """Require the existing completion receipt to bind these exact inputs.

    The supplied runner excerpt stops before metadata construction. Inspect
    values without assuming undocumented key spellings; missing bindings stop
    the export for inspection rather than being inferred from filenames.
    """
    def strings(value):
        if isinstance(value, dict):
            return set().union(*(strings(v) for v in value.values())) if value else set()
        if isinstance(value, list):
            return set().union(*(strings(v) for v in value)) if value else set()
        return {value} if isinstance(value, str) else set()
    values = strings(metadata)
    if score_sha not in values or cache_sha not in values:
        raise ValueError('Score completion metadata does not bind current score/cache digests; inspect it first')
    if not ({manifest['adapter'], manifest['adapter_sha256']} & values):
        raise ValueError('Score completion metadata does not identify the pinned step600 adapter; inspect it first')


def run():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run this data export in an assigned Slurm node, not the head node')
    g.verify_bundle(ROOT)
    start = time.monotonic()
    protocol, manifest, repo, exp, cache_path, protected = g.guard(root=ROOT)
    cfg = g.read(ROOT / 'selection.json')
    if cfg != {'variant': 'A', 'count': 128, 'ranking': 'descending_existing_KL',
               'tie_break': 'example_id_ascending', 'minimum_response_tokens': None,
               'minimum_KL': None, 'EOS_included': True, 'training_authorized': False}:
        raise ValueError('Approved selection configuration changed')
    scores_path = exp / 'full/vulnerability_scores.jsonl'
    skipped_path = exp / 'full/teacher_cache.jsonl.skipped.jsonl'
    cache_meta = g.read(exp / 'full/teacher_cache.jsonl.meta.json')
    if not scores_path.is_file() or g.sha(skipped_path) != cache_meta['skipped_anchors_sha256']:
        raise ValueError('Completed score file or protected skipped evidence missing/changed')
    protected[str(skipped_path)] = g.sha(skipped_path)
    score_meta_path = exp / 'full/vulnerability_scores.jsonl.meta.json'
    if not score_meta_path.is_file():
        raise ValueError('Original score completion metadata missing')
    score_meta = g.read(score_meta_path)
    protected[str(score_meta_path)] = g.sha(score_meta_path)
    verify_score_metadata(score_meta, score_sha=protected[str(scores_path)],
                          cache_sha=protected[str(cache_path)], manifest=manifest)
    ranked, selected, anchors, cache = rank_candidates(
        list(g.rows(Path(manifest['input_anchor_file']))), list(g.rows(cache_path)),
        list(g.rows(scores_path)), list(g.rows(skipped_path)), count=128)

    # Offline tokenizer only: do not load either model or regenerate responses.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        protocol['model']['repository_id'], revision=protocol['model']['revision'],
        use_fast=True, local_files_only=True)
    if tokenizer.eos_token_id != 128001:
        raise ValueError('Pinned tokenizer EOS differs')
    template = protocol['evaluation']['prompt_format'].replace('\\n', '\n')
    exported = []
    for row in selected:
        key = row['example_id']
        saved = cache[key]
        rendered = template.replace('{question}', anchors[key]['prompt'])
        if tokenizer.encode(rendered, add_special_tokens=False) != saved['prompt_token_ids']:
            raise ValueError(f'Selected original prompt/cache tokens differ: {key}')
        if any(token >= len(tokenizer) for token in saved['continuation_token_ids']):
            raise ValueError(f'Continuation token outside pinned vocabulary: {key}')
        exported.append({**anchors[key], **saved, **row,
                         'formatted_prompt': rendered,
                         'cached_teacher_text': tokenizer.decode(
                             saved['continuation_token_ids'], skip_special_tokens=False),
                         'cached_teacher_text_without_special_tokens': tokenizer.decode(
                             saved['continuation_token_ids'], skip_special_tokens=True)})

    unchanged = {path: g.sha(Path(path)) == value for path, value in protected.items()}
    if not all(unchanged.values()):
        raise ValueError('Protected original evidence changed during export')
    out = exp / 'analysis' / OUTPUT_NAME
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / 'ranked_candidates.jsonl', ranked)
    write_jsonl(out / 'damage_A_anchors.jsonl', exported)
    # This subset is not substituted for the original canonical full-pool cache.
    write_jsonl(out / 'selected_teacher_cache.jsonl', [cache[r['example_id']] for r in selected])
    with (out / 'damage_A_review.csv').open('x', newline='', encoding='utf-8-sig') as stream:
        fields = ['selection_rank', 'example_id', 'vulnerability_score',
                  'response_token_count', 'ended_with_eos', 'prompt', 'cached_teacher_text']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        # Prevent spreadsheet formulas in this display-only view. JSONL retains
        # every original character and is the authoritative selection export.
        for row in exported:
            display = dict(row)
            for field in ('example_id', 'prompt', 'cached_teacher_text'):
                if display[field].lstrip().startswith(('=', '+', '-', '@')):
                    display[field] = "'" + display[field]
            writer.writerow(display)
    pool_payload = [(r['example_id'], r['response_token_count'])
                    for r in sorted(ranked, key=lambda row: row['example_id'])]
    pool_sha = hashlib.sha256(json.dumps(
        pool_payload, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    selected_tokens = sum(r['response_token_count'] for r in selected)
    summary = {
        'status': 'complete', 'scope': 'Variant A top-128 membership export only',
        'variant': 'A', 'selected_prompt_count': 128, 'eligible_prompt_count': len(ranked),
        'input_prompt_count': len(anchors), 'original_overlength_count': 26,
        'response_length_filter': None, 'KL_threshold': None,
        'ranking': 'descending stored vulnerability_score; stable example_id ties',
        'response_length_definition': 'Saved teacher continuation; includes native EOS when generated',
        'selected_unique_response_tokens': selected_tokens,
        'selected_response_length_histogram': histogram(selected),
        'eligible_response_length_histogram': histogram(ranked),
        'selected_EOS_only_count': sum(r['response_token_count'] == 1 and r['ended_with_eos'] for r in selected),
        'selected_two_token_count': sum(r['response_token_count'] == 2 for r in selected),
        'highest_selected_KL': selected[0]['vulnerability_score'],
        'lowest_selected_KL': selected[-1]['vulnerability_score'],
        'eligible_pool_ids_and_lengths_sha256': pool_sha,
        'model': protocol['model'], 'discovery_student_step': 600,
        'validation_branch': 'original friend300', 'source_manifest': manifest,
        'score_completion_metadata_as_supplied': score_meta,
        'source_checksums': protected, 'protected_inputs_unchanged': unchanged,
        'selection_configuration': cfg, 'python': sys.version,
        'platform': platform.platform(), 'job_id': os.environ['SLURM_JOB_ID'],
        'node': os.environ.get('SLURMD_NODENAME'),
        'model_loads': 0, 'generation_calls': 0, 'KL_recomputations': 0, 'training_updates': 0,
        'training_ready': False, 'total_training_anchor_token_budget': None,
        'limitations': [
            'High KL measures behavioural drift, not verified harmful forgetting or answer correctness.',
            'Unrestricted ranking may concentrate on instrument questions and short responses.',
            '128 is the selected prompt count, not a matched training-token exposure budget.',
            'Shared anchor-token exposure, interleaving and KL coefficient remain separate decisions.',
            'No teacher responses, EOS positions, original scores or eligible IDs were changed.',
            'Completeness and checksums are verified; all full-pool model logits are not recomputed.'
        ], 'elapsed_seconds': time.monotonic() - start}
    summary['export_sha256'] = {p.name: g.sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    g.write_new(out / 'summary.json', summary)
    print('[COMPLETE] ' + str(out), flush=True)
    print(json.dumps({key: summary[key] for key in (
        'status', 'selected_prompt_count', 'eligible_prompt_count',
        'selected_unique_response_tokens', 'selected_response_length_histogram',
        'highest_selected_KL', 'lowest_selected_KL', 'training_ready')}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        g.verify_bundle(ROOT)
        print(json.dumps({'status': 'package_verified_not_run', 'variant': 'A',
                          'selected_count': 128, 'minimum_response_tokens': None,
                          'training_authorized': False}, indent=2))
    else:
        run()


if __name__ == '__main__':
    main()
