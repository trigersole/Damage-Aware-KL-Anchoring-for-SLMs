#!/usr/bin/env python3
"""Approved cached-response >8-token top128 export; no generation or training."""
import argparse
import csv
import fcntl
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import source_guard as g
import selection_checks as checks

ROOT = Path(__file__).resolve().parent
OUTPUT_NAME = 'damage-B-top128-gt8-20261105-001'


def configuration(root=ROOT):
    g.verify_bundle(root)
    cfg = g.read(root / 'selection.json')
    expected = {'variant': 'B', 'approved': True, 'approval_date_singapore': '2026-11-05',
        'approval': 'Hemanth: Prepare >8-token Version B selection (recommended); data-only, no training',
        'count': 128, 'response_token_count_greater_than': 8, 'EOS_included': True,
        'ranking': 'descending_existing_KL', 'tie_break': 'example_id_ascending',
        'minimum_KL': None, 'teacher_regeneration': False, 'KL_recomputation': False,
        'training_authorized': False}
    if cfg != expected:
        raise ValueError('Approved Variant B selection configuration changed')
    return cfg


def select(ranked, *, count=128, threshold=8):
    if type(count) is not int or count < 1 or type(threshold) is not int or threshold < 0:
        raise ValueError('Invalid selection count/threshold')
    # All cache/score/length/ID checks and original ranking precede this filter.
    eligible = []; excluded = []
    for row in ranked:
        if type(row['response_token_count']) is not int or row['response_token_count'] < 1:
            raise ValueError('Invalid response token count')
        if row['response_token_count'] > threshold:
            eligible.append({**row, 'original_KL_rank': row['selection_rank'],
                             'selection_rank': len(eligible) + 1})
        else:
            excluded.append({**row, 'reason': 'cached_response_token_count_not_greater_than_8'})
    if len(eligible) < count:
        raise ValueError(f'Only{len(eligible)} responses pass the strict>{threshold} rule;need{count}. No relaxation authorized')
    return eligible, eligible[:count], excluded


def verify_A(exp, manifest, ranked, cache, protected):
    directory = exp / 'analysis/damage_A_top128_20261104_001'
    summary = g.read(directory / 'summary.json')
    if (summary['status'], summary['selected_prompt_count'], summary['eligible_prompt_count'],
        summary['selected_unique_response_tokens']) != ('complete', 128, 13743, 255):
        raise ValueError('Original A selection counts/status differ')
    if summary['source_manifest'] != manifest:
        raise ValueError('A and B must use the same discovery source')
    required = {'ranked_candidates.jsonl', 'damage_A_anchors.jsonl',
                'selected_teacher_cache.jsonl', 'damage_A_review.csv'}
    if not required.issubset(summary['export_sha256']):
        raise ValueError('A export receipt incomplete')
    for name, expected in summary['export_sha256'].items():
        if Path(name).name != name or g.sha(directory / name) != expected:
            raise ValueError('Original A output checksum differs')
        protected[str(directory / name)] = expected
    protected[str(directory / 'summary.json')] = g.sha(directory / 'summary.json')
    selected = list(g.rows(directory / 'damage_A_anchors.jsonl'))
    ids = [r['example_id'] for r in selected]
    if ids != [r['example_id'] for r in ranked[:128]]:
        raise ValueError('Original A membership is not unchanged unrestricted top128')
    if list(g.rows(directory / 'selected_teacher_cache.jsonl')) != [cache[key] for key in ids]:
        raise ValueError('Original A cache subset differs')
    if sum(cache[key]['response_token_count'] for key in ids) != 255:
        raise ValueError('Original A unique response-token total differs')
    return set(ids)


def verify_completed(out, identity):
    receipt = g.read(out / 'completion.json')
    if receipt['identity'] != identity:
        raise ValueError('Existing B result uses different source/configuration')
    for name, expected in receipt['files_sha256'].items():
        if Path(name).name != name or g.sha(out / name) != expected:
            raise ValueError('Existing completed B output changed')
    if set(p.name for p in out.iterdir()) != set(receipt['files_sha256']) | {'completion.json'}:
        raise ValueError('Existing completed B output membership differs')
    return g.read(out / 'summary.json')


def print_summary(summary, *, reused=False):
    keys = ['status', 'variant', 'cached_prompt_count', 'eligible_prompt_count',
            'excluded_short_prompt_count', 'selected_prompt_count',
            'selected_unique_response_tokens', 'selected_response_length_histogram',
            'highest_selected_KL', 'lowest_selected_KL', 'overlap_with_A_prompt_count', 'training_ready']
    print(json.dumps({'reused_completed_export': reused, **{k: summary[k] for k in keys}}, indent=2), flush=True)


def run(root=ROOT):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit this export through Slurm,not the head node')
    cfg = configuration(root); started = time.monotonic()
    protocol, manifest, repo, exp, cache_path, protected = g.guard(root=root)
    for relative, expected in manifest['code_sha256'].items():
        protected[str(repo / relative)] = expected
    scores_path = exp / 'full/vulnerability_scores.jsonl'
    skipped_path = exp / 'full/teacher_cache.jsonl.skipped.jsonl'
    meta_path = exp / 'full/vulnerability_scores.jsonl.meta.json'
    cache_meta = g.read(exp / 'full/teacher_cache.jsonl.meta.json')
    if g.sha(skipped_path) != cache_meta['skipped_anchors_sha256']:
        raise ValueError('Original overlength exclusions changed')
    for p in [scores_path, skipped_path, meta_path]:
        protected[str(p)] = g.sha(p)
    score_meta = g.read(meta_path)
    checks.verify_score_metadata(score_meta, score_sha=protected[str(scores_path)],
                                 cache_sha=protected[str(cache_path)], manifest=manifest)
    ranked, _, anchors, cache = checks.rank_candidates(
        list(g.rows(Path(manifest['input_anchor_file']))), list(g.rows(cache_path)),
        list(g.rows(scores_path)), list(g.rows(skipped_path)), count=128)
    eligible, selected, excluded = select(ranked, count=cfg['count'],
                                         threshold=cfg['response_token_count_greater_than'])
    A_ids = verify_A(exp, manifest, ranked, cache, protected)
    bundle = g.sha(root / 'files.sha256.json')
    identity = {'package_manifest_sha256': bundle, 'configuration': cfg,
                'source_manifest': manifest, 'protected_source_sha256': protected}
    results = root / 'results'; results.mkdir(exist_ok=True)
    out = results / OUTPUT_NAME
    with (results / '.selection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if out.exists():
            summary = verify_completed(out, identity)
            print('[COMPLETE] '+str(out), flush=True); print_summary(summary, reused=True)
            return
        # Cached tokenizer only; preserve exact prompt tokens and teacher bytes.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(protocol['model']['repository_id'],
            revision=protocol['model']['revision'], use_fast=True, local_files_only=True)
        if tokenizer.eos_token_id != 128001:
            raise ValueError('Pinned native EOS differs')
        software = {name: metadata.version(name) for name in ['transformers', 'tokenizers', 'huggingface_hub']}
        template = protocol['evaluation']['prompt_format'].replace('\\n', '\n')
        exported = []
        for row in selected:
            key = row['example_id']; saved = cache[key]
            rendered = template.replace('{question}', anchors[key]['prompt'])
            if tokenizer.encode(rendered, add_special_tokens=False) != saved['prompt_token_ids']:
                raise ValueError('Original formatted prompt/cache tokens differ: '+key)
            if any(token >= len(tokenizer) for token in saved['continuation_token_ids']):
                raise ValueError('Cached token is outside pinned vocabulary')
            exported.append({**anchors[key], **saved, **row, 'formatted_prompt': rendered,
                'cached_teacher_text': tokenizer.decode(saved['continuation_token_ids'], skip_special_tokens=False),
                'cached_teacher_text_without_special_tokens': tokenizer.decode(saved['continuation_token_ids'], skip_special_tokens=True)})
        selected_ids = {r['example_id'] for r in selected}
        with tempfile.TemporaryDirectory(prefix='.B-export-', dir=results) as temporary:
            stage = Path(temporary)
            payloads = {'ranked_original_candidates.jsonl': ranked,
                'ranked_eligible_candidates.jsonl': eligible,
                'excluded_short_responses.jsonl': excluded,
                'eligible_not_selected.jsonl': eligible[128:],
                'original_overlength.jsonl': list(g.rows(skipped_path)),
                'damage_B_anchors.jsonl': exported,
                'selected_teacher_cache.jsonl': [cache[r['example_id']] for r in selected]}
            for name, rows in payloads.items():
                checks.write_jsonl(stage / name, rows)
            with (stage / 'damage_B_review.csv').open('x', newline='', encoding='utf-8-sig') as stream:
                fields = ['selection_rank', 'original_KL_rank', 'example_id', 'vulnerability_score',
                          'response_token_count', 'ended_with_eos', 'prompt', 'cached_teacher_text']
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader()
                for row in exported:
                    display = dict(row)
                    for field in ['example_id', 'prompt', 'cached_teacher_text']:
                        if display[field].lstrip().startswith(('=', '+', '-', '@')):
                            display[field] = "'" + display[field]
                    writer.writerow(display)
            comparison = {'A_count': len(A_ids), 'B_count': len(selected_ids),
                'shared_ids': sorted(A_ids & selected_ids), 'only_A_ids': sorted(A_ids - selected_ids),
                'only_B_ids': sorted(selected_ids - A_ids), 'source_pool_same': True,
                'note': 'Response-length restriction is the intended ablation; no benchmark outcomes used'}
            g.write_new(stage / 'comparison_with_A.json', comparison)
            unchanged = {p: g.sha(Path(p)) == h for p, h in protected.items()}
            if not all(unchanged.values()):
                raise ValueError('Original evidence changed during export')
            summary = {'status': 'complete', 'variant': 'B', 'scope': 'Approved data-only >8-token cached-response selection',
                'input_prompt_count': len(anchors), 'cached_prompt_count': len(ranked), 'original_overlength_count': 26,
                'eligible_prompt_count': len(eligible), 'excluded_short_prompt_count': len(excluded),
                'eligible_not_selected_count': len(eligible)-128, 'selected_prompt_count': 128,
                'response_length_filter': 'cached response_token_count > 8;native EOS counted when present',
                'selected_unique_response_tokens': sum(r['response_token_count'] for r in selected),
                'selected_response_length_histogram': checks.histogram(selected),
                'eligible_response_length_histogram': checks.histogram(eligible),
                'excluded_response_length_histogram': checks.histogram(excluded),
                'selected_non_EOS_token_min': min(r['response_token_count']-int(r['ended_with_eos']) for r in selected),
                'highest_selected_KL': selected[0]['vulnerability_score'], 'lowest_selected_KL': selected[-1]['vulnerability_score'],
                'overlap_with_A_prompt_count': len(A_ids & selected_ids),
                'model': protocol['model'], 'discovery_student_step': 600, 'validation_branch': 'original friend300',
                'source_manifest': manifest, 'source_checksums': protected, 'protected_inputs_unchanged': unchanged,
                'score_completion_metadata_as_supplied': score_meta, 'selection_configuration': cfg,
                'package_manifest_sha256': bundle, 'job_id': os.environ['SLURM_JOB_ID'],
                'node': os.environ.get('SLURMD_NODENAME'), 'python': platform.python_version(),
                'software_versions': software,
                'tokenizer': {'repository_id': protocol['model']['repository_id'],
                    'revision': protocol['model']['revision'], 'use_fast': True, 'add_special_tokens': False,
                    'eos_token_id': tokenizer.eos_token_id, 'vocabulary_size': len(tokenizer)},
                'model_loads': 0, 'generation_calls': 0, 'KL_recomputations': 0, 'training_updates': 0,
                'training_ready': False, 'total_training_anchor_token_budget': None,
                'limitations': ['Longer teacher output does not establish correctness or guarantee retention improvement',
                    'Threshold is an approved exploratory choice,not an optimized boundary',
                    '128 whole prompts is not an equal training-token exposure budget',
                    'Original responses,EOS,KL scores and input text stay unchanged; selection is from training-independent cached inputs',
                    'All future runs must begin independently from the untouched base;do not continue A',
                    'Concrete B smoke/full-training launch remains a separate decision'],
                'elapsed_seconds': time.monotonic()-started}
            summary['export_sha256'] = {p.name: g.sha(p) for p in sorted(stage.iterdir())}
            g.write_new(stage / 'summary.json', summary)
            g.write_new(stage / 'completion.json', {'identity': identity,
                'files_sha256': {p.name: g.sha(p) for p in sorted(stage.iterdir())}})
            os.rename(stage, out)
        print('[COMPLETE] '+str(out), flush=True); print_summary(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true'); args = parser.parse_args()
    if args.preflight:
        cfg = configuration()
        print(json.dumps({'status': 'package_verified_not_run', 'selection': cfg}, indent=2))
    else:
        run()


if __name__ == '__main__':
    main()
