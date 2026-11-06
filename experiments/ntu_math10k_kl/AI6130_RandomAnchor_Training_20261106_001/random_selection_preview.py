"""Inspect a canonical cache and preview random whole-prompt selection; no training."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def load_cache(path, expected_sha256):
    if sha(path) != expected_sha256:
        raise ValueError('Canonical teacher cache checksum differs; do not mix generation protocols')
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def preview(cache, *, seed, prompt_count, expected_pool_count):
    if type(seed) is not int or type(prompt_count) is not int or not 0 < prompt_count <= len(cache):
        raise ValueError('Explicit integer seed and valid prompt count required')
    if len(cache) != expected_pool_count:
        raise ValueError('Eligible pool count differs; reconcile inputs before selection')
    by_id = {}
    for row in cache:
        key = row.get('example_id'); prompt = row.get('prompt_token_ids'); response = row.get('continuation_token_ids')
        if not isinstance(key, str) or not key or key in by_id:
            raise ValueError('Invalid or repeated candidate ID')
        for tokens in [prompt, response]:
            if not isinstance(tokens, list) or not tokens or any(type(x) is not int or x < 0 for x in tokens):
                raise ValueError('Invalid cached token IDs')
        if type(row.get('response_token_count')) is not int or row['response_token_count'] != len(response):
            raise ValueError('Cached response-position count differs')
        if len(prompt) > 1792 or len(response) > 256 or len(prompt) + len(response) > 2048:
            raise ValueError('Canonical generation/sequence limit differs')
        if 128001 in response[:-1]:
            raise ValueError('Interior native EOS in cached continuation')
        by_id[key] = row
    order = sorted(by_id)
    random.Random(seed).shuffle(order)
    chosen = order[:prompt_count]
    # This preview explicitly preserves whole responses. It is not a255-position prefix policy.
    selected = [{'example_id': key, 'selection_rank': i,
                 'available_response_tokens': by_id[key]['response_token_count'],
                 'selected_response_tokens': by_id[key]['response_token_count']}
                for i, key in enumerate(chosen, 1)]
    pool = [(key, by_id[key]['response_token_count']) for key in sorted(by_id)]
    fingerprint = hashlib.sha256(json.dumps(pool, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    return {'status': 'whole_response_preview_only', 'method': 'random', 'seed': seed,
            'eligible_pool_sha256': fingerprint, 'eligible_prompt_count': len(cache),
            'anchor_prompt_budget': prompt_count, 'selected_prompt_count': len(selected),
            'anchor_token_budget': sum(r['selected_response_tokens'] for r in selected),
            'scored_position_policy': 'all original cached response positions, including EOS when present',
            'selected': selected, 'training_authorized': False,
            'note': 'Pool/selection/scoring-position recipe requires team agreement before final export or training; no255-position clipping inferred.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--expected-cache-sha256', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--prompt-count', type=int, required=True)
    parser.add_argument('--expected-pool-count', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New preview path; refuses overwrite')
    args = parser.parse_args()
    result = preview(load_cache(args.cache, args.expected_cache_sha256), seed=args.seed,
                     prompt_count=args.prompt_count, expected_pool_count=args.expected_pool_count)
    result['teacher_cache_file_sha256'] = sha(args.cache)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'selected'}, indent=2))


if __name__ == '__main__':
    main()
