"""Unchanged A cache/ranking validators; B applies its approved filter afterward."""
from collections import Counter
import json
import math


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
