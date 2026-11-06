"""Select diverse Math10K anchors from the complete pool with length screening.

Uses the existing protocol's tokenizer, template, generation limits, diversity
encoder and selector. Teacher generation is needed only for selected prompts:
greedy generation always returns at least one token for every length-valid input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .anchor_pipeline import cache_teacher, file_sha256, write_jsonl_exclusive
from .diversity import (
    DiversityConfig, embed_prompts, pool_fingerprint, rank_pool_rows, read_pool,
    select_top_ranked_rows, write_ranked_rows,
)
from .hf_runtime import load_tokenizer
from .selection import Candidate, select_anchors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--pool', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    rows = read_pool(args.pool)
    tokenizer = load_tokenizer(protocol['model'])
    runtime = protocol['runtime']
    template = protocol['evaluation']['prompt_format'].replace('\\n', '\n')
    maximum = runtime['max_sequence_tokens'] - runtime['teacher_max_new_tokens']
    eligible, excluded = [], []
    for row in rows:
        length = len(tokenizer.encode(template.replace('{question}', row['prompt']),
                                      add_special_tokens=False))
        if not length:
            raise ValueError(f"empty tokenized prompt: {row['example_id']}")
        if length > maximum:
            excluded.append(dict(example_id=row['example_id'],
                                 reason='prompt_exceeds_token_limit',
                                 prompt_token_count=length, maximum_prompt_tokens=maximum))
        else:
            eligible.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out = args.output_dir
    write_jsonl_exclusive(out / 'eligible_anchors.jsonl', eligible)
    write_jsonl_exclusive(out / 'excluded_length.jsonl', excluded)
    print(f'Input={len(rows)} eligible={len(eligible)} excluded={len(excluded)}', flush=True)
    settings = protocol['anchor_selection']['diversity']
    config = DiversityConfig(settings['model_id'], settings['model_revision'],
                             settings['algorithm'], settings['seed'], device=runtime['device'])
    vectors = embed_prompts([r['prompt'] for r in eligible], config)
    ranked = rank_pool_rows(eligible, vectors, config)
    write_ranked_rows(ranked, out / 'diversity_ranks.jsonl')
    selected = select_top_ranked_rows(ranked, protocol['anchor_selection']['prompt_budget'])
    selected_path = out / 'diverse_prompts_top128.jsonl'
    write_ranked_rows(selected, selected_path)
    cache_protocol = json.loads(json.dumps(protocol))
    cache_protocol['datasets']['anchors']['eligible_ids_manifest'] = str(selected_path.resolve())
    (out / 'cache_protocol.json').write_text(json.dumps(cache_protocol, indent=2) + '\n')
    cache_path = out / 'selected_teacher_cache.jsonl'
    cache_teacher(cache_protocol, selected_path, cache_path)
    cached = {r['example_id']: r for r in
              (json.loads(line) for line in cache_path.open() if line.strip())}
    if set(cached) != {r['example_id'] for r in selected}:
        raise ValueError('selected teacher cache does not cover every selected prompt')
    allocation = select_anchors(
        [Candidate(r['example_id'], cached[r['example_id']]['response_token_count'],
                   diversity_rank=r['diversity_rank']) for r in selected],
        method='diversity', anchor_token_budget=protocol['anchor_selection']['token_budget'],
        anchor_prompt_budget=len(selected), seed=protocol['seed'])
    allocation['allocation_scope'] = 'already-selected top 128; full selection pool in manifest.json'
    (out / 'response_token_allocation.json').write_text(json.dumps(allocation, indent=2) + '\n')
    manifest = dict(
        source_pool=str(args.pool.resolve()), source_sha256=file_sha256(args.pool),
        input_prompt_count=len(rows), eligible_prompt_count=len(eligible),
        excluded_prompt_count=len(excluded), eligible_pool_sha256=pool_fingerprint(eligible),
        selected_prompt_count=len(selected), selected_sha256=file_sha256(selected_path),
        protocol_source=str(args.protocol.resolve()), protocol_sha256=file_sha256(args.protocol),
        diversity_metadata=selected[0]['diversity_metadata'],
        prompt_template=template, maximum_prompt_tokens=maximum,
        teacher_cache_scope='selected prompts only; length eligibility checked for entire source pool',
        selected_response_positions=sum(r['selected_response_tokens'] for r in allocation['selected']),
        status='complete')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
