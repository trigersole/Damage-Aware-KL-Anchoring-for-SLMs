#!/usr/bin/env python3
"""Approved candidate-vector selection only; no model, generation or training."""
import argparse
from collections import Counter
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import time

import calibration_core as core
import io_utils as io
import rank_fusion as fusion
import selection_checks as old
import source_guard as g
import vector_ranking

ROOT = Path(__file__).resolve().parent


def configuration(root=ROOT):
    g.verify_bundle(root)
    cfg = io.read(root / 'selection.json')
    approval = io.read(root / 'approval.json')
    if not approval['selection_approved'] or approval['training_approved']:
        raise ValueError('Data-only selection approval differs')
    required = {'candidate_count': 13743, 'count': 128, 'seed': 42,
                'damage_weight': .5, 'diversity_weight': .5,
                'response_length_filter': None, 'additional_quality_filter': None,
                'encoder_repo': 'BAAI/bge-m3',
                'encoder_revision': '5617a9f61b028005a4858fdac845db406aefb181',
                'algorithm': 'farthest_first', 'ranking_dtype': 'float64',
                'tie_tolerance': 1e-12, 'teacher_regeneration': False,
                'training_authorized': False}
    if any(cfg.get(k) != v for k, v in required.items()):
        raise ValueError('Approved recipe changed')
    return cfg


def verify_vector_source(cfg, anchors, candidate_ids, budget, root=ROOT):
    import numpy as np
    ident = io.read(root / 'calibration_identity.json')
    fp = io.digest(ident)
    if fp != cfg['calibration_fingerprint']:
        raise ValueError('Pinned original BGE identity differs')
    cache = Path.home() / cfg['calibration_home_relative']
    if io.read(cache / 'identity.json') != ident or io.read(cache / 'status.json')['status'] != 'complete':
        raise ValueError('Original BGE cache identity/completion differs')
    source = Path.home() / ident['config']['default_source_home_relative']
    corpus = core.load_corpus(source, ident['config'])
    if io.read(cache / 'inventory.json') != corpus.inventory():
        raise ValueError('Original BGE inventory differs')
    protected = {str(cache / 'identity.json'): io.file_hash(cache / 'identity.json'),
                 str(cache / 'inventory.json'): io.file_hash(cache / 'inventory.json')}
    for name in ['mapping', 'compiled-embeddings', 'length-audit']:
        budget.check()
        receipt = core.verified_block(cache / name, fp)
        protected[str(cache / name / 'receipt.json')] = io.file_hash(cache / name / 'receipt.json')
        for filename, digest in receipt['files'].items():
            protected[str(cache / name / filename)] = digest
    if io.rows(cache / 'mapping/text_map.jsonl') != list(corpus.mapping()):
        raise ValueError('Original text/vector mapping differs')
    if io.rows(cache / 'length-audit/over_encoder_limit.jsonl'):
        raise ValueError('Original full-text length audit did not pass')
    enc = ident['config']['encoder']
    if (enc['repo_id'], enc['revision'], enc['max_tokens'], enc['dimension'], enc['pooling'], enc['dtype'], enc['normalize_embeddings']) != (
            cfg['encoder_repo'], cfg['encoder_revision'], 8192, 1024, 'CLS', 'float32', True):
        raise ValueError('Pinned full-text BGE encoder policy differs')
    vectors = np.load(cache / 'compiled-embeddings/vectors.npy', mmap_mode='r', allow_pickle=False)
    if vectors.shape != (len(corpus.texts), 1024):
        raise ValueError('Original vector shape differs')
    ids = sorted(candidate_ids)
    indices = []
    for key in ids:
        item = corpus.candidate_by_id.get(key)
        if item is None or item['prompt'] != anchors[key]['prompt']:
            raise ValueError('Candidate ID/text missing or changed in BGE cache: ' + key)
        indices.append(item['text_index'])
    # Only candidate rows enter ranking; reference vectors/texts verify provenance only.
    selected = np.asarray(vectors[indices], dtype=np.float32)
    core.check_vectors(selected, len(ids), 1024)
    mapping = [{'example_id': k, 'text_index': i,
                'prompt_sha256': __import__('hashlib').sha256(anchors[k]['prompt'].encode()).hexdigest()}
               for k, i in zip(ids, indices)]
    encoder_runtimes = []
    for path in sorted(cache.glob('runtime-*.json')):
        protected[str(path)] = io.file_hash(path)
        encoder_runtimes.append({'filename': path.name, 'metadata': io.read(path)})
    receipt = {'calibration_fingerprint': fp, 'candidate_mapping_sha256': io.digest(mapping),
               'candidate_vector_bytes_sha256': __import__('hashlib').sha256(selected.tobytes()).hexdigest(),
               'shape': list(selected.shape), 'encoder': enc,
               'cached_encoder_runtime_records': encoder_runtimes,
               'protected_BGE_files_sha256': protected,
               'reference_vectors_used_for_ranking': False}
    return ids, selected, receipt


def checked_resume(path, identity):
    if not path.exists():
        return None
    saved = io.read(path)
    if saved['identity_sha256'] != identity or saved['state_sha256'] != io.digest(saved['state']):
        raise ValueError('Ranking checkpoint identity/checksum differs')
    return saved['state']


def export_payloads(temp, ranked, ranks, anchors, cache, protocol, cfg, tokenizer, source_receipt, runtime):
    rank_rows = [{'example_id': key, 'diversity_rank': value} for key, value in ranks.items()]
    combined = fusion.select_whole_prompts(ranked, rank_rows, count=cfg['count'], damage_weight=cfg['damage_weight'])
    diversity = fusion.select_whole_prompts(ranked, rank_rows, count=cfg['count'], damage_weight=0.)
    diversity['method'] = 'diversity_farthest_first'
    template = protocol['evaluation']['prompt_format'].replace('\\n', '\n')
    selections = {'diversity_only': diversity, 'damage_diverse': combined}
    summaries = {}
    for mode, selection in selections.items():
        subset = fusion.align_selected_cache(selection, list(cache.values()), sequence_limit=2048)
        exported = []
        for row, saved in zip(selection['selected'], subset):
            key = row['example_id']; rendered = template.replace('{question}', anchors[key]['prompt'])
            if tokenizer.encode(rendered, add_special_tokens=False) != saved['prompt_token_ids']:
                raise ValueError('Cached formatted prompt/token mismatch')
            if any(token >= len(tokenizer) for token in saved['continuation_token_ids']):
                raise ValueError('Cached token outside pinned vocabulary')
            exported.append({**anchors[key], **saved, **row, 'formatted_prompt': rendered,
                             'ended_with_eos': saved['continuation_token_ids'][-1] == tokenizer.eos_token_id,
                             'cached_teacher_text': tokenizer.decode(saved['continuation_token_ids'], skip_special_tokens=False),
                             'cached_teacher_text_without_special_tokens': tokenizer.decode(saved['continuation_token_ids'], skip_special_tokens=True)})
        io.jsonl(temp / f'{mode}_anchors.jsonl', exported)
        io.jsonl(temp / f'{mode}_teacher_cache.jsonl', subset)
        lengths = [r['response_token_count'] for r in exported]
        A_ids = {r['example_id'] for r in ranked[:128]}
        B_ids = {r['example_id'] for r in [r for r in ranked if r['response_token_count'] > 8][:128]}
        chosen = {r['example_id'] for r in exported}
        summaries[mode] = {'selected_prompt_count': len(exported),
                           'selected_unique_response_positions': sum(lengths),
                           'response_length_histogram': dict(sorted(Counter(map(str, lengths)).items(), key=lambda x: int(x[0]))),
                           'responses_ending_native_EOS': sum(r['ended_with_eos'] for r in exported),
                           'minimum_selected_KL': min(r['vulnerability_score'] for r in exported),
                           'maximum_selected_KL': max(r['vulnerability_score'] for r in exported),
                           'overlap_with_A': len(chosen & A_ids), 'overlap_with_B': len(chosen & B_ids),
                           'AB_overlap_basis': 'Membership recomputed from unchanged approved source score order; no benchmark outcomes',
                           'partial_continuations': 0}
    summaries['overlap_between_selected_sets'] = len({r['example_id'] for r in combined['selected']} & {r['example_id'] for r in diversity['selected']})
    io.json_file(temp / 'selection_manifests.json', selections)
    io.jsonl(temp / 'ranked_damage_diverse_candidates.jsonl', fusion.combined_order(ranked, rank_rows, damage_weight=.5))
    io.json_file(temp / 'source_receipt.json', source_receipt)
    summary = {'status': 'complete', 'scope': 'Approved cached-vector data selection only',
               'eligible_prompt_count': len(ranked), 'configuration': cfg, 'sets': summaries,
               'base_model': protocol['model'], 'runtime': runtime,
               'encoder_loads': 0, 'language_model_loads': 0, 'generation_calls': 0, 'training_updates': 0,
               'training_ready': False,
               'limitations': ['Selection weights are prespecified,not optimized using test outcomes.',
                   'Weighted rank fusion is not MMR and does not guarantee pairwise diversity.',
                   'BGE diversity differs from teammate MiniLM;use matching diversity-only control or disclose the confound.',
                   '128 whole prompts does not fix unique response positions or repeated training exposure.',
                   'Teacher responses are preserved,not certified factually correct.',
                   'Concrete training/smoke/evaluation execution remains a later approved stage.']}
    io.json_file(temp / 'summary.json', summary)
    return summary


def run(cfg, seconds, root=ROOT):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Submit via Slurm; no computation on the head node')
    import torch
    if not torch.cuda.is_available():
        raise ValueError('Allocated CUDA GPU required for ranking')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = root / 'results' / cfg['experiment_id']
    runtime = {'job': os.environ['SLURM_JOB_ID'], 'node': os.environ.get('SLURMD_NODENAME'),
               'gpu': torch.cuda.get_device_name(0), 'python': platform.python_version(),
               'packages': {name: metadata.version(name) for name in ['torch', 'numpy', 'transformers', 'tokenizers']},
               'ranking_dtype': 'float64', 'TF32': False}
    with io.lease(output), io.Budget(seconds) as budget:
        try:
            protocol, manifest, repo, exp, cache_path, protected = g.guard(root=root)
            for relative, digest in manifest['code_sha256'].items():
                protected[str(repo / relative)] = digest
            score_path = exp / 'full/vulnerability_scores.jsonl'
            if g.sha(cache_path) != cfg['original_teacher_cache_sha256'] or g.sha(score_path) != cfg['original_scores_sha256']:
                raise ValueError('Original teacher cache/score bytes differ from audited A/B evidence')
            skipped = exp / 'full/teacher_cache.jsonl.skipped.jsonl'
            score_meta = exp / 'full/vulnerability_scores.jsonl.meta.json'
            if g.sha(skipped) != io.read(exp / 'full/teacher_cache.jsonl.meta.json')['skipped_anchors_sha256']:
                raise ValueError('Original exclusions changed')
            for path in [skipped, score_meta]:
                protected[str(path)] = g.sha(path)
            old.verify_score_metadata(io.read(score_meta), score_sha=g.sha(score_path), cache_sha=g.sha(cache_path), manifest=manifest)
            ranked, _, anchors, cache = old.rank_candidates(list(g.rows(Path(manifest['input_anchor_file']))),
                list(g.rows(cache_path)), list(g.rows(score_path)), list(g.rows(skipped)), count=128)
            if io.digest(ranked) != cfg['canonical_ranking_digest']:
                raise ValueError('Original audited KL ranking changed')
            payload = [(r['example_id'], r['response_token_count']) for r in sorted(ranked, key=lambda r: r['example_id'])]
            pool_sha = __import__('hashlib').sha256(json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
            if pool_sha != cfg['pool_ID_response_length_fingerprint']:
                raise ValueError('Common pool IDs/response lengths changed')
            print('[VERIFY] Original13743 teacher/cache/score rows verified; no regeneration', flush=True)
            ids, vectors, vector_receipt = verify_vector_source(cfg, anchors, set(cache), budget, root)
            print('[VERIFY] Candidate-only full-text BGE vectors mapped and verified', flush=True)
            identity = {'configuration': cfg, 'package_manifest_sha256': g.sha(root / 'files.sha256.json'),
                        'source_manifest': manifest, 'protected_original_files_sha256': protected,
                        'vector_receipt': vector_receipt}
            fp = io.digest(identity)
            if (output / 'identity.json').exists():
                if io.read(output / 'identity.json') != identity:
                    raise ValueError('Existing output identity differs; preserve and use a new ID')
            else:
                io.json_file(output / 'identity.json', identity)
            io.atomic_json(output / f'runtime-{runtime["job"]}.json', runtime)
            ranking_folder = output / 'diversity-ranking'
            if ranking_folder.exists():
                core.verified_block(ranking_folder, fp)
                rank_rows = io.rows(ranking_folder / 'ranks.jsonl')
            else:
                checkpoint = output / 'ranking_checkpoint.json'
                resume = checked_resume(checkpoint, fp)
                print(f'[RANK] Resume {len(resume["prefix_indices"]) if resume else 0}/{len(ids)}', flush=True)
                def publish(state):
                    io.atomic_json(checkpoint, {'identity_sha256': fp, 'state': state, 'state_sha256': io.digest(state)})
                    print(f'[RANK] {len(state["prefix_indices"])}/{len(ids)}', flush=True)
                ranks = vector_ranking.farthest_first(ids, vectors, seed=cfg['seed'], device='cuda:0',
                    check=budget.check, publish=publish, resume=resume, every=128)
                def save_ranking(temp):
                    io.jsonl(temp / 'ranks.jsonl', [{'example_id': key, 'diversity_rank': ranks[key]} for key in ids])
                    io.json_file(temp / 'provenance.json', vector_receipt)
                core.block(output, 'diversity-ranking', fp, save_ranking, budget)
                rank_rows = io.rows(ranking_folder / 'ranks.jsonl')
            if {r['example_id'] for r in rank_rows} != set(cache):
                raise ValueError('Saved ranking/common pool mismatch')
            fusion.combined_order(ranked, rank_rows, damage_weight=cfg['damage_weight'])
            ranks = {r['example_id']: r['diversity_rank'] for r in rank_rows}
            if not (output / 'exports').exists():
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(protocol['model']['repository_id'], revision=protocol['model']['revision'], use_fast=True, local_files_only=True)
                if tokenizer.eos_token_id != 128001:
                    raise ValueError('Pinned native EOS differs')
                def export(temp):
                    export_payloads(temp, ranked, ranks, anchors, cache, protocol, cfg, tokenizer, identity, runtime)
                core.block(output, 'exports', fp, export, budget)
            else:
                core.verified_block(output / 'exports', fp)
            for name, digest in {**protected, **vector_receipt['protected_BGE_files_sha256']}.items():
                budget.check()
                if g.sha(Path(name)) != digest:
                    raise ValueError('Original source/vector file changed during selection')
            status = {'state': 'complete', 'experiment_id': cfg['experiment_id'],
                      'output': str(output), 'summary': io.read(output / 'exports/summary.json')}
            io.atomic_json(output / 'status.json', status)
            print(json.dumps(status, indent=2), flush=True)
        except io.Pause:
            io.atomic_json(output / 'status.json', {'state': 'paused', 'resume': 'Repeat the same sbatch command after this job ends; cached stages/ranking checkpoint are reused'})
            print('[PAUSED] Saved progress; repeat the same submission', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Explicit alias; saved stages/checkpoints always resume')
    parser.add_argument('--seconds', type=float, default=3300.)
    args = parser.parse_args()
    cfg = configuration()
    if args.preflight:
        print(json.dumps({'state': 'approved_package_verified_not_run', 'configuration': cfg}, indent=2))
    else:
        run(cfg, args.seconds)


if __name__ == '__main__':
    main()
