"""Validate the approved combined selection and unchanged A/B training controls."""
from pathlib import Path
import state_io as io
import source_guard
import selection_checks
import rank_fusion

ROOT = Path(__file__).resolve().parent


def verify(root=ROOT):
    source_guard.verify_bundle(root)
    plan = io.read(root / 'plan.json')
    approval = io.read(root / 'execution_approval.json')
    if approval['approved'] is not True or approval['plan_sha256'] != io.sha(root / 'plan.json'):
        raise ValueError('Combined execution approval/plan identity differs')
    p = io.read(root / 'protocol.json'); ref = io.read(root / 'reference_training.json')
    for key in ['model', 'seed', 'lora', 'training', 'runtime', 'evaluation', 'datasets']:
        if p[key] != ref[key]:
            raise ValueError('Matched A/B control changed: ' + key)
    if p['experiment_id'] != plan['plan_id'] or p['kl']['coefficient'] != .01:
        raise ValueError('Combined experiment/coefficient changed')
    if {k: v for k, v in p['kl'].items() if k not in ['coefficient', 'teacher_continuation']} != {
            k: v for k, v in ref['kl'].items() if k not in ['coefficient', 'teacher_continuation']}:
        raise ValueError('Matched KL formulation changed')
    if (p['training']['max_steps'], p['runtime']['anchor_tokens_per_update'],
            p['training']['anchor_interleave_every_steps'], p['anchor_selection']['token_budget']) != (3609, 64, 4, 1453):
        raise ValueError('Approved combined inventory/exposure changed')
    if (p['anchor_selection']['method'], p['anchor_selection']['damage_weight']) != ('damage_diverse', .5):
        raise ValueError('Approved combined selector differs')
    if plan['validation']['updates'] != [0] + list(range(300, 3601, 300)) + [3609]:
        raise ValueError('Validation cadence changed')
    train = io.rows(root / 'inputs/train.jsonl'); val = io.rows(root / 'inputs/validation_original_300.jsonl')
    if (len(train), len(val)) != (9619, 300):
        raise ValueError('Original target split sizes differ')
    return p, plan, train, val


def selection_membership(selected, cache, full, scores, diversity_ranks):
    scored = [{'example_id': r['example_id'], 'vulnerability_score': r['vulnerability_score'],
               'response_token_count': full[r['example_id']]['response_token_count']} for r in scores]
    expected = rank_fusion.select_whole_prompts(scored, diversity_ranks, count=128, damage_weight=.5)
    ids = [r['example_id'] for r in selected]
    if len(selected) != 128 or len(cache) != 128 or len(set(ids)) != 128 or ids != [r['example_id'] for r in expected['selected']]:
        raise ValueError('Combined selection is not the approved50:50 top128')
    if sum(r['response_token_count'] for r in cache) != 1453:
        raise ValueError('Combined unique response-position inventory differs')
    for record, row, ranked in zip(selected, cache, expected['selected']):
        if row['example_id'] != record['example_id'] or row != full[record['example_id']]:
            raise ValueError('Combined cache differs from original teacher')
        if any(record[k] != v for k, v in row.items()) or any(record[k] != v for k, v in ranked.items()):
            raise ValueError('Combined tokens/rank-fusion values differ')
        pids, rids = row['prompt_token_ids'], row['continuation_token_ids']
        if not pids or not 1 <= len(rids) <= 256 or len(rids) != row['response_token_count'] or len(pids) > 1792 or len(pids) + len(rids) > 2048:
            raise ValueError('Combined whole-response/context length differs')
        if rids[-1] != 128001 or 128001 in rids[:-1]:
            raise ValueError('Audited combined nativeEOS alignment differs')
    return expected


def verify_sources(p, plan, root=ROOT):
    _, manifest, repo, exp, canonical, protected = source_guard.guard(root=root)
    a_dir = Path(plan['paths']['a_calibration'])
    reference_files = io.read(root / 'a_calibration_files.sha256.json')
    if io.read(a_dir / 'files.sha256.json') != reference_files:
        raise ValueError('A reference bundle differs')
    for rel, digest in reference_files.items():
        if io.sha(a_dir / rel) != digest:
            raise ValueError('Protected A source changed: ' + rel)
        protected[str(a_dir / rel)] = digest
    d = Path(plan['paths']['selection']); spec = plan['selected_anchors']
    for rel, key in [('identity.json', 'selection_identity_sha256'), ('exports/receipt.json', 'exports_receipt_sha256'),
                     ('diversity-ranking/receipt.json', 'ranking_receipt_sha256'),
                     ('exports/damage_diverse_anchors.jsonl', 'selected_anchors_sha256'),
                     ('exports/damage_diverse_teacher_cache.jsonl', 'selected_cache_sha256'),
                     ('exports/selection_manifests.json', 'selection_manifests_sha256')]:
        if io.sha(d / rel) != spec[key]:
            raise ValueError('Audited combined export changed: ' + rel)
        protected[str(d / rel)] = spec[key]
    identity = io.read(d / 'identity.json'); selection_protocol = io.read(root / 'selection_protocol.json')
    if identity['configuration'] != selection_protocol or identity['source_manifest'] != manifest:
        raise ValueError('Combined selection/source protocol differs')
    for stage in ['diversity-ranking', 'exports']:
        receipt = io.read(d / stage / 'receipt.json')
        if receipt['fingerprint'] != io.digest(identity):
            raise ValueError('Selection stage identity differs')
        names = {f.name for f in (d / stage).iterdir() if f.is_file()}
        if names != set(receipt['files']) | {'receipt.json'}:
            raise ValueError('Selection stage file membership differs')
        for name, digest in receipt['files'].items():
            if Path(name).name != name or io.sha(d / stage / name) != digest:
                raise ValueError('Selection stage payload changed')
            protected[str(d / stage / name)] = digest
    status = io.read(d / 'status.json'); summary = io.read(d / 'exports/summary.json')
    if status['state'] != summary['status'] or summary['status'] != 'complete' or status['summary'] != summary:
        raise ValueError('Selection is not complete')
    if summary['sets']['damage_diverse']['selected_unique_response_positions'] != 1453:
        raise ValueError('Audited combined response inventory differs')
    if io.sha(canonical) != selection_protocol['original_teacher_cache_sha256'] or io.sha(exp / 'full/vulnerability_scores.jsonl') != selection_protocol['original_scores_sha256']:
        raise ValueError('Original cached teacher/score bytes changed')
    for filename, digest in identity['protected_original_files_sha256'].items():
        if io.sha(filename) != digest:
            raise ValueError('Original selection-source evidence changed: ' + filename)
        protected[filename] = digest
    full = {r['example_id']: r for r in io.rows(canonical)}
    source_anchors = {r['example_id']: r for r in io.rows(manifest['input_anchor_file'])}
    raw_scores = io.rows(exp / 'full/vulnerability_scores.jsonl')
    skip = io.rows(exp / 'full/teacher_cache.jsonl.skipped.jsonl')
    ranked, _, _, _ = selection_checks.rank_candidates(list(source_anchors.values()), list(full.values()), raw_scores, skip, count=128)
    if io.digest(ranked) != selection_protocol['canonical_ranking_digest']:
        raise ValueError('Original ranked score snapshot differs')
    selected = io.rows(d / 'exports/damage_diverse_anchors.jsonl')
    cache = io.rows(d / 'exports/damage_diverse_teacher_cache.jsonl')
    ranks = io.rows(d / 'diversity-ranking/ranks.jsonl')
    selection_membership(selected, cache, full, raw_scores, ranks)
    for record in selected:
        if record['prompt'] != source_anchors[record['example_id']]['prompt'] or record['formatted_prompt'] != 'Question: ' + record['prompt'] + '\nAnswer:':
            raise ValueError('Selected raw/formatted prompt differs')
    reference = a_dir / 'results/damage-A-kl-calibration-20261104-001'
    a_identity = io.read(reference / 'common_identity.json')
    a_initial = io.read(reference / 'calibration/damage_001/initialization.json')
    a_order = io.read(reference / 'data_order.json')
    for rel in ['common_identity.json', 'calibration/damage_001/initialization.json', 'data_order.json']:
        protected[str(reference / rel)] = io.sha(reference / rel)
    return cache, selected, protected, {'selection_identity_sha256': spec['selection_identity_sha256'],
            'selected_cache_sha256': spec['selected_cache_sha256'], 'source_manifest': manifest,
            'selector': 'approved50:50 normalized rank fusion'}, a_identity, a_initial, a_order
