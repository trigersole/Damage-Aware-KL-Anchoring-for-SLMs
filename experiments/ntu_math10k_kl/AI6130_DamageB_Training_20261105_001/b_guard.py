"""Read-only validation of the approved B selection and unchanged A controls."""
from pathlib import Path
import state_io as io
import source_guard

ROOT = Path(__file__).resolve().parent


def verify(root=ROOT):
    source_guard.verify_bundle(root)
    plan = io.read(root / 'plan.json')
    approval = io.read(root / 'execution_approval.json')
    if approval['approved'] is not True or approval['plan_sha256'] != io.sha(root / 'plan.json'):
        raise ValueError('B approval/plan identity differs')
    p = io.read(root / 'protocol.json'); ref = io.read(root / 'reference_training.json')
    for key in ['model', 'seed', 'lora', 'training', 'runtime', 'evaluation', 'datasets']:
        if p[key] != ref[key]:
            raise ValueError('Matched A control changed: ' + key)
    if p['experiment_id'] != plan['plan_id'] or p['kl']['coefficient'] != .01:
        raise ValueError('B experiment/coefficient changed')
    if {k: v for k, v in p['kl'].items() if k not in ['coefficient', 'teacher_continuation']} != {
            k: v for k, v in ref['kl'].items() if k not in ['coefficient', 'teacher_continuation']}:
        raise ValueError('Matched A KL formulation changed')
    if (p['training']['max_steps'], p['runtime']['anchor_tokens_per_update'],
            p['training']['anchor_interleave_every_steps'], p['anchor_selection']['token_budget']) != (3609, 64, 4, 1749):
        raise ValueError('Approved B budget changed')
    if plan['validation']['updates'] != [0] + list(range(300, 3601, 300)) + [3609]:
        raise ValueError('Approved validation cadence changed')
    train = io.rows(root / 'inputs/train.jsonl'); val = io.rows(root / 'inputs/validation_original_300.jsonl')
    if (len(train), len(val)) != (9619, 300):
        raise ValueError('Original target split sizes differ')
    return p, plan, train, val


def selection_membership(selected, cache, full, scores):
    eligible = [r for r in scores if full[r['example_id']]['response_token_count'] > 8]
    ranked = sorted(eligible, key=lambda r: (-r['vulnerability_score'], r['example_id']))[:128]
    if len(eligible) != 12838 or len(selected) != 128 or len(cache) != 128:
        raise ValueError('B eligibility/count differs')
    ids = [r['example_id'] for r in selected]
    if ids != [r['example_id'] for r in ranked] or len(set(ids)) != 128:
        raise ValueError('B is not the unchanged >8-token top128')
    if sum(r['response_token_count'] for r in cache) != 1749:
        raise ValueError('B unique response-position budget differs')
    for rank, (record, row, score) in enumerate(zip(selected, cache, ranked), 1):
        if row != full[record['example_id']] or row['example_id'] != record['example_id']:
            raise ValueError('B cache differs from original teacher bytes')
        if any(record[k] != v for k, v in row.items()):
            raise ValueError('B selected tokens differ')
        if record['selection_rank'] != rank or record['vulnerability_score'] != score['vulnerability_score']:
            raise ValueError('B ranking/score differs')
        response = row['continuation_token_ids']; prompt = row['prompt_token_ids']
        if len(response) != row['response_token_count'] or len(response) <= 8 or not prompt:
            raise ValueError('B response length differs')
        if response[-1] != 128001 or len(prompt) + len(response) > 2048:
            raise ValueError('Audited B EOS/context differs')


def verify_sources(p, plan, root=ROOT):
    _, manifest, repo, exp, canonical, protected = source_guard.guard(root=root)
    a_dir = Path(plan['paths']['a_calibration'])
    expected = io.read(root / 'a_calibration_files.sha256.json')
    if io.read(a_dir / 'files.sha256.json') != expected:
        raise ValueError('A reference bundle differs')
    for rel, h in expected.items():
        if io.sha(a_dir / rel) != h:
            raise ValueError('Protected A source changed: ' + rel)
        protected[str(a_dir / rel)] = h
    d = Path(plan['paths']['selection']); spec = plan['selected_anchors']
    for name, key in [('completion.json', 'selection_completion_sha256'),
                      ('damage_B_anchors.jsonl', 'selected_anchors_sha256'),
                      ('selected_teacher_cache.jsonl', 'selected_cache_sha256')]:
        if io.sha(d / name) != spec[key]:
            raise ValueError('Audited B selection changed: ' + name)
    completion = io.read(d / 'completion.json')
    for name, h in completion['files_sha256'].items():
        if Path(name).name != name or io.sha(d / name) != h:
            raise ValueError('B export receipt changed: ' + name)
        protected[str(d / name)] = h
    protected[str(d / 'completion.json')] = io.sha(d / 'completion.json')
    summary = io.read(d / 'summary.json')
    if summary['status'] != 'complete' or summary['variant'] != 'B' or summary['source_manifest'] != manifest:
        raise ValueError('B source provenance differs')
    for name, h in summary['source_checksums'].items():
        if io.sha(name) != h:
            raise ValueError('Original B evidence changed: ' + name)
        protected[name] = h
    selected = io.rows(d / 'damage_B_anchors.jsonl')
    cache = io.rows(d / 'selected_teacher_cache.jsonl')
    full = {r['example_id']: r for r in io.rows(canonical)}
    selection_membership(selected, cache, full, io.rows(exp / 'full/vulnerability_scores.jsonl'))
    reference = a_dir / 'results/damage-A-kl-calibration-20261104-001'
    a_identity = io.read(reference / 'common_identity.json')
    a_initial = io.read(reference / 'calibration/damage_001/initialization.json')
    a_order = io.read(reference / 'data_order.json')
    for f in ['common_identity.json', 'calibration/damage_001/initialization.json', 'data_order.json']:
        protected[str(reference / f)] = io.sha(reference / f)
    return cache, selected, protected, {'selection_completion_sha256': io.sha(d / 'completion.json'),
            'selected_cache_sha256': io.sha(d / 'selected_teacher_cache.jsonl'), 'source_manifest': manifest}, a_identity, a_initial, a_order
