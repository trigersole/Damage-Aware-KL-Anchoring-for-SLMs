"""Verify the supplied Random selection and matched A/B training inputs."""

from pathlib import Path

import random_selection_preview
import source_guard
import state_io as io


ROOT = Path(__file__).resolve().parent
MANIFEST_NAME = 'random_whole_response_preview.json'


def verify(root=ROOT, *, require_approval=True):
    source_guard.verify_bundle(root)
    plan_path = root / 'plan.json'
    plan = io.read(plan_path)
    approval = io.read(root / 'execution_approval.json')
    if approval['plan_sha256'] != io.sha(plan_path):
        raise ValueError('Random execution approval has a different plan identity')
    if require_approval and approval['approved'] is not True:
        raise ValueError('Review and approve the matched Random plan before model work')

    protocol = io.read(root / 'protocol.json')
    reference = io.read(root / 'reference_training.json')
    for key in ['model', 'seed', 'lora', 'training', 'runtime', 'evaluation', 'datasets']:
        if protocol[key] != reference[key]:
            raise ValueError('Matched A/B control changed: ' + key)
    kl = protocol['kl']
    expected_kl = {**reference['kl'], 'coefficient': .01,
                   'teacher_continuation': plan['paths']['teacher_cache']}
    if kl != expected_kl or protocol['experiment_id'] != plan['plan_id']:
        raise ValueError('Random experiment or matched KL controls differ')
    if (protocol['training']['max_steps'], protocol['runtime']['anchor_tokens_per_update'],
            protocol['training']['anchor_interleave_every_steps']) != (3609, 64, 4):
        raise ValueError('Matched 57,728-position schedule differs')

    spec = plan['selected_anchors']
    if (spec['method'], spec['seed'], spec['pool_count'], spec['prompt_count'],
            spec['unique_response_positions']) != ('random', 42, 13743, 128, 22100):
        raise ValueError('Supplied Random selection controls differ')
    if protocol['anchor_selection'] != {
        'method': 'random', 'token_budget': spec['unique_response_positions'],
        'damage_weight': None, 'candidate_pool_manifest': plan['paths']['teacher_cache'],
        'selected_ids_manifest': MANIFEST_NAME,
    }:
        raise ValueError('Random selection paths or inventory differ')
    if io.sha(root / MANIFEST_NAME) != spec['manifest_sha256']:
        raise ValueError('Supplied Random manifest changed')
    if plan['validation']['updates'] != [0, *range(300, 3601, 300), 3609]:
        raise ValueError('Matched validation cadence differs')

    train = io.rows(root / 'inputs/train.jsonl')
    val = io.rows(root / 'inputs/validation_original_300.jsonl')
    if (len(train), len(val)) != (9619, 300):
        raise ValueError('Original target split sizes differ')
    return protocol, plan, train, val


def bind_selection(supplied, cache, spec):
    """Reproduce the teammate's full-response selection from canonical cache rows."""
    expected = random_selection_preview.preview(
        cache, seed=spec['seed'], prompt_count=spec['prompt_count'],
        expected_pool_count=spec['pool_count'])
    expected['teacher_cache_file_sha256'] = spec['teacher_cache_sha256']
    if supplied != expected or expected['eligible_pool_sha256'] != spec['pool_sha256']:
        raise ValueError('Supplied Random IDs, ranks or masks differ from the seeded selection')
    if expected['anchor_token_budget'] != spec['unique_response_positions']:
        raise ValueError('Whole-response Random inventory differs')
    rows_by_id = {row['example_id']: row for row in cache}
    selected = [rows_by_id[row['example_id']] for row in supplied['selected']]
    if (sum(row['continuation_token_ids'][-1] == 128001 for row in selected),
            sum(row['response_token_count'] == 256 for row in selected)) != (
                spec['native_eos_count'], spec['generation_cap_count']):
        raise ValueError('Random native EOS or generation-cap counts differ')
    return selected


def verify_selection(plan, root=ROOT):
    _, original_manifest, _, _, canonical, protected = source_guard.guard(root=root)
    spec = plan['selected_anchors']
    if str(canonical) != plan['paths']['teacher_cache'] or protected[str(canonical)] != spec['teacher_cache_sha256']:
        raise ValueError('Random must use the pinned canonical teacher cache')
    manifest_path = root / MANIFEST_NAME
    if io.sha(manifest_path) != spec['manifest_sha256']:
        raise ValueError('Supplied Random manifest changed')
    cache = bind_selection(io.read(manifest_path), io.rows(canonical), spec)
    original_prompts = {row['example_id']: row for row in io.rows(original_manifest['input_anchor_file'])}
    selected = [original_prompts[row['example_id']] for row in cache]
    protected[str(manifest_path)] = spec['manifest_sha256']
    identity = {**spec, 'source_manifest': original_manifest,
                'selected_cache_rows_sha256': io.digest(cache)}
    return cache, selected, protected, identity


def verify_sources(protocol, plan, root=ROOT):
    cache, selected, protected, identity = verify_selection(plan, root)
    calibration = Path(plan['paths']['a_calibration'])
    reference_files = io.read(root / 'a_calibration_files.sha256.json')
    if io.read(calibration / 'files.sha256.json') != reference_files:
        raise ValueError('A reference bundle differs')
    for rel, digest in reference_files.items():
        if io.sha(calibration / rel) != digest:
            raise ValueError('Protected A source changed: ' + rel)
        protected[str(calibration / rel)] = digest

    reference = calibration / 'results/damage-A-kl-calibration-20261104-001'
    paths = [reference / rel for rel in [
        'common_identity.json', 'calibration/damage_001/initialization.json', 'data_order.json']]
    protected.update({str(path): io.sha(path) for path in paths})
    ref_identity, ref_initial, ref_order = (io.read(path) for path in paths)
    return cache, selected, protected, identity, ref_identity, ref_initial, ref_order
