"""Proposed isolated lambda0.01 continuation. Training requires explicit approval."""
import argparse
import copy
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def preflight(root=ROOT):
    for name, expected in read(root / 'files.sha256.json').items():
        if sha(root / name) != expected:
            raise ValueError('Continuation package changed: ' + name)
    p = read(root / 'plan.json')
    if (p['coefficient'], p['parent_updates'], p['total_updates'], p['additional_updates']) != (.01, 600, 3609, 3009):
        raise ValueError('Proposed coefficient or update budget changed')
    if (p['anchor_every_updates'], p['anchor_positions_per_event'], p['total_scored_anchor_positions'],
        p['additional_scored_anchor_positions']) != (4, 64, 57728, 48128):
        raise ValueError('Proposed anchor exposure changed')
    if p['validation_steps'] != [0] + list(range(300, 3601, 300)) + [3609]:
        raise ValueError('Proposed validation/checkpoint cadence changed')
    return p


def approval(root):
    path = root / 'execution_approval.json'
    if not path.exists():
        raise ValueError('Full3609-update budget is not approved. Review plan.json and obtain confirmation first.')
    a = read(path)
    if a.get('approved') is not True or a['plan_sha256'] != sha(root / 'plan.json'):
        raise ValueError('Approval does not match the exact continuation plan')
    return a


def state_digest(io, model, optimizer, scheduler, progress):
    return io.tree_hash({'parameters': io.parameters(model), 'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(), 'progress': progress, 'rng': io.rng_state()})


def handoff(io, folder, parent, model, optimizer, scheduler, parent_identity, identity, *, expected_step=600):
    """Read parent state without modifying it; independently commit and verify new state."""
    verified = folder / 'handoff_verified.json'
    if (folder / 'recovery/current.json').exists():
        progress = io.load(folder, model, optimizer, scheduler, identity)
        if not verified.exists():
            origin = read(folder / 'origin.json')
            if progress['step'] != expected_step or state_digest(io, model, optimizer, scheduler, progress) != origin['state_sha256']:
                raise ValueError('Interrupted handoff cannot be verified')
            io.freeze(verified, {'status': 'complete', 'state_sha256': origin['state_sha256'],
                                 'parameters_optimizer_scheduler_RNG_progress_equal': True})
        return progress
    progress = io.load(parent, model, optimizer, scheduler, parent_identity)
    if progress['step'] != expected_step or progress['phase'] != 'train' or scheduler.last_epoch != expected_step:
        raise ValueError('Parent is not the completed600-update recovery state')
    before = state_digest(io, model, optimizer, scheduler, progress)
    io.freeze(folder / 'origin.json', {'parent': str(parent), 'parent_identity': parent_identity,
              'parent_recovery': read(parent / 'recovery/current.json'), 'state_sha256': before,
              'inherited_validation_steps': [r['step'] for r in progress['validation_history']],
              'inherited_training_updates': expected_step})
    io.save(folder, model, optimizer, scheduler, progress, identity)
    progress = io.load(folder, model, optimizer, scheduler, identity)
    if state_digest(io, model, optimizer, scheduler, progress) != before:
        raise ValueError('Continuation handoff changed parameters, optimizer, scheduler, RNG or progress')
    io.freeze(verified, {'status': 'complete', 'state_sha256': before,
                        'parameters_optimizer_scheduler_RNG_progress_equal': True})
    return progress


def check_progress(progress, batches, limit):
    step = progress['step']
    if not 600 <= step <= limit or len(progress['history']) != step:
        raise ValueError('Continuation step/history differs')
    if [x['step'] for x in progress['history']] != list(range(1, step + 1)):
        raise ValueError('Non-contiguous inherited/continued history')
    if progress['anchor_tokens_seen'] != step // 4 * 64 or progress['anchor_cursor'] != (step // 4 * 64) % 255:
        raise ValueError('Continuation anchor budget/cursor differs')
    if progress['examples_seen'] != sum(len(group) for _, group in batches[:step]):
        raise ValueError('Continuation target cursor differs')
    if progress['response_tokens_seen'] != sum(sum(ex.response_mask[1:]) for _, group in batches[:step] for ex in group):
        raise ValueError('Continuation target-token count differs')


def verify_checkpoint(path, identity):
    receipt = read(path / 'receipt.json')
    step = int(path.name.removeprefix('step_'))
    if receipt['identity'] != identity or receipt['step'] != step or any(sha(path / f) != h for f, h in receipt['files'].items()):
        raise ValueError('Scientific checkpoint receipt/weights differ: ' + str(path))
    return receipt


def select_best(io, folder, parent, parent_identity, identity, progress):
    # Step0 is an untouched reference, not a trained LoRA checkpoint.
    selected = min((r for r in progress['validation_history'] if r['step'] > 0),
                   key=lambda r: (r['validation_loss'], r['step']))
    owner = parent if selected['step'] <= 600 else folder
    path = owner / 'checkpoints' / ('step_%06d' % selected['step'])
    verify_checkpoint(path, parent_identity if owner == parent else identity)
    result = {'step': selected['step'], 'validation_loss': selected['validation_loss'],
              'checkpoint': str(path), 'metric': 'original300 response-token-weighted reference loss includingEOS',
              'exact_tie_break': 'earlier step', 'official_test_used': False}
    io.atomic(folder / 'best_model.json', result)
    return result


def train_loop(io, old, folder, parent, model, teacher, optimizer, scheduler, tokenizer,
               p, plan, train, val, cache, progress, identity, parent_identity, budget):
    from calibration_core import update
    import legacy
    batches = list(legacy.epoch_batches(train, p['seed'], 3, 8))
    if len(batches) != 3609:
        raise ValueError('Original three-epoch ordering differs')
    check_progress(progress, batches, plan['total_updates'])
    started = time.monotonic()
    while not budget.expired():
        if progress['phase'] == 'validate':
            io.checkpoint(folder, model, tokenizer, progress['step'], identity)
            result = old.validate(folder, model, teacher, val, cache, tokenizer, p,
                                  progress['step'], identity, budget)
            if result is None:
                break
            progress['validation_history'].append(result)
            progress['phase'] = 'train'
            io.save(folder, model, optimizer, scheduler, progress, identity)
            select_best(io, folder, parent, parent_identity, identity, progress)
        if progress['step'] == plan['total_updates'] or budget.expired():
            break
        epoch, group = batches[progress['step']]
        metrics, cursor = update(model, teacher, group, optimizer, scheduler, p, tokenizer,
                                 cache, progress['anchor_cursor'], .01, progress['step'] + 1)
        progress['step'] += 1
        progress['anchor_cursor'] = cursor
        progress['examples_seen'] += len(group)
        progress['response_tokens_seen'] += metrics['target_tokens']
        progress['anchor_tokens_seen'] += metrics['anchor_tokens']
        metrics.update(coefficient=.01, examples_seen=progress['examples_seen'],
                       anchor_tokens_seen=progress['anchor_tokens_seen'], epoch=epoch,
                       segment_seconds=time.monotonic()-started)
        progress['history'].append(metrics)
        step = progress['step']
        if step % 10 == 0 or step == plan['total_updates']:
            print(f'[TRAIN] damage_001 {step}/3609 math={metrics["target_loss"]:.6f} '
                  f'KL={metrics["anchor_kl"]} anchor_positions={progress["anchor_tokens_seen"]}', flush=True)
        if step in plan['validation_steps']:
            progress['phase'] = 'validate'
            io.save(folder, model, optimizer, scheduler, progress, identity)
        elif step % 50 == 0:
            io.save(folder, model, optimizer, scheduler, progress, identity)
    check_progress(progress, batches, plan['total_updates'])
    io.save(folder, model, optimizer, scheduler, progress, identity)
    complete = progress['step'] == plan['total_updates'] and progress['phase'] == 'train'
    if complete and [r['step'] for r in progress['validation_history']] != plan['validation_steps']:
        raise ValueError('Full-training validation history is incomplete')
    best = select_best(io, folder, parent, parent_identity, identity, progress)
    status = {'status': 'complete' if complete else 'paused', 'step': progress['step'],
              'planned_steps': 3609, 'coefficient': .01, 'phase': progress['phase'],
              'examples_seen': progress['examples_seen'], 'target_response_tokens_seen': progress['response_tokens_seen'],
              'anchor_positions_seen': progress['anchor_tokens_seen'], 'inherited_updates': 600,
              'best_by_original300_reference_loss': best, 'generation_calls': 0,
              'validation_history': progress['validation_history'], 'identity': identity}
    io.atomic(folder / 'status.json', status)
    if complete:
        io.freeze(folder / 'completion.json', status)
    return status


def run(root=ROOT):
    plan = preflight(root)
    approved = approval(root)  # Refuse before importing model libraries or touching parent state.
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Training requires a Slurm allocation; do not compute on the head node')
    parent_root = Path(plan['parent_package'])
    expected = read(root / 'parent_files.sha256.json')
    if sha(parent_root / 'files.sha256.json') != plan['parent_bundle_sha256'] or expected != read(parent_root / 'files.sha256.json'):
        raise ValueError('Wrong immutable calibration parent package')
    for f, h in expected.items():
        if sha(parent_root / f) != h:
            raise ValueError('Parent calibration source/data changed: ' + f)
    sys.path.insert(0, str(parent_root))
    old = importlib.import_module('calibrate')
    io = importlib.import_module('state_io')
    budget = io.Budget(plan['internal_wall_seconds'])
    for number in [signal.SIGUSR1, signal.SIGTERM, signal.SIGINT]:
        signal.signal(number, budget.signal)
    p0, train_rows, val_rows = old.verify_local(parent_root)
    cache, _, protected, source = old.verify_anchors(p0, parent_root)
    original = parent_root / 'results' / plan['parent_experiment_id']
    parent = original / 'calibration/damage_001'
    case = read(parent / 'status.json')
    wanted = read(root / 'expected_parent_result.json')
    if (case['status'], case['condition'], case['coefficient'], case['step'], case['phase'], case['anchor_tokens_seen']) != ('complete', 'damage_001', .01, 600, 'train', 9600):
        raise ValueError('Parent lambda0.01 calibration is not complete at600')
    if case['validation_history'] != wanted['validation_history']:
        raise ValueError('Parent differs from the reviewed calibration result')
    if read(original / 'calibration/summary.json')['status'] != 'complete':
        raise ValueError('Original three-condition calibration incomplete')
    parent_identity = case['identity']
    if read(parent / 'identity.json') != parent_identity or parent_identity['bundle'] != plan['parent_bundle_sha256']:
        raise ValueError('Parent condition identity differs')
    if (parent_identity['condition'], parent_identity['coefficient'], parent_identity['scope']) != ('damage_001', .01, 'calibration'):
        raise ValueError('Wrong inherited condition')
    if parent_identity['source'] != source or parent_identity['protocol'] != io.digest(p0):
        raise ValueError('Inherited source/protocol does not match original calibration')
    pointer = read(parent / 'recovery/current.json')
    state = parent / 'recovery' / pointer['generation'] / 'state.pt'
    if Path(pointer['generation']).name != pointer['generation'] or not pointer['generation'].startswith('recovery-') or sha(state) != pointer['sha256']:
        raise ValueError('Parent recovery pointer/checksum differs')
    for f in ['identity.json', 'status.json', 'initialization.json', 'recovery/current.json', 'training_history.json', 'validation_history.json']:
        protected[str(parent / f)] = sha(parent / f)
    protected[str(state)] = sha(state)
    for f, h in expected.items():
        protected[str(parent_root / f)] = h
    import torch
    import legacy
    if not torch.cuda.is_available():
        raise ValueError('Allocated CUDA GPU required')
    runtime = old.runtime_info(torch)
    if runtime != parent_identity['runtime']:
        raise ValueError('Resume on the same GPU model/software/precision controls as calibration: ' + str(parent_identity['runtime']))
    folder = root / 'results' / plan['plan_id']
    identity = {'plan_sha256': sha(root / 'plan.json'), 'bundle_sha256': sha(root / 'files.sha256.json'),
                'parent_identity': parent_identity, 'parent_state_sha256': pointer['sha256'], 'runtime': runtime}
    with io.lease(folder):
        io.freeze(folder / 'identity.json', identity)
        io.freeze(folder / 'execution_approval.json', approved)
        if (folder / 'completion.json').exists():
            print(json.dumps({'status': 'already_complete', 'output': str(folder)})); return
        io.atomic(folder / 'jobs' / (os.environ['SLURM_JOB_ID'] + '.json'),
                  {'runtime': runtime, 'node': os.environ.get('SLURMD_NODENAME'), 'identity': identity})
        p = copy.deepcopy(p0)
        p['experiment_id'] = plan['plan_id']; p['kl']['coefficient'] = .01
        p['calibration'].update(steps=3609, conditions={'damage_001': .01}, validation_steps=plan['validation_steps'])
        p['evaluation']['checkpoint_selection_rule'] = plan['selection_rule']
        io.freeze(folder / 'effective_protocol.json', p)
        legacy.set_seed(p['seed']); tokenizer = legacy.load_tokenizer(p['model'])
        train = legacy.make_examples(tokenizer, train_rows, p)
        val = legacy.make_examples(tokenizer, val_rows, p)
        batches = list(legacy.epoch_batches(train, p['seed'], 3, 8))
        data_order = io.digest([[x.example_id for x in b] for _, b in batches[:600]])
        if data_order != read(original / 'data_order.json')['first600_target_batch_ID_sha256']:
            raise ValueError('Target order differs from the inherited prefix')
        if len(batches) != 3609 or sum(len(b) for _, b in batches) != 28857 or sum(sum(x.response_mask[1:]) for x in val) != 39614:
            raise ValueError('Exact-epoch data/validation membership differs')
        io.freeze(folder / 'data_order.json', {'first600_target_batch_ID_sha256': data_order,
                  'full_target_batch_ID_sha256': io.digest([[x.example_id for x in b] for _, b in batches]),
                  'target_presentations': 28857, 'original_rows': 9619, 'seed': p['seed'],
                  'actual_ordering': 'unchanged exact per-epoch shuffle; partial batches are not filled across epochs'})
        teacher = legacy.load_base(p['model'], precision='bf16', quantization='none', device='cuda:0')
        teacher.requires_grad_(False); teacher.eval()
        base = legacy.load_base(p['model'], precision='bf16', quantization='none', device='cuda:0')
        legacy.set_seed(p['seed']); model = legacy.attach_new_lora(base, p['lora'])
        if io.parameter_hash(model) != read(parent / 'initialization.json')['initial_trainable_sha256']:
            raise ValueError('Fresh LoRA initialization differs from parent')
        t = p['training']
        optimizer = torch.optim.AdamW((x for x in model.parameters() if x.requires_grad), lr=t['learning_rate'],
                                     weight_decay=t['weight_decay'], betas=tuple(t['betas']), eps=t['eps'])
        scheduler = old.schedule(optimizer, p)
        try:
            progress = handoff(io, folder, parent, model, optimizer, scheduler, parent_identity, identity)
            print(f'[HANDOFF] verified; resume step={progress["step"]} target_examples={progress["examples_seen"]} '
                  f'anchor_positions={progress["anchor_tokens_seen"]}; unchanged LR schedule', flush=True)
            status = train_loop(io, old, folder, parent, model, teacher, optimizer, scheduler, tokenizer,
                                p, plan, train, val, cache, progress, identity, parent_identity, budget)
            if any(sha(Path(f)) != h for f, h in protected.items()):
                raise ValueError('A protected parent input changed during continuation')
            io.atomic(folder / 'preservation.json', {'protected_sha256': protected, 'all_unchanged': True})
            from plot_full import plot
            plot(folder)
            print(json.dumps({'status': status['status'], 'step': status['step'], 'output': str(folder),
                              'best': status['best_by_original300_reference_loss'],
                              'resume': 'Repeat the same Slurm command'}), flush=True)
        finally:
            del model, base, teacher, optimizer, scheduler
            gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['preflight', 'report', 'authorize', 'train'])
    parser.add_argument('--confirm-plan', help='Exact plan checksum; explicit operator confirmation only')
    args = parser.parse_args()
    plan = preflight()
    if args.stage == 'authorize':
        if args.confirm_plan != sha(ROOT / 'plan.json'):
            raise ValueError('Explicit confirmation must match the reviewed plan checksum')
        path = ROOT / 'execution_approval.json'
        value = {'approved': True, 'plan_sha256': args.confirm_plan,
                 'scope': 'lambda0.01,600-to3609 continuation,57728 total anchor positions,original300 minimum-loss selection',
                 'source': 'Operator explicitly confirms this exact plan after project approval'}
        if path.exists():
            if read(path) != value:
                raise ValueError('Existing approval differs; preserve it')
        else:
            with path.open('x') as stream:
                json.dump(value, stream, indent=2); stream.write('\n')
        print('Exact continuation plan authorized; no training started.')
    elif args.stage == 'preflight':
        print(json.dumps({'status': 'package_verified_not_run', 'plan': plan, 'plan_sha256': sha(ROOT / 'plan.json'),
                          'authorized': (ROOT / 'execution_approval.json').exists()}, indent=2))
    elif args.stage == 'report':
        path = ROOT / 'results' / plan['plan_id'] / 'status.json'
        print(json.dumps(read(path) if path.exists() else {'status': 'not_started', 'authorized': (ROOT / 'execution_approval.json').exists()}, indent=2))
    else:
        run()


if __name__ == '__main__':
    main()
