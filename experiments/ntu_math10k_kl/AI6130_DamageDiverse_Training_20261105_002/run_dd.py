"""Fresh matched damage-plus-diversity training; separate disposable smoke, resumable own state."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import signal
import time
import state_io as io
import dd_guard
import a_helpers as a

ROOT = Path(__file__).resolve().parent


def smoke_cache(cache):
    """Stress an EOS-only response plus longest total sequence/response in the disposable fixture."""
    longest = max(cache, key=lambda r: (len(r['prompt_token_ids']) + r['response_token_count'], r['example_id']))
    response = max(cache, key=lambda r: (r['response_token_count'], r['example_id']))
    eos_only = next((r for r in cache if r['response_token_count'] == 1), None)
    ordered = []
    for row in [eos_only, longest, response]:
        if row is not None and row['example_id'] not in {r['example_id'] for r in ordered}:
            ordered.append(row)
    used = {r['example_id'] for r in ordered}
    return ordered + sorted((r for r in cache if r['example_id'] not in used),
                            key=lambda r: (-len(r['prompt_token_ids']) - r['response_token_count'], r['example_id']))


def check_progress(progress, batches, cache, limit):
    step = progress['step']; positions = sum(r['response_token_count'] for r in cache)
    if not 0 <= step <= limit or [r['step'] for r in progress['history']] != list(range(1, step + 1)):
        raise ValueError('B recovery step/history differs')
    expected = step // 4 * 64
    if progress['anchor_tokens_seen'] != expected or progress['anchor_cursor'] != expected % positions:
        raise ValueError('B recovery anchor exposure/cursor differs')
    if progress['examples_seen'] != sum(len(group) for _, group in batches[:step]):
        raise ValueError('B recovery target cursor differs')
    tokens = sum(sum(ex.response_mask[1:]) for _, group in batches[:step] for ex in group)
    if progress['response_tokens_seen'] != tokens:
        raise ValueError('B recovery target-token counter differs')
    for r in progress['history']:
        if r['anchor_tokens'] != (64 if r['step'] % 4 == 0 else 0):
            raise ValueError('B history has an unmatched anchor event')


def verify_checkpoint(path, identity):
    receipt = io.read(path / 'receipt.json')
    if receipt['identity'] != identity or receipt['step'] != int(path.name.removeprefix('step_')):
        raise ValueError('B checkpoint identity differs')
    if any(Path(f).name != f or io.sha(path / f) != h for f, h in receipt['files'].items()):
        raise ValueError('B checkpoint receipt/weights differ')
    return receipt


def best_checkpoint(folder, history, identity):
    trained = [r for r in history if r['step'] > 0]
    if not trained:
        return None
    best = min(trained, key=lambda r: (r['validation_loss'], r['step']))
    path = folder / 'checkpoints' / ('step_%06d' % best['step'])
    verify_checkpoint(path, identity)
    result = {'step': best['step'], 'validation_loss': best['validation_loss'], 'checkpoint': str(path),
              'metric': 'original300 response-token-weighted reference loss includingEOS',
              'exact_tie_break': 'earlier step', 'official_test_used': False}
    io.atomic(folder / 'best_model.json', result)
    return result


def train_loop(folder, model, teacher, optimizer, scheduler, tokenizer, p, batches, val, cache,
               progress, identity, budget, *, smoke=False):
    from calibration_core import update
    import torch
    limit = 8 if smoke else 3609
    validation_steps = [] if smoke else [0] + list(range(300, 3601, 300)) + [3609]
    check_progress(progress, batches, cache, limit)
    if smoke and progress['step'] >= 4 and not progress['recovery_probe_passed']:
        progress = a.recovery_probe(folder, model, optimizer, scheduler, progress, identity)
    started = time.monotonic()
    while not budget.expired():
        if progress['phase'] == 'validate':
            if progress['step']:
                io.checkpoint(folder, model, tokenizer, progress['step'], identity)
            result = a.validate(folder, model, teacher, val, cache, tokenizer, p,
                                progress['step'], identity, budget)
            if result is None:
                break
            progress['validation_history'].append(result); progress['phase'] = 'train'
            io.save(folder, model, optimizer, scheduler, progress, identity)
            best_checkpoint(folder, progress['validation_history'], identity)
        if progress['step'] == limit or budget.expired():
            break
        epoch, group = batches[progress['step']]
        metrics, cursor = update(model, teacher, group, optimizer, scheduler, p, tokenizer,
                                 cache, progress['anchor_cursor'], .01, progress['step'] + 1)
        if any(x.requires_grad or x.grad is not None for x in teacher.parameters()):
            raise ValueError('Frozen teacher acquired gradients')
        progress['step'] += 1; progress['anchor_cursor'] = cursor
        progress['examples_seen'] += len(group)
        progress['response_tokens_seen'] += metrics['target_tokens']
        progress['anchor_tokens_seen'] += metrics['anchor_tokens']
        metrics.update(coefficient=.01, examples_seen=progress['examples_seen'],
                       anchor_tokens_seen=progress['anchor_tokens_seen'], epoch=epoch,
                       segment_seconds=time.monotonic() - started)
        progress['history'].append(metrics); step = progress['step']
        if smoke or step % 10 == 0 or step == limit:
            print(f'[TRAIN] damage_diverse {step}/{limit} math={metrics["target_loss"]:.6f} '
                  f'KL={metrics["anchor_kl"]} anchor_positions={progress["anchor_tokens_seen"]}', flush=True)
        if smoke and step == 4:
            progress = a.recovery_probe(folder, model, optimizer, scheduler, progress, identity)
        if step in validation_steps:
            progress['phase'] = 'validate'
            io.save(folder, model, optimizer, scheduler, progress, identity)
        elif step % 50 == 0:
            io.save(folder, model, optimizer, scheduler, progress, identity)
    check_progress(progress, batches, cache, limit)
    if scheduler.last_epoch != progress['step']:
        raise ValueError('B learning-rate schedule/recovery cursor differs')
    io.save(folder, model, optimizer, scheduler, progress, identity)
    complete = progress['step'] == limit and progress['phase'] == 'train'
    if complete and smoke and not progress['recovery_probe_passed']:
        raise ValueError('B hardware recovery probe not complete')
    if complete and not smoke and [r['step'] for r in progress['validation_history']] != validation_steps:
        raise ValueError('B validation cadence incomplete')
    best = None if smoke else best_checkpoint(folder, progress['validation_history'], identity)
    status = {'status': 'complete' if complete else 'paused', 'scope': 'hardware-smoke' if smoke else 'full-training',
              'step': progress['step'], 'planned_steps': limit, 'phase': progress['phase'],
              'coefficient': .01, 'examples_seen': progress['examples_seen'],
              'target_response_tokens_seen': progress['response_tokens_seen'],
              'anchor_positions_seen': progress['anchor_tokens_seen'], 'unique_anchor_positions': 1453,
              'selected_prompts': 128, 'anchor_cursor': progress['anchor_cursor'],
              'recovery_probe_passed': progress['recovery_probe_passed'],
              'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3,
              'identity': identity, 'best_by_original300_reference_loss': best,
              'validation_history': progress['validation_history'], 'generation_calls': 0,
              'benchmark_questions': 0, 'imported_training_updates': 0}
    io.atomic(folder / 'status.json', status)
    if complete:
        io.freeze(folder / 'completion.json', status)
    return status


def run(stage, root=ROOT):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Submit model work through Slurm; no head-node computation')
    budget = io.Budget(2700 if stage == 'smoke' else 19800)
    for n in [signal.SIGUSR1, signal.SIGTERM, signal.SIGINT]:
        signal.signal(n, budget.signal)
    p, plan, train_rows, val_rows = dd_guard.verify(root)
    cache, selected, protected, source, ref_identity, ref_initial, ref_order = dd_guard.verify_sources(p, plan, root)
    import torch
    import legacy
    if not torch.cuda.is_available():
        raise ValueError('Allocated CUDA GPU required')
    runtime = a.runtime_info(torch)
    if runtime['gpu'] != 'NVIDIA A40' or runtime != ref_identity['runtime']:
        raise ValueError('Use the matching A40/software controls; found ' + str(runtime))
    out = root / 'results' / p['experiment_id']
    with io.lease(out):
        common = {'bundle': io.sha(root / 'files.sha256.json'), 'protocol': io.digest(p),
                  'source': source, 'runtime': runtime}
        io.freeze(out / 'common_identity.json', common)
        smoke_folder = out / 'hardware-smoke'; folder = smoke_folder if stage == 'smoke' else out / 'full-training'
        identity = {**common, 'scope': 'hardware-smoke' if stage == 'smoke' else 'full-training', 'condition': 'damage_diverse'}
        io.freeze(folder / 'identity.json', identity)
        if stage == 'train':
            passed = io.read(smoke_folder / 'completion.json')
            if (passed['status'] != 'complete' or passed['step'] != 8 or not passed['recovery_probe_passed']
                    or passed['anchor_positions_seen'] != 128 or passed['examples_seen'] != 64
                    or passed['identity'] != {**common, 'scope': 'hardware-smoke', 'condition': 'damage_diverse'}):
                raise ValueError('A matching completed B smoke/recovery check is required')
        if (folder / 'completion.json').exists():
            complete = io.read(folder / 'completion.json')
            if complete['identity'] != identity or complete != io.read(folder / 'status.json'):
                raise ValueError('Completed B receipt changed')
            if complete['best_by_original300_reference_loss']:
                verify_checkpoint(Path(complete['best_by_original300_reference_loss']['checkpoint']), identity)
            if any(io.sha(f) != h for f, h in protected.items()):
                raise ValueError('Protected input changed after B completion')
            io.atomic(folder / 'preservation.json', {'all_unchanged': True, 'protected_sha256': protected})
            print(json.dumps(complete, indent=2)); return
        io.atomic(out / 'jobs' / (os.environ['SLURM_JOB_ID'] + '.json'),
                  {'runtime': runtime, 'node': os.environ.get('SLURMD_NODENAME'), 'stage': stage})
        legacy.set_seed(p['seed']); tokenizer = legacy.load_tokenizer(p['model'])
        train = legacy.make_examples(tokenizer, train_rows, p)
        val = legacy.make_examples(tokenizer, val_rows, p)
        batches = list(legacy.epoch_batches(train, p['seed'], 3, 8))
        if len(batches) != 3609 or sum(len(g) for _, g in batches) != 28857:
            raise ValueError('Exact-epoch B target budget differs')
        if sum(sum(x.response_mask[1:]) for x in val) != 39614:
            raise ValueError('Original300 validation tokenization differs')
        prefix = io.digest([[x.example_id for x in g] for _, g in batches[:600]])
        if prefix != ref_order['first600_target_batch_ID_sha256']:
            raise ValueError('A/B target data order differs')
        for row, item in zip(cache, selected):
            if tokenizer.encode('Question: ' + item['prompt'] + '\nAnswer:', add_special_tokens=False) != row['prompt_token_ids']:
                raise ValueError('B cached prompt tokenization differs')
        io.freeze(out / 'data_order.json', {'first600_target_batch_ID_sha256': prefix,
                  'full_target_batch_ID_sha256': io.digest([[x.example_id for x in g] for _, g in batches]),
                  'target_presentations': 28857, 'seed': p['seed'], 'validation_response_tokens': 39614,
                  'partial_batches': 'Not filled across epochs', 'original300_overlap_disclosed': True})
        active = smoke_cache(cache) if stage == 'smoke' else cache
        fixture = sorted(train, key=lambda ex: (-len(ex.input_ids), ex.example_id))[:8]
        actual_batches = [(1, fixture)] * 8 if stage == 'smoke' else batches
        io.freeze(folder / 'anchor_order.json', {'example_ids': [r['example_id'] for r in active],
                  'scope': 'Length-stress fixture only; never used in full combined training' if stage == 'smoke' else 'Approved50:50 damage/diversity rank-fusion order'})
        torch.cuda.reset_peak_memory_stats()
        teacher = legacy.load_base(p['model'], precision='bf16', quantization='none', device='cuda:0')
        teacher.requires_grad_(False); teacher.eval()
        base = legacy.load_base(p['model'], precision='bf16', quantization='none', device='cuda:0')
        legacy.set_seed(p['seed']); model = legacy.attach_new_lora(base, p['lora'])
        if any('lora_' not in n for n, x in model.named_parameters() if x.requires_grad):
            raise ValueError('Only LoRA parameters may train')
        initial = io.parameter_hash(model)
        if initial != ref_initial['initial_trainable_sha256']:
            raise ValueError('A/B fresh LoRA initialization differs')
        io.freeze(out / 'common_initialization.json', {'initial_trainable_sha256': initial,
                  'fresh_base_repository': p['model'], 'no_imported_adapter': True})
        io.freeze(folder / 'initialization.json', {'initial_trainable_sha256': initial,
                  'attention': getattr(model.config, '_attn_implementation', None),
                  'teacher_frozen': True, 'gradient_checkpointing': bool(getattr(model, 'is_gradient_checkpointing', False))})
        if bool(getattr(model, 'is_gradient_checkpointing', False)) or getattr(model.config, '_attn_implementation', None) != ref_initial['attention']:
            raise ValueError('A/B attention/checkpointing execution differs')
        t = p['training']
        optimizer = torch.optim.AdamW((x for x in model.parameters() if x.requires_grad), lr=t['learning_rate'],
                                     weight_decay=t['weight_decay'], betas=tuple(t['betas']), eps=t['eps'])
        scheduler = a.schedule(optimizer, p)
        progress = dict(step=0, examples_seen=0, response_tokens_seen=0, anchor_tokens_seen=0,
                        anchor_cursor=0, phase='train' if stage == 'smoke' else 'validate',
                        history=[], validation_history=[], recovery_probe_passed=False)
        if (folder / 'recovery/current.json').exists():
            # Only this B scope is a valid recovery source. A and smoke identities cannot load here.
            progress = io.load(folder, model, optimizer, scheduler, identity)
            print(f'[RESUME] {stage} step={progress["step"]} phase={progress["phase"]}', flush=True)
        else:
            io.save(folder, model, optimizer, scheduler, progress, identity)
            print('[FRESH] Untouched base plus fresh LoRA; no A/B/smoke/discovery state imported', flush=True)
        try:
            status = train_loop(folder, model, teacher, optimizer, scheduler, tokenizer, p,
                                actual_batches, [] if stage == 'smoke' else val, active,
                                progress, identity, budget, smoke=stage == 'smoke')
            if any(io.sha(f) != h for f, h in protected.items()):
                raise ValueError('Protected historical input changed')
            io.atomic(folder / 'preservation.json', {'all_unchanged': True, 'protected_sha256': protected})
            if stage == 'train':
                from plot_dd import plot
                plot(folder)
            print(json.dumps({'status': status['status'], 'stage': stage, 'step': status['step'],
                  'anchor_positions': status['anchor_positions_seen'], 'recovery_passed': status['recovery_probe_passed'],
                  'best': status['best_by_original300_reference_loss'], 'output': str(folder),
                  'resume': 'Repeat the same sbatch stage; --resume is an explicit alias'}, indent=2), flush=True)
        finally:
            del model, base, teacher, optimizer, scheduler
            gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['preflight', 'smoke', 'train', 'report'])
    parser.add_argument('--resume', action='store_true', help='Explicit alias; own committed state always resumes')
    args = parser.parse_args()
    p, plan, _, _ = dd_guard.verify()
    if args.stage == 'preflight':
        print(json.dumps({'status': 'package_verified_not_run', 'approved': True, 'planned_updates': 3609,
              'anchors': 128, 'unique_positions': 1453, 'total_exposure': 57728, 'actual_GPU_smoke': 'pending'}, indent=2))
    elif args.stage == 'report':
        out = ROOT / 'results' / p['experiment_id']
        print(json.dumps({s: io.read(out / s / 'status.json') if (out / s / 'status.json').exists()
              else {'status': 'not_started'} for s in ['hardware-smoke', 'full-training']}, indent=2))
    else:
        try:
            run(args.stage)
        except Exception as exc:
            if os.environ.get('SLURM_JOB_ID'):
                io.atomic(ROOT / 'results' / p['experiment_id'] / 'failures' / (os.environ['SLURM_JOB_ID'] + '.json'),
                          {'stage': args.stage, 'status': 'failed', 'type': type(exc).__name__,
                           'message': str(exc), 'resume_preserves_last_committed_state': True})
            raise


if __name__ == '__main__':
    main()
