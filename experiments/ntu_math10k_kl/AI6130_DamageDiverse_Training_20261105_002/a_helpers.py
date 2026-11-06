#!/usr/bin/env python3
"""Approved isolated original300 KL calibration; explicit smoke/train separation."""
from __future__ import annotations
import argparse
import copy
import gc
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import sys
import time

import state_io as io
import source_guard as guard

ROOT = Path(__file__).resolve().parent
CONDITIONS = {'task_only': 0.0, 'damage_001': 0.01, 'damage_005': 0.05}


def verify_local(root=ROOT):
    guard.verify_bundle(root)
    p = io.read(root / 'protocol.json'); old = io.read(root / 'original_protocol.json')
    approved = io.read(root / 'approval.json')
    if approved['conditions'] != CONDITIONS or approved['steps'] != 600 or approved['smoke_steps'] != 8:
        raise ValueError('Approved comparison changed')
    for field in ['model', 'lora', 'seed']:
        if p[field] != old[field]:
            raise ValueError('Original model/LoRA/seed changed')
    for field, value in old['training'].items():
        if field not in ['training_ids_manifest', 'anchor_interleave_every_steps'] and p['training'][field] != value:
            raise ValueError('Original target-training field changed: ' + field)
    if (p['training']['weight_decay'], p['training']['betas'], p['training']['eps']) != (0.01, [0.9, 0.999], 1e-8):
        raise ValueError('Original AdamW defaults changed')
    if (p['training']['anchor_interleave_every_steps'], p['runtime']['anchor_tokens_per_update']) != (4, 64):
        raise ValueError('Approved anchor exposure changed')
    if (p['training']['max_steps'], p['training']['warmup_steps'], p['calibration']['steps']) != (3609, 361, 600):
        raise ValueError('Original schedule or calibration budget changed')
    if p['calibration']['conditions'] != CONDITIONS or p['calibration']['validation_steps'] != [0, 300, 600]:
        raise ValueError('Comparison/cadence changed')
    if (p['runtime']['max_sequence_tokens'], p['runtime']['target_answer_append_eos'], p['runtime']['gradient_checkpointing']) != (2048, True, False):
        raise ValueError('Target tokenization or checkpointing changed')
    if p['evaluation']['prompt_format'].replace('\\n', '\n') != 'Question: {question}\nAnswer:':
        raise ValueError('Original Q/A format changed')
    if {k: p['kl'][k] for k in ['direction', 'temperature', 'token_mask', 'normalization']} != {
            'direction': 'teacher_to_student', 'temperature': 1,
            'token_mask': 'response_tokens', 'normalization': 'token_mean'}:
        raise ValueError('Approved KL formulation changed')
    train = io.rows(root / 'inputs/train.jsonl'); val = io.rows(root / 'inputs/validation_original_300.jsonl')
    if (len(train), len(val)) != (9619, 300):
        raise ValueError('Original data membership changed')
    for path, expected in [('train.jsonl', 'adcbf9771c5ac3556a3b645293d132ad565c9033ef8d82e4c14031cd0f60b2eb'),
                           ('validation_original_300.jsonl', '2139a7c220259c1e1270e5123ef104465e0356d44ac2ab2f4ce930c0781c57ed')]:
        if io.sha(root / 'inputs' / path) != expected:
            raise ValueError('Original target input bytes changed')
    ids = [r['example_id'] for r in train + val]
    if len(ids) != len(set(ids)):
        raise ValueError('Original row IDs overlap or repeat')
    return p, train, val


def verify_anchors(p, root=ROOT):
    _, manifest, repo, exp, canonical, protected = guard.guard(root=root)
    directory = Path(p['calibration']['selection_directory'])
    summary = io.read(directory / 'summary.json')
    if summary['status'] != 'complete' or summary['source_manifest'] != manifest:
        raise ValueError('Wrong damage-A selection provenance')
    if (summary['selected_prompt_count'], summary['eligible_prompt_count'],
            summary['selected_unique_response_tokens']) != (128, 13743, 255):
        raise ValueError('Wrong approved top128 selection counts')
    for name, expected in summary['export_sha256'].items():
        if Path(name).name != name or io.sha(directory / name) != expected:
            raise ValueError('Selected export changed: ' + name)
    for name, expected in summary['source_checksums'].items():
        if io.sha(name) != expected:
            raise ValueError('Original evidence changed since selection: ' + name)
        protected[name] = expected
    selected = io.rows(directory / 'damage_A_anchors.jsonl')
    cache = io.rows(directory / 'selected_teacher_cache.jsonl')
    full = {r['example_id']: r for r in io.rows(canonical)}
    scores = io.rows(exp / 'full/vulnerability_scores.jsonl')
    ranking = sorted(scores, key=lambda r: (-r['vulnerability_score'], r['example_id']))
    if len(selected) != 128 or [r['example_id'] for r in selected] != [r['example_id'] for r in ranking[:128]]:
        raise ValueError('Membership is not the original unrestricted top128')
    if len(cache) != 128 or sum(r['response_token_count'] for r in cache) != 255:
        raise ValueError('Wrong selected teacher tokens')
    for index, (record, cached) in enumerate(zip(selected, cache), 1):
        if cached != full[cached['example_id']] or record['example_id'] != cached['example_id']:
            raise ValueError('Selected cache differs from original canonical record')
        if record['selection_rank'] != index or record['vulnerability_score'] != ranking[index - 1]['vulnerability_score']:
            raise ValueError('Selected rank/score differs')
        for key in cached:
            if record[key] != cached[key]:
                raise ValueError('Selected source tokens differ')
    protected[str(directory / 'summary.json')] = io.sha(directory / 'summary.json')
    protected.update({str(directory / name): expected for name, expected in summary['export_sha256'].items()})
    return cache, selected, protected, {'selection_summary_sha256': io.sha(directory / 'summary.json'),
                                     'selected_cache_sha256': io.sha(directory / 'selected_teacher_cache.jsonl'),
                                     'source_manifest': manifest}


def schedule(optimizer, p):
    import torch
    total, warmup = p['training']['max_steps'], p['training']['warmup_steps']
    def factor(step):
        return ((step + 1) / warmup if step < warmup else
                max(0.0, (total - step) / (total - warmup)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def runtime_info(torch):
    packages = {name: metadata.version(name) for name in ['torch', 'transformers', 'peft', 'accelerate', 'numpy']}
    for name, version in {'torch': '2.11.0+cu128', 'transformers': '4.57.6', 'peft': '0.21.0', 'accelerate': '1.15.0'}.items():
        if packages[name] != version:
            raise ValueError(f'Expected {name} {version}; found {packages[name]}')
    return {'packages': packages, 'python': platform.python_version(),
            'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
            'capability': list(torch.cuda.get_device_capability(0)),
            'matmul_precision': torch.get_float32_matmul_precision(),
            'TF32': torch.backends.cuda.matmul.allow_tf32}


def validate(folder, model, teacher, examples, cache, tokenizer, p, step, identity, budget):
    import torch
    import legacy
    from calibration_core import compact_kl_sum
    model.eval(); teacher.eval(); directory = folder / 'validation' / f'step_{step:06d}'
    records = []
    for i, ex in enumerate(examples):
        path = directory / 'math' / f'{i:04d}.json'
        if path.exists():
            record = io.read(path)
        else:
            if budget.expired():
                return None
            batch = legacy.collate([ex], pad_token_id=tokenizer.pad_token_id, device=p['runtime']['device'])
            with torch.no_grad():
                logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
                loss = float(legacy.target_causal_loss(logits, batch['input_ids'], batch['response_mask']).item())
            record = {'identity': identity, 'step': step, 'example_id': ex.example_id,
                      'response_tokens': sum(ex.response_mask[1:]), 'loss': loss}
            io.atomic(path, record); del logits, batch
        if (record['identity'] != identity or record['step'] != step or record['example_id'] != ex.example_id
                or record['response_tokens'] != sum(ex.response_mask[1:]) or not math.isfinite(record['loss'])):
            raise ValueError('Saved math validation record changed')
        records.append(record)
        if (i + 1) % 100 == 0:
            print(f'[VALIDATION] step={step} math {i+1}/{len(examples)}', flush=True)
    anchor_records = []
    for i, row in enumerate(cache):
        path = directory / 'anchor_diagnostic' / f'{i:04d}.json'
        if path.exists():
            record = io.read(path)
        else:
            if budget.expired():
                return None
            prompt, response = row['prompt_token_ids'], row['continuation_token_ids']
            ex = legacy.TokenizedExample(row['example_id'], tuple(prompt + response),
                                         tuple([False]*len(prompt) + [True]*len(response)))
            batch = legacy.collate([ex], pad_token_id=tokenizer.pad_token_id, device=p['runtime']['device'])
            with torch.no_grad():
                t = teacher(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
                s = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
                value, count = compact_kl_sum(t, s, batch['response_mask'])
            record = {'identity': identity, 'step': step, 'example_id': row['example_id'],
                      'positions': count, 'KL_sum': float(value.item())}
            io.atomic(path, record); del t, s, value, batch
        if (record['identity'] != identity or record['step'] != step or record['example_id'] != row['example_id']
                or record['positions'] != row['response_token_count'] or not math.isfinite(record['KL_sum'])):
            raise ValueError('Saved anchor diagnostic changed')
        anchor_records.append(record)
    response_tokens = sum(r['response_tokens'] for r in records)
    anchor_tokens = sum(r['positions'] for r in anchor_records)
    result = {'step': step, 'questions': len(records), 'response_tokens': response_tokens,
              'validation_loss': sum(r['loss'] * r['response_tokens'] for r in records) / response_tokens,
              'selected_set_anchor_KL': sum(r['KL_sum'] for r in anchor_records) / anchor_tokens,
              'anchor_diagnostic_positions': anchor_tokens,
              'anchor_diagnostic_is_training_set_not_retention': True}
    if step == 0 and abs(result['selected_set_anchor_KL']) > 1e-3:
        raise ValueError('Fresh student and teacher disagree unexpectedly; inspect initialization')
    io.freeze(directory / 'summary.json', result)
    print('[VALIDATION COMPLETE] ' + json.dumps(result), flush=True)
    return result


def check_progress(progress, *, coefficient, smoke):
    step = progress['step']; limit = 8 if smoke else 600
    if not 0 <= step <= limit or len(progress['history']) != step:
        raise ValueError('Recovery step/history count differs')
    if [r['step'] for r in progress['history']] != list(range(1, step + 1)):
        raise ValueError('Recovery history is not contiguous')
    expected = step // 4 * 64 if coefficient else 0
    if progress['anchor_tokens_seen'] != expected or progress['anchor_cursor'] != expected % 255:
        raise ValueError('Recovery anchor exposure/cursor differs')
    if progress['examples_seen'] != step * 8:
        raise ValueError('Recovery target example count differs')


def recovery_probe(folder, model, optimizer, scheduler, progress, identity):
    before = io.tree_hash({'parameters': io.parameters(model), 'optimizer': optimizer.state_dict(),
                           'scheduler': scheduler.state_dict(), 'progress': progress, 'rng': io.rng_state()})
    io.save(folder, model, optimizer, scheduler, progress, identity)
    progress = io.load(folder, model, optimizer, scheduler, identity)
    after = io.tree_hash({'parameters': io.parameters(model), 'optimizer': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict(), 'progress': progress, 'rng': io.rng_state()})
    if before != after:
        raise ValueError('Hardware parameter/optimizer/scheduler/RNG/cursor roundtrip differs')
    progress['recovery_probe_passed'] = True
    io.save(folder, model, optimizer, scheduler, progress, identity)
    return progress


def run_condition(root, stage, name, coefficient, p, train, val, cache, tokenizer, identity, budget):
    import torch
    import legacy
    from calibration_core import update
    smoke = stage == 'smoke'; scope = 'hardware-smoke' if smoke else 'calibration'
    folder = root / 'results' / p['experiment_id'] / scope / name
    case_identity = {**identity, 'condition': name, 'coefficient': coefficient, 'scope': scope}
    folder.mkdir(parents=True, exist_ok=True); io.freeze(folder / 'identity.json', case_identity)
    if (folder / 'status.json').exists() and io.read(folder / 'status.json')['status'] == 'complete':
        print(f'[REUSE COMPLETE] {scope}/{name}', flush=True); return True
    if budget.expired():
        return False
    print(f'[LOAD] {scope}/{name}; fresh base plus new LoRA; coefficient={coefficient}', flush=True)
    torch.cuda.reset_peak_memory_stats()
    teacher = legacy.load_base(p['model'], precision=p['lora']['precision'], quantization='none', device=p['runtime']['device'])
    teacher.requires_grad_(False); teacher.eval()
    base = legacy.load_base(p['model'], precision=p['lora']['precision'], quantization='none', device=p['runtime']['device'])
    legacy.set_seed(p['seed'])  # Exactly match LoRA initialization across conditions.
    model = legacy.attach_new_lora(base, p['lora'])
    initial_hash = io.parameter_hash(model)
    common = root / 'results' / p['experiment_id'] / scope / 'common_initialization.json'
    io.freeze(common, {'shared_identity': identity, 'initial_trainable_sha256': initial_hash})
    effective = copy.deepcopy(p); effective['kl']['coefficient'] = coefficient
    effective['anchor_selection']['method'] = 'vulnerability' if coefficient else 'none'
    if not coefficient:
        effective['runtime']['anchor_tokens_per_update'] = 0
        effective['training']['anchor_interleave_every_steps'] = 0
        effective['anchor_selection']['token_budget'] = 0
    io.freeze(folder / 'effective_protocol.json', effective)
    io.freeze(folder / 'initialization.json', {'initial_trainable_sha256': initial_hash,
                                            'attention': getattr(model.config, '_attn_implementation', None),
                                            'teacher_frozen': all(not x.requires_grad for x in teacher.parameters()),
                                            'gradient_checkpointing': bool(getattr(model, 'is_gradient_checkpointing', False))})
    t = p['training']
    optimizer = torch.optim.AdamW((x for x in model.parameters() if x.requires_grad), lr=t['learning_rate'],
                                 weight_decay=t['weight_decay'], betas=tuple(t['betas']), eps=t['eps'])
    scheduler = schedule(optimizer, p)
    progress = dict(step=0, examples_seen=0, response_tokens_seen=0, anchor_tokens_seen=0,
                    anchor_cursor=0, phase='train' if smoke else 'validate',
                    history=[], validation_history=[], recovery_probe_passed=False)
    if (folder / 'recovery/current.json').exists():
        progress = io.load(folder, model, optimizer, scheduler, case_identity)
        print(f'[RESUME] {name} step={progress["step"]} phase={progress["phase"]}', flush=True)
    else:
        io.save(folder, model, optimizer, scheduler, progress, case_identity)
    check_progress(progress, coefficient=coefficient, smoke=smoke)
    if smoke and progress['step'] >= 4 and not progress['recovery_probe_passed']:
        progress = recovery_probe(folder, model, optimizer, scheduler, progress, case_identity)
    all_batches = list(legacy.epoch_batches(train, p['seed'], t['epochs'], 8))
    if len(all_batches) != 3609:
        raise ValueError('Original exact-epoch scheduler length differs')
    limit = 8 if smoke else 600
    # Smoke stresses the longest targets and anchors without touching validation.
    active_cache = sorted(cache, key=lambda r: (-len(r['prompt_token_ids']), r['example_id'])) if smoke else cache
    fixture = sorted(train, key=lambda ex: (-len(ex.input_ids), ex.example_id))[:8] if smoke else None
    started = time.monotonic()
    try:
        while not budget.expired():
            if progress['phase'] == 'validate':
                if progress['step']:
                    io.checkpoint(folder, model, tokenizer, progress['step'], case_identity)
                result = validate(folder, model, teacher, val, cache, tokenizer, p,
                                  progress['step'], case_identity, budget)
                if result is None:
                    break
                progress['validation_history'].append(result); progress['phase'] = 'train'
                io.save(folder, model, optimizer, scheduler, progress, case_identity)
            if progress['step'] == limit or budget.expired():
                break
            group = fixture if smoke else all_batches[progress['step']][1]
            metrics, cursor = update(model, teacher, group, optimizer, scheduler, p, tokenizer,
                                     active_cache, progress['anchor_cursor'], coefficient, progress['step'] + 1)
            progress['step'] += 1; progress['anchor_cursor'] = cursor
            progress['examples_seen'] += len(group)
            progress['response_tokens_seen'] += metrics['target_tokens']
            progress['anchor_tokens_seen'] += metrics['anchor_tokens']
            metrics.update(coefficient=coefficient, examples_seen=progress['examples_seen'],
                           anchor_tokens_seen=progress['anchor_tokens_seen'],
                           segment_seconds=time.monotonic()-started)
            progress['history'].append(metrics)
            step = progress['step']
            if smoke or step % 10 == 0:
                print(f'[TRAIN] {name} {step}/{limit} math={metrics["target_loss"]:.6f} '
                      f'KL={metrics["anchor_kl"]} anchor_tokens={progress["anchor_tokens_seen"]} '
                      f'peak_GiB={torch.cuda.max_memory_allocated()/1024**3:.2f}', flush=True)
            if smoke and step == 4:
                progress = recovery_probe(folder, model, optimizer, scheduler, progress, case_identity)
            if not smoke and step in [300, 600]:
                progress['phase'] = 'validate'
                io.save(folder, model, optimizer, scheduler, progress, case_identity)
            elif step % p['calibration']['recovery_every_steps'] == 0:
                io.save(folder, model, optimizer, scheduler, progress, case_identity)
        check_progress(progress, coefficient=coefficient, smoke=smoke)
        io.save(folder, model, optimizer, scheduler, progress, case_identity)
        complete = progress['step'] == limit and progress['phase'] == 'train'
        if complete and smoke and not progress['recovery_probe_passed']:
            raise ValueError('Hardware recovery probe incomplete')
        if complete and not smoke and [r['step'] for r in progress['validation_history']] != [0, 300, 600]:
            raise ValueError('Approved validation cadence differs')
        if complete and smoke:
            io.checkpoint(folder, model, tokenizer, 8, case_identity)
        status = dict(status='complete' if complete else 'paused', scope=scope, condition=name,
                      coefficient=coefficient, step=progress['step'], planned_steps=limit,
                      phase=progress['phase'], examples_seen=progress['examples_seen'],
                      target_response_tokens_seen=progress['response_tokens_seen'],
                      anchor_tokens_seen=progress['anchor_tokens_seen'],
                      recovery_probe_passed=progress['recovery_probe_passed'],
                      peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,
                      identity=case_identity, validation_history=progress['validation_history'])
        io.atomic(folder / 'status.json', status)
        print('[CONDITION] ' + json.dumps(status), flush=True)
        return complete
    finally:
        del model, base, teacher, optimizer, scheduler
        gc.collect(); torch.cuda.empty_cache()


def report(root=ROOT):
    p = io.read(root / 'protocol.json'); out = root / 'results' / p['experiment_id']
    result = {'state': 'not_started', 'automatic_coefficient_winner': None, 'scopes': {}}
    for scope in ['hardware-smoke', 'calibration']:
        cells = {}
        for name in CONDITIONS:
            path = out / scope / name / 'status.json'
            cells[name] = io.read(path) if path.exists() else {'status': 'not_started'}
        result['scopes'][scope] = cells
    states = [x['status'] for x in result['scopes']['calibration'].values()]
    result['state'] = 'complete' if all(x == 'complete' for x in states) else 'incomplete'
    result['interpretation'] = 'Stability/learning calibration only; no benchmark retention or automatic lambda selection'
    return result


def run(stage, root=ROOT):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Submit through Slurm; no model work on the head node')
    budget = io.Budget(2700 if stage == 'smoke' else 19800)
    for number in [signal.SIGUSR1, signal.SIGTERM, signal.SIGINT]:
        signal.signal(number, budget.signal)
    p, train_rows, val_rows = verify_local(root)
    cache, selected, protected, source = verify_anchors(p, root)
    import torch
    import legacy
    if not torch.cuda.is_available():
        raise ValueError('Allocated CUDA GPU required')
    runtime = runtime_info(torch)
    out = root / 'results' / p['experiment_id']
    with io.lease(out):
        common_identity = {'bundle': io.sha(root / 'files.sha256.json'), 'protocol': io.digest(p),
                           'source': source, 'runtime': runtime}
        if (out / 'common_identity.json').exists():
            previous = io.read(out / 'common_identity.json')
            if previous['runtime'] != runtime:
                raise ValueError('GPU/software differs from previous allocation: expected ' +
                                 str(previous['runtime']) + '; found ' + str(runtime) +
                                 '. Resume on matching hardware/software; do not edit identity files.')
        io.freeze(out / 'common_identity.json', common_identity)
        io.atomic(out / 'jobs' / (os.environ['SLURM_JOB_ID'] + '.json'),
                  {'runtime': runtime, 'node': os.environ.get('SLURMD_NODENAME'), 'stage': stage,
                   'platform': platform.platform(), 'source_checksums': protected})
        if stage == 'calibrate':
            smoke = io.read(out / 'hardware-smoke/summary.json')
            if smoke['status'] != 'complete' or smoke['identity'] != common_identity:
                raise ValueError('Complete matching smoke on the same hardware/software before calibration')
        legacy.set_seed(p['seed']); tokenizer = legacy.load_tokenizer(p['model'])
        train = legacy.make_examples(tokenizer, train_rows, p)
        val = legacy.make_examples(tokenizer, val_rows, p)
        if sum(sum(ex.response_mask[1:]) for ex in val) != 39614:
            raise ValueError('Original300 validation token count differs')
        for row, item in zip(cache, selected):
            if tokenizer.encode('Question: ' + item['prompt'] + '\nAnswer:', add_special_tokens=False) != row['prompt_token_ids']:
                raise ValueError('Cached anchor prompt tokens differ')
        batches = list(legacy.epoch_batches(train, p['seed'], 3, 8))
        io.freeze(out / 'data_order.json', {
            'first600_target_batch_ID_sha256': io.digest([[x.example_id for x in b] for _, b in batches[:600]]),
            'initialization_seed': p['seed'], 'validation_questions': 300,
            'validation_response_tokens': 39614, 'max_target_tokens': max(len(x.input_ids) for x in train),
            'original300_overlap_disclosed': True})
        statuses = []
        for name, coefficient in CONDITIONS.items():
            if budget.expired():
                break
            done = run_condition(root, stage, name, coefficient, p, train, val, cache,
                                 tokenizer, common_identity, budget)
            statuses.append(done)
            if not done:
                break
        unchanged = {path: io.sha(path) == value for path, value in protected.items()}
        if not all(unchanged.values()):
            raise ValueError('Original protected experiment changed during calibration')
        complete = len(statuses) == 3 and all(statuses)
        scope = 'hardware-smoke' if stage == 'smoke' else 'calibration'
        io.atomic(out / scope / 'summary.json', {'status': 'complete' if complete else 'paused',
                  'identity': common_identity, 'protected_inputs_unchanged': unchanged,
                  'conditions': list(CONDITIONS), 'steps_per_condition': 8 if stage == 'smoke' else 600,
                  'generation_calls': 0, 'retention_evaluation_calls': 0,
                  'automatic_coefficient_selection': False})
        if stage == 'calibrate':
            from plot_curves import plot
            plot(root)
        print(json.dumps({'status': 'complete' if complete else 'paused', 'stage': stage,
                          'output': str(out), 'resume': 'Repeat the same sbatch command'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['preflight', 'smoke', 'calibrate', 'report', 'plot'])
    parser.add_argument('--resume', action='store_true', help='Explicit alias; calibrated state always resumes safely')
    args = parser.parse_args()
    if args.stage == 'preflight':
        p, train, val = verify_local()
        print(json.dumps({'status': 'package_verified_not_run', 'train': len(train), 'validation': len(val),
                          'conditions': CONDITIONS, 'steps_each': 600, 'anchor_exposure_each': 9600}, indent=2))
    elif args.stage == 'report':
        print(json.dumps(report(), indent=2))
    elif args.stage == 'plot':
        from plot_curves import plot
        plot(ROOT)
    else:
        try:
            run(args.stage)
        except Exception as exc:
            if os.environ.get('SLURM_JOB_ID'):
                p = io.read(ROOT / 'protocol.json')
                io.atomic(ROOT / 'results' / p['experiment_id'] / 'failures' /
                          (os.environ['SLURM_JOB_ID'] + '.json'),
                          {'stage': args.stage, 'status': 'failed', 'type': type(exc).__name__,
                           'message': str(exc), 'resume_preserves_last_committed_state': True})
            raise


if __name__ == '__main__':
    main()
