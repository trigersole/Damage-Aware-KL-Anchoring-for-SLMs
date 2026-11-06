"""Fixed random-only inference; audit and reuse immutable base/task-only/A records."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import state_io as state
import eval_io as io
import random_guard
import run_random
import a_evaluate as a

ROOT = Path(__file__).resolve().parent
REFERENCES = ['base', 'task_only', 'damage_A']


def verify_reference_files(plan, root=ROOT):
    package = Path(plan['paths']['a_evaluation'])
    manifest = io.read(root / 'a_evaluation_files.sha256.json')
    if io.read(package / 'files.sha256.json') != manifest:
        raise ValueError('A evaluator bundle differs')
    protected = {}
    for rel, h in manifest.items():
        if io.sha(package / rel) != h:
            raise ValueError('A evaluation source changed: ' + rel)
        protected[str(package / rel)] = h
    ref = package / plan['paths']['a_evaluation_output']
    for rel, h in io.read(root / 'reference_results.sha256.json').items():
        if io.sha(ref / rel) != h:
            raise ValueError('Audited base/task-only/A result changed: ' + rel)
        protected[str(ref / rel)] = h
    if io.read(ref / 'status.json')['state'] != 'complete':
        raise ValueError('Reference comparison is not complete')
    if io.read(ref / 'evaluation.json') != io.read(root / 'reference_evaluation.json'):
        raise ValueError('Reference protocol differs')
    return ref, io.read(ref / 'identity.json'), protected


def reference_cell(ref, ref_identity, name, mode, expected):
    folder = ref / mode / name
    identity = {'run': io.digest(ref_identity), 'mode': mode, 'benchmark': name,
                'dataset_sha256': io.sha(ref / 'prepared' / name / 'rows.jsonl')}
    if io.read(folder / 'identity.json') != identity:
        raise ValueError('Reference cell identity differs')
    records = io.recover(folder / 'items.jsonl', expected, identity, completed=True)
    receipt = io.read(folder / 'completion.json')
    if (len(records) != len(expected) or io.summarize(records) != io.read(folder / 'summary.json')
            or receipt != {'identity': identity, 'items_sha256': io.sha(folder / 'items.jsonl'),
                           'summary_sha256': io.sha(folder / 'summary.json'), 'count': len(records)}):
        raise ValueError('Reference completion/summary differs')
    return records


def report(out, ref, p):
    result = {'state': 'complete', 'remaining': [], 'benchmarks': {},
              'scope': 'Fixed selected random inference; base/task-only/A reused with pinned provenance'}
    for name in p['benchmarks']:
        entry = {mode: io.read(ref / mode / name / 'summary.json') for mode in REFERENCES}
        folder = out / 'random' / name
        if (folder / 'completion.json').exists():
            entry['random'] = io.read(folder / 'summary.json')
            after = io.rows(folder / 'items.jsonl')
            for mode in REFERENCES:
                before = io.rows(ref / mode / name / 'items.jsonl')
                if [(r['key'], r['row_sha256']) for r in before] != [(r['key'], r['row_sha256']) for r in after]:
                    raise ValueError('Random/reference paired rows differ')
                entry['Random_vs_' + mode + '_percentage_points'] = 100 * (entry['random']['accuracy'] - entry[mode]['accuracy'])
                entry['Random_vs_' + mode + '_paired'] = {
                    'gained_correct': sum(not b['correct'] and x['correct'] for b, x in zip(before, after)),
                    'lost_correct': sum(b['correct'] and not x['correct'] for b, x in zip(before, after))}
        else:
            result['remaining'].append({'mode': 'random', 'benchmark': name})
        result['benchmarks'][name] = entry
    if result['remaining']:
        result['state'] = 'paused' if out.exists() else 'not_started'
    return result


def matching_runtime(runtime, reference, training):
    return (runtime == reference
            and all(runtime[k] == v for k, v in training.items() if k != 'packages')
            and all(runtime['packages'].get(k) == v for k, v in training['packages'].items()))


def run(root=ROOT):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Submit inference through Slurm; no head-node computation')
    protocol, plan, train, val = random_guard.verify(root)
    p = io.read(root / 'reference_evaluation.json'); config = io.read(root / 'evaluation.json')
    if not config['approved'] or config['reference_profile_sha256'] != io.sha(root / 'reference_evaluation.json'):
        raise ValueError('Frozen Random evaluation approval differs')
    training = root / 'results' / protocol['experiment_id'] / 'full-training'
    complete = io.read(training / 'completion.json')
    if complete['status'] != 'complete' or complete['step'] != 3609 or complete['anchor_positions_seen'] != 57728:
        raise ValueError('Finish approved Random training before final testing')
    history = complete['validation_history']
    winner = min((r for r in history if r['step'] > 0), key=lambda r: (r['validation_loss'], r['step']))
    best = io.read(training / 'best_model.json')
    if best != complete['best_by_original300_reference_loss'] or best['step'] != winner['step'] or best['validation_loss'] != winner['validation_loss']:
        raise ValueError('Random checkpoint was not selected by the frozen development rule')
    adapter = Path(best['checkpoint']); receipt = run_random.verify_checkpoint(adapter, complete['identity'])
    ref, ref_identity, protected = verify_reference_files(plan, root)
    import torch
    from transformers import AutoTokenizer
    if not torch.cuda.is_available():
        raise ValueError('Allocated CUDA GPU required')
    runtime = a.runtime(torch)
    train_runtime = complete['identity']['runtime']
    if not matching_runtime(runtime, ref_identity['runtime'], train_runtime):
        raise ValueError('Use the matching A40/evaluation/training software controls')
    protected.update({str(adapter / f): h for f, h in receipt['files'].items()})
    for f in ['receipt.json']:
        protected[str(adapter / f)] = io.sha(adapter / f)
    for f in ['completion.json', 'best_model.json', 'identity.json']:
        protected[str(training / f)] = io.sha(training / f)
    sources = {'random': {'directory': str(adapter), 'selected_step': best['step'],
               'weights_sha256': io.sha(adapter / 'adapter_model.safetensors'),
               'config_sha256': io.sha(adapter / 'adapter_config.json')}}
    identity = {'bundle': io.sha(root / 'files.sha256.json'), 'protocol': io.digest(config),
                'reference_protocol': io.digest(p), 'reference_identity': ref_identity,
                'sources': sources, 'runtime': runtime}
    out = root / 'results' / config['evaluation_id']
    budget = a.Budget(19800)
    for n in [signal.SIGUSR1, signal.SIGTERM, signal.SIGINT]:
        signal.signal(n, budget.stop)
    with state.lease(out):
        io.freeze(out / 'identity.json', identity)
        io.freeze(out / 'evaluation.json', {'profile': p, 'random_evaluation': config, 'random_selected': best})
        io.atomic(out / 'jobs' / (os.environ['SLURM_JOB_ID'] + '.json'),
                  {'runtime': runtime, 'node': os.environ.get('SLURMD_NODENAME')})
        from data_eval import prepare, normalize
        _, anchors, selection_files, _ = random_guard.verify_selection(plan, root)
        protected.update(selection_files)
        known = {'math10k_train': [r['prompt'] for r in train], 'original300': [r['prompt'] for r in val],
                 'selected_random_anchors': [r['prompt'] for r in anchors]}
        known = {k: {normalize(x) for x in values} for k, values in known.items()}
        tokenizer = AutoTokenizer.from_pretrained(p['model']['repo_id'], revision=p['model']['revision'], use_fast=True)
        prepared = prepare(p, out, tokenizer, io.digest(identity), known, budget.expired)
        if prepared is not None:
            for name in p['benchmarks']:
                for mode in REFERENCES:
                    reference_cell(ref, ref_identity, name, mode, prepared[name])
            io.freeze(out / 'reference_reuse.json', {'status': 'verified', 'root': str(ref),
                      'identity': ref_identity, 'modes': REFERENCES, 'reference_files': io.read(root / 'reference_results.sha256.json'),
                      'no_reference_inference_or_changes': True})
            probe_p = {**p, 'modes': ['random']}
            ready = a.smoke_all(probe_p, out, sources, io.digest(identity), budget.expired)
            pending = []
            if ready:
                for name in p['benchmarks']:
                    cell_id = {'run': io.digest(identity), 'mode': 'random', 'benchmark': name,
                               'dataset_sha256': io.sha(out / 'prepared' / name / 'rows.jsonl')}
                    folder = out / 'random' / name
                    if (folder / 'items.jsonl').exists():
                        io.cell(None, prepared[name], folder, cell_id, budget.expired, a.score)
                    else:
                        pending.append((name, cell_id))
                if pending and not budget.expired():
                    a.set_evaluation_seed(p['seed'])
                    print('[LOAD] selected random; frozen inference only', flush=True)
                    backend = a.AuditedBackend(a.backend_protocol(p, 'random', sources))
                    try:
                        for name, cell_id in pending:
                            if not io.cell(backend, prepared[name], out / 'random' / name, cell_id, budget.expired, a.score):
                                break
                    finally:
                        del backend; gc.collect(); torch.cuda.empty_cache()
        if any(io.sha(f) != h for f, h in protected.items()):
            raise ValueError('Historical output or selected model changed')
        io.atomic(out / 'preservation.json', {'all_unchanged': True, 'protected_sha256': protected})
        result = report(out, ref, p); io.atomic(out / 'status.json', result)
        from plot_random import plot_evaluation
        plot_evaluation(out, result)
        print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['run', 'report'])
    parser.add_argument('--resume', action='store_true', help='Explicit alias; saved predictions always resume')
    args = parser.parse_args()
    _, plan, _, _ = random_guard.verify()
    if args.stage == 'report':
        config = io.read(ROOT / 'evaluation.json')
        ref = Path(plan['paths']['a_evaluation']) / plan['paths']['a_evaluation_output']
        print(json.dumps(report(ROOT / 'results' / config['evaluation_id'], ref,
                                io.read(ROOT / 'reference_evaluation.json')), indent=2))
    else:
        try:
            run()
        except Exception:
            if os.environ.get('SLURM_JOB_ID'):
                config = io.read(ROOT / 'evaluation.json')
                io.atomic(ROOT / 'results' / config['evaluation_id'] / 'failures' / (os.environ['SLURM_JOB_ID'] + '.json'),
                          {'status': 'failed', 'details': 'See Slurm stderr; saved results preserved'})
            raise


if __name__ == '__main__':
    main()
