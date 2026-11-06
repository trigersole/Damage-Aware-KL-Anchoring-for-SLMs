"""Standalone scientific plots from saved random records; never runs inference."""
import csv
import json
import os
from pathlib import Path


def canvas(folder):
    folder.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(folder / '.mplconfig'))
    os.environ.setdefault('XDG_CACHE_HOME', str(folder / '.cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def rolling(values, size):
    return [sum(values[max(0, i-size+1):i+1]) / min(i+1, size) for i in range(len(values))]


def plot(folder, *, synthetic=False):
    out = folder / 'curves'; plt = canvas(out)
    history = json.loads((folder / 'training_history.json').read_text())
    validation = json.loads((folder / 'validation_history.json').read_text())
    status = json.loads((folder / 'status.json').read_text())
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    axes = axes.ravel()
    x = [r['step'] for r in history]
    axes[0].plot(x, [r['target_loss'] for r in history], alpha=.2, color='#1976D2')
    axes[0].plot(x, rolling([r['target_loss'] for r in history], 20), color='#1976D2')
    axes[1].plot([r['step'] for r in validation], [r['validation_loss'] for r in validation], 'o-')
    anchors = [r for r in history if r['anchor_tokens']]
    ax = [r['step'] for r in anchors]
    axes[2].plot(ax, rolling([r['anchor_kl'] for r in anchors], 10))
    axes[3].plot([r['step'] for r in validation], [r['selected_set_anchor_KL'] for r in validation], 'o-')
    ratios = [r for r in anchors if r['target_gradient_norm'] > 0]
    axes[4].plot([r['step'] for r in ratios],
                 [r['weighted_anchor_gradient_norm'] / r['target_gradient_norm'] for r in ratios])
    axes[5].plot(x, [r['learning_rate_used'] for r in history])
    titles = ['Math training loss: raw + mean20', 'Original300 validation: known training overlap',
              'Anchor KL: mean10 events', 'Fixed22,100 training positions: not retention',
              'Weighted anchor/math gradient-norm ratio', 'Unchanged3,609-update learning-rate schedule']
    for axis, title in zip(axes, titles):
        axis.set_title(title); axis.set_xlabel('Optimizer update'); axis.grid(alpha=.2)
    fig.suptitle('Random, lambda0.01 — ' + status['status'] + '\nFresh base; seed-42 random whole responses')
    if synthetic:
        fig.text(.5, .5, 'SYNTHETIC LAYOUT FIXTURE — NO RANDOM RESULT', ha='center', fontsize=20, color='red', rotation=25)
    fig.savefig(out / 'training_curves.png', dpi=160)
    fig.savefig(out / 'training_curves.svg'); plt.close(fig)
    fields = list(dict.fromkeys(k for r in history for k in r))
    with (out / 'training_history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(history)
    if validation:
        with (out / 'validation_history.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(validation[0])); writer.writeheader(); writer.writerows(validation)


def plot_evaluation(out, result):
    folder = out / 'summary'; plt = canvas(folder)
    names = list(result['benchmarks']); modes = ['base', 'task_only', 'damage_A', 'random']
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    rows = []
    for j, mode in enumerate(modes):
        data = [(i, result['benchmarks'][name][mode]) for i, name in enumerate(names) if mode in result['benchmarks'][name]]
        ax.bar([i + (j-1.5)*.18 for i, _ in data], [100*r['accuracy'] for _, r in data], width=.18, label=mode)
        rows += [{'benchmark': names[i], 'mode': mode, **r} for i, r in data]
    ax.set_xticks(range(len(names)), names); ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 100)
    ax.set_title('Fixed five-benchmark comparison — ' + result['state'] + '\nBase/task-only/A reused; selected random model uses the same profile')
    ax.legend(); ax.grid(axis='y', alpha=.2)
    fig.savefig(folder / 'accuracy.png', dpi=160); fig.savefig(folder / 'accuracy.svg'); plt.close(fig)
    fields = ['benchmark', 'mode', 'correct', 'total', 'accuracy', 'parse_failures']
    with (folder / 'accuracy.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
