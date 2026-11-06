"""Loss diagnostics only; deliberately does not rank coefficients by total loss."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import state_io as io


def rolling(values, window):
    return [sum(values[max(0, i-window+1):i+1]) / min(i+1, window) for i in range(len(values))]


def plot(root, *, synthetic=False):
    p = io.read(root / 'protocol.json'); out = root / 'results' / p['experiment_id']
    os.environ.setdefault('MPLCONFIGDIR', str(out / '.mplconfig'))
    os.environ.setdefault('XDG_CACHE_HOME', str(out / '.cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    axes = axes.ravel()
    names = ['task_only', 'damage_001', 'damage_005']
    colors = ['#455A64', '#1976D2', '#E65100']; all_rows = []; reports = {}
    for name, color in zip(names, colors):
        folder = out / 'calibration' / name
        history = io.read(folder / 'training_history.json') if (folder / 'training_history.json').exists() else []
        val = io.read(folder / 'validation_history.json') if (folder / 'validation_history.json').exists() else []
        status = io.read(folder / 'status.json') if (folder / 'status.json').exists() else {'status': 'not_started'}
        reports[name] = {'status': status['status'], 'committed_updates': len(history),
                         'validation_history': val, 'lambda': p['calibration']['conditions'][name],
                         'anchor_tokens_seen': sum(r['anchor_tokens'] for r in history)}
        label = f"{name} (lambda={p['calibration']['conditions'][name]:g})"
        if history:
            x = [r['step'] for r in history]; y = [r['target_loss'] for r in history]
            axes[0].plot(x, y, color=color, alpha=0.15, linewidth=0.6)
            axes[0].plot(x, rolling(y, 20), color=color, label=label)
        anchor = [r for r in history if r['is_anchor_update']]
        if anchor:
            x = [r['step'] for r in anchor]
            axes[2].plot(x, [r['anchor_kl'] for r in anchor], color=color, alpha=.2)
            axes[2].plot(x, rolling([r['anchor_kl'] for r in anchor], 10), color=color, label=label)
            axes[5].plot(x, rolling([r['weighted_anchor_kl'] for r in anchor], 10), color=color, label=label)
            valid = [r for r in anchor if r['target_gradient_norm'] > 0]
            axes[4].plot([r['step'] for r in valid],
                         [r['weighted_anchor_gradient_norm']/r['target_gradient_norm'] for r in valid],
                         color=color, label=label)
        if val:
            x = [r['step'] for r in val]
            axes[1].plot(x, [r['validation_loss'] for r in val], 'o-', color=color, label=label)
            axes[3].plot(x, [r['selected_set_anchor_KL'] for r in val], 'o-', color=color, label=label)
        for record in history:
            all_rows.append({'condition': name, **record})
    titles = [
        ('Math training loss: raw + rolling mean (20)', 'Token-mean cross-entropy'),
        ('Original 300 validation: known training overlap', 'Response-token-weighted reference loss'),
        ('KL on matching anchor updates: raw + mean (10)', 'Unweighted token-mean KL'),
        ('Fixed 255 anchor positions: training-set diagnostic', 'Full selected-set mean KL; not retention'),
        ('Gradient norms before clipping', 'Weighted anchor norm / math norm'),
        ('Weighted KL at anchor updates: mean (10)', 'Lambda times token-mean KL')]
    for axis, (title, ylabel) in zip(axes, titles):
        axis.set_title(title, fontsize=10); axis.set_xlabel('Optimizer update'); axis.set_ylabel(ylabel, fontsize=9)
        axis.grid(alpha=.2)
        if axis.lines:
            axis.legend(fontsize=7)
    axes[4].set_yscale('symlog', linthresh=1e-4)
    complete = all(r['status'] == 'complete' for r in reports.values())
    title = ('SYNTHETIC CPU FIXTURE — NO EXPERIMENTAL RESULTS' if synthetic else
             'Damage-A coefficient calibration — ' + ('complete' if complete else 'partial progress'))
    figure.suptitle(title, fontsize=13)
    charts = out / 'curves'; charts.mkdir(parents=True, exist_ok=True)
    figure.savefig(charts / 'loss_curves.png', dpi=160)
    figure.savefig(charts / 'loss_curves.svg'); plt.close(figure)
    # Total objectives are isolated by run to discourage choosing the smallest
    # number across different lambda values.
    fig, panels = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    for axis, name, color in zip(panels, names, colors):
        records = [r for r in all_rows if r['condition'] == name]
        if records:
            axis.plot([r['step'] for r in records], [r['total_loss'] for r in records], color=color, alpha=.3)
            axis.plot([r['step'] for r in records], rolling([r['total_loss'] for r in records], 20), color=color)
        axis.set_title(name); axis.set_ylabel('Its own total objective'); axis.set_xlabel('Optimizer update'); axis.grid(alpha=.2)
    fig.suptitle(('SYNTHETIC FIXTURE — ' if synthetic else '') + 'Do not rank coefficients by total loss', fontsize=12)
    fig.savefig(charts / 'total_objectives.png', dpi=140); plt.close(fig)
    fields = ['condition', 'step', 'coefficient', 'target_loss', 'target_tokens', 'anchor_kl',
              'weighted_anchor_kl', 'total_loss', 'anchor_tokens', 'anchor_tokens_seen',
              'target_gradient_norm', 'weighted_anchor_gradient_norm', 'combined_gradient_norm_before_clip',
              'learning_rate_used', 'learning_rate_next']
    with (charts / 'curves.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(all_rows)
    io.atomic(charts / 'report.json', {'scope': 'Calibration only; no retention scores', 'synthetic': synthetic,
              'automatic_coefficient_winner': None, 'conditions': reports,
              'limits': ['Known overlap in original300 validation', 'No comparison by total objective',
                         'Lower anchor KL is not retained capability', '600 steps are not full training']})
    print('[CURVES] ' + str(charts / 'loss_curves.png'), flush=True)
