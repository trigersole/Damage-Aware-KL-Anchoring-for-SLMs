"""Scientific training diagnostics, with inherited prefix clearly marked."""
import csv
import json
import os
from pathlib import Path


def rolling(values, size):
    return [sum(values[max(0, i-size+1):i+1]) / min(i+1, size) for i in range(len(values))]


def plot(folder, *, synthetic=False):
    out = folder / 'curves'; out.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(out / '.mplconfig'))
    os.environ.setdefault('XDG_CACHE_HOME', str(out / '.cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history = json.loads((folder / 'training_history.json').read_text())
    validation = json.loads((folder / 'validation_history.json').read_text())
    status = json.loads((folder / 'status.json').read_text())
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    axes = axes.ravel()
    x = [r['step'] for r in history]; loss = [r['target_loss'] for r in history]
    axes[0].plot(x, loss, color='0.65', alpha=.25, linewidth=.6)
    axes[0].plot(x, rolling(loss, 20), color='#1976D2')
    axes[1].plot([r['step'] for r in validation], [r['validation_loss'] for r in validation], 'o-', color='#1976D2')
    anchors = [r for r in history if r['anchor_tokens']]
    if anchors:
        ax = [r['step'] for r in anchors]
        axes[2].plot(ax, [r['anchor_kl'] for r in anchors], color='#1976D2', alpha=.2)
        axes[2].plot(ax, rolling([r['anchor_kl'] for r in anchors], 10), color='#1976D2')
        valid = [r for r in anchors if r['target_gradient_norm'] > 0]
        axes[4].plot([r['step'] for r in valid], [r['weighted_anchor_gradient_norm']/r['target_gradient_norm'] for r in valid], color='#1976D2')
        axes[5].plot(ax, rolling([r['weighted_anchor_kl'] for r in anchors], 10), color='#1976D2')
    axes[3].plot([r['step'] for r in validation], [r['selected_set_anchor_KL'] for r in validation], 'o-', color='#1976D2')
    titles = ['Math training loss: raw + mean20', 'Original300 validation: known training overlap',
              'Raw anchor KL: training diagnostic', 'Fixed255 training-anchor positions: not retention',
              'Weighted anchor/math gradient-norm ratio', 'Weighted anchor KL: mean10']
    for a, title in zip(axes, titles):
        a.set_title(title); a.set_xlabel('Optimizer update'); a.grid(alpha=.2)
        a.axvline(600, color='0.3', linestyle='--', linewidth=.8)
    fig.suptitle('Damage-A lambda0.01 — ' + status['status'] + '\nUpdates1–600 inherited unchanged; continuation starts601')
    if synthetic:
        fig.text(.5,.5,'LAYOUT FIXTURE — NO FULL-TRAINING RESULT',ha='center',va='center',
                 fontsize=24,rotation=28,color='red',alpha=.3)
    fig.savefig(out / 'training_curves.png', dpi=160)
    fig.savefig(out / 'training_curves.svg'); plt.close(fig)
    fields = list(dict.fromkeys(k for r in history for k in r))
    with (out / 'curves.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(history)
    (out / 'report.json').write_text(json.dumps({'status': status['status'], 'step': status['step'],
        'inherited_updates': 600, 'coefficient': .01, 'validation_history': validation,
        'best': status['best_by_original300_reference_loss'], 'retention_scores': None,
        'synthetic_layout_fixture': synthetic}, indent=2)+'\n')
