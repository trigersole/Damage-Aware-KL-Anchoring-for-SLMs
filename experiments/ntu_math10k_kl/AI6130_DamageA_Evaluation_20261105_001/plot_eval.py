"""Full-cell benchmark summaries only; missing results never plotted as zero."""
import csv
import os


def plot(p,out,report,*,synthetic=False):
    folder=out/'summary';folder.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(folder/'.mplconfig'));os.environ.setdefault('XDG_CACHE_HOME',str(folder/'.cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    rows=[];colors=['#546E7A','#E68A2E','#1976D2']
    for ax,(name,entry) in zip(axes.ravel(),report['benchmarks'].items()):
        for i,mode in enumerate(p['modes']):
            if mode not in entry:continue
            result=entry[mode];value=100*result['accuracy']
            ax.bar(i,value,color=colors[i]);ax.text(i,value+1,f"{result['correct']}/{result['total']}",ha='center',fontsize=8)
            rows.append({'benchmark':name,'condition':mode,**{k:result[k] for k in ['correct','total','accuracy','parse_failures']}})
        ax.set_xticks(range(3),p['modes'],rotation=15);ax.set_ylim(0,110);ax.set_ylabel('Accuracy (%)');ax.set_title(name)
    axes.ravel()[-1].axis('off')
    fig.suptitle('Fixed256-token profile — '+report['state']+'\nOnly completed cells shown; no model/checkpoint tuning')
    if synthetic:fig.text(.5,.5,'SYNTHETIC LAYOUT FIXTURE — NO BENCHMARK RESULTS',ha='center',rotation=25,color='red',alpha=.3,fontsize=22)
    fig.savefig(folder/'accuracy.png',dpi=160);fig.savefig(folder/'accuracy.svg');plt.close(fig)
    with (folder/'accuracy.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['benchmark','condition','correct','total','accuracy','parse_failures']);writer.writeheader();writer.writerows(rows)
