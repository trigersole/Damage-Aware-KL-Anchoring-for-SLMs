"""Per-item durable inference output; completed scientific outputs are immutable."""
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid


def read(p):return json.loads(Path(p).read_text())


def rows(p):
    with Path(p).open() as f:return [json.loads(s) for s in f if s.strip()]


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name('.'+p.name+'.'+uuid.uuid4().hex)
    with tmp.open('x') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)


def freeze(p,value):
    if p.exists():
        if read(p)!=value:raise ValueError('Frozen evaluation identity/result differs: '+str(p))
    else:atomic(p,value)


def recover(path,expected,identity,*,completed=False):
    if not path.exists():return []
    result=[];end=0
    with path.open('rb') as f:
        for line in f:
            if not line.endswith(b'\n'):
                if completed:raise ValueError('Completed result contains a torn line')
                break
            r=json.loads(line);index=len(result)
            if index>=len(expected) or r['identity']!=identity or r['key']!=expected[index]['key'] or r['row_sha256']!=digest(expected[index]):
                raise ValueError('Saved per-item order/source/identity differs')
            if r['record_sha256']!=digest({k:v for k,v in r.items() if k!='record_sha256'}):
                raise ValueError('Saved prediction checksum differs')
            result.append(r);end=f.tell()
    if end!=path.stat().st_size:
        with path.open('r+b') as f:f.truncate(end)  # Uncommitted partial tail only.
    return result


def summarize(records):
    n=len(records)
    if not n:raise ValueError('Cannot summarize an empty benchmark')
    r={'correct':sum(x['correct'] for x in records),'total':n,
       'accuracy':sum(x['correct'] for x in records)/n,'parse_failures':sum(x['parse_failure'] for x in records)}
    if 'generated_token_ids' in records[0]:
        r['generation_diagnostics']={'cap':256,'at_cap':sum(x['at_cap'] for x in records),
          'native_eos':sum(x['ended_with_eos'] for x in records),
          'note':'Descriptive only; no post-test token-budget tuning'}
    return r


def cell(backend,expected,folder,identity,expired,score):
    folder.mkdir(parents=True,exist_ok=True);freeze(folder/'identity.json',identity)
    final=folder/'items.jsonl';partial=folder/'items.jsonl.partial'
    records=recover(final if final.exists() else partial,expected,identity,completed=final.exists())
    if final.exists():
        if len(records)!=len(expected) or read(folder/'summary.json')!=summarize(records):
            raise ValueError('Completed cell count/summary differs')
        receipt={'identity':identity,'items_sha256':sha(final),'summary_sha256':sha(folder/'summary.json'),'count':len(records)}
        freeze(folder/'completion.json',receipt);return True
    initial=len(records);started=time.monotonic()
    print(f"[{identity['mode']}/{identity['benchmark']}] Resume {initial}/{len(expected)}",flush=True)
    with partial.open('a') as f:
        for row in expected[initial:]:
            if expired():return False
            result=score(backend,row)
            record={'identity':identity,'key':row['key'],'row_sha256':digest(row),**result}
            if any(not math.isfinite(x) for x in record.get('choice_scores',[])):
                raise ValueError('Nonfinite choice score')
            record['record_sha256']=digest(record)
            f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
            records.append(record);done=len(records)
            if done==initial+1 or done%50==0 or done==len(expected):
                eta=(time.monotonic()-started)/(done-initial)*(len(expected)-done)/60
                print(f"[{identity['mode']}/{identity['benchmark']}] {done}/{len(expected)} | cell ETA {eta:.1f} min",flush=True)
    freeze(folder/'summary.json',summarize(records));os.replace(partial,final)
    freeze(folder/'completion.json',{'identity':identity,'items_sha256':sha(final),'summary_sha256':sha(folder/'summary.json'),'count':len(records)})
    return True


def report(p,out):
    result={'state':'complete','remaining':[],'benchmarks':{},'scope':'Fixed-model evaluation; no training/tuning'}
    for name in p['benchmarks']:
        entry={}
        for mode in p['modes']:
            folder=out/mode/name
            if (folder/'completion.json').exists() and (folder/'summary.json').exists():entry[mode]=read(folder/'summary.json')
            else:result['remaining'].append({'mode':mode,'benchmark':name})
        if all(mode in entry for mode in p['modes']):
            entry['damage_vs_task_only_percentage_points']=100*(entry['damage_A']['accuracy']-entry['task_only']['accuracy'])
            entry['task_only_vs_base_percentage_points']=100*(entry['task_only']['accuracy']-entry['base']['accuracy'])
            entry['damage_vs_base_percentage_points']=100*(entry['damage_A']['accuracy']-entry['base']['accuracy'])
            before=rows(out/'task_only'/name/'items.jsonl');after=rows(out/'damage_A'/name/'items.jsonl')
            if [(r['key'],r['row_sha256']) for r in before]!=[(r['key'],r['row_sha256']) for r in after]:raise ValueError('Paired benchmark rows differ')
            entry['damage_vs_task_only_paired']={
              'gained_correct':sum(not b['correct'] and a['correct'] for b,a in zip(before,after)),
              'lost_correct':sum(b['correct'] and not a['correct'] for b,a in zip(before,after))}
        result['benchmarks'][name]=entry
    if result['remaining']:result['state']='paused' if out.exists() else 'not_started'
    return result
