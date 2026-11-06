"""Unchanged standard-library source guards extracted from the successful A exporter."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
WORK=Path("/home/msai/hemanth005/AI6130_Srihari_Step5_20261102")
COMMIT="ab02b28be9f491c1cd8fb4cc025a91b26455a779"


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def rows(p):
    with p.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_new(p,value):
    with p.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())


def verify_bundle(root=ROOT):
    for name,h in read(root/'files.sha256.json').items():
        if sha(root/name)!=h:raise ValueError('Diagnostic package changed: '+name)


def guard(work=WORK,root=ROOT):
    exp=work/'experiment_256_v1';repo=work/'code_git_ab02b28b'
    expected=read(root/'expected_source_manifest.json');actual=read(exp/'manifest.json')
    if actual!=expected:raise ValueError('Step5 manifest differs from the supplied pinned experiment')
    current=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if current!=COMMIT:raise ValueError('Wrong source commit')
    for rel,h in expected['code_sha256'].items():
        if sha(repo/rel)!=h:raise ValueError('Changed pinned source: '+rel)
    protocol_path=exp/'config/protocol_256.json';p=read(protocol_path)
    adapter=Path(actual['adapter']);saved=adapter.parent.parent/'protocol.json'
    checks=[(protocol_path,actual['protocol_sha256']),(Path(actual['input_anchor_file']),actual['input_anchor_sha256']),
            (adapter/'adapter_model.safetensors',actual['adapter_sha256']),(saved,actual['saved_checkpoint_protocol_sha256'])]
    for path,h in checks:
        if sha(path)!=h:raise ValueError('Changed protected input: '+str(path))
    old=read(saved)
    if old['datasets']['math10k']['validation_size']!=300 or p['datasets']['math10k']['validation_size']!=300:
        raise ValueError('This diagnostic requires the original friend300 branch')
    if not str(adapter).endswith('/step_000600') or old['model']!=p['model']:
        raise ValueError('Wrong original step600/base provenance')
    if p['model']!={'repository_id':'meta-llama/Llama-3.2-3B','revision':'13afe5124825b4f3751f836b40dafda64c1ed062'}:
        raise ValueError('Wrong teacher checkpoint')
    if (p['runtime']['teacher_max_new_tokens'],p['runtime']['max_sequence_tokens'])!=(256,2048):
        raise ValueError('Original cap/context changed')
    if p['evaluation']['prompt_format'].replace('\\n','\n')!='Question: {question}\nAnswer:':
        raise ValueError('Original prompt changed')
    cache=exp/'full/teacher_cache.jsonl';cm=read(cache.with_name('teacher_cache.jsonl.meta.json'))
    if cm['cache_sha256']!=sha(cache) or cm['source_sha256']!=actual['input_anchor_sha256']:
        raise ValueError('Teacher cache source/checksum mismatch')
    if (cm['cached_anchor_count'],cm['input_anchor_count'],cm['skipped_anchor_count'])!=(13743,13769,26):
        raise ValueError('Wrong complete cache counts')
    protected={str(path):h for path,h in checks}
    protected.update({str(exp/'manifest.json'):sha(exp/'manifest.json'),str(cache):sha(cache),str(cache.with_name('teacher_cache.jsonl.meta.json')):sha(cache.with_name('teacher_cache.jsonl.meta.json'))})
    scores=exp/'full/vulnerability_scores.jsonl'
    if scores.exists():protected[str(scores)]=sha(scores)
    return p,actual,repo,exp,cache,protected
