"""Full fixed labelled splits; input-only overlap flags never change membership."""
import hashlib
import json
import os
import re
import unicodedata
import urllib.request
from zipfile import ZipFile
import eval_io as io


def normalize(text):
    return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',text).casefold()))


def choice_prompt(question,labels,texts):
    if not isinstance(question,str) or not question.strip():raise ValueError('Empty question')
    if not 2<=len(labels)<=5 or len(labels)!=len(texts) or len(set(labels))!=len(labels):raise ValueError('Invalid option labels')
    if any(not isinstance(x,str) or not x.strip() for x in texts):raise ValueError('Empty option text')
    return (f'Question: {question}\n'+'Options:\n'+'\n'.join(f'{a}. {b}' for a,b in zip(labels,texts))
            +'\n\nSelect the correct answer option.\nRespond with only the option letter.\nAnswer:')


def canonical(name,doc,index):
    from kl_anchors.evaluate_math10k import choice_data
    from kl_anchors.metrics import extract_gsm8k_final_answer
    if name=='gsm8k':
        q=doc['question'];prompt=f'Question: {q}\nAnswer:'
        if extract_gsm8k_final_answer(doc['answer']) is None:raise ValueError('Invalid GSM8K gold number')
        tails=[]
    else:
        q,labels,texts,gold=choice_data(name,doc)
        if not 0<=gold<len(labels):raise ValueError('Invalid explicit gold index')
        prompt=choice_prompt(q,labels,texts);tails=[' '+a for a in labels]
    return {'key':f'{name}:{index}','benchmark':name,'source_index':index,'raw':doc,
            'question':q,'prompt':prompt,'choice_continuations':tails}


def load_social(spec,out):
    path=out/'socialiqa-train-dev.zip'
    if not path.exists():
        with urllib.request.urlopen(spec['url'],timeout=90) as response:data=response.read()
        if hashlib.sha256(data).hexdigest()!=spec['sha256']:raise ValueError('Social IQa archive checksum differs')
        tmp=path.with_suffix('.partial');tmp.write_bytes(data);os.replace(tmp,path)
    if io.sha(path)!=spec['sha256']:raise ValueError('Cached Social IQa archive differs')
    with ZipFile(path) as z:data=z.read(spec['member']);labels=z.read(spec['labels_member'])
    if hashlib.sha256(data).hexdigest()!=spec['member_sha256'] or hashlib.sha256(labels).hexdigest()!=spec['labels_sha256']:
        raise ValueError('Social IQa inputs or label alignment differs')
    docs=[json.loads(x) for x in data.decode().splitlines() if x.strip()]
    gold=[x.strip() for x in labels.decode().splitlines() if x.strip()]
    if len(docs)!=1954 or len(gold)!=len(docs):raise ValueError('Social IQa labelled split count differs')
    return [{**r,'label':v} for r,v in zip(docs,gold)]


def prepare(p,out,tokenizer,identity,known,expired):
    from datasets import load_dataset
    result={}
    for name,spec in p['benchmarks'].items():
        if expired():return None
        folder=out/'prepared'/name;folder.mkdir(parents=True,exist_ok=True)
        path=folder/'rows.jsonl';manifest=folder/'manifest.json'
        if manifest.exists():
            m=io.read(manifest)
            if m['identity']!=identity or m['rows_sha256']!=io.sha(path) or m['source']!=spec:
                raise ValueError('Prepared dataset changed')
            result[name]=io.rows(path)
            if len(result[name])!=spec['expected_count']:raise ValueError('Prepared count changed')
            continue
        print('[PREPARE] '+name,flush=True)
        docs=load_social(spec,out) if name=='social_iqa' else [dict(r) for r in load_dataset(spec['repo_id'],name=spec.get('config'),revision=spec['revision'],split=spec['split'])]
        if len(docs)!=spec['expected_count']:raise ValueError('Unexpected full split count: '+name)
        examples=[];maximum=0;flags={k:[] for k in known}
        for i,doc in enumerate(docs):
            row=canonical(name,doc,i)
            n=len(tokenizer.encode(row['prompt'],add_special_tokens=False))
            extra=p['generation']['max_new_tokens'] if name=='gsm8k' else max(len(tokenizer.encode(x,add_special_tokens=False)) for x in row['choice_continuations'])
            if not n or n+extra>p['context_tokens']:raise ValueError('Context limit exceeded; no truncation: '+row['key'])
            maximum=max(maximum,n+extra)
            for k,values in known.items():
                if normalize(row['question']) in values or normalize(row['prompt']) in values:flags[k].append(row['key'])
            examples.append(row)
        tmp=path.with_suffix('.partial')
        with tmp.open('w') as f:
            for row in examples:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
        io.freeze(manifest,{'identity':identity,'source':spec,'count':len(examples),'rows_sha256':io.sha(path),
                           'maximum_prompt_plus_allowance_tokens':maximum,'exact_surface_flags':flags,
                           'excluded_rows':0,'scope':'Input-only exact surface screen; not semantic/pretraining isolation'})
        result[name]=examples
    return result
