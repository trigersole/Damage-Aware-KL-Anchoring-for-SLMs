"""Isolated teacher/step600 EOS diagnostic, two cached Dolly prompts only.

No training, prompt modification, forced continuation or KL-score replacement.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
WORK=Path('/home/msai/hemanth005/AI6130_Srihari_Step5_20261102')
COMMIT='ab02b28be9f491c1cd8fb4cc025a91b26455a779'
OUTPUT_NAME='teacher_eos_probe_20261104_001'
TARGETS=(('dholak_sitar',('Dholak','Sitar')),('hydraulophone_mandriola',('Hydraulophone','Mandriola')))


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


def select_cases(anchors,cached):
    found=[]
    for name,terms in TARGETS:
        matches=[a for a in anchors if all(re.search(r'\b'+re.escape(t)+r'\b',a['prompt'],re.I) for t in terms)]
        if len(matches)!=1:
            raise ValueError(f'{name}: expected one matching original prompt, found {len(matches)}; do not silently substitute')
        a=matches[0]
        if a['example_id'] not in cached:raise ValueError('Selected prompt missing from complete teacher cache')
        c=cached[a['example_id']]
        if c['response_token_count']!=len(c['continuation_token_ids']):raise ValueError('Cached token count mismatch')
        found.append({'case':name,'example_id':a['example_id'],'original_prompt':a['prompt'],**c})
    return found


def generation_kwargs(protocol,tokenizer):
    # Exactly the visible original run_step5.py generation controls. No changes
    # to EOS/min length, beam count, penalties or model generation defaults.
    return {'do_sample':False,'max_new_tokens':protocol['runtime']['teacher_max_new_tokens'],
            'pad_token_id':tokenizer.pad_token_id,'eos_token_id':tokenizer.eos_token_id}


def token_summary(vector,tokenizer,eos,letters,torch,top_k=10):
    vector=vector.detach().float()
    if torch.isnan(vector).any() or torch.isposinf(vector).any() or not torch.isfinite(vector).any():
        raise ValueError('Invalid raw/processed token vector')
    probabilities=vector.softmax(-1)
    values,indices=torch.topk(vector,min(top_k,vector.numel()))
    top=[]
    for value,i in zip(values.cpu().tolist(),indices.cpu().tolist()):
        top.append({'id':i,'text':tokenizer.decode([i],skip_special_tokens=False),'logit':value if math_finite(value) else None,'probability':float(probabilities[i].item())})
    def info(i):
        value=float(vector[i].item())
        return {'id':i,'score':value if math_finite(value) else None,'probability':float(probabilities[i].item()),'rank_with_ties':int((vector>vector[i]).sum().item())+1}
    return {'argmax_id':int(vector.argmax().item()),'top':top,'eos':info(eos),'letters':{name:info(ids[0]) for name,ids in letters.items() if len(ids)==1},'negative_infinity_count':int(torch.isneginf(vector).sum().item())}


def math_finite(value):
    import math
    return math.isfinite(value)


def inspect_positions(observation,continuation,tokenizer,torch,limit=4):
    if len(observation.scores)!=len(continuation) or len(observation.logits)!=len(continuation):
        raise ValueError('Diagnostic assumes greedy one-score-per-generated-token output')
    letters={x:tokenizer.encode(' '+x,add_special_tokens=False) for x in 'ABCD'}
    result=[]
    for index in range(min(limit,len(continuation))):
        raw=observation.logits[index][0].detach().float()
        processed=observation.scores[index][0].detach().float()
        finite=torch.isfinite(raw)&torch.isfinite(processed)
        delta=float((raw[finite]-processed[finite]).abs().max().item()) if finite.any() else None
        result.append({'position':index+1,'generated_prefix_ids':continuation[:index],'selected_token_id':continuation[index],
                       'selected_text':tokenizer.decode([continuation[index]],skip_special_tokens=False),
                       'raw':token_summary(raw,tokenizer,tokenizer.eos_token_id,letters,torch),
                       'processed':token_summary(processed,tokenizer,tokenizer.eos_token_id,letters,torch),
                       'raw_and_processed_identical':bool(torch.equal(raw,processed)),
                       'maximum_absolute_finite_score_change':delta})
    return result


def probe(model,tokenizer,case,protocol,torch,seed_fn):
    template=protocol['evaluation']['prompt_format'].replace('\\n','\n')
    prompt=template.replace('{question}',case['original_prompt'])
    ids=tokenizer.encode(prompt,add_special_tokens=False)
    if ids!=case['prompt_token_ids']:raise ValueError('Rendered prompt tokens differ from saved teacher input')
    if len(ids)+256>2048:raise ValueError('Selected case violates original eligibility')
    inputs=torch.tensor([ids],dtype=torch.long,device=protocol['runtime']['device'])
    kwargs=generation_kwargs(protocol,tokenizer)
    native=copy.deepcopy(model.generation_config)
    effective=copy.deepcopy(native)
    effective.update(**kwargs)
    if effective.num_beams!=1 or effective.do_sample:
        raise ValueError('Inherited generation is not single-beam greedy; inspect before proceeding')
    model.eval();seed_fn(int(protocol['seed']))
    with torch.inference_mode():
        ordinary=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),**kwargs)
    plain=ordinary[0,len(ids):].tolist();del ordinary
    # Keep the same256 budget even for the instrumentation call: changing it
    # could move a forced-EOS boundary and invalidate this comparison.
    seed_fn(int(protocol['seed']))
    with torch.inference_mode():
        observed=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),**kwargs,
                                return_dict_in_generate=True,output_scores=True,output_logits=True)
    instrumented=observed.sequences[0,len(ids):].tolist()
    result={'case':case['case'],'example_id':case['example_id'],'prompt':prompt,'prompt_token_ids':ids,
            'cached_teacher_ids':case['continuation_token_ids'],'ordinary_generation_ids':plain,
            'instrumented_generation_ids':instrumented,'ordinary_matches_instrumented':plain==instrumented,
            'ordinary_matches_cached_teacher':plain==case['continuation_token_ids'],
            'response_with_special_tokens':tokenizer.decode(plain,skip_special_tokens=False),
            'response_without_special_tokens':tokenizer.decode(plain,skip_special_tokens=True),
            'ordinary_ended_with_eos':bool(plain and plain[-1]==tokenizer.eos_token_id),
            'cached_teacher_response_with_special_tokens':tokenizer.decode(case['continuation_token_ids'],skip_special_tokens=False),
            'tokenizer_special_token_ids':{'bos':tokenizer.bos_token_id,'eos':tokenizer.eos_token_id,'pad':tokenizer.pad_token_id},
            'native_generation_config':native.to_dict(),'effective_generation_config':effective.to_dict(),
            'explicit_original_generation_kwargs':kwargs,'instrumented_positions':inspect_positions(observed,instrumented,tokenizer,torch)}
    del inputs,observed
    return result


def run():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Submit through Slurm; no model computation on head node')
    verify_bundle()
    protocol,manifest,repo,exp,cache,protected=guard()
    anchors=list(rows(Path(manifest['input_anchor_file'])))
    if len(anchors)!=13769 or len({r['example_id'] for r in anchors})!=13769:raise ValueError('Wrong anchor membership/IDs')
    wanted={a['example_id'] for a in anchors if any(all(re.search(r'\b'+re.escape(t)+r'\b',a['prompt'],re.I) for t in terms) for _,terms in TARGETS)}
    cached={r['example_id']:r for r in rows(cache) if r['example_id'] in wanted}
    cases=select_cases(anchors,cached)
    out=exp/'analysis'/OUTPUT_NAME
    out.mkdir(parents=True,exist_ok=False)
    write_new(out/'source_receipt.json',{'manifest':manifest,'protected_file_sha256_before':protected,'probe_source_sha256':sha(Path(__file__)),'job':os.environ['SLURM_JOB_ID'],'validation_branch':300,'scope':'Two cached prompts; ordinary teacher and step600 generation plus score instrumentation; no KL recomputation/training'})
    try:
        sys.path.insert(0,str(repo/'src'))
        import torch
        import kl_anchors.hf_runtime as runtime
        if Path(runtime.__file__).resolve()!=repo/'src/kl_anchors/hf_runtime.py':raise ValueError('Wrong imported model loader')
        if not torch.cuda.is_available():raise ValueError('CUDA allocation unavailable')
        versions={p:metadata.version(p) for p in ('torch','transformers','peft','accelerate')}
        if versions['torch']!='2.11.0+cu128' or versions['transformers']!='4.57.6' or versions['peft']!='0.21.0':raise ValueError('Runtime versions differ from completed experiment')
        write_new(out/'runtime.json',{'packages':versions,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'python':sys.version})
        tokenizer=runtime.load_tokenizer(protocol['model'])
        summaries=[];start=time.monotonic()
        for mode in ('teacher','step600'):
            runtime.set_seed(int(protocol['seed']))
            model=runtime.load_base(protocol['model'],precision=protocol['lora']['precision'],quantization=protocol['lora']['quantization'],device=protocol['runtime']['device'])
            if mode=='step600':model=runtime.attach_saved_lora(model,manifest['adapter'])
            model.eval()
            for case in cases:
                print(f"[PROBE] model={mode} case={case['case']} id={case['example_id']}",flush=True)
                result=probe(model,tokenizer,case,protocol,torch,runtime.set_seed)
                write_new(out/f"{mode}-{case['case']}.json",result)
                summary={'model':mode,'case':case['case'],'response_token_count':len(result['ordinary_generation_ids']),
                         'ended_with_eos':result['ordinary_ended_with_eos'],'ordinary_matches_instrumented':result['ordinary_matches_instrumented'],
                         'teacher_replay_matches_cache':result['ordinary_matches_cached_teacher'] if mode=='teacher' else None,
                         'text':result['response_with_special_tokens'],
                         'positions':[{'position':p['position'],'selected_id':p['selected_token_id'],'raw_argmax_id':p['raw']['argmax_id'],
                                       'processed_argmax_id':p['processed']['argmax_id'],'raw_eos_probability':p['raw']['eos']['probability'],
                                       'raw_eos_rank':p['raw']['eos']['rank_with_ties'],'processed_eos_probability':p['processed']['eos']['probability'],
                                       'scores_unchanged':p['raw_and_processed_identical']} for p in result['instrumented_positions']]}
                summaries.append(summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
            del model;gc.collect();torch.cuda.empty_cache()
        unchanged={path:sha(Path(path))==h for path,h in protected.items()}
        if not all(unchanged.values()):raise ValueError('Protected completed experiment changed during diagnostic; stop for review')
        write_new(out/'summary.json',{'status':'complete','models':['teacher','step600'],'questions':2,'ordinary_generations':4,'instrumented_replays':4,
                                     'records':summaries,'protected_files_unchanged':unchanged,'elapsed_seconds':time.monotonic()-start,
                                     'interpretation_limit':'Logits can explain the immediate greedy/processor mechanism, not why a response pattern was learned. Divergent replay or instrumentation must be inspected before causal interpretation. No KL-score decomposition is performed.'})
        print('[COMPLETE] '+str(out/'summary.json'),flush=True)
    except Exception as error:
        write_new(out/'failure.json',{'status':'failed','type':type(error).__name__,'message':str(error),'old_outputs_not_modified_by_probe':True})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    verify_bundle()
    if args.preflight:
        m=read(ROOT/'expected_source_manifest.json')
        print(json.dumps({'status':'prepared_not_run','adapter':m['adapter'],'commit':m['code_commit'],'cases':[name for name,_ in TARGETS],'models':['teacher','step600'],'changes_to_generation_protocol':False},indent=2))
    else:run()


if __name__=='__main__':main()
