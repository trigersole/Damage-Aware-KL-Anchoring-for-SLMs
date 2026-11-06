"""Approved five-benchmark,three-model inference; source checks+probes+resumable cells."""
import argparse
import fcntl
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

import eval_io as io
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'snapshot'))
from kl_anchors.evaluate import HuggingFaceBackend,set_evaluation_seed
from kl_anchors.evaluate_math10k import eval_row


class Budget:
    def __init__(self,seconds):self.start=time.monotonic();self.seconds=seconds;self.requested=False
    def stop(self,*_):self.requested=True
    def expired(self):return self.requested or time.monotonic()-self.start>=self.seconds


class AuditedBackend(HuggingFaceBackend):
    def generate(self,prompt,*,max_new_tokens):
        # Same tokenizer/generate/decode arguments as the frozen friend backend;
        # additional bookkeeping does not call the model a second time.
        inputs=self.tokenizer(prompt,return_tensors='pt',add_special_tokens=self.add_special_tokens)
        inputs={k:v.to(self._input_device()) for k,v in inputs.items()}
        pad=self.tokenizer.pad_token_id
        if pad is None:pad=self.tokenizer.eos_token_id
        with self.torch.inference_mode():
            output=self.model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,num_beams=1,pad_token_id=pad)
        continuation=output[0,inputs['input_ids'].shape[1]:]
        ids=continuation.detach().cpu().tolist()
        eos=getattr(getattr(self.model,'generation_config',None),'eos_token_id',None)
        if eos is None:eos=self.tokenizer.eos_token_id
        eos_ids=list(eos) if isinstance(eos,(list,tuple)) else [eos]
        ended=bool(ids and ids[-1] in eos_ids)
        self.last_generation={'generated_token_ids':ids,'generated_tokens':len(ids),
             'at_cap':len(ids)==max_new_tokens,'ended_with_eos':ended,'native_eos_ids':eos_ids,
             'cap_reached_without_eos':len(ids)==max_new_tokens and not ended}
        return self.tokenizer.decode(continuation,skip_special_tokens=self.skip_special_tokens)


def score(backend,row):
    result=eval_row(backend,row['benchmark'],row['raw'],row['source_index'],256)
    if result['prompt']!=row['prompt']:raise ValueError('Friend scorer/input audit prompt differs')
    if row['benchmark']=='gsm8k':result.update(backend.last_generation)
    return result


def backend_protocol(p,mode,sources):
    model={**p['model'],'dtype':'bfloat16','device_map':'cuda:0','trust_remote_code':False,'quantization':'none'}
    if mode!='base':model['adapter_path']=sources[mode]['directory']
    return {'model':model,'tokenizer':{**p['model'],'use_fast':True,'add_special_tokens':False},
            'generation':p['generation']}


def verify(root=ROOT):
    for name,h in io.read(root/'files.sha256.json').items():
        if io.sha(root/name)!=h:raise ValueError('Evaluation source/config changed: '+name)
    p=io.read(root/'evaluation.json')
    if p['approved'] is not True or p['generation']!={'max_new_tokens':256,'do_sample':False,'num_beams':1,'skip_special_tokens':True}:
        raise ValueError('Approved generation profile changed')
    if p['modes']!=['base','task_only','damage_A'] or list(p['benchmarks'])!=['gsm8k','arc_easy','arc_challenge','openbookqa','social_iqa']:
        raise ValueError('Approved conditions/benchmarks changed')
    if [x['expected_count'] for x in p['benchmarks'].values()]!=[1319,2376,1172,500,1954] or p['tokenizer']['add_special_tokens'] is not False:
        raise ValueError('Approved full splits/tokenization changed')
    return p


def runtime(torch):
    packages={name:metadata.version(name) for name in ['torch','transformers','datasets','peft','accelerate','numpy']}
    for name,wanted in {'torch':'2.11.0+cu128','transformers':'4.57.6','datasets':'4.8.5','peft':'0.21.0','accelerate':'1.15.0'}.items():
        if packages[name]!=wanted:raise ValueError('Unexpected evaluation software: '+name)
    return {'packages':packages,'python':platform.python_version(),'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,
            'capability':list(torch.cuda.get_device_capability(0)),
            'matmul_precision':torch.get_float32_matmul_precision(),'TF32':torch.backends.cuda.matmul.allow_tf32}


def smoke_all(p,out,sources,identity,expired,factory=AuditedBackend):
    from data_eval import canonical
    for mode in p['modes']:
        file=out/'smoke'/(mode+'.json')
        if file.exists():
            if io.read(file)['identity']!=identity or io.read(file)['status']!='complete':raise ValueError('Synthetic probe identity/status differs')
            continue
        if expired():return False
        print('[PROBE] '+mode+';synthetic inputs only',flush=True)
        set_evaluation_seed(p['seed']);backend=factory(backend_protocol(p,mode,sources))
        try:
            fixture=canonical('social_iqa',{'context':'Synthetic hardware fixture.','question':'Choose a marker.',
                              'answerA':'First marker','answerB':'Second marker','answerC':'Third marker','label':'1'},0)
            scored=score(backend,fixture)
            if len(scored['choice_scores'])!=3 or not all(math.isfinite(x) for x in scored['choice_scores']):raise ValueError('Synthetic score probe failed')
            text=backend.generate('Question: Synthetic inference probe.\nAnswer:',max_new_tokens=8)
            if not 1<=backend.last_generation['generated_tokens']<=8:raise ValueError('Synthetic generation probe failed')
            io.freeze(file,{'identity':identity,'status':'complete','scope':'Synthetic generation/MC probes;no benchmark accuracy',
                            'response':text,'generation':backend.last_generation,'choice_scores':scored['choice_scores']})
        finally:
            del backend;gc.collect()
            import torch
            torch.cuda.empty_cache()
    return True


def run(root=ROOT):
    p=verify(root)
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('Submit model work through Slurm,not the head node')
    budget=Budget(p['max_wall_seconds'])
    for number in [signal.SIGUSR1,signal.SIGTERM,signal.SIGINT]:signal.signal(number,budget.stop)
    out=root/'results'/p['evaluation_id'];out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from source_eval import verify_sources
        sources,protected,known,parent_runtime=verify_sources(p,root,out)
        import torch
        from transformers import AutoTokenizer
        if not torch.cuda.is_available():raise ValueError('Allocated CUDA GPU required')
        r=runtime(torch)
        if any(r[k]!=parent_runtime[k] for k in ['python','gpu','cuda','capability','matmul_precision','TF32']):
            raise ValueError('Use matching training GPU/software controls: '+str(parent_runtime))
        if any(r['packages'][name]!=version for name,version in parent_runtime['packages'].items()):
            raise ValueError('Training/evaluation package versions differ')
        identity={'bundle':io.sha(root/'files.sha256.json'),'protocol':io.digest(p),'sources':sources,'runtime':r}
        io.freeze(out/'identity.json',identity);io.freeze(out/'evaluation.json',p)
        io.atomic(out/'jobs'/(os.environ['SLURM_JOB_ID']+'.json'),{'node':os.environ.get('SLURMD_NODENAME'),'runtime':r})
        from data_eval import prepare,normalize
        known={k:{normalize(x) for x in text} for k,text in known.items()}
        tokenizer=AutoTokenizer.from_pretrained(p['model']['repo_id'],revision=p['model']['revision'],use_fast=True,trust_remote_code=False)
        prepared=prepare(p,out,tokenizer,io.digest(identity),known,budget.expired)
        ready=prepared is not None and smoke_all(p,out,sources,io.digest(identity),budget.expired)
        if ready:
            for mode in p['modes']:
                pending=[]
                for name in p['benchmarks']:
                    cellid={'run':io.digest(identity),'mode':mode,'benchmark':name,
                            'dataset_sha256':io.sha(out/'prepared'/name/'rows.jsonl')}
                    if (out/mode/name/'items.jsonl').exists():
                        io.cell(None,prepared[name],out/mode/name,cellid,budget.expired,score)
                    else:pending.append((name,cellid))
                if not pending:continue
                if budget.expired():break
                set_evaluation_seed(p['seed']);print('[LOAD] '+mode+';inference only',flush=True)
                backend=AuditedBackend(backend_protocol(p,mode,sources))
                try:
                    for name,cellid in pending:
                        if not io.cell(backend,prepared[name],out/mode/name,cellid,budget.expired,score):break
                finally:
                    del backend;gc.collect();torch.cuda.empty_cache()
                if budget.expired():break
        if any(io.sha(f)!=h for f,h in protected.items()):raise ValueError('Protected model/source changed during inference')
        io.atomic(out/'preservation.json',{'all_unchanged':True,'protected_sha256':protected})
        result=io.report(p,out);io.atomic(out/'status.json',result)
        from plot_eval import plot
        plot(p,out,result)
        print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['preflight','run','report']);args=parser.parse_args()
    p=verify()
    if args.stage=='preflight':print(json.dumps({'status':'package_verified_not_run','models':p['modes'],'benchmarks':list(p['benchmarks']),
         'max_new_tokens':256,'expected_total_predictions':21963},indent=2))
    elif args.stage=='report':print(json.dumps(io.report(p,ROOT/'results'/p['evaluation_id']),indent=2))
    else:
        try:run()
        except Exception as exc:
            if os.environ.get('SLURM_JOB_ID'):
                io.atomic(ROOT/'results'/p['evaluation_id']/'failures'/(os.environ['SLURM_JOB_ID']+'.json'),
                          {'status':'failed','error_type':type(exc).__name__,'details':'See Slurm stderr;existing committed results preserved'})
            raise


if __name__=='__main__':main()
