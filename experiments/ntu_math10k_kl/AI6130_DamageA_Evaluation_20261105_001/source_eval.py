"""Read-only model/source provenance checks before any benchmark inference."""
from pathlib import Path
import eval_io as io


def architecture(directory,model):
    c=io.read(directory/'adapter_config.json')
    if (c['base_model_name_or_path'],c['r'],c['lora_alpha'],c['lora_dropout'])!=(model['repo_id'],128,256,0):
        raise ValueError('Unexpected adapter architecture: '+str(directory))
    if set(c['target_modules'])!={'q_proj','k_proj','v_proj','o_proj'}:raise ValueError('Unexpected adapter modules')


def verify_sources(p,root,out):
    full=Path(p['full_training_package']);manifest=io.read(root/'full_training_files.sha256.json')
    if io.sha(full/'files.sha256.json')!=p['full_training_bundle_sha256'] or io.read(full/'files.sha256.json')!=manifest:
        raise ValueError('Full training package identity differs')
    protected={str(full/'files.sha256.json'):io.sha(full/'files.sha256.json')}
    for f,h in manifest.items():
        if io.sha(full/f)!=h:raise ValueError('Full training source/config changed: '+f)
        protected[str(full/f)]=h
    plan=io.read(full/'plan.json');approval=io.read(full/'execution_approval.json')
    if approval['approved'] is not True or approval['plan_sha256']!=io.sha(full/'plan.json'):raise ValueError('Full run approval differs')
    protected[str(full/'execution_approval.json')]=io.sha(full/'execution_approval.json')
    run=full/'results'/plan['plan_id'];status=io.read(run/'completion.json')
    if io.read(run/'status.json')!=status:raise ValueError('Completion/status differ')
    if (status['status'],status['step'],status['coefficient'],status['examples_seen'],status['target_response_tokens_seen'],status['anchor_positions_seen'])!=('complete',3609,.01,28857,3875691,57728):
        raise ValueError('Expected complete original300 damage-A run')
    expected=io.read(root/'expected_full_report.json')
    history=io.read(run/'validation_history.json');best=io.read(run/'best_model.json')
    if history!=expected['validation_history'] or best!=expected['best'] or status['best_by_original300_reference_loss']!=best:
        raise ValueError('Fixed validation history/selected adapter differs')
    chosen=min((r for r in history if r['step']>0),key=lambda r:(r['validation_loss'],r['step']))
    if chosen['step']!=2400 or best['step']!=p['selected_step'] or best['official_test_used'] is not False:
        raise ValueError('Review selected checkpoint before evaluation')
    damage=run/'checkpoints/step_002400'
    if str(damage)!=best['checkpoint']:raise ValueError('Selected checkpoint path differs')
    receipt=io.read(damage/'receipt.json')
    if receipt['identity']!=status['identity'] or receipt['step']!=2400 or status['identity']['bundle_sha256']!=p['full_training_bundle_sha256']:
        raise ValueError('Selected checkpoint receipt differs')
    for f,h in receipt['files'].items():
        if io.sha(damage/f)!=h:raise ValueError('Selected checkpoint changed: '+f)
        protected[str(damage/f)]=h
    protected[str(damage/'receipt.json')]=io.sha(damage/'receipt.json')
    for f in ['completion.json','status.json','best_model.json','validation_history.json','handoff_verified.json']:
        protected[str(run/f)]=io.sha(run/f)
    if io.read(run/'handoff_verified.json')['status']!='complete':raise ValueError('Parent handoff not verified')
    parent=Path(plan['parent_package']);parent_manifest=io.read(full/'parent_files.sha256.json')
    if io.sha(parent/'files.sha256.json')!=plan['parent_bundle_sha256']:raise ValueError('Original calibration parent differs')
    protected[str(parent/'files.sha256.json')]=io.sha(parent/'files.sha256.json')
    for f,h in parent_manifest.items():
        if io.sha(parent/f)!=h:raise ValueError('Original parent source/data differs: '+f)
        protected[str(parent/f)]=h
    parent_protocol=io.read(parent/'protocol.json')
    if parent_protocol['model']!={'repository_id':p['model']['repo_id'],'revision':p['model']['revision']}:
        raise ValueError('Wrong pinned base model')
    selection=Path(parent_protocol['calibration']['selection_directory'])
    selection_summary=io.read(selection/'summary.json')
    source=status['identity']['parent_identity']['source']
    if io.sha(selection/'summary.json')!=source['selection_summary_sha256']:raise ValueError('Anchor-selection provenance differs')
    protected[str(selection/'summary.json')]=io.sha(selection/'summary.json')
    selected=selection/'damage_A_anchors.jsonl'
    if io.sha(selected)!=selection_summary['export_sha256']['damage_A_anchors.jsonl']:raise ValueError('Selected anchor inputs differ')
    protected[str(selected)]=io.sha(selected)
    known={'train':[r['prompt'] for r in io.rows(parent/'inputs/train.jsonl')],
           'original300':[r['prompt'] for r in io.rows(parent/'inputs/validation_original_300.jsonl')],
           'selected_anchors':[r['prompt'] for r in io.rows(selected)]}
    if [len(known[k]) for k in ['train','original300','selected_anchors']]!=[9619,300,128]:raise ValueError('Known input manifests differ')
    baseline=Path(p['baseline']['local_directory'])
    if not baseline.exists():
        from huggingface_hub import snapshot_download
        baseline=Path(snapshot_download(p['baseline']['fallback_repo_id'],revision=p['baseline']['fallback_revision'],
                      allow_patterns=['adapter_config.json','adapter_model.safetensors']))
    for f,k in [('adapter_model.safetensors','weights_sha256'),('adapter_config.json','config_sha256')]:
        if io.sha(baseline/f)!=p['baseline'][k]:raise ValueError('Original task-only adapter bytes differ')
        protected[str(baseline/f)]=io.sha(baseline/f)
    architecture(baseline,p['model']);architecture(damage,p['model'])
    sources={'base':{'model':p['model']},
             'task_only':{'directory':str(baseline),'weights_sha256':io.sha(baseline/'adapter_model.safetensors'),
                          'config_sha256':io.sha(baseline/'adapter_config.json'),'step':2400,'provenance':p['baseline']['provenance']},
             'damage_A':{'directory':str(damage),'weights_sha256':io.sha(damage/'adapter_model.safetensors'),
                         'config_sha256':io.sha(damage/'adapter_config.json'),'step':2400,'original300_validation_loss':best['validation_loss'],
                         'checkpoint_training_anchor_exposure':38400,'full_trajectory_anchor_exposure':57728}}
    return sources,protected,known,status['identity']['runtime']
