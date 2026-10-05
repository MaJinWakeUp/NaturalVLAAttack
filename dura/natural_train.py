"""Train anchored DURA trajectories; select by action error only after appearance gates."""
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader

from .adapters import load_policy
from .attack import DURAAttack
from .cli import read_config
from .config import AttackConfig
from .data import ManifestDataset, collate_observations, cycle_batches, load_image, save_image
from .diffusion import StableDiffusionBackend
from .naturalness import appearance_report, eligible_best, check_limits, validate_fixed_limits
from .objective import TargetObjective
from .render import PatchRenderer
from .validate_patch import balanced_indices, score_patch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    cfg=read_config(args.config)
    settings=cfg['naturalness']
    attack=AttackConfig(**cfg['attack'])
    if attack.anchor_weight <= 0:
        raise ValueError('Natural training requires the clean diffusion anchor')
    if 'refinement' in cfg:
        raise ValueError('Unconstrained post-DURA refinement is not allowed in the natural workflow')
    sizes=settings['step_sizes']
    if not sizes or len(set(sizes))!=len(sizes) or any(not np.isfinite(s) or s<=0 for s in sizes):
        raise ValueError('Choose unique positive finite attack step sizes')
    limits=settings['limits']
    if settings.get('fixed_limits'):
        validate_fixed_limits(limits)
    check_limits({key:0. for key in limits},limits)
    if any(str(cfg[section].get('device','cpu')).startswith('cuda') for section in ('policy','diffusion')) and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; run training on a GPU node.')
    output=Path(cfg['output'])
    final=Path(cfg['final_output']) if cfg.get('final_output') else None
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    if final and final.exists(): raise FileExistsError(final)
    training=ManifestDataset(cfg['manifest'],cfg.get('image_size',224))
    validation=ManifestDataset(cfg['validation_manifest'],cfg.get('image_size',224))
    if {r['image'] for r in training.records}&{r['image'] for r in validation.records}:
        raise ValueError('Training and selection frames overlap')
    output.mkdir(parents=True,exist_ok=True)
    (output/'config.json').write_text(json.dumps(cfg,indent=2))
    provenance={'method':'Algorithm 1 trajectories followed by additional appearance-gated candidate selection',
                'paper_settings':{'anchor_weight':.2,'ddim_steps':200,'strength':.5},
                'additional_choices':['step-size search','appearance acceptance thresholds','validation candidate selection'],
                'code_sha256':{name:hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest() for name in ('dura/attack.py','dura/diffusion.py','dura/naturalness.py','dura/natural_train.py')},
                'versions':{name:importlib.metadata.version(name) for name in ('torch','torchvision','diffusers','transformers','numpy')},
                'seed_sha256':hashlib.sha256(Path(cfg['seed_patch']).read_bytes()).hexdigest()}
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    random.seed(attack.seed); np.random.seed(attack.seed); torch.manual_seed(attack.seed)
    backend=StableDiffusionBackend(**cfg['diffusion'])
    policy=load_policy(cfg['policy'])
    renderer=PatchRenderer(cfg['patch_box'])
    seed=load_image(cfg['seed_patch'],cfg.get('seed_size',512))[None].to(backend.device)
    save_image(seed,output/'seed.png')
    scene_reference = None
    if settings.get('scene_reference_patch'):
        reference_path=(Path(args.config).resolve().parent/settings['scene_reference_patch']).resolve()
        scene_reference=load_image(reference_path,cfg.get('seed_size',512))[None].to(backend.device)
        provenance['scene_reference']={'path':str(reference_path),'sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest()}
        (output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    selected_indices=balanced_indices(validation,cfg.get('validation_frames',40))
    observations=collate_observations([validation[i] for i in selected_indices]).to(backend.device)
    def inspect(patch):
        return appearance_report(observations.images,patch,seed,renderer,observations.metadata,cfg['patch_box'],limits)
    report={'status':'running','method':provenance['method'],'selection_data_role':'tuning, not independent ASR validation',
            'seed_appearance':inspect(seed),'historical_appearance':{},'candidates':[],
            'selected_candidate':None,'visual_review':'required; not automatically inferred from the gate'}
    report['reference_policy']='Acceptance uses the current seed; optional original-scene comparison is diagnostic only.'
    if scene_reference is not None:
        report['seed_vs_original_scene_reference']=appearance_report(observations.images,seed,scene_reference,renderer,observations.metadata,cfg['patch_box'],limits)
    for name,path in settings.get('comparison_patches',{}).items():
        path=(Path(args.config).resolve().parent/path).resolve()
        report['historical_appearance'][name]=inspect(load_image(path)[None].to(backend.device))
    for number,size in enumerate(sizes):
        candidate_cfg=replace(attack,step_size=float(size))
        folder=output/f'candidate_{number:02d}'
        folder.mkdir()
        objective=TargetObjective(policy,renderer,**cfg['objective'],policy_batch_size=cfg.get('policy_batch_size',1))
        loader=DataLoader(training,batch_size=cfg.get('batch_size',4),shuffle=True,collate_fn=collate_observations,
                          generator=torch.Generator().manual_seed(attack.seed))
        print(json.dumps({'candidate':number,'attack_config':asdict(candidate_cfg)}),flush=True)
        with (folder/'history.jsonl').open('x') as log:
            def callback(row):
                log.write(json.dumps(row)+'\n'); log.flush()
                if row['update']%10==0: print(json.dumps({'candidate':number,**row}),flush=True)
            result=DURAAttack(backend,objective,candidate_cfg).run(seed,cycle_batches(loader),callback)
        save_image(result.patch,folder/'patch.png')
        patch=load_image(folder/'patch.png')[None].to(backend.device)
        torch.save({'patch':patch.cpu(),'latent':result.latent.cpu(),'attack_config':asdict(candidate_cfg)},folder/'patch.pt')
        save_image(renderer(observations.images[:1],patch,observations.metadata[:1])[0,0],folder/'preview.png')
        appearance=inspect(patch)
        candidate={'id':number,'step_size':size,'scratch_folder':str(folder),'appearance':appearance,'action_score':None}
        if scene_reference is not None:
            candidate['appearance_vs_original_scene_reference']=appearance_report(observations.images,patch,scene_reference,renderer,observations.metadata,cfg['patch_box'],limits)
        if appearance['gate']['accepted']:
            candidate['action_score']=score_patch(policy,renderer,validation,patch,cfg['objective'],cfg.get('validation_frames',40),backend.device)
        report['candidates'].append(candidate)
        (folder/'report.json').write_text(json.dumps(candidate,indent=2))
        (output/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'candidate_result':candidate}),flush=True)
    best=eligible_best(report['candidates'])
    if best is None:
        report['status']='no_candidate_passed_appearance_limits'
        (output/'report.json').write_text(json.dumps(report,indent=2))
        raise RuntimeError('No acceptable candidate. Nothing published; inspect scratch diagnostics.')
    report['status']='appearance_gate_passed_pending_visual_and_rollout_review'
    report['selected_candidate']=best['id']
    report['selection_rule']='Appearance gate first, then lowest validation decoded-action MSE, then CE.'
    (output/'report.json').write_text(json.dumps(report,indent=2))
    for name in ('patch.png','patch.pt','preview.png'):
        shutil.copy2(Path(best['scratch_folder'])/name,output/name)
    if final:
        final.mkdir(parents=True,exist_ok=False)
        for name in ('patch.png','preview.png','config.json','provenance.json','report.json'):
            shutil.copy2(output/name,final/name)
    print(json.dumps({'selected_candidate':best['id'],'step_size':best['step_size'],'action_score':best['action_score'],'final_output':str(final)}),flush=True)


if __name__=='__main__': main()
