"""Unconstrained whole-atlas search, validation rollouts, and automatic final ASR."""
from __future__ import annotations

from dataclasses import replace
import gc
import json
from pathlib import Path
import shutil
import signal

import numpy as np
from PIL import Image
import torch

from . import train_object_texture as training
from .collect_object_states import collect_states, load_records, run_episode
from .config import AttackConfig
from .data import load_image
from .diffusion import StableDiffusionBackend
from .object_texture import NativeTextureWorld, digest, fingerprint, load_mapping, write_json


def reuse_collection(cfg):
    """Copy only compatible original training/calibration/validation observations."""
    root = Path(cfg['output'])
    source = cfg['full_surface'].get('reuse_collection')
    if not source or (root/'collection_complete.json').exists():
        return
    source = Path(source)
    previous = json.loads((source/'config.json').read_text())
    for key in ('policy', 'collection', 'suite', 'task_id', 'task_order', 'instance',
                'texture_sha256', 'init_sha256', 'libero_root'):
        if cfg[key] != previous[key]:
            raise ValueError(f'Cannot reuse collection with changed {key}')
    if any(cfg['splits'][s] != previous['splits'][s] for s in ('train','calibration','validation')):
        raise ValueError('Reused collection splits differ')
    marker = source/'collection_complete.json'
    if not marker.exists():
        raise ValueError('Reused original collection is incomplete')
    shutil.copytree(source/'states', root/'states', dirs_exist_ok=True)
    manifest = {str(p.relative_to(root)):digest(p) for p in (root/'states').rglob('*') if p.is_file()}
    write_json(root/'collection_reuse.json', {'source':str(source),
        'source_config_sha256':fingerprint(previous),'files':manifest,
        'scope':'Original training, calibration and validation only; no old test episodes copied.'})
    shutil.copyfile(marker, root/'collection_complete.json')


def preset_panel(name, size):
    colors = {'black':[0,0,0], 'white':[255,255,255], 'red':[255,0,0],
              'green':[0,255,0], 'blue':[0,0,255]}
    if name in colors:
        return np.broadcast_to(np.array(colors[name], dtype=np.uint8),(size,size,3)).copy()
    if name == 'checker':
        y,x = np.indices((size,size)); gray = ((x//max(1,size//8)+y//max(1,size//8))%2*255).astype('uint8')
        return np.repeat(gray[:,:,None],3,axis=2)
    if name == 'noise':
        coarse = np.random.default_rng(2026).integers(0,256,(16,16,3),dtype=np.uint8)
        return np.array(Image.fromarray(coarse).resize((size,size),Image.Resampling.NEAREST))
    raise ValueError(f'Unknown full-surface preset: {name}')


def save_texture(path, atlas):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.stem+'.tmp.png')
    Image.fromarray(atlas).save(temporary)
    temporary.replace(path)


def stop_check():
    if training._stop_requested:
        raise InterruptedError('Slurm signal received; completed full-surface work saved')


def attach_references(root, policy, instruction, records):
    path = root/'reference_actions.json'
    cached = json.loads(path.read_text()) if path.exists() else {}
    missing = [r for r in records if r['snapshot'] not in cached]
    for start in range(0,len(missing),4):
        batch = missing[start:start+4]
        images = torch.stack([training.tensor_image(r['image']) for r in batch])
        with torch.no_grad():
            actions = policy.predict_actions(images,[instruction]*len(batch),[{}]*len(batch))
        if actions.shape != (len(batch),7) or not torch.isfinite(actions).all():
            raise ValueError('Invalid original reference actions')
        for row, action in zip(batch, actions.detach().float().cpu().tolist()):
            cached[row['snapshot']] = {'image_sha256':row['canonical_sha256'],'action':action}
        write_json(path,cached)
        stop_check()
    for row in records:
        saved = cached[row['snapshot']]
        if saved['image_sha256'] != row['canonical_sha256']:
            raise ValueError('Reference action belongs to a different observation')
        row['reference_action'] = saved['action']


def validation_result(original, candidate):
    clean = {r['trial']:r for r in original}
    changed = {r['trial']:r for r in candidate}
    if set(clean) != set(changed) or len(clean) != len(original) or len(changed) != len(candidate):
        raise ValueError('Unpaired validation episodes')
    for index in clean:
        for field in ('canonical_state_sha256','canonical_image_sha256'):
            if clean[index][field] != changed[index][field]:
                raise ValueError('Validation canonical state/render mismatch')
    eligible = [i for i in clean if clean[i]['success']]
    induced = sum(not changed[i]['success'] for i in eligible)
    return {'episodes':len(clean),'original_successes':len(eligible),
            'induced_failures':induced,'conditional_asr':induced/len(eligible) if eligible else None,
            'failures':sum(not r['success'] for r in candidate),
            'mean_steps':float(np.mean([r['steps'] for r in candidate]))}


def candidate_rank(candidate):
    validation = candidate['validation']
    return (-validation['induced_failures'], -validation['mean_steps'],
            candidate['validation_objective'], candidate['id'])


def validation_rollouts(root, world, policy, atlas, folder):
    reports, original = [], []
    for index in world.cfg['splits']['validation']:
        baseline = json.loads((root/'states/validation'/f'episode_{index:03d}'/'report.json').read_text())
        expected = (baseline['canonical_image_sha256'],baseline['canonical_state_sha256'])
        path = folder/'validation'/f'state_{index:03d}.json'
        if path.exists():
            report = json.loads(path.read_text())
        else:
            report,_ = run_episode(world,policy,index,atlas,expected_canonical=expected)
            write_json(path,report)
            print(json.dumps({'validation':folder.name,'state':index,'success':report['success']}),flush=True)
        original.append(baseline); reports.append(report)
        stop_check()
    return reports, validation_result(original,reports)


def select_full(cfg):
    root = Path(cfg['output'])
    if cfg.get('profile') != 'full_surface_unconstrained' or cfg['appearance'].get('enabled',True):
        raise ValueError('Unconstrained selection requires the explicit full-surface profile')
    complete = json.loads((root/'training_complete.json').read_text())
    if complete['config_sha256'] != fingerprint(cfg):
        raise ValueError('Training configuration changed')
    if (root/'selection.json').exists():
        selected = json.loads((root/'selection.json').read_text())
        if selected['config_sha256'] != fingerprint(cfg):
            raise ValueError('Frozen selection configuration changed')
        return selected
    if (root/'test_started.json').exists():
        raise ValueError('Cannot select again after test access')
    candidates = json.loads((root/'candidates.json').read_text())
    eligible = [c for c in candidates if c.get('status') == 'complete' and
                c['validation']['original_successes'] >= cfg['acceptance']['baseline_validation_successes']]
    if not eligible:
        raise ValueError('No candidate has valid matched validation controls')
    winner = min(eligible,key=candidate_rank)
    selected = {**winner,'config_sha256':fingerprint(cfg),'review_decisions':{},
                'selection_rule':'validation induced failures, longer rollout, action divergence, fixed ID',
                'appearance_constraints_enabled':False,'held_out_test_opened':False}
    for field,sha in [('texture','texture_sha256'),('benign_texture','benign_sha256')]:
        if digest(root/selected[field]) != selected[sha]:
            raise ValueError('Candidate texture changed before selection')
    write_json(root/'selection.json',selected)
    write_json(root/'status.json',{'status':'selected_for_test','candidate':selected['id']})
    return selected


def train_full(cfg):
    from .adapters import load_policy
    from .collect_object_states import prepare
    root = prepare(cfg)
    if (root/'training_complete.json').exists():
        return
    policy = load_policy(cfg['policy'])
    if not (root/'collection_complete.json').exists():
        collect_states(cfg,policy)
    mapping = load_mapping(cfg)
    train, calibration, validation = (load_records(cfg,s) for s in ('train','calibration','validation'))
    world = NativeTextureWorld(cfg)
    candidates = []
    write_json(root/'review_queue.json',[])  # Appearance is recorded, never used as a gate.
    try:
        objective = training.NativeObjective(world,mapping,policy,cfg['objective'])
        objective.verify_records(train+calibration+validation)
        attach_references(root,policy,world.instruction,train+validation)

        def assess(identity, atlas, benign_path, queries, kind):
            folder = root/'candidates'/identity; path = folder/'report.json'
            if path.exists():
                candidate = json.loads(path.read_text())
            else:
                texture_path = folder/'texture.png'
                save_texture(texture_path,atlas)
                benign = np.array(Image.open(root/benign_path).convert('RGB'))
                _, validation_metrics = validation_rollouts(root,world,policy,atlas,folder)
                appearance = training.appearance(world,mapping,atlas,benign,calibration,cfg)
                scorer = training.NativeObjective(world,mapping,policy,cfg['objective'])
                scores = [(float(scorer.atlas_score(atlas,validation[i:i+4])),len(validation[i:i+4]))
                          for i in range(0,len(validation),4)]
                score = sum(value*n for value,n in scores)/sum(n for _,n in scores)
                preview = folder/'preview.png'
                training.contact_sheet(world,atlas,benign,calibration[0]['state'],preview)
                candidate = {'id':identity,'status':'complete','kind':kind,'validation':validation_metrics,
                    'validation_action_mse':-score,'validation_objective':score,
                    'objective':'negative weighted action divergence from original',
                    'appearance':appearance,'texture':str(texture_path.relative_to(root)),
                    'texture_sha256':digest(texture_path),'benign_texture':str(benign_path),
                    'benign_sha256':digest(root/benign_path),'preview':str(preview.relative_to(root)),
                    'optimization_policy_frame_queries':queries}
                write_json(path,candidate)
            candidates.append(candidate)
            write_json(root/'candidates.json',candidates)
            stop_check()

        for name in cfg['full_surface']['presets']:
            atlas = mapping.compose(preset_panel(name,cfg['mapping']['working_size']))
            assess('preset_'+name,atlas,'assets/original.png',0,'unoptimized full-surface baseline')

        backend = StableDiffusionBackend(**cfg['diffusion'])
        seed_panel = load_image(root/'assets/seed_panel.png')[None].to(backend.device)
        attack = AttackConfig(**cfg['attack'])
        for seed in cfg['search']['seeds']:
            benign_path = Path('benign')/f'seed_{seed}'/'texture.png'
            if not (root/benign_path).exists():
                benign_panel = training.benign_panel(backend,seed_panel,replace(attack,seed=seed,step_size=0))
                save_texture(root/benign_path,mapping.compose(benign_panel))
            benign = np.array(Image.open(root/benign_path).convert('RGB'))
            validation_rollouts(root,world,policy,benign,(root/benign_path).parent)
            for size in cfg['search']['step_sizes']:
                identity = f'seed_{seed}_step_{size:g}'
                folder = root/'candidates'/identity
                if (folder/'report.json').exists():
                    candidates.append(json.loads((folder/'report.json').read_text()))
                    write_json(root/'candidates.json',candidates)
                    continue
                try:
                    objective = training.NativeObjective(world,mapping,policy,cfg['objective'])
                    panel = training.trajectory(backend,objective,seed_panel,train,cfg,
                                                replace(attack,seed=seed,step_size=size),folder)
                except FloatingPointError as error:
                    failure = {'id':identity,'status':'numerical_failure','error':str(error),
                               'optimization_policy_frame_queries':objective.observation_queries}
                    write_json(folder/'report.json',failure)
                    candidates.append(failure); write_json(root/'candidates.json',candidates)
                    continue
                assess(identity,mapping.compose(panel),benign_path,objective.observation_queries,'optimized diffusion texture')
    finally:
        world.close()
    write_json(root/'training_complete.json',{'status':'ready_for_test','candidate_count':len(candidates),
        'config_sha256':fingerprint(cfg),'appearance_constraints_enabled':False,'held_out_test_opened':False,
        'optimization_policy_frame_queries':sum(c['optimization_policy_frame_queries'] for c in candidates)})


def run(cfg):
    if cfg.get('profile') != 'full_surface_unconstrained':
        raise ValueError('Use the full-surface configuration for this stage')
    from .evaluate_object_texture import native_preflight, evaluate
    training._stop_requested = False
    signal.signal(signal.SIGUSR1,training.request_stop)
    signal.signal(signal.SIGTERM,training.request_stop)
    native_preflight(cfg)
    train_full(cfg)
    stop_check()
    selected = select_full(cfg)
    print(json.dumps({'selected':selected['id'],'validation':selected['validation']}),flush=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    evaluate(cfg)
