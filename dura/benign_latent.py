"""Local latent search around a completed benign diffusion reconstruction."""
from dataclasses import replace
import gc
import json
from pathlib import Path
import signal

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from . import train_object_texture as training
from .collect_object_states import collect_states, load_records, prepare
from .config import AttackConfig
from .data import load_image
from .diffusion import StableDiffusionBackend
from .full_surface_attack import (save_texture, stop_check, attach_references,
                                  validation_rollouts, validation_result)
from .object_texture import (NativeTextureWorld, array_digest, digest, fingerprint,
                             load_mapping, write_json)


def latent_delta(parameters, shape):
    return F.interpolate(parameters,size=shape,mode='bilinear',align_corners=False)


@torch.no_grad()
def project(parameters, radius, shape):
    rms = float(latent_delta(parameters,shape).square().mean().sqrt())
    parameters.mul_(min(1.,radius/max(rms,1e-12)))


class LocalDecoder:
    def __init__(self, backend, latent, reference):
        self.backend, self.latent = backend, latent.detach()
        self.reference = training.tensor_image(reference)[None].to(latent.device)
        with torch.no_grad():
            self.base = backend.decode(latent).detach()

    @torch.no_grad()
    def decode(self, parameters, radius):
        bounded = parameters.detach().clone()
        project(bounded,radius,self.latent.shape[-2:])
        delta = latent_delta(bounded,self.latent.shape[-2:])
        # Subtract the same decoder's zero-delta result: the origin matches the
        # saved benign PNG exactly, including its prior 8-bit quantization.
        panel = (self.reference+(self.backend.decode(self.latent+delta)-self.base)).clamp(0,1)
        return panel,delta


def antithetic_gradient(parameters, sigma, pairs, loss, generator):
    gradient = torch.zeros_like(parameters)
    values = []
    for _ in range(pairs):
        direction = torch.randn(parameters.shape,device=parameters.device,generator=generator)
        plus, minus = float(loss(parameters+sigma*direction)), float(loss(parameters-sigma*direction))
        if not np.isfinite([plus,minus]).all():
            raise FloatingPointError('Nonfinite direct latent query')
        gradient.add_((plus-minus)*direction/(2*sigma*pairs))
        values.extend((plus,minus))
    return gradient,float(np.mean(values))


def optimize(decoder, objective, records, cfg, seed, radius, folder, callback):
    settings = cfg['latent_optimizer']; folder.mkdir(parents=True,exist_ok=True)
    shape = (1,decoder.latent.shape[1],settings['grid_size'],settings['grid_size'])
    parameters = torch.nn.Parameter(torch.zeros(shape,device=decoder.latent.device))
    optimizer = torch.optim.Adam([parameters],lr=settings['learning_rate'])
    generator = torch.Generator(device=parameters.device).manual_seed(seed+17000)
    signature = {'config':fingerprint(cfg),'seed':seed,'radius':radius,
                 'latent_sha256':array_digest(decoder.latent.cpu().numpy())}
    path = folder/'checkpoint.pt'; start,history = 0,[]
    if path.exists():
        saved = torch.load(path,map_location='cpu')
        if saved['signature'] != signature:
            raise ValueError('Direct latent checkpoint configuration changed')
        parameters.data.copy_(saved['parameters'].to(parameters.device))
        optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator'])
        start,history = saved['completed'],saved['history']
        objective.observation_queries = saved['observation_queries']
        objective.candidate_queries = saved['candidate_queries']
        if start % settings['validate_every'] == 0 or start == settings['updates']:
            callback(start,parameters.detach(),objective.observation_queries)
    for update in range(start,settings['updates']):
        batch = training.balanced_batch(records,seed,update,cfg['search']['batch_size'])
        @torch.no_grad()
        def loss(query):
            panel,delta = decoder.decode(query,radius)
            atlas = objective.mapping.compose(training.uint8_panel(panel[0]))
            attack = objective.atlas_score(atlas,batch)
            objective.candidate_queries += 1
            regularization = (settings['rgb_weight']*(panel-decoder.reference).square().mean()+
                              settings['latent_weight']*delta.square().mean())
            return attack+regularization
        gradient,score = antithetic_gradient(parameters.detach(),settings['sigma'],settings['pairs'],loss,generator)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite direct latent gradient')
        optimizer.zero_grad(set_to_none=True)
        parameters.grad = gradient
        torch.nn.utils.clip_grad_norm_([parameters],settings['max_gradient_norm'])
        optimizer.step(); project(parameters,radius,decoder.latent.shape[-2:])
        row = {'update':update+1,'loss':score,'gradient_norm':float(gradient.norm()),
               'latent_delta_rms':float(latent_delta(parameters.detach(),decoder.latent.shape[-2:]).square().mean().sqrt()),
               'policy_frame_queries':objective.observation_queries,'candidate_queries':objective.candidate_queries}
        history.append(row)
        training.save_checkpoint(path,{'signature':signature,'completed':update+1,
            'parameters':parameters.detach().cpu(),'optimizer':optimizer.state_dict(),
            'generator':generator.get_state(),'history':history,
            'observation_queries':objective.observation_queries,'candidate_queries':objective.candidate_queries})
        write_json(folder/'history.json',history)
        print(json.dumps({'optimizer':folder.name,**row}),flush=True)
        stop_check()
        if (update+1) % settings['validate_every'] == 0 or update+1 == settings['updates']:
            callback(update+1,parameters.detach(),objective.observation_queries)
    return {'updates':len(history),'optimization_policy_frame_queries':objective.observation_queries}


def validation_metrics(original, benign, adversarial):
    result = validation_result(original,adversarial)
    by_index = {r['trial']:r for r in benign}
    attacked = {r['trial']:r for r in adversarial}
    validation_result(original,benign)  # Verify every control's canonical pairing.
    eligible = [r['trial'] for r in original if r['success'] and by_index[r['trial']]['success']]
    failures = sum(not attacked[i]['success'] for i in eligible)
    result.update(both_controls_successes=len(eligible),optimization_induced_failures=failures,
                  optimization_conditional_asr=failures/len(eligible) if eligible else None)
    return result


def rank(candidate):
    v = candidate['validation']
    return (-v['optimization_induced_failures'],-v['induced_failures'],
            candidate['appearance']['rmse_benign'],-v['mean_steps'],candidate['id'])


def select_latent(cfg):
    root = Path(cfg['output']); path = root/'selection.json'
    completed = json.loads((root/'training_complete.json').read_text())
    if completed['config_sha256'] != fingerprint(cfg):
        raise ValueError('Latent training settings changed')
    if path.exists():
        selected = json.loads(path.read_text())
        if selected['config_sha256'] != fingerprint(cfg):
            raise ValueError('Latent selection settings changed')
        return selected
    if (root/'test_started.json').exists():
        raise ValueError('Cannot reselect after test access')
    candidates = json.loads((root/'candidates.json').read_text())
    eligible = [c for c in candidates if c['status']=='complete' and c['appearance']['accepted'] and
        c['validation']['both_controls_successes'] >= cfg['acceptance']['baseline_validation_successes']]
    if not eligible:
        raise ValueError('No candidate passed benign-relative appearance and matched validation gates')
    winner = min(eligible,key=rank)
    selected = {**winner,'config_sha256':fingerprint(cfg),'review_decisions':{},'held_out_test_opened':False,
                'appearance_reference':'unoptimized reference texture',
                'selection_rule':'new failures beyond both controls, original-induced failures, lower benign distortion, longer rollout, ID'}
    for field,sha in [('texture','texture_sha256'),('benign_texture','benign_sha256')]:
        if digest(root/selected[field]) != selected[sha]:
            raise ValueError('Latent candidate image changed')
    write_json(path,selected)
    write_json(root/'status.json',{'status':'selected_for_test','candidate':selected['id']})
    return selected


def train(cfg):
    from .adapters import load_policy
    root = prepare(cfg)
    if (root/'training_complete.json').exists():
        return
    policy = load_policy(cfg['policy']); mapping = load_mapping(cfg)
    if not (root/'collection_complete.json').exists():
        collect_states(cfg,policy)
    records,calibration,validation = (load_records(cfg,s) for s in ('train','calibration','validation'))
    world = NativeTextureWorld(cfg)
    candidates,optimizer_runs = {},{}
    if (root/'candidates.json').exists():
        candidates = {c['id']:c for c in json.loads((root/'candidates.json').read_text())}
    write_json(root/'review_queue.json',[])
    try:
        training.NativeObjective(world,mapping,policy,cfg['objective']).verify_records(records+calibration+validation)
        backend = StableDiffusionBackend(**cfg['diffusion'])
        reconstruction = cfg.get('initialization', {}).get('mode', 'reconstruction') == 'reconstruction'
        original_panel = load_image(root/'assets/seed_panel.png')[None].to(backend.device) if reconstruction else None
        attack = AttackConfig(**cfg['attack'])
        original_reports = [json.loads((root/'states/validation'/f'episode_{i:03d}'/'report.json').read_text())
                            for i in cfg['splits']['validation']]
        for seed in cfg['search']['seeds']:
            directory = root/'benign'/f'seed_{seed}'; directory.mkdir(parents=True,exist_ok=True)
            latent_path = directory/'latent.pt'; benign_path = directory/'texture.png'
            if not latent_path.exists():
                if reconstruction:
                    panel,latent = training.benign_panel(backend,original_panel,replace(attack,seed=seed),return_latent=True)
                    initialization = {'mode':'reconstruction','seed':seed,'original_texture_encoded':True}
                else:
                    from .texture_initialization import independent_reference
                    panel,latent,initialization = independent_reference(backend,cfg,seed)
                save_texture(benign_path,mapping.compose(panel))
                training.save_checkpoint(latent_path,{'latent':latent.cpu(),'panel':panel,
                    'texture_sha256':digest(benign_path),'config_sha256':fingerprint(cfg),
                    'initialization':initialization})
            saved = torch.load(latent_path,map_location='cpu')
            if saved['config_sha256'] != fingerprint(cfg) or saved['texture_sha256'] != digest(benign_path):
                raise ValueError('Benign reference latent or image changed')
            latent = saved['latent'].to(backend.device)
            initialization = saved.get('initialization', {'mode':'reconstruction','seed':seed,'original_texture_encoded':True})
            write_json(directory/'initialization.json',initialization)
            benign = np.array(Image.open(benign_path).convert('RGB'))
            decoder = LocalDecoder(backend,latent,saved['panel'])
            benign_reports,_ = validation_rollouts(root,world,policy,benign,directory)
            # Optimize differences from benign-policy actions, so previously benign
            # failures are not counted as newly induced optimization successes.
            benign_records = []
            world.set_texture(benign)
            for row in records:
                world.restore_state(row['state']); image = world.image()
                benign_records.append({**row,'image':image,'canonical_sha256':array_digest(image)})
            attach_references(directory,policy,world.instruction,benign_records)

            def assess(identity, panel, queries, kind, radius=0., update=0, override=None):
                folder = root/'candidates'/identity; path = folder/'report.json'
                if path.exists():
                    candidate = json.loads(path.read_text())
                    if digest(root/candidate['texture']) != candidate['texture_sha256']:
                        raise ValueError('Saved latent candidate texture changed')
                    candidates[identity] = candidate
                    write_json(root/'candidates.json',list(candidates.values()))
                    return
                atlas = mapping.compose(panel)
                texture = folder/'texture.png'
                if texture.exists() and not np.array_equal(np.array(Image.open(texture)),atlas):
                    raise ValueError('A resumed candidate differs from its saved validation texture')
                save_texture(texture,atlas)
                appearance = training.appearance(world,mapping,atlas,benign,calibration,cfg)
                metrics = None
                if appearance['accepted']:
                    reports = override if override is not None else validation_rollouts(root,world,policy,atlas,folder)[0]
                    metrics = validation_metrics(original_reports,benign_reports,reports)
                preview = folder/'preview.png'
                training.contact_sheet(world,atlas,benign,calibration[0]['state'],preview)
                candidate = {'id':identity,'kind':kind,'seed':seed,'radius':radius,'update':update,
                    'status':'complete' if appearance['accepted'] else 'appearance_rejected',
                    'appearance':appearance,'validation':metrics,'texture':str(texture.relative_to(root)),
                    'texture_sha256':digest(texture),'benign_texture':str(benign_path.relative_to(root)),
                    'benign_sha256':digest(benign_path),'preview':str(preview.relative_to(root)),
                    'initialization':initialization,
                    'optimization_policy_frame_queries_to_snapshot':queries}
                write_json(path,candidate); candidates[identity] = candidate
                write_json(root/'candidates.json',list(candidates.values()))
                stop_check()

            assess(f'benign_seed_{seed}',saved['panel'],0,'unchanged benign reference',override=benign_reports)
            for radius in cfg['latent_optimizer']['radii']:
                identity = f'seed_{seed}_radius_{radius:g}'
                objective = training.NativeObjective(world,mapping,policy,cfg['objective'])
                def callback(update,parameters,queries):
                    panel,_ = decoder.decode(parameters,radius)
                    assess(f'{identity}_update_{update:03d}',training.uint8_panel(panel[0]),queries,
                           'direct benign-latent optimization',radius,update)
                optimizer_runs[identity] = optimize(decoder,objective,benign_records,cfg,seed,radius,
                                                     root/'optimizers'/identity,callback)
                write_json(root/'optimizer_runs.json',optimizer_runs)
    finally:
        world.close()
    write_json(root/'training_complete.json',{'status':'ready_for_test','candidate_count':len(candidates),
        'config_sha256':fingerprint(cfg),'held_out_test_opened':False,
        'optimization_policy_frame_queries':sum(r['optimization_policy_frame_queries'] for r in optimizer_runs.values())})


def run(cfg):
    if cfg.get('profile') != 'benign_latent':
        raise ValueError('Use the benign-latent profile for this stage')
    from .evaluate_object_texture import native_preflight,evaluate
    training._stop_requested = False
    signal.signal(signal.SIGUSR1,training.request_stop)
    signal.signal(signal.SIGTERM,training.request_stop)
    native_preflight(cfg); train(cfg); stop_check(); select_latent(cfg)
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    evaluate(cfg)
