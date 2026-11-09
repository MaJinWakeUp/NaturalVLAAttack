"""Resumable native-rendering DURA search with appearance and baseline gates."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import signal
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from .attack import score_direction
from .config import AttackConfig
from .data import load_image
from .diffusion import StableDiffusionBackend
from .naturalness import ssim
from .object_texture import (NativeTextureWorld, array_digest, crop_object, digest,
                             fingerprint, load_config, load_mapping, write_json)
from .collect_object_states import collect_states, load_records, prepare, run_episode


_stop_requested = False


def request_stop(signum, frame):
    global _stop_requested
    _stop_requested = True


def tensor_image(image):
    return torch.from_numpy(image.copy()).permute(2, 0, 1).float()/255


def uint8_panel(tensor):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError('Nonfinite decoded diffusion panel')
    return tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().numpy()


class NativeObjective:
    def __init__(self, world, mapping, policy, settings):
        self.world, self.mapping, self.policy = world, mapping, policy
        self.kind = settings.get('kind', 'target')
        if self.kind not in ('target', 'action_divergence', 'reference_token_nll'):
            raise ValueError('Unknown native action objective')
        self.target = torch.tensor(settings['target'], dtype=torch.float32)
        self.weights = torch.tensor(settings['weights'], dtype=torch.float32)
        if settings.get('dimensions') is not None:
            mask = torch.zeros_like(self.weights); mask[settings['dimensions']] = 1
            self.weights *= mask
        self.candidate_queries = 0
        self.observation_queries = 0

    def verify_records(self, records):
        self.world.original()
        for row in records:
            self.world.restore_state(row['state'])
            if array_digest(self.world.image()) != row['canonical_sha256']:
                raise ValueError(f"Cross-process RGB replay differs: {row['snapshot']}")

    @torch.no_grad()
    def atlas_score(self, atlas, records):
        self.world.set_texture(atlas)
        images = []
        for row in records:
            self.world.restore_state(row['state'])
            images.append(tensor_image(self.world.image()))
        if self.kind == 'reference_token_nll':
            # Smooth teacher-forced confidence supplies a signal even when decoded
            # discrete action tokens do not change under a small texture perturbation.
            losses = []
            for pixels,row in zip(images,records):
                reference = torch.tensor(row['reference_action'],dtype=torch.float32)
                logits,targets,weights = self.policy.target_logits(
                    pixels[None],[self.world.instruction],[{}],reference,self.weights)
                ce = torch.nn.functional.cross_entropy(logits.float().transpose(1,2),targets,reduction='none')
                losses.append(-(ce*weights.to(ce)).sum(-1).mean())
            self.observation_queries += len(records)
            value = torch.stack(losses).mean()
            if not torch.isfinite(value):
                raise FloatingPointError('Nonfinite benign-reference token loss')
            return value
        actions = self.policy.predict_actions(torch.stack(images), [self.world.instruction]*len(records), [{}]*len(records))
        self.observation_queries += len(records)
        if not torch.isfinite(actions).all():
            raise FloatingPointError('Nonfinite policy action during a texture query')
        reference = (torch.tensor([row['reference_action'] for row in records], device=actions.device,
                                  dtype=actions.dtype) if self.kind == 'action_divergence' else self.target.to(actions))
        value = ((actions-reference).square()*self.weights.to(actions)).sum(-1).mean()
        return -value if self.kind == 'action_divergence' else value

    @torch.no_grad()
    def __call__(self, panels, records):
        scores = []
        for panel in panels:
            scores.append(self.atlas_score(self.mapping.compose(uint8_panel(panel)), records))
            self.candidate_queries += 1
        return torch.stack(scores).to(panels.device)


def balanced_batch(records, seed, index, count):
    groups = {}
    for row in records:
        groups.setdefault(row['state_index'], []).append(row)
    keys = sorted(groups)
    rng = np.random.default_rng(seed*100003+index)
    selected = rng.choice(len(keys), size=count, replace=count > len(keys))
    return [groups[keys[k]][int(rng.integers(len(groups[keys[k]])))] for k in selected]


def save_checkpoint(path, state):
    temporary = path.with_name(path.name+'.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def trajectory(backend, objective, seed_panel, records, cfg, attack, folder):
    """Same anchored update as DURAAttack, with per-update RNG/state checkpoints."""
    folder.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=backend.device).manual_seed(attack.seed)
    timesteps = backend.timesteps(attack)
    with torch.no_grad():
        clean = backend.encode(seed_panel)
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        latent = backend.add_noise(clean, noise, timesteps[0])
        anchors, anchor = [], latent.clone()
        for timestep in timesteps:
            anchors.append(anchor.detach().cpu() if attack.anchor_weight else None)
            if attack.anchor_weight:
                anchor = backend.step(anchor, timestep)
    checkpoint = folder/'checkpoint.pt'
    signature = {'config':fingerprint(cfg), 'seed':attack.seed, 'step_size':attack.step_size}
    history, start = [], 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu')
        if saved['signature'] != signature:
            raise ValueError('Checkpoint settings changed')
        latent = saved['latent'].to(backend.device)
        generator.set_state(saved['generator_state'])
        start, history = saved['completed'], saved['history']
        objective.candidate_queries = saved['candidate_queries']
        objective.observation_queries = saved['observation_queries']
    started = time.monotonic()
    for index in range(start, len(timesteps)):
        timestep = timesteps[index]
        batch = balanced_batch(records, attack.seed, index, cfg['search']['batch_size'])
        with torch.no_grad():
            mixed = ((1-attack.anchor_weight)*latent+attack.anchor_weight*anchors[index].to(latent)
                     if attack.anchor_weight else latent)
            denoised = backend.step(mixed, timestep).detach()
            direction, loss = score_direction(denoised, backend.query_alpha(timestep, attack.query_alpha),
                attack.samples, attack.candidate_batch_size,
                lambda candidates:objective(backend.decode(candidates), batch), generator, attack.baseline)
            if not np.isfinite(loss) or not torch.isfinite(direction).all():
                raise FloatingPointError('Nonfinite native-renderer query score')
            rms_limit = cfg['search'].get('gradient_rms_max')
            if rms_limit is not None:
                direction *= min(1., rms_limit/max(float(direction.square().mean().sqrt()), 1e-12))
            latent = denoised-attack.step_size*direction
        record = {'update':index+1, 'loss':loss, 'gradient_norm':float(direction.norm()),
                  'candidate_queries':objective.candidate_queries,
                  'policy_frame_queries':objective.observation_queries,
                  'segment_seconds':time.monotonic()-started}
        history.append(record)
        save_checkpoint(checkpoint, {'signature':signature, 'latent':latent.cpu(),
            'generator_state':generator.get_state(), 'completed':index+1, 'history':history,
            'candidate_queries':objective.candidate_queries, 'observation_queries':objective.observation_queries})
        write_json(folder/'history.json', history)
        print(json.dumps({'candidate':folder.name, **record}), flush=True)
        if _stop_requested:
            raise InterruptedError('Slurm signal received; update checkpoint saved')
    with torch.no_grad():
        return uint8_panel(backend.decode(latent)[0])


@torch.no_grad()
def benign_panel(backend, seed_panel, attack, return_latent=False):
    generator = torch.Generator(device=backend.device).manual_seed(attack.seed)
    timesteps = backend.timesteps(attack)
    clean = backend.encode(seed_panel)
    noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    latent = backend.add_noise(clean, noise, timesteps[0])
    for timestep in timesteps:
        latent = backend.step(latent, timestep)
    panel = uint8_panel(backend.decode(latent)[0])
    return (panel,latent.detach()) if return_latent else panel


def appearance(world, mapping, atlas, reference, records, cfg):
    metrics = mapping.texture_report(atlas, reference, cfg['appearance'])
    distinct = {row['state_index'] for row in records}
    if len(distinct) < cfg['appearance']['min_calibration_trajectories']:
        raise ValueError('Insufficient visible calibration trajectories')
    candidate_crops, reference_crops, original_crops = [], [], []
    for texture, destination in ((reference, reference_crops), (atlas, candidate_crops)):
        world.set_texture(texture)
        for row in records:
            world.restore_state(row['state'])
            destination.append(tensor_image(crop_object(world.image(), row['mask'],
                cfg['appearance']['crop_border'], cfg['appearance']['crop_size'])))
    for row in records:
        original_crops.append(tensor_image(crop_object(row['image'], row['mask'],
            cfg['appearance']['crop_border'], cfg['appearance']['crop_size'])))
    with torch.no_grad():
        similarities = ssim(torch.stack(candidate_crops), torch.stack(reference_crops)).tolist()
        original_similarities = ssim(torch.stack(candidate_crops), torch.stack(original_crops)).tolist()
    metrics.update(ssim_min=min(similarities), ssim_original_min=min(original_similarities),
                   ssim_per_frame=similarities, calibration_states=sorted(distinct))
    metrics['accepted'] = bool(metrics['accepted'] and np.isfinite(similarities).all() and
        (not cfg['appearance'].get('enabled', True) or min(similarities) >= cfg['appearance']['ssim_min']))
    return metrics


def contact_sheet(world, atlas, reference, state, path):
    """Six close camera views for appearance review; restored before any task rollouts."""
    world.restore_state(state)
    model, data, mj = world.model, world.data, world.mujoco
    camera = world.sim.model.camera_name2id(world.cfg['collection']['camera'])
    if model.cam_bodyid[camera] != 0:
        raise ValueError('Review camera must be attached to the world body')
    old_pos, old_quat, old_fovy = model.cam_pos[camera].copy(), model.cam_quat[camera].copy(), float(model.cam_fovy[camera])
    target = data.geom_xpos[world.geom_ids[0]].copy()
    sheet = Image.new('RGB', (6*180, 3*200), 'white')
    draw = ImageDraw.Draw(sheet)
    try:
        for column, angle in enumerate(np.linspace(0, 2*np.pi, 6, endpoint=False)):
            position = target+np.array([.25*np.cos(angle), .25*np.sin(angle), .13])
            z = position-target; z /= np.linalg.norm(z)
            x = np.cross([0., 0., 1.], z); x /= np.linalg.norm(x)
            y = np.cross(z, x)
            model.cam_pos[camera] = position
            mj.mju_mat2Quat(model.cam_quat[camera], np.column_stack((x,y,z)).ravel())
            model.cam_fovy[camera] = 35
            world.sim.forward()
            for row, texture in enumerate((world.original_gpu_texture[::-1] if world.flip_texture else world.original_gpu_texture, reference, atlas)):
                world.set_texture(texture)
                image = world.image(); mask = world.object_mask()
                crop = crop_object(image, mask, border=8, size=180)
                sheet.paste(Image.fromarray(crop), (180*column, 200*row+20))
                draw.text((180*column+4, 200*row+3), ('Original','Benign','Candidate')[row], fill='black')
        sheet.save(path)
    finally:
        model.cam_pos[camera], model.cam_quat[camera], model.cam_fovy[camera] = old_pos, old_quat, old_fovy
        world.restore_state(state); world.original()


def train(cfg):
    from .adapters import load_policy
    root = prepare(cfg)
    complete = root/'training_complete.json'
    if complete.exists():
        print(complete.read_text()); return
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    policy = load_policy(cfg['policy'])
    if not (root/'collection_complete.json').exists():
        collect_states(cfg, policy)
    mapping = load_mapping(cfg)
    training, calibration, validation = (load_records(cfg,s) for s in ('train','calibration','validation'))
    world = NativeTextureWorld(cfg)
    backend = StableDiffusionBackend(**cfg['diffusion'])
    seed_panel = load_image(root/'assets/seed_panel.png')[None].to(backend.device)
    attack_base = AttackConfig(**cfg['attack'])
    candidates, review_queue = [], []
    review_dir = root/'review'; review_dir.mkdir(exist_ok=True)
    try:
        verifier = NativeObjective(world, mapping, policy, cfg['objective'])
        verifier.verify_records(training+calibration+validation)
        for seed in cfg['search']['seeds']:
            benign_dir = root/'benign'/f'seed_{seed}'
            benign_dir.mkdir(parents=True, exist_ok=True)
            benign_path = benign_dir/'texture.png'
            if not benign_path.exists():
                panel = benign_panel(backend, seed_panel, replace(attack_base, seed=seed, step_size=0))
                Image.fromarray(mapping.compose(panel)).save(benign_path)
            benign = np.array(Image.open(benign_path).convert('RGB'))
            benign_report_path = benign_dir/'report.json'
            if benign_report_path.exists():
                benign_report = json.loads(benign_report_path.read_text())
            else:
                benign_report = {'appearance':appearance(world,mapping,benign,mapping.original,calibration,cfg),
                                 'validation_rollouts':[]}
                for state_index in cfg['splits']['validation']:
                    original = json.loads((root/'states/validation'/f'episode_{state_index:03d}'/'report.json').read_text())
                    expected = (original['canonical_image_sha256'], original['canonical_state_sha256'])
                    report, _ = run_episode(world,policy,state_index,benign,expected_canonical=expected)
                    benign_report['validation_rollouts'].append(report)
                benign_report['validation_successes'] = sum(r['success'] for r in benign_report['validation_rollouts'])
                write_json(benign_report_path, benign_report)
            for size in cfg['search']['step_sizes']:
                identity = f'seed_{seed}_step_{size:g}'
                folder = root/'candidates'/identity
                folder.mkdir(parents=True, exist_ok=True)
                report_path, texture_path = folder/'report.json', folder/'texture.png'
                if report_path.exists():
                    candidate = json.loads(report_path.read_text())
                else:
                    objective = NativeObjective(world,mapping,policy,cfg['objective'])
                    try:
                        panel = trajectory(backend,objective,seed_panel,training,cfg,
                                           replace(attack_base, seed=seed, step_size=size),folder)
                    except FloatingPointError as error:
                        candidate = {'id':identity,'seed':seed,'step_size':size,'status':'numerical_failure',
                            'error':str(error),'appearance':{'accepted':False},
                            'benign_appearance':benign_report['appearance'],
                            'benign_validation_successes':benign_report['validation_successes'],
                            'validation_action_mse':None,
                            'optimization_policy_frame_queries':objective.observation_queries}
                        write_json(report_path,candidate)
                        candidates.append(candidate)
                        write_json(root/'candidates.json',candidates)
                        continue
                    Image.fromarray(mapping.compose(panel)).save(texture_path)
                    texture = np.array(Image.open(texture_path).convert('RGB'))
                    appearance_report = appearance(world,mapping,texture,benign,calibration,cfg)
                    score = None
                    if appearance_report['accepted'] and benign_report['appearance']['accepted']:
                        # Limit policy batches to four; includes every held-out snapshot.
                        scores = [float(objective.atlas_score(texture,validation[i:i+4]))
                                  for i in range(0,len(validation),4)]
                        counts = [len(validation[i:i+4]) for i in range(0,len(validation),4)]
                        score = float(np.average(scores,weights=counts))
                    candidate = {'id':identity, 'seed':seed, 'step_size':size, 'appearance':appearance_report,
                                 'validation_action_mse':score, 'benign_appearance':benign_report['appearance'],
                                 'benign_validation_successes':benign_report['validation_successes'],
                                 'texture':str(texture_path.relative_to(root)), 'texture_sha256':digest(texture_path),
                                 'benign_texture':str(benign_path.relative_to(root)), 'benign_sha256':digest(benign_path),
                                 'optimization_policy_frame_queries':attack_base.updates*cfg['search']['samples']*cfg['search']['batch_size']}
                    write_json(report_path,candidate)
                candidates.append(candidate)
                required = cfg['acceptance']['baseline_validation_successes']
                original_successes = sum(json.loads((root/'states/validation'/f'episode_{i:03d}'/'report.json').read_text())['success']
                                         for i in cfg['splits']['validation'])
                qualifies = (candidate['appearance']['accepted'] and candidate['benign_appearance']['accepted'] and
                    candidate['validation_action_mse'] is not None and candidate['benign_validation_successes'] >= required and
                    original_successes >= required)
                if qualifies:
                    sheet = review_dir/f'review_{len(review_queue):02d}.png'
                    if not sheet.exists():
                        contact_sheet(world,np.array(Image.open(root/candidate['texture']).convert('RGB')),benign,
                                      calibration[0]['state'],sheet)
                    review_queue.append({'id':candidate['id'], 'sheet':str(sheet.relative_to(root)),
                        'sheet_sha256':digest(sheet), 'texture_sha256':candidate['texture_sha256'],
                        'benign_sha256':candidate['benign_sha256'],
                        'mask_sha256':digest(root/'assets/editable_mask.png')})
                write_json(root/'candidates.json',candidates)
                write_json(root/'review_queue.json',review_queue)
                if _stop_requested:
                    raise InterruptedError('Slurm signal received; completed candidate saved')
    finally:
        world.close()
    status = 'awaiting_visual_review' if review_queue else 'no_eligible_candidate'
    result = {'status':status, 'candidate_count':len(candidates), 'review_count':len(review_queue),
              'optimization_policy_frame_queries':sum(c['optimization_policy_frame_queries'] for c in candidates),
              'held_out_test_opened':False, 'config_sha256':fingerprint(cfg)}
    write_json(complete,result)
    write_json(root/'status.json',result)
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    train(load_config(args.config))


if __name__=='__main__':
    main()
