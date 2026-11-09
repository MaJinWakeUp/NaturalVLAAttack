"""Collect original-texture trajectories and replayable native simulator states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .object_texture import (NativeTextureWorld, array_digest, audit_test_states,
                             fingerprint, load_config, load_mapping, save_mapping, write_json)


def prepare(cfg):
    root = Path(cfg['output']); root.mkdir(parents=True, exist_ok=True)
    if cfg.get('initialization'):
        from .texture_initialization import prepare_initialization
        prepare_initialization(cfg)
    if cfg.get('initial_state_bank'):
        from .initial_states import prepare_state_bank
        prepare_state_bank(cfg)
    marker = root/'config.json'
    if marker.exists():
        if fingerprint(json.loads(marker.read_text())) != fingerprint(cfg):
            raise ValueError('Run configuration changed; choose a new run directory')
    else:
        write_json(marker, cfg)
    if fingerprint(load_config(marker)) != fingerprint(cfg):
        raise ValueError('Frozen configuration does not round-trip through the runtime loader')
    audit_path = root/'test_state_audit.json'
    if not audit_path.exists():
        write_json(audit_path, audit_test_states(cfg))
    if not (root/'assets/mapping.json').exists():
        save_mapping(load_mapping(cfg), root/'assets')
    if cfg.get('profile') in ('full_surface_unconstrained','benign_latent'):
        from .full_surface_attack import reuse_collection
        reuse_collection(cfg)
    return root


def policy_action(policy, image, instruction):
    import torch
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float()[None]/255
    action = policy.predict_actions(tensor, [instruction], [{}])[0].detach().cpu().numpy()
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError('Policy must produce one finite seven-dimensional action')
    return action


def controller_context(world):
    controllers = []
    for robot in world.env.robots:
        values = {}
        for key, value in robot.controller.__dict__.items():
            if isinstance(value, np.ndarray):
                values[key] = value.tolist()
            elif isinstance(value, (str, bool, int, float)):
                values[key] = value
        controllers.append(values)
    rng = np.random.get_state()
    return {'controllers':controllers,
            'gripper_commands':[np.asarray(robot.gripper.current_action).tolist() for robot in world.env.robots if robot.has_gripper],
            'numpy_rng':[rng[0], rng[1].tolist(), rng[2], rng[3], rng[4]],
            'environment_timestep':int(world.env.env.timestep)}


def run_episode(world, policy, state_index, atlas=None, collect=False, video=None,
                expected_canonical=None):
    """Every condition starts from a reset controller, settled state, and original render."""
    from .evaluate_libero import executable_action
    started = time.monotonic()
    start_state = world.reset(state_index)
    canonical = world.image()
    canonical_sha = array_digest(canonical)
    canonical_state_sha = array_digest(start_state)
    if expected_canonical and (canonical_sha, canonical_state_sha) != tuple(expected_canonical):
        raise ValueError('Canonical state/render mismatch; no condition policy actions were taken')
    if atlas is not None:
        world.set_texture(atlas)
    snapshots, actions = [], []
    writer = None
    if video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(str(video), fps=20)
    success = False
    try:
        for step in range(world.cfg['collection']['horizon']):
            image = world.image()
            if collect:
                snapshots.append({'state':world.state().copy(), 'image':image,
                                  'context':controller_context(world), 'step':step})
            if writer:
                writer.append_data(image)
            action = policy_action(policy, image, world.instruction)
            executed = executable_action(action, 'openvla')
            actions.append(executed.tolist())
            _, _, done, info = world.env.step(executed)
            success = bool(world.env.check_success())
            if success:
                break
            if done:
                break
    finally:
        if writer:
            writer.close()
        world.original()
    report = {'suite':world.cfg['suite'], 'task_id':world.cfg['task_id'], 'trial':state_index,
              'success':success, 'steps':len(actions), 'actions':actions,
              'canonical_image_sha256':canonical_sha, 'canonical_state_sha256':canonical_state_sha,
              'elapsed_seconds':time.monotonic()-started}
    return report, snapshots


def policy_visible_pixels(mask, crop_area):
    h, w = mask.shape
    hh, ww = round(h*crop_area**.5), round(w*crop_area**.5)
    top, left = int(round((h-hh)/2)), int(round((w-ww)/2))
    return int(mask[top:top+hh, left:left+ww].sum())


def collect_states(cfg, policy=None):
    from .adapters import load_policy
    root = prepare(cfg)
    if policy is None:
        policy = load_policy(cfg['policy'])
    world = NativeTextureWorld(cfg)
    try:
        for split in ('train', 'calibration', 'validation'):
            records = []
            for state_index in cfg['splits'][split]:
                folder = root/'states'/split/f'episode_{state_index:03d}'
                complete = folder/'report.json'
                if complete.exists():
                    saved = json.loads(complete.read_text())
                    for row in saved['snapshots']:
                        if not (root/row['snapshot']).exists():
                            raise ValueError(f'Incomplete saved episode: {folder}')
                    records.extend(saved['snapshots'])
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                print(json.dumps({'collect':split, 'state':state_index}), flush=True)
                report, frames = run_episode(world, policy, state_index, collect=True)
                rows = []
                indices = np.unique(np.linspace(0, len(frames)-1, cfg['collection']['frames_per_episode'], dtype=int))
                for index in indices:
                    frame = frames[int(index)]
                    world.restore_state(frame['state'])
                    image = world.image()
                    if array_digest(image) != array_digest(frame['image']):
                        np.savez_compressed(folder/'replay_mismatch.npz',state=frame['state'],
                                            expected=frame['image'],actual=image)
                        Image.fromarray(frame['image']).save(folder/'replay_expected.png')
                        Image.fromarray(image).save(folder/'replay_actual.png')
                        raise ValueError('Saved full state does not reproduce the policy RGB observation')
                    mask = world.object_mask()
                    path = folder/f"step_{frame['step']:04d}.npz"
                    np.savez_compressed(path, state=frame['state'], image=image, mask=mask,
                                        context_json=json.dumps(frame['context']))
                    Image.fromarray(image).save(path.with_suffix('.png'))
                    visible = policy_visible_pixels(mask, cfg['policy']['center_crop_area'])
                    row = {'snapshot':str(path.relative_to(root)), 'state_index':state_index,
                           'step':frame['step'], 'split':split, 'canonical_sha256':array_digest(image),
                           'object_pixels':int(mask.sum()), 'policy_visible_pixels':visible,
                           'eligible':visible >= cfg['collection']['min_object_pixels']}
                    rows.append(row)
                report['snapshots'] = rows
                write_json(complete, report)
                records.extend(rows)
            write_json(root/'states'/f'{split}.json', records)
            if not any(row['eligible'] for row in records):
                raise ValueError(f'No visible object snapshots in {split}')
    finally:
        world.close()
    write_json(root/'collection_complete.json', {'splits':{s:len(cfg['splits'][s]) for s in ('train','calibration','validation')}})
    return policy


def load_records(cfg, split, eligible=True):
    root = Path(cfg['output'])
    rows = json.loads((root/'states'/f'{split}.json').read_text())
    result = []
    for row in rows:
        if eligible and not row['eligible']:
            continue
        with np.load(root/row['snapshot'], allow_pickle=False) as source:
            result.append({**row, 'state':source['state'].copy(), 'image':source['image'].copy(),
                           'mask':source['mask'].copy()})
    if not result:
        raise ValueError(f'No eligible records in {split}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    collect_states(load_config(args.config))


if __name__ == '__main__':
    main()
