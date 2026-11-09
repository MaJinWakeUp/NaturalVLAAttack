"""Freeze additional LIBERO placement-sampler initializations without policy queries."""
import copy
import json
from pathlib import Path

import numpy as np

from .object_texture import NativeTextureWorld, array_digest, digest, reset_gripper_commands, write_json


def load_state_bank(settings, state_shape):
    path = Path(settings['path'])
    if not settings.get('sha256') or digest(path) != settings['sha256']:
        raise ValueError('Initial-state bank is missing or changed; run preparation first')
    with np.load(path,allow_pickle=False) as data:
        ids, states = data['ids'], data['states']
        if (not np.issubdtype(ids.dtype,np.integer) or ids.tolist() != settings['ids'] or states.shape != (len(ids),*state_shape) or
                not np.isfinite(states).all() or len(set(ids.tolist())) != len(ids)):
            raise ValueError('Invalid additional initial-state bank')
        return {int(index):state.copy() for index,state in zip(ids,states)}


def prepare_state_bank(cfg):
    settings = cfg['initial_state_bank']; path = Path(settings['path'])
    if path.exists():
        sha = digest(path)
        if settings.get('sha256') and sha != settings['sha256']:
            raise ValueError('Frozen initial-state bank changed')
        with np.load(path,allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata']))
            if (data['ids'].tolist() != settings['ids'] or
                    metadata['init_sha256'] != cfg['init_sha256'] or
                    metadata['task_name'] != cfg['task_name'] or
                    metadata['collection_seed'] != cfg['collection']['seed']):
                raise ValueError('Existing state bank belongs to a different configuration')
        settings['sha256'] = sha
        return
    if settings.get('sha256'):
        raise FileNotFoundError('Frozen state bank is missing; do not regenerate it silently')
    if any(i < 1000 for i in settings['ids']) or len(set(settings['ids'])) != len(settings['ids']):
        raise ValueError('Fresh initialization IDs must be unique and at least 1000')
    bootstrap = copy.deepcopy(cfg)
    bootstrap.pop('initial_state_bank')
    bootstrap['splits']['test'] = [20]
    world = NativeTextureWorld(bootstrap,render=False)
    states = []
    try:
        prior = {array_digest(np.asarray(state,dtype=np.float64)) for state in world.initial_states}
        for index in settings['ids']:
            seed = cfg['collection']['seed']+index
            np.random.seed(seed); world.env.seed(seed); world.env.reset()
            reset_gripper_commands(world.env.robots)
            world.sim.forward()
            state = np.asarray(world.env.get_sim_state(),dtype=np.float64).copy()
            if (state.shape != world.initial_states.shape[1:] or not np.isfinite(state).all() or
                    array_digest(state) in prior or world.env.check_success()):
                raise ValueError('Invalid or duplicated freshly sampled initialization')
            prior.add(array_digest(state)); states.append(state)
    finally:
        world.close()
    metadata = {'task_name':cfg['task_name'],'init_sha256':cfg['init_sha256'],
                'collection_seed':cfg['collection']['seed'],
                'method':'Native LIBERO placement sampler, reset seed = collection seed + ID',
                'policy_queries':0,'rendered_images':0}
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.stem+'.tmp.npz')
    np.savez_compressed(temporary,ids=np.array(settings['ids']),states=np.stack(states),metadata=json.dumps(metadata))
    temporary.replace(path)
    settings['sha256'] = digest(path)
    write_json(path.with_suffix('.json'),{**metadata,'ids':settings['ids'],'sha256':settings['sha256']})
