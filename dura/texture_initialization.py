"""Reproducible material references from prompts or checksum-pinned NumPy latents."""
from pathlib import Path
import shutil

import numpy as np
import torch

from .object_texture import array_digest, digest


def validate_initialization(cfg):
    settings = cfg.get('initialization', {'mode': 'reconstruction'})
    mode = settings.get('mode')
    if mode not in ('reconstruction', 'prompt', 'saved_latent'):
        raise ValueError('Initialization mode must be reconstruction, prompt or saved_latent')
    if mode == 'reconstruction':
        return
    size = cfg['mapping']['working_size']
    if type(size) is not int or size < 8 or size % 8:
        raise ValueError('Stable Diffusion v1.5 working size must be divisible by 8')
    seeds = {str(s) for s in cfg['search']['seeds']}
    prompts = settings.get('prompts', {})
    if prompts and (set(prompts) != seeds or any(not isinstance(p, str) or not p.strip() for p in prompts.values())):
        raise ValueError('Prompts must have one string-keyed entry per search seed')
    if mode == 'saved_latent':
        if settings.get('space') not in ('noise', 'clean'):
            raise ValueError('Saved latent space must explicitly be noise or clean')
        files = settings.get('files', {})
        if set(files) != seeds:
            raise ValueError('Supply one saved latent file per search seed')
        for item in files.values():
            sha = item.get('sha256', '')
            if not item.get('path') or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
                raise ValueError('Saved latent files require a path and lowercase SHA-256')
    denoise = mode == 'prompt' or settings.get('space') == 'noise'
    if denoise:
        prompt = settings.get('prompt', '')
        if not prompts and (not isinstance(prompt, str) or not prompt.strip()):
            raise ValueError('Prompt/noise initialization requires a material prompt')
        guidance = settings.get('guidance_scale', 7.5)
        steps = settings.get('ddim_steps', 50)
        if (not np.isfinite(guidance) or guidance <= 1 or type(steps) is not int or not 1 <= steps <= 1000 or
                not isinstance(settings.get('negative_prompt', ''), str)):
            raise ValueError('Invalid initialization guidance, DDIM steps or negative prompt')


def read_latent(path, sha256, size):
    path = Path(path)
    if path.suffix != '.npy' or digest(path) != sha256:
        raise ValueError('Saved latent must be a checksum-matching .npy file')
    value = np.load(path, allow_pickle=False)
    if (value.shape != (1, 4, size//8, size//8) or not np.issubdtype(value.dtype, np.floating) or
            not np.isfinite(value).all()):
        raise ValueError('Saved latent must be finite floating point with shape [1,4,H/8,W/8]')
    value = value.astype(np.float32)
    if not np.isfinite(value).all():
        raise ValueError('Saved latent overflows float32')
    return torch.from_numpy(value.copy())


def prepare_initialization(cfg):
    """Freeze external inputs before a job is queued; never overwrite them on resume."""
    settings = cfg.get('initialization', {})
    if settings.get('mode') != 'saved_latent':
        return
    for seed, item in settings['files'].items():
        target = Path(cfg['output'])/'assets/initial_latents'/f'seed_{seed}.npy'
        source = target if target.exists() else Path(item['path'])
        read_latent(source, item['sha256'], cfg['mapping']['working_size'])
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.tmp')
            shutil.copyfile(source, temporary)
            temporary.replace(target)
        if digest(target) != item['sha256']:
            raise ValueError('Initial latent changed while freezing')


@torch.no_grad()
def independent_reference(backend, cfg, seed):
    """Return a reference without reading, encoding or noising the original atlas."""
    from .train_object_texture import uint8_panel
    settings = cfg['initialization']; size = cfg['mapping']['working_size']
    mode = settings['mode']
    if mode not in ('prompt', 'saved_latent'):
        raise ValueError('Expected independent texture initialization')
    source = None
    if mode == 'saved_latent':
        source = settings['files'][str(seed)]
        value = read_latent(Path(cfg['output'])/'assets/initial_latents'/f'seed_{seed}.npy',
                            source['sha256'], size)
    else:
        value = torch.randn((1, 4, size//8, size//8), generator=torch.Generator().manual_seed(seed))
    input_sha = array_digest(value.numpy())
    clean = mode == 'saved_latent' and settings['space'] == 'clean'
    prompt = settings.get('prompts', {}).get(str(seed), settings.get('prompt', ''))
    if clean:
        latent = value.to(backend.device)
    else:
        backend.set_prompt(prompt, settings.get('negative_prompt', ''), settings.get('guidance_scale', 7.5))
        latent = backend.from_noise(value, settings.get('ddim_steps', 50))
    panel = uint8_panel(backend.decode(latent)[0])
    metadata = {'mode': mode, 'space': 'clean' if clean else 'noise', 'seed': seed,
                'prompt': None if clean else prompt,
                'negative_prompt': None if clean else settings.get('negative_prompt', ''),
                'guidance_scale': None if clean else settings.get('guidance_scale', 7.5),
                'ddim_steps': 0 if clean else settings.get('ddim_steps', 50),
                'source_file_sha256': source['sha256'] if source else None,
                'input_tensor_sha256': input_sha, 'latent_sha256': array_digest(latent.cpu().numpy()),
                'original_texture_encoded': False}
    return panel, latent.detach(), metadata
