"""Immutable atlas mapping and native LIBERO texture rendering.

GPU/model imports are lazy so configuration, mapping, and report checks run on CPU.
The external benchmark assets are never written.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw
import yaml

from .config import expand_path, read_config


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def reset_gripper_commands(robots):
    """Match a fresh gripper constructor; robosuite 1.4 soft reset omits this state."""
    for robot in robots:
        if getattr(robot, 'has_gripper', False):
            robot.gripper.current_action = np.zeros(robot.gripper.dof)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_config(path):
    path = Path(path).resolve()
    text = path.read_text()
    specific = json.loads(text) if path.suffix.lower() == '.json' else yaml.safe_load(text)
    if specific.get('base_config'):
        cfg = read_config(path.parent / specific['base_config'])
        for key in ('manifest', 'validation_manifest', 'seed_patch', 'naturalness',
                    'patch_box', 'validation_frames', 'validate_actions'):
            cfg.pop(key, None)
        cfg.update(specific)
        cfg.pop('base_config')
        cfg['policy'].update(cfg.pop('policy_override'))
        cfg['diffusion']['local_files_only'] = True
        cfg['diffusion']['load_safety_checker'] = False
        cfg['attack'].update(mode='blackbox', samples=cfg['search']['samples'],
                             baseline=cfg['search']['baseline'],
                             query_alpha=cfg['search']['query_alpha'])
    else:
        cfg = specific
    cfg['libero_root'] = expand_path(os.environ.get('DURA_LIBERO_ROOT', cfg['libero_root']), path.parent)
    for key in ('output', 'final_output'):
        value = os.environ.get('DURA_FINAL_ROOT') if key == 'final_output' else None
        cfg[key] = expand_path(value or cfg[key], path.parent)
    if cfg.get('initial_state_bank'):
        cfg['initial_state_bank']['path'] = expand_path(cfg['initial_state_bank']['path'], path.parent)
    for item in cfg.get('initialization', {}).get('files', {}).values():
        item['path'] = expand_path(item['path'], path.parent)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    required = {'train', 'calibration', 'validation', 'reserved', 'test'}
    splits = cfg['splits']
    if set(splits) != required or any(not rows for rows in splits.values()):
        raise ValueError('Specify all five nonempty state splits')
    flat = [i for rows in splits.values() for i in rows]
    if any(type(i) is not int or i < 0 for i in flat) or len(flat) != len(set(flat)):
        raise ValueError('State splits must be disjoint nonnegative integer indices')
    if cfg['suite'] != 'libero_object' or cfg['task_id'] != 0 or cfg['task_order'] != 0:
        raise ValueError('This pilot is fixed to LIBERO Object task 0/order 0')
    if cfg['task_name'] != 'pick_up_the_alphabet_soup_and_place_it_in_the_basket':
        raise ValueError('Wrong pilot task')
    from .config import AttackConfig
    attack = AttackConfig(**cfg['attack'])
    unconstrained = cfg.get('profile') == 'full_surface_unconstrained'
    benign_latent = cfg.get('profile') == 'benign_latent'
    if unconstrained and (cfg['mapping'].get('mode') != 'full' or cfg['appearance'].get('enabled', True)):
        raise ValueError('Full-surface profile requires full mapping and disabled appearance gates')
    if unconstrained and (cfg.get('evaluation',{}).get('primary_reference') != 'original' or
            cfg.get('objective',{}).get('kind') != 'action_divergence'):
        raise ValueError('Full-surface profile requires original-conditioned ASR and action divergence')
    if benign_latent:
        from .texture_initialization import validate_initialization
        validate_initialization(cfg)
        settings = cfg.get('latent_optimizer', {})
        if (cfg['mapping'].get('mode') != 'full' or not cfg['appearance'].get('enabled',True) or
                cfg['appearance'].get('reference') != 'benign' or
                cfg['objective'].get('kind') != 'reference_token_nll'):
            raise ValueError('Benign latent optimization requires full mapping and benign-relative appearance bounds')
        if (not settings.get('radii') or any(not np.isfinite(r) or r <= 0 for r in settings['radii']) or
                settings.get('updates',0) < 1 or settings.get('pairs',0) < 1 or
                settings.get('sigma',0) <= 0 or settings.get('learning_rate',0) <= 0 or
                settings.get('grid_size',0) < 1 or settings.get('validate_every',0) < 1):
            raise ValueError('Invalid direct latent optimizer settings')
        if (len(set(settings['radii'])) != len(settings['radii']) or
                any(type(settings[k]) is not int for k in ('updates','pairs','grid_size','validate_every')) or
                any(not np.isfinite(settings[k]) or settings[k] <= 0
                    for k in ('sigma','learning_rate','max_gradient_norm')) or
                any(not np.isfinite(settings[k]) or settings[k] < 0 for k in ('rgb_weight','latent_weight')) or
                2*settings['pairs'] != cfg['search']['samples']):
            raise ValueError('Inconsistent or nonfinite latent search settings')
        if not cfg.get('initial_state_bank') or cfg['initial_state_bank']['ids'] != splits['test']:
            raise ValueError('This follow-up requires a frozen bank of fresh test initializations')
    if attack.mode != 'blackbox' or (attack.anchor_weight <= 0 and not (unconstrained or benign_latent)):
        raise ValueError('Native texture optimization requires black-box queries and an anchor')
    search = cfg['search']
    if (search['samples'] < 2 or search['batch_size'] < 1 or
            len(set(search['seeds'])) != len(search['seeds']) or
            len(set(search['step_sizes'])) != len(search['step_sizes']) or
            not search['seeds'] or not search['step_sizes'] or
            any(not np.isfinite(x) or x <= 0 for x in search['step_sizes'])):
        raise ValueError('Invalid bounded search')
    if (attack.samples != search['samples'] or attack.baseline != search['baseline'] or
            attack.query_alpha != search['query_alpha']):
        raise ValueError('Search settings and algorithm query settings differ')
    if cfg['collection']['frames_per_episode'] < 1:
        raise ValueError('Need a positive frame count')
    if search.get('gradient_rms_max') is not None and (not np.isfinite(search['gradient_rms_max']) or
                                                      search['gradient_rms_max'] <= 0):
        raise ValueError('Gradient numerical limit must be positive and finite')
    if not 0 < cfg['mapping']['max_surface_fraction'] <= 1:
        raise ValueError('Invalid surface budget')
    if not 0 <= cfg['appearance']['ssim_min'] <= 1 or cfg['appearance']['rmse_max'] < 0:
        raise ValueError('Invalid appearance bounds')
    tolerance = cfg['acceptance']['physics_atol']
    if type(tolerance) not in (int, float) or not np.isfinite(tolerance) or not 0 < tolerance <= 1e-8:
        raise ValueError('Physics tolerance must be numeric and at most 1e-8')


def fingerprint(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def read_mesh(path):
    """Read the actual MuJoCo binary mesh, including its per-vertex UVs."""
    data = Path(path).read_bytes()
    nv, nn, nt, nf = struct.unpack_from('<4i', data)
    expected = 16 + 12*nv + 12*nn + 8*nt + 12*nf
    if min(nv, nf) < 1 or nt != nv or len(data) != expected:
        raise ValueError('Expected a MuJoCo mesh with one UV per vertex')
    vertices = np.frombuffer(data, '<f4', nv*3, 16).reshape(-1, 3).astype(float)
    uv = np.frombuffer(data, '<f4', nt*2, 16+12*(nv+nn)).reshape(-1, 2).astype(float)
    faces = np.frombuffer(data, '<i4', nf*3, 16+12*(nv+nn)+8*nt).reshape(-1, 3)
    if not np.isfinite(vertices).all() or not np.isfinite(uv).all() or faces.min() < 0 or faces.max() >= nv:
        raise ValueError('Invalid mesh data')
    return vertices, uv, faces


def polygon_area(vertices):
    v = np.asarray(vertices)
    return 0.0 if len(v) < 3 else abs(float(np.dot(v[:, 0], np.roll(v[:, 1], -1)) -
                                               np.dot(v[:, 1], np.roll(v[:, 0], -1)))) / 2


def clip_rectangle(poly, box):
    """Sutherland–Hodgman clipping; exact affine area ratio within a UV triangle."""
    result = [np.array(p, dtype=float) for p in poly]
    for axis, bound, direction in ((0, box[0], 1), (0, box[2], -1),
                                    (1, box[1], 1), (1, box[3], -1)):
        previous, result = result, []
        if not previous:
            break
        a = previous[-1]
        for b in previous:
            inside_a = direction*(a[axis]-bound) >= 0
            inside_b = direction*(b[axis]-bound) >= 0
            if inside_a != inside_b:
                ratio = (bound-a[axis])/(b[axis]-a[axis])
                result.append(a + ratio*(b-a))
            if inside_b:
                result.append(b)
            a = b
    return result


def surface_fraction(vertices, uv_pixels, faces, box):
    triangle = vertices[faces]
    areas = np.linalg.norm(np.cross(triangle[:, 1]-triangle[:, 0],
                                    triangle[:, 2]-triangle[:, 0]), axis=1)/2
    if not areas.sum() > 0:
        raise ValueError('Mesh has no surface area')
    edited = 0.0
    for triangle_uv, area in zip(uv_pixels[faces], areas):
        uv_area = polygon_area(triangle_uv)
        if uv_area > 1e-12:
            edited += area * min(1.0, polygon_area(clip_rectangle(triangle_uv, box))/uv_area)
    return float(edited / areas.sum())


class AtlasMapping:
    def __init__(self, original, vertices, uv, faces, settings):
        self.original = np.asarray(original, dtype=np.uint8).copy()
        self.settings = dict(settings)
        self.height, self.width = self.original.shape[:2]
        self.panel = tuple(settings['panel'])
        self.editable_box = tuple(settings['editable'])
        l, t, r, b = self.panel
        x0, y0, x1, y1 = self.editable_box
        if not (0 <= l <= x0 < x1 <= r <= self.width and
                0 <= t <= y0 < y1 <= b <= self.height):
            raise ValueError('Editable box must be inside panel and original atlas')
        full = settings.get('mode', 'panel') == 'full'
        if full and (self.panel != (0, 0, self.width, self.height) or self.editable_box != self.panel):
            raise ValueError('Full-surface mapping must cover the entire atlas')
        if settings['feather'] < 0 or (settings['feather'] == 0 and not full):
            raise ValueError('A positive boundary taper is required')
        if uv.min() < -1e-5 or uv.max() > 1+1e-5:
            raise ValueError('Pilot expects texture coordinates in [0,1]')
        # MuJoCo 2.3.7 uploads tex_rgb in file row order and uses mesh UVs directly.
        # This asset's native texture buffer matches the PNG without a vertical flip.
        if settings.get('uv_origin', 'top') == 'top':
            pixels = uv * [self.width-1, self.height-1]
        elif settings['uv_origin'] == 'bottom':
            pixels = uv * [self.width-1, -(self.height-1)] + [0, self.height-1]
        else:
            raise ValueError('Unknown UV origin')
        used = Image.new('1', (self.width, self.height))
        draw = ImageDraw.Draw(used)
        for triangle in pixels[faces]:
            draw.polygon([tuple(p) for p in triangle], fill=1)
        yy, xx = np.mgrid[:self.height, :self.width]
        distance = np.minimum.reduce((xx-x0, x1-1-xx, yy-y0, y1-1-yy))
        self.mask = np.ones((self.height, self.width), dtype=bool) if full else (distance >= 0) & np.array(used, dtype=bool)
        if not self.mask.any():
            raise ValueError('Editable rectangle misses used texels')
        self.alpha = (np.ones_like(distance, dtype=np.float32) if full else
                      np.clip(distance / settings['feather'], 0, 1).astype(np.float32) * self.mask)
        # Full-atlas replacement also changes degenerate-UV triangles' sampled texels.
        self.surface_fraction = 1.0 if full else surface_fraction(vertices, pixels, faces, self.editable_box)
        if self.surface_fraction > settings['max_surface_fraction']:
            raise ValueError(f'Editable surface exceeds budget: {self.surface_fraction}')
        # Include ordinary mask boundaries and actual mesh UV joins in continuity checks.
        self.edge_a, self.edge_b = self._edges(vertices, pixels, faces)

    def _edges(self, vertices, pixels, faces):
        a, b = [], []
        for shift in (0, 1):
            first = self.mask[:-1, :] if shift == 0 else self.mask[:, :-1]
            second = self.mask[1:, :] if shift == 0 else self.mask[:, 1:]
            yy, xx = np.where(first != second)
            a.extend(zip(yy, xx))
            b.extend(zip(yy+(shift == 0), xx+(shift == 1)))
        joins = {}
        for face in faces:
            for i, j in ((0, 1), (1, 2), (2, 0)):
                left, right = int(face[i]), int(face[j])
                kl, kr = tuple(np.round(vertices[left], 6)), tuple(np.round(vertices[right], 6))
                if kl > kr:
                    kl, kr, left, right = kr, kl, right, left
                key = (kl, kr)
                edge = pixels[[left, right]]
                if key in joins:
                    other = joins[key]
                    if np.max(np.abs(other-edge)) > 1:
                        for f in np.linspace(0, 1, 8):
                            p, q = np.rint(edge[0]*(1-f)+edge[1]*f).astype(int), np.rint(other[0]*(1-f)+other[1]*f).astype(int)
                            p = np.clip(p, [0, 0], [self.width-1, self.height-1])
                            q = np.clip(q, [0, 0], [self.width-1, self.height-1])
                            if self.mask[p[1], p[0]] or self.mask[q[1], q[0]]:
                                a.append((p[1], p[0])); b.append((q[1], q[0]))
                else:
                    joins[key] = edge
        return np.asarray(a, dtype=int).reshape(-1, 2), np.asarray(b, dtype=int).reshape(-1, 2)

    def seed_panel(self):
        return np.array(Image.fromarray(self.original).crop(self.panel).resize(
            (self.settings['working_size'],)*2, Image.Resampling.LANCZOS))

    def compose(self, panel):
        panel = np.asarray(panel)
        if panel.dtype != np.uint8:
            if not np.isfinite(panel).all():
                raise ValueError('Nonfinite texture candidate')
            panel = np.rint(np.clip(panel, 0, 1)*255).astype(np.uint8)
        l, t, r, b = self.panel
        resized = np.asarray(Image.fromarray(panel).resize((r-l, b-t), Image.Resampling.LANCZOS))
        result = self.original.copy()
        alpha = self.alpha[t:b, l:r, None]
        result[t:b, l:r] = np.rint(result[t:b, l:r]*(1-alpha)+resized*alpha).astype(np.uint8)
        if not np.array_equal(result[~self.mask], self.original[~self.mask]):
            raise AssertionError('Protected texture pixels changed')
        return result

    def seam_energy(self, atlas):
        if not len(self.edge_a):
            return 0.0
        atlas = np.asarray(atlas, dtype=np.float32)/255
        return float(np.abs(atlas[tuple(self.edge_a.T)]-atlas[tuple(self.edge_b.T)]).mean())

    def texture_report(self, atlas, benign, bounds):
        atlas, benign = np.asarray(atlas), np.asarray(benign)
        def rmse(reference):
            return float(np.sqrt(np.mean(((atlas[self.mask].astype(float)-reference[self.mask])/255)**2)))
        values = {'rmse_original': rmse(self.original), 'rmse_benign': rmse(benign),
                  'surface_fraction': self.surface_fraction, 'seam_energy': self.seam_energy(atlas),
                  'original_seam_energy': self.seam_energy(self.original),
                  'protected_pixels_exact': bool(np.array_equal(atlas[~self.mask], self.original[~self.mask]))}
        benign_reference = bounds.get('reference') == 'benign'
        values['reference_seam_energy'] = self.seam_energy(benign) if benign_reference else values['original_seam_energy']
        seam_limit = values['reference_seam_energy']*(1+bounds['seam_relative_increase'])+bounds['seam_absolute_slack']
        values['accepted'] = bool(values['protected_pixels_exact'] and
            (values['rmse_benign'] if benign_reference else max(values['rmse_original'], values['rmse_benign'])) <= bounds['rmse_max'] and
            values['seam_energy'] <= seam_limit and
            self.surface_fraction <= self.settings['max_surface_fraction'])
        values['constraints_enabled'] = bounds.get('enabled', True)
        if not values['constraints_enabled']:
            values['accepted'] = bool(values['protected_pixels_exact'] and np.isfinite(atlas).all())
        return values


def load_mapping(cfg):
    root = Path(cfg['libero_root'])
    path = root/cfg['texture_relative']
    if digest(path) != cfg['texture_sha256']:
        raise ValueError('Original LIBERO texture hash changed')
    return AtlasMapping(np.array(Image.open(path).convert('RGB')),
                        *read_mesh(root/cfg['mesh_relative']), cfg['mapping'])


def save_mapping(mapping, folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mapping.original).save(folder/'original.png')
    Image.fromarray(mapping.seed_panel()).save(folder/'seed_panel.png')
    Image.fromarray(mapping.mask.astype('uint8')*255).save(folder/'editable_mask.png')
    preview = mapping.original.copy()
    preview[mapping.mask] = np.rint(.4*preview[mapping.mask]+.6*np.array([255, 0, 0])).astype('uint8')
    Image.fromarray(preview).resize((768, 768)).save(folder/'mapping_preview.png')
    write_json(folder/'mapping.json', {'settings':mapping.settings, 'surface_fraction':mapping.surface_fraction,
                                     'mask_sha256':digest(folder/'editable_mask.png'),
                                     'original_sha256':array_digest(mapping.original)})


class NativeTextureWorld:
    """One unchanged LIBERO environment; direct renders avoid stale sensor caches."""
    def __init__(self, cfg, render=True):
        import mujoco
        import torch
        sys.path.insert(0, cfg['libero_root'])
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs.env_wrapper import ControlEnv
        self.cfg, self.mujoco, self.render_enabled = cfg, mujoco, render
        if render:
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
        suite = benchmark.get_benchmark_dict()[cfg['suite']](task_order_index=cfg['task_order'])
        task = suite.get_task(cfg['task_id'])
        if task.name != cfg['task_name']:
            raise ValueError('Installed task order differs from the frozen protocol')
        self.instruction = task.language
        init_path = Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file
        if digest(init_path) != cfg['init_sha256']:
            raise ValueError('Initial-state file changed')
        self.initial_states = torch.load(init_path, map_location='cpu')
        self.extra_initial_states = {}
        bank = cfg.get('initial_state_bank')
        if bank:
            from .initial_states import load_state_bank
            self.extra_initial_states = load_state_bank(bank, self.initial_states.shape[1:])
        if any(i >= len(self.initial_states) and i not in self.extra_initial_states
               for rows in cfg['splits'].values() for i in rows):
            raise ValueError('State split exceeds installed initial-state count')
        self.env = ControlEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),
                              use_camera_obs=False, has_offscreen_renderer=render,
                              hard_reset=False, camera_names=[cfg['collection']['camera']],
                              camera_heights=cfg['collection']['camera_size'], camera_widths=cfg['collection']['camera_size'])
        self.env.seed(cfg['collection']['seed']); self.env.reset()
        self.sim = self.env.sim
        self.model = self.sim.model._model
        self.data = self.sim.data._data
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        bodies = [i for i in range(self.model.nbody)
                  if (self.sim.model.body_id2name(i) or '').startswith(cfg['instance']+'_')]
        self.geom_ids = np.where(np.isin(self.model.geom_bodyid, bodies))[0]
        visual = self.geom_ids[(self.model.geom_group[self.geom_ids] == 1) & (self.model.geom_matid[self.geom_ids] >= 0)]
        materials = np.unique(self.model.geom_matid[visual])
        textures = np.unique(self.model.mat_texid[materials]); textures = textures[textures >= 0]
        if len(textures) != 1:
            raise ValueError(f'Expected one target texture, found {textures}')
        self.texture_id = int(textures[0])
        using_materials = np.where(self.model.mat_texid == self.texture_id)[0]
        using_geoms = np.where(np.isin(self.model.geom_matid, using_materials))[0]
        if set(using_geoms)-set(self.geom_ids):
            raise ValueError('Target texture is shared with another instance; clone its XML binding first')
        h, w = int(self.model.tex_height[self.texture_id]), int(self.model.tex_width[self.texture_id])
        offset = int(self.model.tex_adr[self.texture_id])
        self.texture_view = self.model.tex_rgb[offset:offset+h*w*3].reshape(h, w, 3)
        self.original_gpu_texture = self.texture_view.copy()
        original = np.array(Image.open(Path(cfg['libero_root'])/cfg['texture_relative']).convert('RGB'))
        self.flip_texture = None
        for flip in (False, True):
            if original.shape == self.texture_view.shape and np.array_equal(original[::-1] if flip else original, self.texture_view):
                self.flip_texture = flip
                break
        if self.flip_texture is None:
            raise ValueError('Compiled texture differs from the original RGB atlas')
        if self.flip_texture != (cfg['mapping'].get('uv_origin', 'top') == 'bottom'):
            raise ValueError('Atlas mapping orientation differs from the native texture buffer')
        self.current_atlas = original.copy()
        self.render_count = 0

    def state(self):
        state = np.empty(self.mujoco.mj_stateSize(self.model, self.state_spec))
        self.mujoco.mj_getState(self.model, self.data, state, self.state_spec)
        return state

    def restore_state(self, state):
        self.mujoco.mj_setState(self.model, self.data, np.asarray(state), self.state_spec)
        self.synchronize()

    def synchronize(self):
        """Derive camera/geometry poses at current qpos without changing integration state.

        mj_step integrates positions after computing derived transforms. Rendering those
        cached transforms would disagree with rendering a restored copy of the state.
        Preserve solver warm-start history while recomputing the derived quantities.
        """
        warmstart = self.data.qacc_warmstart.copy()
        self.sim.forward()
        self.data.qacc_warmstart[:] = warmstart

    def set_texture(self, atlas):
        atlas = np.asarray(atlas, dtype=np.uint8)
        if atlas.shape != self.current_atlas.shape:
            raise ValueError('Texture shape differs from compiled atlas')
        self.texture_view[:] = atlas[::-1] if self.flip_texture else atlas
        if self.render_enabled:
            context = self.sim._render_context_offscreen
            context.gl_ctx.make_current()
            # Use native MjModel, bypassing robosuite 1.4's proxy upload wrapper.
            self.mujoco.mjr_uploadTexture(self.model, context.con, self.texture_id)
        self.current_atlas = atlas

    def original(self):
        atlas = self.original_gpu_texture[::-1] if self.flip_texture else self.original_gpu_texture
        self.set_texture(atlas)

    def reset(self, index):
        self.original()
        seed = self.cfg['collection']['seed'] + int(index)
        np.random.seed(seed); self.env.seed(seed); self.env.reset()
        reset_gripper_commands(self.env.robots)
        initial = self.extra_initial_states[index] if index in self.extra_initial_states else self.initial_states[index]
        self.env.set_init_state(initial)
        for _ in range(self.cfg['collection']['settle_steps']):
            self.env.step([0., 0., 0., 0., 0., 0., -1.])
        return self.state()

    def _raw(self, segmentation=False):
        from robosuite import macros
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
        settings = self.cfg['collection']
        self.synchronize()
        self.render_count += 1
        raw = self.sim.render(width=settings['camera_size'], height=settings['camera_size'],
                              camera_name=settings['camera'], segmentation=segmentation)
        return raw[::IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]]

    def image(self):
        from .evaluate_libero import policy_image
        return policy_image(self._raw(), 'openvla', self.cfg['collection']['image_size'])

    def object_mask(self):
        raw = self._raw(segmentation=True)
        valid = (raw[..., 0] == int(self.mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(raw[..., 1], self.geom_ids)
        size = self.cfg['collection']['image_size']
        return np.array(Image.fromarray(valid[::-1, ::-1]).resize((size, size), Image.Resampling.NEAREST), dtype=bool)

    def canonical(self):
        atlas = self.current_atlas
        self.original()
        image = self.image()
        self.set_texture(atlas)
        return image

    def close(self):
        self.original()
        self.env.close()


def crop_object(image, mask, border=4, size=128):
    yy, xx = np.where(mask)
    if not len(xx):
        raise ValueError('Object is invisible')
    x0, y0 = max(0, int(xx.min())-border), max(0, int(yy.min())-border)
    x1, y1 = min(image.shape[1], int(xx.max())+border+1), min(image.shape[0], int(yy.max())+border+1)
    return np.array(Image.fromarray(image).crop((x0, y0, x1, y1)).resize((size, size), Image.Resampling.BILINEAR))


def audit_test_states(cfg):
    """Inspect earlier compact ASR reports and raw rollout manifests, without policy calls."""
    root = Path(cfg['output']).parent
    test = set(cfg['splits']['test'])
    conflicts, files = [], 0
    for pattern in ('asr_report.json', 'rollouts.jsonl'):
        for path in root.rglob(pattern):
            if Path(cfg['output']) in path.parents:
                continue
            files += 1
            if pattern.endswith('.jsonl'):
                rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
            else:
                data = json.loads(path.read_text())
                if not isinstance(data, dict):
                    continue
                rows = data.get('comparison', {}).get('paired_successes', data.get('paired_successes', []))
                context = dict(data)
                config_path = path.parent/'config.json'
                if config_path.exists():
                    context = {**json.loads(config_path.read_text()), **context}
            for row in rows:
                inherited = {} if pattern.endswith('.jsonl') else context
                index = row.get('trial', row.get('state'))
                if (row.get('suite', inherited.get('suite')) == cfg['suite'] and
                        row.get('task_id', inherited.get('task_id')) == cfg['task_id'] and index in test):
                    conflicts.append({'path':str(path), 'state':index})
    report = {'files_examined':files, 'conflicts':conflicts,
              'scope':'Prior scratch rollout and ASR reports; this audit does not establish independence from policy pretraining.'}
    if conflicts:
        raise ValueError(f'Reserved test states appeared in prior evaluation: {conflicts[:5]}')
    return report
