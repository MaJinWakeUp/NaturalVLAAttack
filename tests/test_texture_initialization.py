"""Independent material generation, saved-latent provenance and CFG regression tests."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from dura.config import PROJECT_ROOT
from dura.diffusion import StableDiffusionBackend
from dura.object_texture import digest, load_config, validate_config
from dura.texture_initialization import independent_reference, prepare_initialization, read_latent


class FakeMaterialBackend:
    device = torch.device('cpu')

    def __init__(self):
        self.calls = []

    def set_prompt(self, *args):
        self.calls.append(('prompt', args))

    def from_noise(self, noise, steps):
        self.calls.append(('denoise', steps))
        return noise * .5

    def decode(self, latent):
        return torch.nn.functional.interpolate(latent[:, :3].sigmoid(), scale_factor=8)

    def encode(self, image):
        raise AssertionError('Independent initialization must not encode the original texture')


class TextureInitializationTests(unittest.TestCase):
    def config(self):
        return load_config(PROJECT_ROOT/'configs/object_texture_prompt_latent_v1.yaml')

    def test_prompt_generation_is_seeded_independent_and_records_prompt(self):
        cfg = self.config(); backend = FakeMaterialBackend()
        panel, latent, metadata = independent_reference(backend, cfg, 0)
        again, repeated, meta_again = independent_reference(backend, cfg, 0)
        _, other, _ = independent_reference(backend, cfg, 1)
        self.assertTrue(torch.equal(latent, repeated))
        self.assertTrue(np.array_equal(panel, again))
        self.assertEqual(metadata, meta_again)
        self.assertFalse(torch.equal(latent, other))
        self.assertEqual(panel.shape, (512, 512, 3))
        self.assertFalse(metadata['original_texture_encoded'])
        self.assertIn('copper', metadata['prompt'])
        self.assertIn('human', backend.calls[0][1][1])
        self.assertEqual(backend.calls[1], ('denoise', 50))

    def test_saved_clean_latent_is_frozen_decoded_directly_and_checksum_bound(self):
        cfg = self.config()
        with TemporaryDirectory() as directory:
            root = Path(directory); source = root/'custom.npy'
            tensor = np.ones((1, 4, 64, 64), dtype=np.float32)*.25
            np.save(source, tensor)
            cfg['output'] = str(root/'run')
            cfg['initialization'].update(mode='saved_latent', space='clean',
                files={str(seed): {'path':str(source), 'sha256':digest(source)} for seed in (0, 1)})
            validate_config(cfg); prepare_initialization(cfg)
            source.unlink()  # Resume uses the immutable scratch copy.
            prepare_initialization(cfg)
            backend = FakeMaterialBackend()
            _, latent, metadata = independent_reference(backend, cfg, 0)
            self.assertTrue(np.array_equal(latent.numpy(), tensor))
            self.assertEqual(backend.calls, [])
            self.assertEqual(metadata['ddim_steps'], 0)
            self.assertIsNone(metadata['prompt'])
            frozen = root/'run/assets/initial_latents/seed_0.npy'
            frozen.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                prepare_initialization(cfg)

    def test_saved_noise_uses_prompt_and_full_denoising(self):
        cfg = self.config()
        with TemporaryDirectory() as directory:
            root = Path(directory); source = root/'noise.npy'
            tensor = np.ones((1, 4, 64, 64), dtype=np.float32)
            np.save(source, tensor)
            cfg['output'] = str(root/'run')
            cfg['initialization'].update(mode='saved_latent', space='noise',
                files={str(seed): {'path':str(source), 'sha256':digest(source)} for seed in (0, 1)})
            prepare_initialization(cfg); backend = FakeMaterialBackend()
            _, latent, metadata = independent_reference(backend, cfg, 1)
            self.assertTrue(np.array_equal(latent.numpy(), tensor*.5))
            self.assertIn('ceramic', backend.calls[0][1][0])
            self.assertEqual(metadata['space'], 'noise')

    def test_invalid_saved_latents_and_configuration_fail_early(self):
        cfg = self.config()
        with TemporaryDirectory() as directory:
            path = Path(directory)/'latent.npy'
            for value in (np.zeros((4, 64, 64)), np.zeros((1, 4, 64, 64),dtype=np.int32),
                          np.full((1, 4, 64, 64),np.nan)):
                np.save(path, value)
                with self.assertRaises(ValueError):
                    read_latent(path, digest(path), 512)
        for settings in ({'mode':'bad'}, {'mode':'prompt','prompt':''},
                         {'mode':'saved_latent','space':'ambiguous'},
                         {'mode':'prompt','prompt':'copper','guidance_scale':float('nan')},
                         {'mode':'prompt','prompts':{0:'copper',1:'ceramic'}}):
            invalid = copy.deepcopy(cfg); invalid['initialization'] = settings
            with self.assertRaises(ValueError):
                validate_config(invalid)

    def test_new_budget_and_final_states_are_disjoint_from_prior_evaluation(self):
        cfg = self.config(); old = load_config(PROJECT_ROOT/'configs/object_texture_benign_latent.yaml')
        self.assertFalse(set(cfg['splits']['test']) & set(old['splits']['test']))
        self.assertEqual(len(cfg['splits']['test']), 30)
        self.assertEqual(cfg['appearance']['ssim_min'], .8)
        self.assertEqual(cfg['appearance']['rmse_max'], .2)
        settings = cfg['latent_optimizer']
        queries = len(cfg['search']['seeds'])*len(settings['radii'])*settings['updates']*2*settings['pairs']*cfg['search']['batch_size']
        self.assertEqual(queries, 25600)

    def test_backend_negative_prompt_and_batched_guidance(self):
        backend = StableDiffusionBackend.__new__(StableDiffusionBackend)
        backend.device = torch.device('cpu'); backend.dtype = torch.float32
        observed = {}

        class Tokenizer:
            model_max_length = 3
            def __call__(self, prompts, **kwargs):
                observed['prompts'] = prompts
                return SimpleNamespace(input_ids=torch.arange(len(prompts)).reshape(-1, 1))

        class Encoder:
            config = SimpleNamespace(use_attention_mask=False)
            def __call__(self, tokens, **kwargs):
                return (tokens.float()[:, :, None],)

        def unet(sample, timestep, encoder_hidden_states):
            observed['embedding_order'] = encoder_hidden_states[:, 0, 0].tolist()
            return SimpleNamespace(sample=encoder_hidden_states[:, :, :, None].expand_as(sample))

        backend.pipe = SimpleNamespace(tokenizer=Tokenizer(), text_encoder=Encoder(), unet=unet)
        backend.scheduler = SimpleNamespace(scale_model_input=lambda value,t:value,
            step=lambda prediction,t,latent,eta:SimpleNamespace(prev_sample=prediction))
        backend.set_prompt('copper', 'human', 7.5)
        value = backend.step(torch.zeros((2, 4, 2, 2)), torch.tensor(1))
        self.assertEqual(observed['prompts'], ['human', 'copper'])
        self.assertEqual(observed['embedding_order'], [0., 0., 1., 1.])
        self.assertTrue(torch.equal(value, torch.full_like(value, 7.5)))
        backend.set_prompt('ceramic', guidance_scale=1.)
        self.assertEqual(observed['prompts'], ['ceramic'])
        self.assertEqual(backend.step(torch.zeros((2,4,2,2)), torch.tensor(1)).shape,(2,4,2,2))

    def test_backend_noise_path_uses_every_timestep_and_noise_sigma(self):
        backend = StableDiffusionBackend.__new__(StableDiffusionBackend)
        backend.device = torch.device('cpu')
        observed = []
        def set_timesteps(steps, device):
            backend.scheduler.timesteps = list(reversed(range(steps)))
        backend.scheduler = SimpleNamespace(set_timesteps=set_timesteps, init_noise_sigma=2.)
        def step(value, timestep):
            observed.append(timestep)
            return value+1
        backend.step = step
        result = backend.from_noise(torch.ones((1,4,2,2)),3)
        self.assertEqual(observed, [2,1,0])
        self.assertTrue(torch.equal(result, torch.full_like(result, 5.)))


if __name__ == '__main__':
    unittest.main()
