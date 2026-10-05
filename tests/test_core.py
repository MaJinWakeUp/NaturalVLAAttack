"""Small CPU checks of equations/contracts; no models, downloads, or experiments."""

import itertools
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from dura.attack import DURAAttack, score_direction
from dura.config import AttackConfig
from dura.data import ObservationBatch, ManifestDataset, load_image, save_image
from dura.defenses import transform
from dura.metrics import rollout_metrics
from dura.objective import TargetObjective
from dura.quality import boundary_seam_energy, patch_tv, rgb_to_lab
from dura.render import PatchRenderer
from dura.adapters.openvla import OpenVLAAdapter


class TinyPolicy:
    def predict_actions(self, images, instructions, metadata):
        return images.mean((2, 3))

    def target_logits(self, images, instructions, metadata, target, weights):
        scores = images.mean((2, 3))
        logits = torch.stack((scores, -scores), -1)
        targets = torch.zeros_like(scores, dtype=torch.long)
        return logits, targets, weights.to(images)[None].expand_as(scores)


class TinyDiffusion:
    device = torch.device('cpu')

    def timesteps(self, config):
        return torch.arange(config.updates, 0, -1)

    def encode(self, patch):
        return patch.clone()

    def decode(self, latent):
        return latent.sigmoid()

    def add_noise(self, clean, noise, timestep):
        return clean + noise * .1

    def step(self, latent, timestep):
        return latent * .9

    def query_alpha(self, timestep, convention):
        return .8


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_paper_update_budget(self):
        self.assertEqual(AttackConfig().updates, 100)
        with self.assertRaises(ValueError):
            AttackConfig(ddim_steps=1, strength=.5)
        with self.assertRaises(ValueError):
            AttackConfig(samples=1, baseline='leave_one_out')

    def test_renderer_exact_box_and_gradient(self):
        images = torch.zeros(2, 3, 8, 8)
        patch = torch.ones(1, 3, 2, 2, requires_grad=True)
        rendered = PatchRenderer((1, 2, 3, 4))(images, patch, [{}, {}])
        expected = torch.zeros(1, 2, 3, 8, 8)
        expected[:, :, :, 2:6, 1:4] = 1
        torch.testing.assert_close(rendered, expected)
        rendered.sum().backward()
        self.assertAlmostEqual(float(patch.grad.sum()), 72, places=4)

    def test_quad_and_box_agree_and_candidates_are_distinct(self):
        images = torch.rand(1, 3, 10, 10)
        patches = torch.rand(2, 3, 4, 4)
        renderer = PatchRenderer((2, 3, 4, 5))
        a = renderer(images, patches, [{}])
        b = renderer(images, patches, [{'quad': [[2,3], [6,3], [6,8], [2,8]]}])
        torch.testing.assert_close(a, b)
        self.assertFalse(torch.equal(a[0], a[1]))

    def test_mse_is_weighted_sum_then_batch_mean(self):
        batch = ObservationBatch(torch.zeros(2,3,4,4), ['a','b'], [{},{}])
        objective = TargetObjective(TinyPolicy(), PatchRenderer((0,0,4,4)),
                                    target=[0,0,0], weights=[1,2,3], policy_batch_size=1)
        loss = objective(torch.stack((torch.ones(3,2,2), torch.zeros(3,2,2))), batch, 'blackbox')
        torch.testing.assert_close(loss, torch.tensor([6.,0.]))
        self.assertEqual(objective.candidate_queries, 2)
        self.assertEqual(objective.observation_queries, 4)

    def test_whitebox_gradient_reaches_patch(self):
        batch = ObservationBatch(torch.zeros(2,3,4,4), ['a','b'], [{},{}])
        objective = TargetObjective(TinyPolicy(), PatchRenderer((0,0,4,4)),
                                    target=[0,0,0], weights=[1,2,3])
        patch = torch.full((1,3,2,2), .4, requires_grad=True)
        loss = objective(patch, batch, 'whitebox').sum()
        grad, = torch.autograd.grad(loss, patch)
        self.assertTrue((grad < 0).all())

    def test_score_estimator_matches_smoothed_quadratic_gradient(self):
        u = torch.tensor([[[[.4, -.2]]]])
        target = torch.tensor([[[[.1, .3]]]])
        alpha = .7
        expected = 2*alpha*u - 2*alpha**.5*target
        for baseline in ('none', 'leave_one_out'):
            g, _ = score_direction(u, alpha, 40000, 400,
                                   lambda z: (z-target).square().flatten(1).sum(1),
                                   torch.Generator().manual_seed(10), baseline)
            torch.testing.assert_close(g, expected, atol=.035, rtol=.05)

    def test_leave_one_out_removes_constant_loss(self):
        g, loss = score_direction(torch.zeros(1,1,2,2), .8, 8, 2,
                                  lambda z: torch.full((len(z),), 7.),
                                  torch.Generator().manual_seed(1), 'leave_one_out')
        torch.testing.assert_close(g, torch.zeros_like(g), atol=1e-5, rtol=0)
        self.assertEqual(loss, 7.)

    def test_blackbox_query_has_no_autograd(self):
        def loss(z):
            self.assertFalse(torch.is_grad_enabled())
            return z.flatten(1).square().sum(1)
        score_direction(torch.zeros(1,1,2,2), .8, 2, 1, loss,
                        torch.Generator().manual_seed(0))

    def test_attack_both_modes_and_query_counts(self):
        batch = ObservationBatch(torch.zeros(2,3,4,4), ['a','b'], [{},{}])
        seed = torch.zeros(1,3,2,2)
        for mode in ('whitebox', 'blackbox'):
            objective = TargetObjective(TinyPolicy(), PatchRenderer((0,0,4,4)),
                                        [0,0,0], [1,1,1])
            cfg = AttackConfig(mode=mode, ddim_steps=4, strength=.5, samples=4, step_size=.01)
            result = DURAAttack(TinyDiffusion(), objective, cfg).run(seed, itertools.repeat(batch))
            self.assertEqual(len(result.history), 2)
            self.assertFalse(result.patch.requires_grad)
            self.assertTrue(torch.isfinite(result.patch).all())
            self.assertEqual(objective.candidate_queries, 2 if mode == 'whitebox' else 8)
            self.assertEqual(objective.observation_queries, 4 if mode == 'whitebox' else 16)

    def test_zero_step_follows_clean_anchor(self):
        cfg = AttackConfig(ddim_steps=4, strength=.5, step_size=0)
        seed = torch.zeros(1,3,2,2)
        batch = ObservationBatch(torch.zeros(1,3,4,4), ['a'], [{}])
        objective = TargetObjective(TinyPolicy(), PatchRenderer((0,0,4,4)), [0,0,0], [1,1,1])
        result = DURAAttack(TinyDiffusion(), objective, cfg).run(seed, itertools.repeat(batch))
        noise = torch.randn(seed.shape, generator=torch.Generator().manual_seed(0))
        torch.testing.assert_close(result.latent, noise * .1 * .9**2)

    def test_ap_is_macro_average_not_pooled_steps(self):
        records = [{'success': False, 'actions': [[0], [1]]},
                   {'success': True, 'actions': [[0]]*8}]
        result = rollout_metrics(records, [0], 0)
        self.assertEqual(result['asr'], .5)
        self.assertEqual(result['ap'], .75)
        with self.assertRaises(ValueError):
            rollout_metrics([{'success': False, 'actions': []}], [0], 0)

    def test_local_artifacts(self):
        white = torch.ones(1,3,8,8)
        torch.testing.assert_close(patch_tv(white), torch.zeros(1))
        torch.testing.assert_close(boundary_seam_energy(white, (0,4,4,4)), torch.zeros(1))
        lab = rgb_to_lab(white)
        torch.testing.assert_close(lab[...,0], torch.full((1,8,8), 100.), atol=.001, rtol=0)

    def test_defenses_are_bounded_and_reproducible(self):
        images = torch.rand(1,3,8,8)
        for kind, strength in [('jpeg', 30), ('bit_depth', 3), ('gaussian', .1)]:
            a = transform(images, kind, strength, torch.Generator().manual_seed(3))
            b = transform(images, kind, strength, torch.Generator().manual_seed(3))
            torch.testing.assert_close(a, b)
            self.assertTrue(((a >= 0) & (a <= 1)).all())

    def test_manifest_paths_and_rgb_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_image(torch.ones(3,4,4), root/'frame.png')
            (root/'train.jsonl').write_text('{"image":"frame.png","instruction":"pick up cup"}\n')
            data = ManifestDataset(root/'train.jsonl', image_size=4)
            tensor, instruction, row = data[0]
            self.assertEqual(instruction, 'pick up cup')
            self.assertTrue(Path(row['image']).is_absolute())
            torch.testing.assert_close(tensor, torch.ones(3,4,4))

    def test_openvla_target_tokens_match_digitize(self):
        adapter = OpenVLAAdapter.__new__(OpenVLAAdapter)
        adapter.device = torch.device('cpu')
        adapter.action_dim = 3
        adapter.stats = {'q01': [-2,-2,-1], 'q99': [2,2,1], 'mask': [True,True,False]}
        adapter.model = SimpleNamespace(bins=np.linspace(-1,1,256), vocab_size=32000)
        ids = adapter._target_ids(torch.tensor([-2.,0.,1.]))
        expected = 32000 - np.digitize([-1.,0.,1.], adapter.model.bins)
        np.testing.assert_equal(ids.numpy(), expected)

    def test_openvla_causal_logit_alignment(self):
        adapter = OpenVLAAdapter.__new__(OpenVLAAdapter)
        adapter.device = torch.device('cpu')
        adapter.action_dim = 3
        adapter.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
        adapter._target_ids = lambda target: torch.tensor([4,5,6])
        adapter._prompt_ids = lambda text: [1,2] if text == 'short' else [1,2,3,4]
        adapter._pixels = lambda images: images
        def model(**kwargs):
            ids = kwargs['input_ids']
            self.assertEqual(ids[0,:4].tolist(), [1,2,4,5])
            self.assertEqual(ids[1,:6].tolist(), [1,2,3,4,4,5])
            length = ids.shape[1]+7
            return SimpleNamespace(logits=torch.arange(length).float()[None,:,None].expand(2,-1,8))
        adapter.model = model
        logits, labels, weights = adapter.target_logits(torch.zeros(2,3,4,4), ['short','long'],
                                                         [{},{}], torch.zeros(3), torch.ones(3))
        torch.testing.assert_close(logits[:,:,0], torch.tensor([[8.,9.,10.], [10.,11.,12.]]))
        self.assertEqual(labels.tolist(), [[4,5,6], [4,5,6]])


if __name__ == '__main__':
    unittest.main()
