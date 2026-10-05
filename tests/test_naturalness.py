import unittest
import torch
from dura.naturalness import ssim, appearance_report, check_limits, eligible_best
from dura.render import PatchRenderer


class NaturalnessTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.limits={'patch_rgb_rmse_max':.06,'patch_tv_max':.04,'boundary_increase_max':5.,'local_ssim_drop_max':.01}

    def test_ssim_identity_and_large_distortion(self):
        torch.manual_seed(1)
        a=torch.rand(2,3,32,32)
        torch.testing.assert_close(ssim(a,a),torch.ones(2),atol=1e-5,rtol=0)
        self.assertTrue((ssim(a,1-a)<.1).all())

    def test_benign_seed_passes_but_checkerboard_is_rejected(self):
        image=torch.full((2,3,224,224),.5)
        seed=torch.full((1,3,50,50),.5)
        renderer=PatchRenderer([0,174,50,50])
        ok=appearance_report(image,seed,seed,renderer,[{},{}],[0,174,50,50],self.limits)
        self.assertTrue(ok['gate']['accepted'])
        patch=(torch.arange(50)[:,None]+torch.arange(50)[None,:])%2
        patch=patch.float()[None,None].expand(1,3,-1,-1)
        bad=appearance_report(image,patch,seed,renderer,[{},{}],[0,174,50,50],self.limits)
        self.assertFalse(bad['gate']['accepted'])
        self.assertIn('patch_tv_max',bad['gate']['violations'])
        self.assertIn('patch_rgb_rmse_max',bad['gate']['violations'])

    def test_nonfinite_metrics_fail_closed(self):
        metrics={key:0. for key in self.limits}
        metrics['patch_rgb_rmse_max']=float('nan')
        self.assertFalse(check_limits(metrics,self.limits)['accepted'])
        with self.assertRaises(ValueError): check_limits(metrics,{'unknown':1.})

    def test_better_attack_cannot_override_failed_appearance(self):
        def row(ok,mse): return {'appearance':{'gate':{'accepted':ok}},'action_score':{'target_weighted_mse':mse,'target_ce':mse}}
        natural=row(True,.4)
        noisy=row(False,.01)
        self.assertIs(eligible_best([natural,noisy]),natural)
        self.assertIsNone(eligible_best([noisy]))

    def test_each_attack_update_uses_matching_anchor_and_denoiser(self):
        import itertools
        from types import SimpleNamespace
        from dura.attack import DURAAttack
        from dura.config import AttackConfig
        calls=[]
        case=self
        class Backend:
            device=torch.device('cpu')
            def timesteps(self,config): return torch.tensor([2,1])
            def encode(self,seed): return seed
            def add_noise(self,clean,noise,timestep): return clean
            def decode(self,z): return z
            def step(self,z,timestep):
                case.assertFalse(torch.is_grad_enabled())
                calls.append(z.clone())
                return z*.9
        class Objective:
            candidate_queries=0
            observation_queries=0
            def __call__(self,patch,batch,mode): return patch.flatten(1).sum(-1)
        batch=SimpleNamespace(to=lambda device:None)
        result=DURAAttack(Backend(),Objective(),AttackConfig(ddim_steps=2,strength=1,anchor_weight=.2,step_size=1)).run(torch.zeros(1,1,1,1),itertools.repeat(batch))
        self.assertEqual(len(calls),4)  # two clean anchors and two attack denoising steps
        torch.testing.assert_close(calls[-1],torch.full_like(calls[-1],-.8))
        torch.testing.assert_close(result.latent,torch.full_like(result.latent,-1.72))
