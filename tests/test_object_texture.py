"""Appearance budgets, simulator matching, review binding, and exact resume checks."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np
import torch

from dura.config import AttackConfig, PROJECT_ROOT
from dura.object_texture import (AtlasMapping, clip_rectangle, polygon_area, surface_fraction,
    fingerprint, load_config, validate_config, write_json, digest, reset_gripper_commands, NativeTextureWorld)
from dura.evaluate_object_texture import summarize, wilson, record_review, select
from dura.train_object_texture import trajectory, save_checkpoint


class FakeBackend:
    device = torch.device('cpu')
    def timesteps(self, config): return torch.tensor([2, 1])
    def encode(self, image): return image.clone()
    def add_noise(self, image, noise, timestep): return image+.1*noise
    def step(self, latent, timestep): return .9*latent
    def query_alpha(self, timestep, convention): return .9
    def decode(self, latent): return latent.sigmoid()


class FakeObjective:
    def __init__(self): self.candidate_queries = self.observation_queries = 0
    def __call__(self, panels, records):
        self.candidate_queries += len(panels)
        self.observation_queries += len(panels)*len(records)
        return panels.flatten(1).square().mean(1)


class ObjectTextureTests(unittest.TestCase):
    def test_native_poses_match_snapshot_without_changing_solver_state(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest('Optional MuJoCo dependency is unavailable')
        model=mujoco.MjModel.from_xml_string('''<mujoco><option timestep="0.01" gravity="0 0 0"/>
          <worldbody><body name="moving" pos="0 0 1"><freejoint/>
          <geom type="box" size="0.1 0.1 0.1" mass="1"/>
          <camera name="attached" pos="0 0 1"/></body></worldbody></mujoco>''')
        data=mujoco.MjData(model)
        data.qvel[0]=1
        mujoco.mj_forward(model,data)
        mujoco.mj_step(model,data)
        self.assertNotEqual(float(data.geom_xpos[0,0]),float(data.qpos[0]))
        world=NativeTextureWorld.__new__(NativeTextureWorld)
        world.model,world.data,world.mujoco=model,data,mujoco
        world.state_spec=mujoco.mjtState.mjSTATE_INTEGRATION
        world.sim=SimpleNamespace(forward=lambda:mujoco.mj_forward(model,data))
        state=world.state().copy()
        world.synchronize()
        self.assertTrue(np.array_equal(state,world.state()))
        self.assertAlmostEqual(float(data.geom_xpos[0,0]),float(data.qpos[0]))
        geometry,camera=data.geom_xpos.copy(),data.cam_xpos.copy()
        data.qpos[0]+=1
        world.sim.forward()
        world.restore_state(state)
        self.assertTrue(np.array_equal(state,world.state()))
        self.assertTrue(np.array_equal(geometry,data.geom_xpos))
        self.assertTrue(np.array_equal(camera,data.cam_xpos))

    def config(self):
        return load_config(PROJECT_ROOT/'configs/object_texture_alphabet_soup.yaml')

    def mapping(self):
        vertices = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],dtype=float)
        faces = np.array([[0,1,2],[0,2,3]])
        settings = {'panel':[0,0,64,64],'editable':[15,15,45,45],
                    'feather':4,'working_size':32,'max_surface_fraction':.3}
        original = np.full((64,64,3),120,dtype=np.uint8)
        return AtlasMapping(original,vertices,vertices[:,:2],faces,settings)

    def test_uv_clipping_measures_surface_area(self):
        triangle = np.array([[0,0],[1,0],[0,1]])
        self.assertAlmostEqual(polygon_area(clip_rectangle(triangle,[0,0,.5,.5])),.25)
        vertices = np.column_stack((triangle,np.zeros(3)))
        self.assertAlmostEqual(surface_fraction(vertices,triangle,np.array([[0,1,2]]),[0,0,.5,.5]),.5)

    def test_composition_preserves_protected_texels_and_boundary(self):
        mapping = self.mapping()
        result = mapping.compose(np.ones((32,32,3),dtype=float))
        self.assertTrue(np.array_equal(result[~mapping.mask],mapping.original[~mapping.mask]))
        self.assertTrue(np.array_equal(result[15,15:45],mapping.original[15,15:45]))
        self.assertTrue(np.all(result[25:35,25:35]==255))
        with self.assertRaises(ValueError):
            mapping.compose(np.full((32,32,3),np.nan))

    def test_surface_limit_is_enforced_even_with_small_atlas_pixel_fraction(self):
        mapping = self.mapping()
        vertices = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],dtype=float)
        settings = {**mapping.settings,'max_surface_fraction':.01}
        with self.assertRaises(ValueError):
            AtlasMapping(mapping.original,vertices,vertices[:,:2],np.array([[0,1,2],[0,2,3]]),settings)

    def test_native_png_row_order_controls_uv_coverage(self):
        vertices=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype=float)
        uv=np.array([[.05,.05],[.4,.05],[.05,.4]])
        settings={'panel':[0,0,64,64],'editable':[5,5,16,16],
                  'feather':2,'working_size':32,'max_surface_fraction':1,'uv_origin':'top'}
        mapping=AtlasMapping(np.zeros((64,64,3),dtype=np.uint8),vertices,uv,np.array([[0,1,2]]),settings)
        self.assertGreater(mapping.surface_fraction,0)
        with self.assertRaises(ValueError):
            AtlasMapping(mapping.original,vertices,uv,np.array([[0,1,2]]),{**settings,'uv_origin':'bottom'})

    def test_appearance_failure_cannot_pass_on_good_similarity_alone(self):
        mapping = self.mapping()
        result = mapping.compose(np.ones((32,32,3),dtype=float))
        cfg = self.config()
        report = mapping.texture_report(result,mapping.original,cfg['appearance'])
        self.assertFalse(report['accepted'])
        self.assertTrue(report['protected_pixels_exact'])

    def test_state_splits_and_query_budget_are_consistent(self):
        cfg = self.config()
        self.assertEqual(len(cfg['splits']['test']),20)
        self.assertEqual(len(cfg['search']['seeds'])*len(cfg['search']['step_sizes'])*
            AttackConfig(**cfg['attack']).updates*cfg['search']['samples']*cfg['search']['batch_size'],76800)
        bad = copy.deepcopy(cfg); bad['splits']['test'][0]=0
        with self.assertRaises(ValueError): validate_config(bad)
        bad = copy.deepcopy(cfg); bad['search']['samples']=2
        with self.assertRaises(ValueError): validate_config(bad)

    def test_frozen_json_preserves_scientific_notation_and_config_hash(self):
        cfg=self.config()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'config.json'
            write_json(path,cfg)
            self.assertIn('1e-08',path.read_text())
            loaded=load_config(path)
            self.assertIsInstance(loaded['acceptance']['physics_atol'],float)
            self.assertEqual(fingerprint(loaded),fingerprint(cfg))

    def test_retyped_physics_tolerance_is_rejected_before_gpu_execution(self):
        cfg=self.config(); cfg['acceptance']['physics_atol']='1e-08'
        with self.assertRaises(ValueError): validate_config(cfg)

    def test_failed_runtime_stage_writes_machine_readable_status(self):
        from dura.object_texture_pilot import main
        with tempfile.TemporaryDirectory() as folder:
            cfg=self.config(); cfg['output']=folder
            with patch('sys.argv',['pilot','--config','unused','native-check']), \
                 patch('dura.object_texture_pilot.load_config',return_value=cfg), \
                 patch('dura.evaluate_object_texture.native_preflight',side_effect=RuntimeError('test failure')):
                with self.assertRaises(RuntimeError): main()
            status=json.loads((Path(folder)/'status.json').read_text())
            self.assertEqual(status['status'],'failed')
            self.assertEqual(status['error'],'test failure')
            self.assertEqual(status['stage'],'native-check')

    def test_soft_reset_clears_previous_gripper_command(self):
        gripper=SimpleNamespace(dof=1,current_action=np.array([-1.,1.]))
        robot=SimpleNamespace(has_gripper=True,gripper=gripper)
        reset_gripper_commands([robot])
        self.assertTrue(np.array_equal(gripper.current_action,np.zeros(1)))

    def rows(self):
        rows = {}
        for name,successes in [('original',[True,True,False]),('benign',[True,True,True]),('adversarial',[False,True,False])]:
            rows[name] = [{'suite':'libero_object','task_id':0,'trial':30+i,'success':value,
                          'canonical_image_sha256':f'image{i}','canonical_state_sha256':f'state{i}'}
                         for i,value in enumerate(successes)]
        return rows

    def test_conditional_asr_has_matched_baseline_denominator(self):
        summary = summarize(self.rows(),self.config()['acceptance'])
        self.assertEqual(summary['conditional_episodes'],2)
        self.assertEqual(summary['conditional_failures'],1)
        self.assertEqual(summary['conditional_asr'],.5)
        self.assertAlmostEqual(summary['failure_rates']['adversarial'],2/3)
        self.assertFalse(summary['meets_effect_criteria'])
        self.assertIsNone(wilson(0,0))

    def test_matching_checks_state_and_render_before_outcome(self):
        for field in ('canonical_image_sha256','canonical_state_sha256'):
            rows = self.rows(); rows['benign'][0][field]='different'
            with self.assertRaises(ValueError): summarize(rows,self.config()['acceptance'])
        rows = self.rows(); rows['adversarial'].pop()
        with self.assertRaises(ValueError): summarize(rows,self.config()['acceptance'])

    def test_zero_eligible_controls_is_undefined_asr(self):
        rows=self.rows()
        for row in rows['original']: row['success']=False
        result=summarize(rows,self.config()['acceptance'])
        self.assertIsNone(result['conditional_asr'])
        self.assertIsNone(result['conditional_95pct_wilson'])

    def test_resumed_trajectory_matches_uninterrupted_rng_and_latent(self):
        cfg = self.config()
        attack = AttackConfig(mode='blackbox',ddim_steps=4,strength=.5,samples=2,
                              baseline='leave_one_out',candidate_batch_size=1,seed=3)
        records=[{'state_index':i} for i in range(4)]
        panel=torch.full((1,3,8,8),.5)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            expected=trajectory(FakeBackend(),FakeObjective(),panel,records,cfg,attack,root/'full')
            def interrupt_after_first(path,state):
                save_checkpoint(path,state)
                if state['completed']==1: raise InterruptedError('simulate interruption')
            with patch('dura.train_object_texture.save_checkpoint',side_effect=interrupt_after_first):
                with self.assertRaises(InterruptedError):
                    trajectory(FakeBackend(),FakeObjective(),panel,records,cfg,attack,root/'resumed')
            objective=FakeObjective()
            actual=trajectory(FakeBackend(),objective,panel,records,cfg,attack,root/'resumed')
            self.assertTrue(np.array_equal(expected,actual))
            self.assertEqual(objective.candidate_queries,4)
            self.assertEqual(objective.observation_queries,16)

    def review_fixture(self,root):
        cfg=self.config(); cfg['output']=str(root)
        paths=['review/review_00.png','assets/editable_mask.png','candidates/c0/texture.png','benign/texture.png']
        for name in paths:
            path=root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(name.encode())
        item={'id':'c0','sheet':paths[0],'sheet_sha256':digest(root/paths[0]),
              'mask_sha256':digest(root/paths[1]),'texture_sha256':digest(root/paths[2]),
              'benign_sha256':digest(root/paths[3])}
        candidate={**item,'texture':paths[2],'benign_texture':paths[3],
                   'validation_action_mse':.01,'appearance':{'rmse_original':.01}}
        write_json(root/'review_queue.json',[item]); write_json(root/'candidates.json',[candidate])
        write_json(root/'training_complete.json',{'config_sha256':fingerprint(cfg)})
        return cfg,item

    def test_selection_requires_blinded_hash_bound_visual_review(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); cfg,item=self.review_fixture(root)
            with self.assertRaises(ValueError): select(cfg)
            record_review(cfg,'c0',True,'test reviewer','Plausible label; protected text intact')
            (root/'candidates/c0/texture.png').write_bytes(b'changed')
            with self.assertRaises(ValueError): select(cfg)

    def test_review_after_test_access_does_not_mutate_decisions(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); cfg,item=self.review_fixture(root)
            write_json(root/'test_started.json',{})
            with self.assertRaises(ValueError):
                record_review(cfg,'c0',True,'reviewer','too late')
            self.assertFalse((root/'review/decisions/c0.json').exists())


if __name__=='__main__': unittest.main()
