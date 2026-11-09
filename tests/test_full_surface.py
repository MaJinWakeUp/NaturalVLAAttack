"""Scientific and runtime regressions for unconstrained object-texture attacks."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np
import torch
from PIL import Image

from dura.config import AttackConfig, PROJECT_ROOT
from dura.object_texture import (AtlasMapping, audit_test_states, digest, fingerprint,
                                 array_digest, load_config, validate_config, write_json)
from dura.train_object_texture import NativeObjective
from dura.full_surface_attack import candidate_rank, preset_panel, select_full, validation_result
from dura.evaluate_object_texture import summarize


class FullSurfaceTests(unittest.TestCase):
    def config(self):
        return load_config(PROJECT_ROOT/'configs/object_texture_full_surface.yaml')

    def mapping(self, degenerate=False):
        vertices=np.array([[0.,0,0],[1,0,0],[0,1,0]])
        uv=np.zeros((3,2)) if degenerate else vertices[:,:2]
        settings={'mode':'full','panel':[0,0,64,64],'editable':[0,0,64,64],
                  'feather':0,'working_size':32,'max_surface_fraction':1.0,'uv_origin':'top'}
        return AtlasMapping(np.full((64,64,3),120,dtype=np.uint8),vertices,uv,np.array([[0,1,2]]),settings)

    def test_full_mapping_changes_every_texel_including_boundary_and_unused_atlas(self):
        mapping=self.mapping()
        result=mapping.compose(np.zeros((32,32,3),dtype=np.uint8))
        self.assertTrue(np.all(result==0))
        self.assertTrue(mapping.mask.all())
        self.assertTrue(np.all(mapping.alpha==1))
        self.assertEqual(mapping.surface_fraction,1.0)
        self.assertEqual(self.mapping(degenerate=True).surface_fraction,1.0)

    def test_full_mapping_rejects_partial_atlas(self):
        mapping=self.mapping()
        settings={**mapping.settings,'panel':[0,0,63,64],'editable':[0,0,63,64]}
        vertices=np.array([[0.,0,0],[1,0,0],[0,1,0]])
        with self.assertRaises(ValueError):
            AtlasMapping(mapping.original,vertices,vertices[:,:2],np.array([[0,1,2]]),settings)

    def test_disabled_appearance_bounds_allow_large_distortion(self):
        mapping=self.mapping();cfg=self.config()
        report=mapping.texture_report(np.zeros_like(mapping.original),mapping.original,
            {**cfg['appearance'],'rmse_max':0.,'seam_absolute_slack':0.})
        self.assertGreater(report['rmse_original'],.4)
        self.assertTrue(report['accepted'])
        self.assertFalse(report['constraints_enabled'])

    def test_zero_anchor_is_explicit_and_old_profile_remains_constrained(self):
        cfg=self.config();self.assertEqual(cfg['attack']['anchor_weight'],0.)
        validate_config(cfg)
        bad=copy.deepcopy(cfg);bad.pop('profile')
        with self.assertRaises(ValueError):validate_config(bad)
        bad=copy.deepcopy(cfg);bad['appearance']['enabled']=True
        with self.assertRaises(ValueError):validate_config(bad)
        bad=copy.deepcopy(cfg);bad['search']['gradient_rms_max']=float('nan')
        with self.assertRaises(ValueError):validate_config(bad)

    def test_budget_and_fresh_test_split_are_frozen(self):
        cfg=self.config()
        self.assertEqual(AttackConfig(**cfg['attack']).updates,80)
        self.assertEqual(len(cfg['search']['seeds'])*len(cfg['search']['step_sizes'])*80*32*4,40960)
        self.assertEqual(len(cfg['splits']['test']),9)
        self.assertFalse(set(cfg['splits']['test']) & (set(range(30,50))|{22}))

    def test_audit_detects_previous_native_report_and_ignores_unopened_states(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);old=root/'old';old.mkdir()
            write_json(old/'config.json',{'suite':'libero_object','task_id':0})
            write_json(old/'asr_report.json',{'paired_successes':[{'state':30,'original':True}]})
            cfg=self.config();cfg['output']=str(root/'new')
            self.assertFalse(audit_test_states(cfg)['conflicts'])
            cfg['splits']['test']=[30]
            with self.assertRaises(ValueError):audit_test_states(cfg)

    def test_divergence_minimization_rewards_changed_actions(self):
        settings=self.config()['objective']
        world=SimpleNamespace(set_texture=lambda atlas:None,restore_state=lambda state:None,
                              image=lambda:np.zeros((8,8,3),dtype=np.uint8),instruction='task')
        policy=SimpleNamespace(predict_actions=lambda images,instructions,contexts:torch.ones((len(images),7)))
        objective=NativeObjective(world,None,policy,settings)
        changed=objective.atlas_score(None,[{'state':None,'reference_action':[0.]*7}])
        same=objective.atlas_score(None,[{'state':None,'reference_action':[1.]*7}])
        self.assertLess(float(changed),float(same))
        self.assertEqual(float(same),0.)

    def rows(self):
        return {name:[{'suite':'libero_object','task_id':0,'trial':i,'success':value,
                       'canonical_image_sha256':f'i{i}','canonical_state_sha256':f's{i}','steps':50}
                      for i,value in enumerate(successes)]
                for name,successes in [('original',[True,True,False]),
                                      ('benign',[False,True,True]),('adversarial',[False,True,False])]}

    def test_primary_original_asr_does_not_hide_benign_failures(self):
        result=summarize(self.rows(),self.config()['acceptance'],primary='original')
        self.assertEqual(result['conditional_asr'],.5)
        self.assertEqual(result['conditional_episodes'],2)
        self.assertEqual(result['both_controls_conditioned_asr'],0.)
        self.assertEqual(result['original_conditioned_asr'],.5)

    def test_validation_outcomes_take_priority_over_surrogate(self):
        stronger={'id':'strong','validation':{'induced_failures':3,'mean_steps':150},'validation_objective':-.01}
        weaker={'id':'weak','validation':{'induced_failures':1,'mean_steps':280},'validation_objective':-10.}
        self.assertLess(candidate_rank(stronger),candidate_rank(weaker))
        rows=self.rows();result=validation_result(rows['original'],rows['adversarial'])
        self.assertEqual(result['induced_failures'],1)
        self.assertEqual(result['conditional_asr'],.5)
        rows['adversarial'][0]['canonical_state_sha256']='different'
        with self.assertRaises(ValueError):validation_result(rows['original'],rows['adversarial'])

    def test_selection_needs_no_naturalness_review_and_is_frozen_before_test(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);cfg=self.config();cfg['output']=str(root)
            candidate={'id':'c0','status':'complete','validation':{'original_successes':5,
                       'induced_failures':3,'mean_steps':220},'validation_objective':-.1,
                       'texture':'texture.png','benign_texture':'benign.png'}
            for name in ('texture.png','benign.png'):(root/name).write_bytes(name.encode())
            candidate.update(texture_sha256=digest(root/'texture.png'),benign_sha256=digest(root/'benign.png'))
            write_json(root/'candidates.json',[candidate])
            write_json(root/'training_complete.json',{'config_sha256':fingerprint(cfg)})
            selected=select_full(cfg)
            self.assertEqual(selected['id'],'c0')
            self.assertEqual(selected['review_decisions'],{})
            self.assertFalse((root/'test_started.json').exists())
            self.assertEqual(select_full(cfg),selected)
            (root/'selection.json').unlink();write_json(root/'test_started.json',{})
            with self.assertRaises(ValueError):select_full(cfg)

    def test_nonoptimized_controls_cover_entire_output_and_are_repeatable(self):
        self.assertTrue(np.all(preset_panel('black',32)==0))
        self.assertEqual(set(np.unique(preset_panel('checker',32))),{0,255})
        self.assertTrue(np.array_equal(preset_panel('noise',32),preset_panel('noise',32)))

    def test_full_run_automatically_reaches_held_out_evaluation(self):
        from dura.full_surface_attack import run
        cfg=self.config()
        calls=[]
        with patch('dura.evaluate_object_texture.native_preflight',side_effect=lambda cfg:calls.append('preflight')), \
             patch('dura.full_surface_attack.train_full',side_effect=lambda cfg:calls.append('train')), \
             patch('dura.full_surface_attack.select_full',side_effect=lambda cfg:(calls.append('select') or {'id':'c','validation':{}})), \
             patch('dura.evaluate_object_texture.evaluate',side_effect=lambda cfg:calls.append('evaluate')), \
             patch('dura.full_surface_attack.signal.signal'), \
             patch('dura.full_surface_attack.torch.cuda.is_available',return_value=False):
            run(cfg)
        self.assertEqual(calls,['preflight','train','select','evaluate'])

    def test_real_training_orchestration_with_fake_policy_and_diffusion(self):
        from dura.full_surface_attack import train_full
        from dura import train_object_texture
        mapping=self.mapping()
        class Backend:
            device=torch.device('cpu')
            def __init__(self,**kwargs):pass
            def timesteps(self,attack):return torch.tensor([2,1])
            def encode(self,panel):return panel.clone()
            def add_noise(self,panel,noise,timestep):return panel+.1*noise
            def step(self,latent,timestep):return .9*latent
            def decode(self,latent):return latent.sigmoid()
            def query_alpha(self,timestep,convention):return .9
        class World:
            instruction='task'
            def __init__(self,cfg):self.cfg=cfg;self.current=mapping.original.copy()
            def set_texture(self,atlas):self.current=atlas.copy()
            def original(self):self.current=mapping.original.copy()
            def restore_state(self,state):pass
            def image(self):return np.array(Image.fromarray(self.current).resize((8,8)))
            def close(self):pass
        policy=SimpleNamespace(predict_actions=lambda images,instructions,contexts:
            images.mean(dim=(1,2,3))[:,None].expand(-1,7))
        with TemporaryDirectory() as directory:
            root=Path(directory);cfg=self.config();cfg['output']=str(root)
            cfg['full_surface']['presets']=['black']
            cfg['search'].update(seeds=[0],step_sizes=[2.],samples=2)
            cfg['attack'].update(ddim_steps=4,strength=.5,samples=2)
            (root/'assets').mkdir()
            Image.fromarray(mapping.original).save(root/'assets/original.png')
            Image.fromarray(mapping.seed_panel()).save(root/'assets/seed_panel.png')
            write_json(root/'collection_complete.json',{})
            records={}
            for split in ('train','calibration','validation'):
                records[split]=[]
                for index in cfg['splits'][split]:
                    image=np.full((8,8,3),120,dtype=np.uint8)
                    records[split].append({'state_index':index,'snapshot':f'{split}/{index}',
                        'state':np.zeros(1),'image':image,'canonical_sha256':array_digest(image)})
            for index in cfg['splits']['validation']:
                write_json(root/'states/validation'/f'episode_{index:03d}'/'report.json',
                    {'trial':index,'success':True,'steps':50,'canonical_image_sha256':f'i{index}',
                     'canonical_state_sha256':f's{index}'})
            def rollout(world,policy,index,atlas,expected_canonical):
                success=bool(atlas.mean()>1)
                return {'trial':index,'success':success,'steps':50 if success else 280,
                        'canonical_image_sha256':expected_canonical[0],
                        'canonical_state_sha256':expected_canonical[1]},[]
            def preview(world,atlas,benign,state,path):Image.fromarray(atlas).save(path)
            with patch('dura.collect_object_states.prepare',return_value=root), \
                 patch('dura.adapters.load_policy',return_value=policy), \
                 patch('dura.full_surface_attack.load_mapping',return_value=mapping), \
                 patch('dura.full_surface_attack.load_records',side_effect=lambda cfg,split:records[split]), \
                 patch('dura.full_surface_attack.NativeTextureWorld',World), \
                 patch('dura.full_surface_attack.StableDiffusionBackend',Backend), \
                 patch('dura.full_surface_attack.run_episode',side_effect=rollout), \
                 patch('dura.full_surface_attack.training.appearance',return_value={'accepted':True,'constraints_enabled':False}), \
                 patch('dura.full_surface_attack.training.contact_sheet',side_effect=preview):
                save_checkpoint=train_object_texture.save_checkpoint
                def interrupt(path,state):
                    save_checkpoint(path,state)
                    if state['completed']==1:raise InterruptedError('Simulate interrupted GPU allocation')
                with patch('dura.train_object_texture.save_checkpoint',side_effect=interrupt):
                    with self.assertRaises(InterruptedError):train_full(cfg)
                train_full(cfg)
                train_full(cfg)  # Completed searches are reused, without further rollouts.
            complete=json.loads((root/'training_complete.json').read_text())
            self.assertEqual(complete['candidate_count'],2)
            self.assertEqual(complete['optimization_policy_frame_queries'],16)
            selected=select_full(cfg)
            self.assertEqual(selected['id'],'preset_black')
            self.assertEqual(selected['validation']['conditional_asr'],1.)
            self.assertFalse((root/'test_started.json').exists())


if __name__=='__main__':unittest.main()
