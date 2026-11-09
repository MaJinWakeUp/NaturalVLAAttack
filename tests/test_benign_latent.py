"""Local latent identity, bounded updates, resumability and new state-bank checks."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np
import torch

from dura import train_object_texture as training
from dura.benign_latent import LocalDecoder,project,latent_delta,antithetic_gradient,optimize,validation_metrics,rank
from dura.config import PROJECT_ROOT
from dura.initial_states import prepare_state_bank,load_state_bank
from dura.object_texture import load_config,validate_config,digest,NativeTextureWorld


class FakeBackend:
    device=torch.device('cpu')
    def decode(self,latent):return latent[:,:3].sigmoid()


class FakeObjective:
    def __init__(self):
        self.mapping=SimpleNamespace(compose=lambda panel:panel)
        self.observation_queries=self.candidate_queries=0
    def atlas_score(self,atlas,records):
        self.observation_queries+=len(records)
        return torch.tensor(-float(atlas.mean())/255)


class BenignLatentTests(unittest.TestCase):
    def config(self):return load_config(PROJECT_ROOT/'configs/object_texture_benign_latent.yaml')

    def test_fresh_run_collects_observations_before_training_and_reuses_completed_collection(self):
        from dura.benign_latent import train
        cfg=self.config();policy=object()
        with TemporaryDirectory() as directory:
            root=Path(directory);cfg['output']=str(root)
            marker=root/'collection_complete.json'
            def collect(settings,loaded_policy):
                self.assertIs(loaded_policy,policy)
                marker.write_text('{}')
            def read(settings,split):
                self.assertTrue(marker.exists(),'Training attempted to read an absent collection')
                raise RuntimeError('Stop before loading simulator/model training resources')
            with patch('dura.benign_latent.prepare',return_value=root), \
                 patch('dura.adapters.load_policy',return_value=policy), \
                 patch('dura.benign_latent.load_mapping',return_value=None), \
                 patch('dura.benign_latent.collect_states',side_effect=collect) as collector, \
                 patch('dura.benign_latent.load_records',side_effect=read):
                for _ in range(2):
                    with self.assertRaisesRegex(RuntimeError,'Stop before loading'):
                        train(cfg)
                collector.assert_called_once_with(cfg,policy)

    def decoder(self):
        latent=torch.zeros((1,4,16,16))
        reference=np.full((16,16,3),128,dtype=np.uint8)
        return LocalDecoder(FakeBackend(),latent,reference)

    def test_zero_delta_exactly_preserves_quantized_benign_image(self):
        decoder=self.decoder()
        panel,delta=decoder.decode(torch.zeros((1,4,4,4)),.1)
        self.assertTrue(torch.equal(panel,decoder.reference))
        self.assertTrue(np.all(training.uint8_panel(panel[0])==128))
        self.assertEqual(float(delta.abs().max()),0.)

    def test_projected_latent_change_respects_interpolated_rms_bound(self):
        parameters=torch.ones((1,4,4,4))*5
        project(parameters,.05,(16,16))
        self.assertLessEqual(float(latent_delta(parameters,(16,16)).square().mean().sqrt()),.050001)
        decoder=self.decoder();_,delta=decoder.decode(torch.ones((1,4,4,4))*10,.15)
        self.assertLessEqual(float(delta.square().mean().sqrt()),.150001)

    def test_antithetic_gradient_direction_matches_known_quadratic(self):
        theta=torch.tensor([[[[.5]]]])
        gradient,_=antithetic_gradient(theta,.025,2048,lambda x:x.square().sum(),
                                      torch.Generator().manual_seed(5))
        self.assertAlmostEqual(float(gradient),1.,delta=.1)
        with self.assertRaises(FloatingPointError):
            antithetic_gradient(theta,.025,1,lambda x:float('nan'),torch.Generator())

    def test_optimizer_resume_preserves_adam_rng_queries_and_pending_validation(self):
        cfg=self.config();cfg['latent_optimizer'].update(updates=3,pairs=2,validate_every=2,grid_size=4)
        records=[{'state_index':i} for i in range(4)]
        decoder=self.decoder();training._stop_requested=False
        with TemporaryDirectory() as directory:
            root=Path(directory);expected_callbacks=[]
            optimize(decoder,FakeObjective(),records,cfg,0,.05,root/'full',
                lambda n,p,q:expected_callbacks.append(n))
            expected=torch.load(root/'full/checkpoint.pt',map_location='cpu')
            for interrupted_at in (1,2):
                folder=root/f'resume_{interrupted_at}';callbacks=[]
                original=training.save_checkpoint
                def interrupt(path,state):
                    original(path,state)
                    if state['completed']==interrupted_at:raise InterruptedError('Simulate interruption')
                with patch('dura.train_object_texture.save_checkpoint',side_effect=interrupt):
                    with self.assertRaises(InterruptedError):
                        optimize(decoder,FakeObjective(),records,cfg,0,.05,folder,lambda n,p,q:callbacks.append(n))
                objective=FakeObjective()
                optimize(decoder,objective,records,cfg,0,.05,folder,lambda n,p,q:callbacks.append(n))
                actual=torch.load(folder/'checkpoint.pt',map_location='cpu')
                self.assertTrue(torch.equal(actual['parameters'],expected['parameters']))
                self.assertTrue(torch.equal(actual['generator'],expected['generator']))
                self.assertEqual(actual['history'],expected['history'])
                self.assertEqual(objective.observation_queries,48)
                self.assertEqual(callbacks,expected_callbacks)

    def test_appearance_bound_is_relative_to_benign_not_original_label(self):
        from test_full_surface import FullSurfaceTests
        mapping=FullSurfaceTests().mapping();cfg=self.config()
        benign=np.zeros_like(mapping.original)
        report=mapping.texture_report(benign,benign,cfg['appearance'])
        self.assertGreater(report['rmse_original'],cfg['appearance']['rmse_max'])
        self.assertEqual(report['rmse_benign'],0.)
        self.assertTrue(report['accepted'])

    def test_preexisting_benign_failures_do_not_count_as_optimizer_successes(self):
        def rows(flags):
            return [{'trial':i,'success':s,'steps':50,'canonical_state_sha256':f's{i}',
                     'canonical_image_sha256':f'i{i}'} for i,s in enumerate(flags)]
        original=rows([True,True,True]);benign=rows([False,True,True])
        unchanged=validation_metrics(original,benign,rows([False,True,True]))
        changed=validation_metrics(original,benign,rows([False,False,True]))
        self.assertEqual(unchanged['induced_failures'],1)
        self.assertEqual(unchanged['optimization_induced_failures'],0)
        self.assertEqual(changed['optimization_conditional_asr'],.5)
        a={'id':'a','validation':unchanged,'appearance':{'rmse_benign':0.}}
        b={'id':'b','validation':changed,'appearance':{'rmse_benign':.01}}
        self.assertLess(rank(b),rank(a))

    def test_settings_reject_nonfinite_queries_and_inconsistent_budgets(self):
        cfg=self.config();validate_config(cfg)
        self.assertEqual(len(cfg['splits']['test']),20)
        self.assertGreaterEqual(min(cfg['splits']['test']),1000)
        self.assertEqual(2*len(cfg['latent_optimizer']['radii'])*60*16*4,15360)
        for key,value in [('sigma',float('nan')),('pairs',7),('learning_rate',0.)]:
            bad=copy.deepcopy(cfg);bad['latent_optimizer'][key]=value
            with self.assertRaises(ValueError):validate_config(bad)

    def test_token_objective_changes_without_decoded_action_changes(self):
        class World:
            instruction='task'
            def set_texture(self,atlas):self.atlas=atlas
            def restore_state(self,state):pass
            def image(self):return np.full((8,8,3),self.atlas,dtype=np.uint8)
        def logits(images,instructions,metadata,target,weights):
            confidence=images.mean().expand(1,7)
            return torch.stack((confidence,torch.zeros_like(confidence)),-1),torch.zeros((1,7),dtype=torch.long),weights[None]
        policy=SimpleNamespace(target_logits=logits)
        objective=training.NativeObjective(World(),None,policy,self.config()['objective'])
        records=[{'state':None,'reference_action':[0.]*7}]
        original=objective.atlas_score(120,records)
        altered=objective.atlas_score(100,records)
        self.assertLess(float(altered),float(original))
        self.assertEqual(objective.observation_queries,2)

    def test_state_bank_prepares_once_and_rejects_tampering(self):
        class FakeWorld:
            initial_states=np.zeros((2,3))
            def __init__(self,cfg,render=False):
                self.robots=[];self.sim=SimpleNamespace(forward=lambda:None)
                self.env=SimpleNamespace(seed=lambda seed:setattr(self,'seed',seed),reset=lambda:None,
                    robots=[],check_success=lambda:False,
                    get_sim_state=lambda:np.random.default_rng(self.seed).normal(size=3))
            def close(self):pass
        with TemporaryDirectory() as directory:
            cfg=self.config();cfg['initial_state_bank'].update(path=str(Path(directory)/'states.npz'),
                ids=[1000,1001],sha256=None)
            with patch('dura.initial_states.NativeTextureWorld',FakeWorld) as factory:
                prepare_state_bank(cfg)
                sha=cfg['initial_state_bank']['sha256'];prepare_state_bank(cfg)
            bank=load_state_bank(cfg['initial_state_bank'],(3,))
            self.assertEqual(set(bank),{1000,1001})
            self.assertFalse(np.array_equal(bank[1000],bank[1001]))
            self.assertEqual(sha,digest(cfg['initial_state_bank']['path']))
            with self.assertRaises(ValueError):load_state_bank(cfg['initial_state_bank'],(4,))
            Path(cfg['initial_state_bank']['path']).write_bytes(b'changed')
            with self.assertRaises(ValueError):prepare_state_bank(cfg)

    def test_native_reset_uses_frozen_new_state_instead_of_builtin_index(self):
        world=NativeTextureWorld.__new__(NativeTextureWorld)
        world.cfg={'collection':{'seed':0,'settle_steps':0}}
        world.initial_states=np.zeros((2,3));world.extra_initial_states={1000:np.array([1.,2.,3.])}
        captured=[]
        world.original=lambda:None;world.state=lambda:world.extra_initial_states[1000]
        world.env=SimpleNamespace(seed=lambda seed:None,reset=lambda:None,robots=[],
                                  set_init_state=lambda state:captured.append(state.copy()))
        world.reset(1000)
        self.assertTrue(np.array_equal(captured[0],[1.,2.,3.]))


if __name__=='__main__':unittest.main()
