"""Checks of scratch configuration and held-out sample selection."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import yaml

from dura.cli import read_config
from dura.validate_patch import balanced_indices


class WorkflowTests(unittest.TestCase):
    def test_optional_paths_are_resolved_relative_to_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = {'manifest':'data/train.jsonl', 'seed_patch':'data/seed.png', 'output':'runs/test',
                   'final_output':'final/test', 'validation_manifest':'data/val.jsonl',
                   'policy': {'checkpoint':'openvla/model'}, 'diffusion': {'checkpoint':'./weights'}}
            path = root/'config.yaml'
            path.write_text(yaml.safe_dump(cfg))
            actual = read_config(path)
            self.assertEqual(actual['final_output'], str(root/'final/test'))
            self.assertEqual(actual['validation_manifest'], str(root/'data/val.jsonl'))
            self.assertEqual(actual['policy']['checkpoint'], 'openvla/model')
            self.assertEqual(actual['diffusion']['checkpoint'], str(root/'weights'))

    def test_held_out_selection_covers_tasks_before_second_frame(self):
        data = SimpleNamespace(records=[{'instruction':x} for x in ['a','a','a','b','b','c']])
        self.assertEqual(balanced_indices(data, 3), [0,3,5])
        self.assertEqual(balanced_indices(data, 10), [0,3,5,1,4,2])
        with self.assertRaises(ValueError):
            balanced_indices(data, 0)

    def test_run_files_stay_in_scratch(self):
        path = Path(__file__).resolve().parents[1]/'configs/libero_relaxed.yaml'
        with patch.dict('os.environ', {'DURA_SCRATCH':'/tmp/test-dura-scratch',
                                      'HF_HOME':'/tmp/test-dura-hf'}):
            cfg = read_config(path)
        for key in ('manifest','validation_manifest','output'):
            self.assertTrue(cfg[key].startswith('/tmp/test-dura-scratch/'), key)
        for component in ('policy','diffusion'):
            self.assertEqual(cfg[component]['cache_dir'], '/tmp/test-dura-hf/hub')
        self.assertTrue(Path(cfg['seed_patch']).is_file())
        self.assertIn('/final_patches/', cfg['final_output'])
