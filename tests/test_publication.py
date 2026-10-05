"""Reproduction checks for portable artifacts and experiment preparation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dura.config import PROJECT_ROOT, read_config
from dura.task_experiments import parser, prepare, SUITES


class PublicationTests(unittest.TestCase):
    def test_catalog_preserves_fourteen_exact_patches_and_shared_seed(self):
        root = PROJECT_ROOT / 'final_patches/libero_task_suite_relaxed_v1'
        summary = json.loads((root/'summary.json').read_text())
        self.assertEqual(len(summary['results']), 14)
        for row in summary['results']:
            report_path = root / row['report']
            report = json.loads(report_path.read_text())
            patch_path = root / row['patch']
            self.assertEqual(hashlib.sha256(patch_path.read_bytes()).hexdigest(), row['patch_sha256'])
            self.assertEqual(row['patch_sha256'], report['patch_sha256'])
            cfg = read_config(report_path.parent/'config.json')
            self.assertEqual(hashlib.sha256(Path(cfg['seed_patch']).read_bytes()).hexdigest(), report['seed_sha256'])
            self.assertTrue(report['comparison']['initial_observations_match'])
            training = json.loads((report_path.parent/'report.json').read_text())
            selected = training['candidates'][training['selected_candidate']]
            self.assertTrue(selected['appearance']['gate']['accepted'])
            self.assertFalse((report_path.parent/'patch.pt').exists())

    def test_prepare_all_task_and_suite_configs_with_external_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / 'data'
            for suite in {s['suite'] for s in SUITES.values()}:
                directory = data/suite
                directory.mkdir(parents=True)
                for split in ['train', 'val']:
                    rows = [{'image': f'{split}{i}.png', 'instruction':f'task {i}'} for i in range(10)]
                    (directory/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            args = parser().parse_args(['prepare', '--scratch', str(root/'run'),
                '--final-root', str(root/'final'), '--data-root', str(data), '--libero-root', '/unused'])
            with patch('dura.task_experiments.task_languages', return_value=[f'task {i}' for i in range(10)]):
                prepare(args)
            plan = json.loads((root/'run/plan.json').read_text())
            self.assertEqual(len(plan['entries']), 14)
            self.assertEqual({e['name'] for e in plan['entries'] if e['kind']=='suite'}, set(SUITES))
            cfg = read_config(root/'run/spatial_task_03/config.yaml')
            rows = list(map(json.loads, Path(cfg['manifest']).read_text().splitlines()))
            self.assertEqual([r['instruction'] for r in rows], ['task 3'])
            self.assertTrue(Path(cfg['seed_patch']).is_file())
            self.assertFalse((root/'final').exists())

    def test_saved_patch_plan_does_not_need_data_and_keeps_reports_in_scratch(self):
        published = PROJECT_ROOT / 'final_patches/libero_task_suite_relaxed_v1'
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)/'run'
            args = parser().parse_args(['prepare', '--scratch',str(run),
                '--final-root',str(published), '--reuse-final-patches', '--experiments','spatial_suite',
                '--libero-root','/unused', '--offset','27'])
            with patch('dura.task_experiments.task_languages', side_effect=AssertionError('LIBERO was loaded')):
                prepare(args)
            plan = json.loads((run/'plan.json').read_text())
            self.assertEqual(plan['final_root'],str(run/'results'))
            entry = plan['entries'][0]
            self.assertEqual(entry['initial_state_offset'],27)
            self.assertEqual(entry['report_output'],str(run/'reports/spatial_suite'))
            self.assertEqual(entry['final'],str(published/'spatial_suite'))


if __name__ == '__main__':
    unittest.main()
