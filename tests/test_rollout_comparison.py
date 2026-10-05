import copy
import unittest
from dura.compare_rollouts import summarize_conditions


def row(task, success, digest='same'):
    return {'suite':'libero_spatial', 'task_id':task, 'trial':20,
            'success':success, 'initial_image_sha256':digest, 'actions':[[0.],[1.]]}


class ComparisonTests(unittest.TestCase):
    def test_conditional_rate_excludes_baseline_failures(self):
        conditions={'clean':[row(0,True),row(1,False),row(2,True)],
                    'seed':[row(0,True),row(1,False),row(2,False)],
                    'adversarial':[row(0,False),row(1,False),row(2,True)]}
        report=summarize_conditions(conditions,[0.],[0.])
        self.assertEqual(report['conditions']['adversarial']['asr'],2/3)
        self.assertEqual(report['conditional_on_clean_success']['rate'],.5)
        self.assertEqual(report['conditional_on_both_baselines_succeeding']['rate'],1.)

    def test_mismatched_initial_states_rejected(self):
        conditions={name:[row(0,True)] for name in ('clean','seed','adversarial')}
        conditions['adversarial'][0]['initial_image_sha256']='different'
        with self.assertRaises(ValueError):
            summarize_conditions(conditions,[0.],[0.])

    def test_missing_pairs_rejected(self):
        conditions={name:[row(0,True)] for name in ('clean','seed','adversarial')}
        conditions['seed']=[]
        with self.assertRaises(ValueError):
            summarize_conditions(conditions,[0.],[0.])

    def test_no_clean_success_has_undefined_conditional_rate(self):
        conditions={name:[row(0,False)] for name in ('clean','seed','adversarial')}
        result=summarize_conditions(conditions,[0.],[0.])
        self.assertIsNone(result['conditional_on_clean_success']['rate'])
