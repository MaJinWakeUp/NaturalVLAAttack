"""Summarize matched clean/seed/adversarial rollouts without conflating failures."""
import argparse
import json
from pathlib import Path
from .metrics import rollout_metrics


def key(row):
    return row['suite'], row['task_id'], row['trial']


def summarize_conditions(conditions, target, tolerance, dimensions=None):
    if set(conditions) != {'clean', 'seed', 'adversarial'}:
        raise ValueError('Expected clean, seed, and adversarial conditions')
    indexed = {}
    for name, rows in conditions.items():
        indexed[name] = {key(row):row for row in rows}
        if len(indexed[name]) != len(rows):
            raise ValueError(f'Duplicate rollout key in {name}')
    keys = set(indexed['clean'])
    if not keys or any(set(rows) != keys for rows in indexed.values()):
        raise ValueError('Conditions must contain the same nonempty set of task/initial-state pairs')
    for k in keys:
        checksums = {indexed[name][k]['initial_image_sha256'] for name in indexed}
        if len(checksums) != 1:
            raise ValueError(f'Initial observations differ across conditions: {k}')
    metrics = {name:rollout_metrics(rows, target, tolerance, dimensions) for name,rows in conditions.items()}
    clean_successes = [k for k in keys if indexed['clean'][k]['success']]
    both_baselines = [k for k in clean_successes if indexed['seed'][k]['success']]
    def conditional(reference):
        failures = [k for k in reference if not indexed['adversarial'][k]['success']]
        return {'reference_successes':len(reference), 'adversarial_failures':len(failures),
                'rate':len(failures)/len(reference) if reference else None,
                'failed_pairs':[list(k) for k in sorted(failures)]}
    pairs = [{'suite':k[0], 'task_id':k[1], 'trial':k[2],
              **{name:indexed[name][k]['success'] for name in indexed}} for k in sorted(keys)]
    return {'paired_rollouts_per_condition':len(keys), 'conditions':metrics,
            'adversarial_failure_rate_minus_clean':metrics['adversarial']['asr']-metrics['clean']['asr'],
            'adversarial_failure_rate_minus_seed':metrics['adversarial']['asr']-metrics['seed']['asr'],
            'conditional_on_clean_success':conditional(clean_successes),
            'conditional_on_both_baselines_succeeding':conditional(both_baselines),
            'target':list(target), 'tolerance':tolerance, 'dimensions':dimensions,
            'initial_observations_match':True, 'paired_successes':pairs}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--target',type=float,nargs='+',required=True)
    parser.add_argument('--tolerance',type=float,nargs='+',required=True)
    parser.add_argument('--dimensions',type=int,nargs='+')
    args=parser.parse_args()
    root=Path(args.root)
    conditions={name:[json.loads(line) for line in (root/name/'rollouts.jsonl').read_text().splitlines()]
                for name in ('clean','seed','adversarial')}
    result=summarize_conditions(conditions,args.target,args.tolerance,args.dimensions)
    with (root/'comparison.json').open('x') as output:
        json.dump(result,output,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
