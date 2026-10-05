"""Fixed held-out frame comparisons; these are not rollout ASR/AP results."""

from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from .data import collate_observations
from .objective import TargetObjective


def balanced_indices(dataset, limit):
    if limit < 1:
        raise ValueError('validation_frames must be positive')
    groups = defaultdict(list)
    for index, record in enumerate(dataset.records):
        groups[record['instruction']].append(index)
    selected = []
    for offset in range(max(map(len, groups.values()))):
        for group in groups.values():
            if offset < len(group):
                selected.append(group[offset])
                if len(selected) >= limit:
                    return selected
    return selected


@torch.no_grad()
def compare_patches(policy, renderer, dataset, seed, patch, objective_config, limit, device,
                    evaluate_actions=True):
    target = torch.tensor(objective_config['target'], device=device, dtype=torch.float32)
    weights = torch.tensor(objective_config['weights'], device=device, dtype=torch.float32)
    dimensions = objective_config.get('dimensions')
    if dimensions is not None:
        mask = torch.zeros_like(weights)
        mask[dimensions] = 1
        weights *= mask
    selected = balanced_indices(dataset, limit)
    records = []
    for index in selected:
        batch = collate_observations([dataset[index]]).to(device)
        cases = {'no_patch': batch.images,
                 'seed_patch': renderer(batch.images, seed, batch.metadata)[0],
                 'optimized_patch': renderer(batch.images, patch, batch.metadata)[0]}
        row = {'image': dataset.records[index]['image'], 'instruction': batch.instructions[0]}
        for name, images in cases.items():
            logits, tokens, token_weights = policy.target_logits(
                images, batch.instructions, batch.metadata, target, weights)
            ce = (F.cross_entropy(logits.float().transpose(1, 2), tokens, reduction='none') * token_weights).sum(-1)
            entry = {'target_ce': float(ce.mean())}
            if evaluate_actions:
                actions = policy.predict_actions(images, batch.instructions, batch.metadata)
                mse = ((actions-target).square()*weights).flatten(1).sum(-1).mean()
                entry.update({'target_weighted_mse': float(mse), 'decoded_actions': actions.cpu().tolist()})
            row[name] = entry
        records.append(row)
        print(f"Held-out comparison {len(records)}/{len(selected)}", flush=True)
    names = ('no_patch', 'seed_patch', 'optimized_patch')
    metrics = ['target_ce'] + (['target_weighted_mse'] if evaluate_actions else [])
    return {'frames': len(records), 'description': 'Fixed recorded frames; not closed-loop ASR or AP',
            'means': {name:{metric:float(np.mean([r[name][metric] for r in records]))
                           for metric in metrics} for name in names}, 'records': records}


@torch.no_grad()
def score_patch(policy, renderer, dataset, patch, objective_config, limit, device):
    objective = TargetObjective(policy, renderer, **objective_config)
    target, weights = objective.target.to(device), objective.weights.to(device)
    records = []
    for index in balanced_indices(dataset, limit):
        batch = collate_observations([dataset[index]]).to(device)
        image = renderer(batch.images, patch, batch.metadata)[0]
        logits, tokens, token_weights = policy.target_logits(image, batch.instructions, batch.metadata, target, weights)
        ce = (F.cross_entropy(logits.float().transpose(1, 2), tokens, reduction='none') * token_weights).sum(-1)
        actions = policy.predict_actions(image, batch.instructions, batch.metadata)
        records.append({'image':dataset.records[index]['image'], 'target_ce':float(ce.mean()),
                        'target_weighted_mse':float(((actions-target).square()*weights).sum(-1).mean())})
    return {'frames':len(records), 'target_ce':float(np.mean([r['target_ce'] for r in records])),
            'target_weighted_mse':float(np.mean([r['target_weighted_mse'] for r in records]))}
