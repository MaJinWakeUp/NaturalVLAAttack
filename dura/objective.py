"""Target-only weighted action losses (Eqs. 2–4)."""

import torch
import torch.nn.functional as F


class TargetObjective:
    def __init__(self, policy, renderer, target, weights=(1, 1, 1, .5, .5, .5, .2),
                 dimensions=None, policy_batch_size=1):
        self.policy, self.renderer = policy, renderer
        self.target = torch.as_tensor(target, dtype=torch.float32)
        self.weights = torch.as_tensor(weights, dtype=torch.float32)
        if self.target.ndim not in (1, 2) or self.target.shape[-1] != len(self.weights):
            raise ValueError("target must be [D] or [horizon,D] with D weights")
        if not torch.isfinite(self.target).all() or not torch.isfinite(self.weights).all():
            raise ValueError("target and weights must be finite")
        if (self.weights < 0).any():
            raise ValueError("weights must be nonnegative")
        if dimensions is not None:
            if not dimensions or any(d < 0 or d >= len(self.weights) for d in dimensions):
                raise ValueError("Invalid attacked dimensions")
            mask = torch.zeros_like(self.weights)
            mask[dimensions] = 1
            self.weights *= mask
        if not self.weights.any() or policy_batch_size < 1:
            raise ValueError("At least one positive weight and positive policy_batch_size required")
        self.policy_batch_size = policy_batch_size
        self.candidate_queries = 0
        self.observation_queries = 0

    def __call__(self, patches, batch, mode):
        rendered = self.renderer(batch.images, patches, batch.metadata)
        n, b = rendered.shape[:2]
        images = rendered.flatten(0, 1)
        instructions, metadata = batch.instructions * n, batch.metadata * n
        losses = []
        for start in range(0, n*b, self.policy_batch_size):
            stop = min(start + self.policy_batch_size, n*b)
            chunk = images[start:stop]
            if mode == "whitebox":
                logits, targets, token_weights = self.policy.target_logits(
                    chunk, instructions[start:stop], metadata[start:stop], self.target, self.weights)
                if logits.shape[:-1] != targets.shape or targets.shape != token_weights.shape:
                    raise ValueError("Adapter must return logits [B,L,V], targets/weights [B,L]")
                ce = F.cross_entropy(logits.float().transpose(1, 2), targets, reduction="none")
                loss = (ce * token_weights.to(ce)).sum(-1)
            elif mode == "blackbox":
                actions = self.policy.predict_actions(chunk, instructions[start:stop], metadata[start:stop])
                actions = torch.as_tensor(actions, dtype=torch.float32, device=chunk.device)
                target, weights = self.target.to(actions), self.weights.to(actions)
                if actions.ndim not in (2, 3) or actions.shape[0] != len(chunk):
                    raise ValueError("Policy actions must be [B,D] or [B,horizon,D]")
                if actions.shape[-1] != target.shape[-1]:
                    raise ValueError("Policy and target action dimensions differ")
                if target.ndim == 2 and actions.shape[1:] != target.shape:
                    raise ValueError("Target horizon must match policy horizon")
                loss = ((actions-target).square() * weights).flatten(1).sum(-1)
            else:
                raise ValueError("Unknown access mode")
            losses.append(loss)
        self.candidate_queries += n
        self.observation_queries += n*b
        return torch.cat(losses).reshape(n, b).mean(-1)
