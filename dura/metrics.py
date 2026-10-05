"""Rollout failure rate and per-trajectory target precision (Eqs. 10–11)."""

import numpy as np


def target_matches(actions, target, tolerance, dimensions=None):
    actions = np.asarray(actions, dtype=float)
    target = np.asarray(target, dtype=float)
    tolerance = np.asarray(tolerance, dtype=float)
    if actions.ndim != 2 or target.shape != (actions.shape[-1],):
        raise ValueError("Expected executed actions [steps,D] and target [D]")
    if not np.isfinite(actions).all() or not np.isfinite(target).all():
        raise ValueError("Actions and target must be finite")
    if not np.isfinite(tolerance).all() or (tolerance < 0).any():
        raise ValueError("Tolerance must be finite and nonnegative")
    matched = np.abs(actions-target) <= tolerance
    if dimensions is not None:
        if not dimensions or any(i < 0 or i >= actions.shape[-1] for i in dimensions):
            raise ValueError("Invalid target dimensions")
        matched = matched[:, dimensions]
    return matched.all(axis=-1)


def rollout_metrics(rollouts, target, tolerance, dimensions=None):
    if not rollouts:
        raise ValueError("At least one rollout is required")
    precisions = []
    for rollout in rollouts:
        if not isinstance(rollout.get("success"), bool):
            raise ValueError("Each rollout needs a boolean success field")
        if not len(rollout["actions"]):
            raise ValueError("Rollout has no executed actions; AP is undefined")
        precisions.append(float(target_matches(rollout["actions"], target, tolerance, dimensions).mean()))
    return {"rollouts": len(rollouts),
            "asr": sum(not row["success"] for row in rollouts)/len(rollouts),
            "ap": float(np.mean(precisions)), "per_rollout_ap": precisions}
