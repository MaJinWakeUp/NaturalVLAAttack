"""Validated algorithm settings; unspecified paper choices remain configurable."""

from dataclasses import dataclass
import math
import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def scratch_root():
    return Path(os.environ.get("DURA_SCRATCH", "/scratch/jin7/dura")).expanduser().resolve()


def expand_path(value, base):
    variables = {
        "DURA_SCRATCH": str(scratch_root()),
        "HF_HOME": os.environ.get("HF_HOME", "/scratch/jin7/huggingface_cache"),
    }
    value = str(value)
    for name, replacement in variables.items():
        value = value.replace("${" + name + "}", replacement)
    value = os.path.expandvars(os.path.expanduser(value))
    if "$" in value:
        raise ValueError(f"Unresolved environment variable in path: {value}")
    return str((base / value).resolve())


def read_config(path):
    """Read YAML/JSON and resolve storage paths relative to the config file."""
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    for name in ("manifest", "validation_manifest", "seed_patch", "output", "final_output"):
        if config.get(name):
            config[name] = expand_path(config[name], path.parent)
    for name in ("policy", "diffusion"):
        section = config.get(name, {})
        checkpoint = section.get("checkpoint", "")
        if checkpoint.startswith((".", "/", "$", "~")):
            section["checkpoint"] = expand_path(checkpoint, path.parent)
        if section.get("cache_dir"):
            section["cache_dir"] = expand_path(section["cache_dir"], path.parent)
    settings = config.get("naturalness", {})
    if settings.get("scene_reference_patch"):
        settings["scene_reference_patch"] = expand_path(settings["scene_reference_patch"], path.parent)
    return config


@dataclass(frozen=True)
class AttackConfig:
    mode: str = "whitebox"
    ddim_steps: int = 200
    strength: float = 0.5
    anchor_weight: float = 0.2
    step_size: float = 1.0
    samples: int = 2048
    candidate_batch_size: int = 1
    baseline: str = "none"
    query_alpha: str = "step"
    candidate_denoise: bool = False
    seed: int = 0

    def __post_init__(self):
        if self.mode not in {"whitebox", "blackbox"}:
            raise ValueError("mode must be whitebox or blackbox")
        if self.ddim_steps < 1 or not 0 < self.strength <= 1:
            raise ValueError("ddim_steps must be positive; strength must be in (0, 1]")
        if int(self.ddim_steps * self.strength) < 1:
            raise ValueError("strength leaves no active denoising steps")
        if not 0 <= self.anchor_weight <= 1:
            raise ValueError("anchor_weight must be in [0, 1]")
        if not math.isfinite(self.step_size) or self.step_size < 0:
            raise ValueError("step_size must be finite and nonnegative")
        if self.samples < 1 or self.candidate_batch_size < 1:
            raise ValueError("sample counts must be positive")
        if self.baseline not in {"none", "leave_one_out"}:
            raise ValueError("baseline must be none or leave_one_out")
        if self.baseline == "leave_one_out" and self.samples < 2:
            raise ValueError("leave_one_out requires at least two samples")
        if self.query_alpha not in {"step", "cumulative", "ddim_transition"}:
            raise ValueError("unknown query_alpha convention")

    @property
    def updates(self):
        return int(self.ddim_steps * self.strength)
