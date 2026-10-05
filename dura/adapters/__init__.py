"""Policy adapters use RGB float tensors; black-box outputs are decoded actions."""

import importlib


def load_policy(config):
    options = dict(config)
    kind = options.pop("kind", "openvla")
    if kind == "openvla":
        from .openvla import OpenVLAAdapter
        return OpenVLAAdapter(**options)
    if kind == "factory":
        module, name = options.pop("factory").split(":", 1)
        return getattr(importlib.import_module(module), name)(**options)
    raise ValueError(f"Unknown policy kind: {kind}")
