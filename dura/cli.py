"""Command entry points for the current natural-patch experiments."""
import argparse
import sys

from .config import read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "experiments", "evaluate", "prepare-data"])
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    modules = {
        "train": "natural_train", "experiments": "task_experiments",
        "evaluate": "evaluate_libero", "prepare-data": "prepare_rlds",
    }
    from importlib import import_module
    sys.argv = [sys.argv[0], *remaining]
    import_module("dura." + modules[args.command]).main()


if __name__ == "__main__":
    main()
