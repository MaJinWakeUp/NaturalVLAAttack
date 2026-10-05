"""Download the pinned models without loading them onto a GPU."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dura.config import PROJECT_ROOT, read_config
from dura.task_experiments import SUITES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(PROJECT_ROOT / 'configs/libero_relaxed.yaml'))
    parser.add_argument('--suites', nargs='+', choices=list(SUITES), default=list(SUITES))
    args = parser.parse_args()
    config = read_config(args.config)
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoProcessor
    diffusion = config['diffusion']
    snapshot_download(diffusion['checkpoint'], revision=diffusion['revision'], cache_dir=diffusion['cache_dir'])
    for name in args.suites:
        policy = {**config['policy'], **{k: SUITES[name][k] for k in ('checkpoint','revision','unnorm_key')}}
        snapshot_download(policy['checkpoint'], revision=policy['revision'], cache_dir=policy['cache_dir'])
        options = {k: policy[k] for k in ('revision','code_revision','cache_dir','trust_remote_code')}
        AutoConfig.from_pretrained(policy['checkpoint'], **options)
        AutoProcessor.from_pretrained(policy['checkpoint'], **options)
        print(f"Cached {name}: {policy['revision']}", flush=True)


if __name__ == '__main__':
    main()
