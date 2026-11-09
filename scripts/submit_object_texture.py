"""Freeze a source snapshot, preflight Slurm, and optionally submit one GPU job."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from dura.config import scratch_root
from dura.object_texture import digest, fingerprint, load_config, write_json
from dura.collect_object_states import prepare


def freeze_source(root):
    destination = root/'source'
    manifest_path = root/'code_provenance.json'
    if destination.exists():
        manifest = json.loads(manifest_path.read_text())
        for relative,sha in manifest['files'].items():
            if digest(destination/relative) != sha:
                raise ValueError('Frozen source snapshot changed')
        return destination
    destination.mkdir(parents=True)
    files = {}
    for directory in ('dura','configs','scripts','docs'):
        for path in (PROJECT/directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py','.yaml','.sh','.sbatch','.md'):
                relative = path.relative_to(PROJECT)
                target = destination/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(path,target)
                files[str(relative)] = digest(target)
    write_json(manifest_path,{'files':files,'source_project':str(PROJECT),
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()})
    return destination


def cache_check(cfg):
    # No model tensors are loaded, and no network access is requested.
    os.environ.setdefault('HF_HOME','/scratch/jin7/huggingface_cache')
    os.environ.setdefault('HF_MODULES_CACHE',str(Path(os.environ['HF_HOME'])/'modules'))
    from huggingface_hub import snapshot_download
    paths = {}
    for key in ('policy','diffusion'):
        section = cfg[key]
        paths[key] = snapshot_download(repo_id=section['checkpoint'],revision=section['revision'],
            cache_dir=section['cache_dir'],local_files_only=True)
    policy_root = Path(paths['policy'])
    index = json.loads((policy_root/'model.safetensors.index.json').read_text())
    for name in set(index['weight_map'].values()):
        if not (policy_root/name).is_file():
            raise FileNotFoundError(policy_root/name)
    for name in ('model_index.json','unet/config.json','unet/diffusion_pytorch_model.safetensors',
                 'vae/config.json','vae/diffusion_pytorch_model.safetensors',
                 'text_encoder/config.json','text_encoder/model.safetensors'):
        if not (Path(paths['diffusion'])/name).is_file():
            raise FileNotFoundError(Path(paths['diffusion'])/name)
    from transformers import AutoConfig, AutoProcessor
    settings = cfg['policy']
    remote = {key:settings[key] for key in ('revision','code_revision','cache_dir','trust_remote_code')}
    for loader in (AutoConfig,AutoProcessor):
        loader.from_pretrained(settings['checkpoint'],local_files_only=True,**remote)
    return paths


def active_job_state(job_id):
    # Old jobs disappear from squeue; querying one by ID can then return an error.
    # Query this user's active jobs so an expired ID is safely treated as inactive.
    output = subprocess.check_output(
        ['squeue', '-h', '--user', str(os.getuid()), '-o', '%i|%T'], text=True)
    for line in output.splitlines():
        identity, state = line.split('|', 1)
        if identity.strip() == str(job_id):
            return state.strip()
    return ''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(PROJECT/'configs/object_texture_alphabet_soup.yaml'))
    parser.add_argument('--stage',choices=['native-check','train','full-run','latent-run','evaluate'],default='train')
    parser.add_argument('--run-root')
    parser.add_argument('--final-root')
    parser.add_argument('--account')
    parser.add_argument('--exclude',help='Comma-separated nodes to avoid, for example the occupied desktop node')
    parser.add_argument('--time',help='Default: 15 minutes for native-check, 6 hours for training/evaluation')
    parser.add_argument('--memory',help='Default: 8G for native-check, 48G for training/evaluation')
    parser.add_argument('--afterok',type=int,help='Start only after this prerequisite job succeeds')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--submit',action='store_true',help='Submit after scheduler validation; otherwise prepare only')
    args = parser.parse_args()
    if args.afterok is not None and args.afterok < 1:
        parser.error('--afterok must be a positive job ID')
    walltime = args.time or ('00:15:00' if args.stage=='native-check' else '06:00:00')
    memory = args.memory or ('8G' if args.stage=='native-check' else '48G')
    cfg = load_config(args.config)
    if args.run_root:
        cfg['output'] = str(Path(args.run_root).resolve())
    if args.final_root:
        cfg['final_output'] = str(Path(args.final_root).resolve())
    root = Path(cfg['output']); root.mkdir(parents=True,exist_ok=True)
    with (root/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        # Preparing assets and checking cached snapshots consumes no GPU.
        prepare(cfg)
        if args.stage=='evaluate':
            from dura.evaluate_object_texture import select
            select(cfg)
        cached = cache_check(cfg)
        code = freeze_source(root)
        config_path = root/'config.json'
        config_sha = fingerprint(cfg)
        if fingerprint(json.loads(config_path.read_text())) != config_sha:
            raise ValueError('Frozen run config differs')
        # Exercise the exact frozen module/configuration combination before requesting a GPU.
        runtime_check = subprocess.run([sys.executable,'-m','dura.object_texture_pilot',
            '--config',str(config_path),'prepare'],cwd=code,text=True,capture_output=True)
        if runtime_check.returncode:
            raise RuntimeError(f'Frozen runtime preflight failed: {runtime_check.stderr}')
        write_json(root/'runtime_preflight.json',{'accepted':True,'config_sha256':config_sha,
            'source':str(code),'stdout':runtime_check.stdout,'stderr':runtime_check.stderr})
        record_path = root/f'slurm_{args.stage}.json'
        if record_path.exists():
            old = json.loads(record_path.read_text())
            if not args.resume:
                raise FileExistsError(f'Job already recorded: {old}; use --resume after checking its status')
            state = active_job_state(old['job_id'])
            if state:
                raise RuntimeError(f"Recorded job {old['job_id']} is still active: {state}")
        logs = root/'logs'; logs.mkdir(exist_ok=True)
        command = ['sbatch','--parsable','--job-name',f'dura-label-{args.stage}',
                   '--time',walltime,'--mem',memory,'--chdir',str(code),
                   '--output',str(logs/f'{args.stage}-%j.out'),
                   '--error',str(logs/f'{args.stage}-%j.err')]
        if args.account:
            command += ['--account',args.account]
        if args.exclude:
            command += ['--exclude',args.exclude]
        if args.afterok:
            command += ['--dependency',f'afterok:{args.afterok}','--kill-on-invalid-dep=yes']
        command += [str(code/'scripts/object_texture_pilot.sbatch'),str(code),str(config_path),args.stage,
                    cfg['libero_root'],os.environ.get('DURA_DATASETS_ROOT','/scratch/jin7/datasets/VLA'),
                    str(scratch_root()),os.environ.get('HF_HOME','/scratch/jin7/huggingface_cache')]
        write_json(root/f'submission_plan_{args.stage}.json',{'command':command,'models_cached':cached,
            'config_sha256':config_sha,'resource_request':{'gpus':1,'gpu_type':'a100','gpu_memory_gb':40,
                'cpus':4,'memory':memory,'walltime':walltime},'afterok':args.afterok})
        if not args.submit:
            print(json.dumps({'prepared':str(root),'command':command,'submitted':False},indent=2)); return
        # --test-only checks policy/resources without allocating a job.
        checked = subprocess.run(command[:1]+['--test-only']+command[1:],text=True,capture_output=True)
        if checked.returncode:
            raise RuntimeError(f'Slurm submission preflight failed: {checked.stderr}')
        result = subprocess.run(command,text=True,capture_output=True,check=True)
        output = result.stdout.strip().splitlines()
        token = output[-1].split(';')[0]
        if not token.isdigit():
            write_json(root/'submission_uncertain.json',{'stdout':result.stdout,'stderr':result.stderr,'command':command})
            raise RuntimeError('Scheduler response was unexpected; inspect queue before resubmitting')
        record = {'job_id':int(token),'stage':args.stage,'run':str(root),'source':str(code),
                  'config_sha256':config_sha,'scheduler_stdout':result.stdout,'scheduler_stderr':result.stderr}
        write_json(record_path,record)
        print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
