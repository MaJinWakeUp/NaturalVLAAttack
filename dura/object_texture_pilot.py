"""Stage entry points for the alphabet-soup object-texture pilot."""
import argparse
import json
import os
from pathlib import Path

from .object_texture import load_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    sub = parser.add_subparsers(dest='stage',required=True)
    for stage in ('prepare','native-check','train','full-run','latent-run','select','status'):
        sub.add_parser(stage)
    review = sub.add_parser('review')
    review.add_argument('--candidate',required=True)
    review.add_argument('--decision',choices=['accept','reject'],required=True)
    review.add_argument('--reviewer',required=True)
    review.add_argument('--notes',required=True)
    evaluation = sub.add_parser('evaluate')
    evaluation.add_argument('--no-video',action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    tracked = args.stage in ('native-check','train','full-run','latent-run','evaluate')
    status_path = Path(cfg['output'])/'status.json'
    context = {'stage':args.stage,'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
    if tracked:
        write_json(status_path,{'status':'running',**context})
    try:
        execute(args,cfg)
    except Exception as error:
        if tracked:
            write_json(status_path,{'status':'failed','error':str(error),
                                   'error_type':type(error).__name__,**context})
        raise
    if tracked and json.loads(status_path.read_text()).get('status') == 'running':
        write_json(status_path,{'status':'completed',**context})


def execute(args,cfg):
    if args.stage == 'prepare':
        from .collect_object_states import prepare
        print(prepare(cfg))
    elif args.stage == 'native-check':
        from .evaluate_object_texture import native_preflight
        print(json.dumps(native_preflight(cfg),indent=2))
    elif args.stage == 'train':
        from .evaluate_object_texture import native_preflight
        from .train_object_texture import train
        native_preflight(cfg)
        train(cfg)
    elif args.stage == 'full-run':
        from .full_surface_attack import run
        run(cfg)
    elif args.stage == 'latent-run':
        from .benign_latent import run
        run(cfg)
    elif args.stage == 'review':
        from .evaluate_object_texture import record_review
        record_review(cfg,args.candidate,args.decision=='accept',args.reviewer,args.notes)
    elif args.stage == 'select':
        from .evaluate_object_texture import select
        print(json.dumps(select(cfg),indent=2))
    elif args.stage == 'evaluate':
        from .evaluate_object_texture import evaluate
        evaluate(cfg,not args.no_video)
    else:
        root = Path(cfg['output'])
        path = root/'status.json'
        print(path.read_text() if path.exists() else json.dumps({'status':'not_started','run':str(root)}))


if __name__ == '__main__':
    main()
