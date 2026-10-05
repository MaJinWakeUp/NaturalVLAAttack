"""Export a bounded, episode-disjoint RLDS sample into scratch storage."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, help='Local TFDS/RLDS version directory')
    parser.add_argument('--output', required=True, help='Export directory in scratch storage')
    parser.add_argument('--image-key', default='image')
    parser.add_argument('--split', default='train')
    parser.add_argument('--train-episodes-per-task', type=int, default=8)
    parser.add_argument('--val-episodes-per-task', type=int, default=2)
    parser.add_argument('--frames-per-episode', type=int, default=8)
    parser.add_argument('--image-size', type=int, default=224)
    args = parser.parse_args()
    if min(args.train_episodes_per_task, args.val_episodes_per_task, args.frames_per_episode, args.image_size) < 1:
        parser.error('Counts and image size must be positive')
    output = Path(args.output).resolve()
    specification = vars(args).copy()
    specification['output'] = str(output)
    specification['dataset'] = str(Path(args.dataset).resolve())
    summary_path = output/'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary['specification'] != specification:
            raise ValueError('Existing export has different settings; choose a new output directory')
        for split in ('train', 'val'):
            rows = [json.loads(line) for line in (output/f'{split}.jsonl').read_text().splitlines()]
            if len(rows) != summary['frames'][split] or not all((output/r['image']).is_file() for r in rows):
                raise ValueError('Existing export is incomplete')
        if not (output/'seed.png').is_file():
            raise ValueError('Existing export has no seed patch')
        print(json.dumps(summary, indent=2))
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Partial export exists; use a new output directory')
    import numpy as np
    from PIL import Image
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    (output/'frames').mkdir()
    builder = tfds.builder_from_directory(specification['dataset'])
    options = tf.data.Options()
    options.threading.private_threadpool_size = 4
    episodes = builder.as_dataset(split=args.split, shuffle_files=False).with_options(options)
    used = defaultdict(lambda: Counter())
    counts = Counter()
    seed_source = None
    with (output/'train.jsonl').open('x') as train, (output/'val.jsonl').open('x') as val:
        for episode_id, episode in enumerate(episodes):
            first = next(iter(episode['steps']), None)
            if first is None:
                continue
            language = first['language_instruction'].numpy().decode('utf-8').strip()
            if not language:
                continue
            # Keep whole episodes disjoint, stratified by exact task instruction.
            if used[language]['val'] < args.val_episodes_per_task:
                split, handle = 'val', val
            elif used[language]['train'] < args.train_episodes_per_task:
                split, handle = 'train', train
            else:
                continue
            steps = list(episode['steps'].as_numpy_iterator())
            selected = np.unique(np.linspace(0, len(steps)-1, args.frames_per_episode, dtype=int))
            for step_id in selected:
                step = steps[int(step_id)]
                image = step['observation'][args.image_key]
                # RLDS already carries the training orientation; do not rotate it again.
                resized = tf.image.resize(image, [args.image_size]*2, method='lanczos3', antialias=True)
                image = tf.cast(tf.clip_by_value(tf.round(resized), 0, 255), tf.uint8).numpy()
                relative = f'frames/ep{episode_id:05d}_s{step_id:04d}.png'
                Image.fromarray(image).save(output/relative)
                instruction = step['language_instruction'].decode('utf-8').strip() or language
                row = {'image': relative, 'instruction': instruction, 'rollout_id': episode_id,
                       'step': int(step_id), 'split': split}
                if 'state' in step['observation']:
                    row['state'] = step['observation']['state'].tolist()
                handle.write(json.dumps(row)+'\n')
                counts[split] += 1
                if seed_source is None and split == 'train':
                    # A benign background crop at the rendering location, not optimized content.
                    x, y, w, h = 0, args.image_size-50, 50, 50
                    Image.fromarray(image[y:y+h, x:x+w]).save(output/'seed.png')
                    seed_source = {'image': relative, 'box': [x,y,w,h],
                                   'description': 'Unmodified lower-left scene crop'}
            used[language][split] += 1
            print(json.dumps({'episode': episode_id, 'task': language, 'split': split,
                              'frames': dict(counts)}), flush=True)
    if min(counts['train'], counts['val']) < 1:
        raise ValueError('Export must contain both training and held-out observations')
    summary = {'specification': specification, 'frames': dict(counts),
               'episodes_by_instruction': {k:dict(v) for k,v in used.items()}, 'seed_source': seed_source}
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
