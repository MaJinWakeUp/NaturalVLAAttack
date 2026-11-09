"""Reviewed selection, physics checks, and paired held-out texture evaluation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .object_texture import (NativeTextureWorld, array_digest, digest, fingerprint,
                             load_config, load_mapping, write_json)
from .collect_object_states import prepare, run_episode


def wilson(successes, total):
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes/total
    center = (p+z*z/(2*total))/(1+z*z/total)
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return [max(0.,center-half),min(1.,center+half)]


def summarize(rows, acceptance, primary='both'):
    if set(rows) != {'original','benign','adversarial'}:
        raise ValueError('Need exactly three conditions')
    indexed = {}
    for name, records in rows.items():
        indexed[name] = {r['trial']:r for r in records}
        if len(indexed[name]) != len(records):
            raise ValueError('Duplicate test state')
    keys = sorted(indexed['original'])
    if not keys or any(set(indexed[c]) != set(keys) for c in indexed):
        raise ValueError('Unpaired or empty test conditions')
    pairs = []
    for key in keys:
        records = [indexed[c][key] for c in indexed]
        signatures = {(r['suite'],r['task_id'],r['canonical_image_sha256'],r['canonical_state_sha256']) for r in records}
        if len(signatures) != 1:
            raise ValueError('Canonical state/render mismatch across test conditions')
        pairs.append({'state':key, **{c:bool(indexed[c][key]['success']) for c in indexed}})
    if primary not in ('both', 'original'):
        raise ValueError('Unknown conditional ASR reference')
    original_eligible = [p for p in pairs if p['original']]
    both_eligible = [p for p in original_eligible if p['benign']]
    eligible = both_eligible if primary == 'both' else original_eligible
    failures = sum(not p['adversarial'] for p in eligible)
    rates = {c:sum(not p[c] for p in pairs)/len(pairs) for c in indexed}
    conditional = failures/len(eligible) if eligible else None
    differences = {}
    for control in ('original','benign'):
        differences[control] = {'failure_rate_increase':rates['adversarial']-rates[control],
            'new_failures':sum(p[control] and not p['adversarial'] for p in pairs),
            'recoveries':sum(not p[control] and p['adversarial'] for p in pairs)}
    advance = (len(eligible) >= acceptance['eligible_test_states'] and conditional is not None and
               conditional >= acceptance['conditional_asr'] and
               all(sum(p[c] for p in pairs) >= acceptance['baseline_test_successes']
                   for c in (('original','benign') if primary == 'both' else ('original',))) and
               differences['benign']['failure_rate_increase']+1e-12 >= acceptance['failure_increase_over_benign'])
    return {'episodes_per_condition':len(pairs), 'failure_rates':rates,
            'conditional_reference':primary,
            'original_conditioned_asr':sum(not p['adversarial'] for p in original_eligible)/len(original_eligible) if original_eligible else None,
            'both_controls_conditioned_asr':sum(not p['adversarial'] for p in both_eligible)/len(both_eligible) if both_eligible else None,
            'conditional_asr':conditional, 'conditional_failures':failures,
            'conditional_episodes':len(eligible), 'conditional_95pct_wilson':wilson(failures,len(eligible)),
            'paired_differences':differences, 'paired_successes':pairs,
            'canonical_states_and_renders_match':True, 'meets_effect_criteria':bool(advance)}


def dynamics_digest(model):
    import hashlib
    fields = ('body_mass','body_inertia','body_pos','body_quat','jnt_type','jnt_pos','jnt_axis',
              'jnt_range','geom_type','geom_size','geom_pos','geom_quat','geom_friction',
              'geom_solimp','geom_solref','geom_contype','geom_conaffinity','dof_damping',
              'dof_armature','dof_frictionloss','actuator_gear','actuator_gainprm','actuator_biasprm')
    h = hashlib.sha256()
    for field in fields:
        h.update(field.encode()); h.update(np.ascontiguousarray(getattr(model,field)).tobytes())
    return h.hexdigest()


def physics_check(world, atlas, state_index=None, steps=10):
    index = world.cfg['splits']['calibration'][0] if state_index is None else state_index
    original_dynamics = dynamics_digest(world.model)
    paths = []
    for texture in (None, atlas):
        start = world.reset(index)
        if texture is not None:
            world.set_texture(texture)
        if dynamics_digest(world.model) != original_dynamics:
            raise ValueError('Appearance override modified dynamics parameters')
        states = [start]
        for _ in range(steps):
            world.env.step([0.,0.,0.,0.,0.,0.,-1.])
            states.append(world.state())
        paths.append(np.stack(states))
    world.original()
    difference = float(np.abs(paths[0]-paths[1]).max())
    if not np.isfinite(difference) or difference > world.cfg['acceptance']['physics_atol']:
        raise ValueError(f'Texture conditions changed fixed-action dynamics: {difference}')
    return {'state_index':index, 'steps':steps, 'max_state_difference':difference,
            'tolerance':world.cfg['acceptance']['physics_atol'], 'dynamics_sha256':original_dynamics,
            'accepted':True}


def replay_check(cfg, root, world):
    """Exact RGB/segmentation replay after motion and after creating a fresh renderer."""
    frames = []
    for state_index in (cfg['splits']['train'][0],cfg['splits']['calibration'][0]):
        world.reset(state_index)
        for step in range(40):
            image = world.image()
            state = world.state().copy()
            mask = world.object_mask()
            if not np.array_equal(state,world.state()):
                raise ValueError('Rendering changed the integration state')
            frames.append({'state':state,'image':image,'mask':mask,'initial_state':state_index,'step':step})
            # Exercise moving camera/robot transforms without any policy query or test-state access.
            action = [.15*np.sin(step/5),.1*np.cos(step/7),.05*np.sin(step/3),0.,0.,0.,-1.]
            world.env.step(action)

    def verify(renderer, phase):
        maximum = 0
        for ordinal,frame in enumerate(reversed(frames)):
            renderer.restore_state(frame['state'])
            actual = renderer.image()
            error = int(np.abs(actual.astype(int)-frame['image'].astype(int)).max())
            maximum = max(maximum,error)
            if error or not np.array_equal(renderer.object_mask(),frame['mask']):
                np.savez_compressed(root/f'replay_failure_{phase}.npz',state=frame['state'],
                                    expected=frame['image'],actual=actual,mask=frame['mask'])
                Image.fromarray(frame['image']).save(root/f'replay_expected_{phase}.png')
                Image.fromarray(actual).save(root/f'replay_actual_{phase}.png')
                raise ValueError(f'Native replay failed during {phase}, frame {ordinal}; max RGB error {error}')
        return maximum

    same_context = verify(world,'same_context')
    fresh = NativeTextureWorld(cfg)
    try:
        fresh_context = verify(fresh,'fresh_context')
    finally:
        fresh.close()
    return {'accepted':True,'frames':len(frames),'checks':2*len(frames),
            'moving_scene_states':sorted({f['initial_state'] for f in frames}),
            'same_context_max_rgb_error':same_context,'fresh_context_max_rgb_error':fresh_context,
            'segmentation_exact':True,'policy_queries':0,'held_out_states_opened':False}


def native_preflight(cfg):
    root = prepare(cfg)
    marker = root/'native_preflight.json'
    if marker.exists():
        existing = json.loads(marker.read_text())
        if existing.get('replay',{}).get('accepted') and existing.get('schema_version') == 2:
            return existing
    world = NativeTextureWorld(cfg)
    mapping = load_mapping(cfg)
    try:
        state = world.reset(cfg['splits']['calibration'][0])
        before = world.image()
        # Technical renderer test deliberately makes the whole target conspicuous.
        # This texture is never scored by the policy or used as an attack candidate.
        probe = 255-mapping.original
        world.set_texture(probe)
        changed = world.image()
        if np.array_equal(before,changed):
            raise ValueError('Native texture upload did not change the rendered target')
        world.original()
        if not np.array_equal(before,world.image()) or not np.array_equal(state,world.state()):
            raise ValueError('Texture update/restoration changed canonical rendering or state')
        Image.fromarray(np.concatenate((before,changed),axis=1)).save(root/'native_upload_check.png')
        result = {'schema_version':2,'native_upload_changes_image':True, 'original_restored_exactly':True,
                  'state_size':len(state), 'texture_id':world.texture_id,
                  'texture_vertical_flip':world.flip_texture,
                  'physics':physics_check(world,probe), 'surface_fraction':mapping.surface_fraction,
                  'replay':replay_check(cfg,root,world)}
        write_json(marker,result)
        return result
    finally:
        world.close()


def check_review_item(root, item):
    if digest(root/item['sheet']) != item['sheet_sha256'] or digest(root/'assets/editable_mask.png') != item['mask_sha256']:
        raise ValueError('Review sheet or editable mask changed')
    candidates = {c['id']:c for c in json.loads((root/'candidates.json').read_text())}
    candidate = candidates[item['id']]
    for field,sha in (('texture','texture_sha256'),('benign_texture','benign_sha256')):
        if digest(root/candidate[field]) != item[sha]:
            raise ValueError('Reviewed texture changed')
    return candidate


def record_review(cfg, candidate_id, accepted, reviewer, notes):
    if not reviewer.strip() or not notes.strip():
        raise ValueError('Record the reviewer and an actual appearance assessment')
    root = Path(cfg['output'])
    if (root/'test_started.json').exists():
        raise ValueError('Cannot authorize a new appearance decision after opening test results')
    queue = json.loads((root/'review_queue.json').read_text())
    matches = [item for item in queue if item['id']==candidate_id]
    if len(matches) != 1:
        raise ValueError('Candidate is not in the appearance-review queue')
    item = matches[0]
    check_review_item(root,item)
    write_json(root/'review/decisions'/f'{candidate_id}.json',
               {**item,'accepted':bool(accepted),'reviewer':reviewer,'notes':notes,
                'held_out_test_opened':(root/'test_started.json').exists()})


def select(cfg):
    if cfg.get('profile') == 'benign_latent':
        from .benign_latent import select_latent
        return select_latent(cfg)
    if cfg.get('profile') == 'full_surface_unconstrained':
        from .full_surface_attack import select_full
        return select_full(cfg)
    root = Path(cfg['output'])
    complete = json.loads((root/'training_complete.json').read_text())
    if complete['config_sha256'] != fingerprint(cfg):
        raise ValueError('Training settings changed')
    if (root/'selection.json').exists():
        selected = json.loads((root/'selection.json').read_text())
        if selected['config_sha256'] != fingerprint(cfg):
            raise ValueError('Frozen selection settings changed')
        return selected
    if (root/'test_started.json').exists():
        raise ValueError('Cannot select a new candidate after accessing test results')
    queue = json.loads((root/'review_queue.json').read_text())
    if not queue:
        raise ValueError('No candidate passed the numeric appearance/baseline gates')
    eligible = []
    for item in queue:
        decision_path = root/'review/decisions'/f"{item['id']}.json"
        if not decision_path.exists():
            raise ValueError(f"Appearance review pending: {item['id']}")
        review = json.loads(decision_path.read_text())
        candidate = check_review_item(root,item)
        if any(review.get(key) != value for key,value in item.items()) or review['held_out_test_opened']:
            raise ValueError('Appearance decision is stale or not blinded to test outcomes')
        if review['accepted']:
            eligible.append(candidate)
    if not eligible:
        raise ValueError('No visually accepted candidate')
    order = {c['id']:i for i,c in enumerate(json.loads((root/'candidates.json').read_text()))}
    winner = min(eligible,key=lambda c:(c['validation_action_mse'],c['appearance']['rmse_original'],order[c['id']]))
    selection = {**winner,'config_sha256':fingerprint(cfg),
                 'review_decisions':{item['id']:digest(root/'review/decisions'/f"{item['id']}.json") for item in queue},
                 'held_out_test_opened':False}
    write_json(root/'selection.json',selection)
    write_json(root/'status.json',{'status':'selected_for_test','candidate':winner['id']})
    return selection


def evaluate(cfg, videos=True):
    from .adapters import load_policy
    root = Path(cfg['output'])
    selection = select(cfg)
    for field,sha in (('texture','texture_sha256'),('benign_texture','benign_sha256')):
        if digest(root/selection[field]) != selection[sha]:
            raise ValueError('Frozen selected texture changed')
    for identity,sha in selection['review_decisions'].items():
        if digest(root/'review/decisions'/f'{identity}.json') != sha:
            raise ValueError('Visual review changed after selection')
    marker = root/'test_started.json'
    frozen = {'selection_sha256':digest(root/'selection.json'),'config_sha256':fingerprint(cfg)}
    if marker.exists() and json.loads(marker.read_text()) != frozen:
        raise ValueError('Test selection changed; do not reuse held-out test results')
    textures = {'original':load_mapping(cfg).original,
                'benign':np.array(Image.open(root/selection['benign_texture']).convert('RGB')),
                'adversarial':np.array(Image.open(root/selection['texture']).convert('RGB'))}
    world = NativeTextureWorld(cfg)
    policy = load_policy(cfg['policy'])
    rows = {condition:[] for condition in textures}
    directory = root/'test'; directory.mkdir(exist_ok=True)
    try:
        write_json(root/'physics_selected.json',physics_check(world,textures['adversarial']))
        # The marker is written before the first reserved state is opened.
        write_json(marker,frozen)
        names = list(textures)
        for ordinal,state_index in enumerate(cfg['splits']['test']):
            expected = None
            for condition in names[ordinal%3:]+names[:ordinal%3]:
                folder = directory/f'state_{state_index:03d}'/condition
                folder.mkdir(parents=True,exist_ok=True)
                path = folder/'report.json'
                if path.exists():
                    report = json.loads(path.read_text())
                    if report.get('selection_sha256') != frozen['selection_sha256']:
                        raise ValueError('Completed test episode belongs to a different selection')
                    actual = (report['canonical_image_sha256'],report['canonical_state_sha256'])
                    if expected and actual != expected:
                        raise ValueError('Completed test controls are not matched')
                else:
                    report,_ = run_episode(world,policy,state_index,textures[condition],
                        expected_canonical=expected,video=folder/'rollout.mp4' if videos else None)
                    report.update(condition=condition,selection_sha256=frozen['selection_sha256'])
                    write_json(path,report)
                expected = (report['canonical_image_sha256'],report['canonical_state_sha256'])
                rows[condition].append(report)
                print(json.dumps({'test_state':state_index,'condition':condition,'success':report['success']}),flush=True)
                if cfg.get('profile') in ('full_surface_unconstrained','benign_latent'):
                    from .full_surface_attack import stop_check
                    stop_check()
    finally:
        world.close()
    summary = summarize(rows,cfg['acceptance'],cfg.get('evaluation',{}).get('primary_reference','both'))
    summary.update(status='evaluated',candidate=selection['id'],
                   selection_sha256=frozen['selection_sha256'],config_sha256=fingerprint(cfg),
                   appearance_accepted=True if cfg['appearance'].get('enabled',True) else None,
                   appearance_constraints_enabled=cfg['appearance'].get('enabled',True),
                   visual_review_recorded=bool(selection['review_decisions']),
                   interpretation=cfg.get('interpretation','Single-task exploratory pilot; Wilson interval is descriptive.'))
    if cfg.get('initial_state_bank'):
        summary['initial_state_bank_sha256'] = cfg['initial_state_bank']['sha256']
        summary['initialization_source'] = 'Fresh native LIBERO placement-sampler resets'
    write_json(root/'asr_report.json',summary)
    export_final(cfg,selection,summary)
    write_json(root/'status.json',summary)
    print(json.dumps(summary,indent=2))
    return summary


def export_final(cfg,selection,summary):
    root, final = Path(cfg['output']),Path(cfg['final_output'])
    marker = final/'report.json'
    if final.exists():
        if marker.exists() and json.loads(marker.read_text()).get('selection_sha256') == summary['selection_sha256']:
            return
        raise FileExistsError(f'Refusing to overwrite an existing final result: {final}')
    temporary = final.with_name(final.name+'.exporting')
    temporary.mkdir(parents=True,exist_ok=True)
    for source,name in ((root/'assets/original.png','original.png'),(root/selection['benign_texture'],'benign.png'),
        (root/selection['texture'],'adversarial.png'),(root/'assets/editable_mask.png','editable_mask.png'),
        (root/'assets/mapping.json','mapping.json')):
        shutil.copyfile(source,temporary/name)
    for item in json.loads((root/'review_queue.json').read_text()):
        if item['id']==selection['id']:
            shutil.copyfile(root/item['sheet'],temporary/'preview.png')
    if selection.get('preview'):
        shutil.copyfile(root/selection['preview'],temporary/'preview.png')
    if (root/'candidates.json').exists():
        shutil.copyfile(root/'candidates.json',temporary/'candidates.json')
    portable = json.loads(json.dumps(cfg))
    portable.update(libero_root='${DURA_LIBERO_ROOT}',output='${DURA_SCRATCH}/runs/'+root.name,final_output='.')
    for section in ('policy','diffusion'):
        portable[section]['cache_dir']='${HF_HOME}/hub'
    portable.get('full_surface',{}).pop('reuse_collection',None)
    if cfg.get('initial_state_bank'):
        shutil.copyfile(cfg['initial_state_bank']['path'],temporary/'initial_states.npz')
        portable['initial_state_bank']['path'] = './initial_states.npz'
    for seed, item in portable.get('initialization', {}).get('files', {}).items():
        target = temporary/'initial_latents'/f'seed_{seed}.npy'
        target.parent.mkdir(parents=True,exist_ok=True)
        source = root/'assets/initial_latents'/target.name
        if digest(source) != item['sha256']:
            raise ValueError('Frozen input latent changed before export')
        shutil.copyfile(source,target)
        item['path'] = './initial_latents/'+target.name
    write_json(temporary/'config.json',portable)
    write_json(temporary/'report.json',summary)
    write_json(temporary/'selection.json',selection)
    write_json(temporary/'provenance.json',{'source_snapshot':json.loads((root/'code_provenance.json').read_text()) if (root/'code_provenance.json').exists() else None,
        'physics':json.loads((root/'physics_selected.json').read_text()),'texture_sha256':selection['texture_sha256'],
        'seed_texture_sha256':selection['benign_sha256']})
    temporary.rename(final)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--no-video',action='store_true')
    args=parser.parse_args()
    evaluate(load_config(args.config),not args.no_video)


if __name__=='__main__':
    main()
