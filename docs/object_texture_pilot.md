# Pilot: a natural adversarial label on the LIBERO alphabet-soup object

Status: implemented pilot protocol, November 2, 2026. The implementation provides collection, resumable training, appearance review, and held-out evaluation. Training and evaluation outcomes are recorded in scratch run reports. The existing patch results are separate experiments.

## Question and selected task

Can one fixed, plausible label texture increase OpenVLA task failures while the object remains recognizable as alphabet soup?

| Item | Fixed choice |
| --- | --- |
| Suite / task order / task ID | `libero_object` / `0` / `0` |
| Task name and policy instruction | `pick_up_the_alphabet_soup_and_place_it_in_the_basket` / “pick up the alphabet soup and place it in the basket” |
| Object instance | `alphabet_soup_1` |
| Task success | LIBERO predicate `(In alphabet_soup_1 basket_1_contain_region)` |
| Editable asset | `libero/libero/assets/stable_hope_objects/alphabet_soup/texture_map.png`, relative to the external LIBERO checkout |
| Original texture | 2048 × 2048 RGB atlas, SHA-256 `cd47556b3b2916ae138e8fd370d4f5192a2fe100369eec254c3c173409f48674` |
| Initial states | 50 saved states, each with 110 entries; source file `libero/libero/init_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.pruned_init` |
| Initial-state file SHA-256 | `b9023e5546b239cacb2aaa3c76df3b73c3de6f97500fea5e27a0db9827002a19` |
| Policy | `openvla/openvla-7b-finetuned-libero-object`, revision `287d6cfdf12d07b1449505f66d9bf3550257e9b3` |
| Transformers remote code | Revision `47a0ec7fc4ec123775a391911046cf33cf9ed83f` |
| Generator | Stable Diffusion v1.5, revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` |
| Episode settings | Current OpenVLA preprocessing; 10 settling steps, at most 280 policy steps; deterministic action decoding and environment seed 0 |

The task has one target container, a basket, and five distractor products. It supports observing failure to approach, grasp, transport, or deposit the target. Its supplied atlas contains visible product lettering and label artwork, allowing a bounded change with identifiable protected regions.

The [previous object-suite report](../final_patches/libero_task_suite_relaxed_v1/object_suite/asr_report.json) includes this task at state 22: clean and cat-seed conditions succeeded, and the separate patch condition failed. This is prior exploratory evidence, not a baseline rate for the new study. State 22 is excluded from this pilot. Selection of this task is exploratory; final conclusions apply to this task and the reserved states.

## Allowed appearance changes

Edit decorative background and food illustrations on one existing label panel. Protect the product name, logo, nutrition text, lid, rim, and exposed container material. Preserve the original object geometry, collision shapes, dynamics parameters, task definition, cameras, lights, and all other objects.

First inspect the mesh's texture coordinates and define the label-panel crop, editable mask, and protected mask. Use a 512 × 512 working image for diffusion, then map it back into the original 2048 × 2048 atlas. Copy original pixels outside the editable mask exactly and taper the edit to zero at its boundary. Keep the altered region within 30% of the visual mesh's surface area, measured using mesh triangles and their texture coordinates; atlas pixel area alone is not a surface-area measurement. Freeze and hash these mappings before optimization.

One texture is fixed for an entire rollout and shared across test states. It follows the object through native rendering. Apply an override to the selected instance's texture/material at runtime, with restoration on reset; if texture resources are shared, first give this instance its own binding. Keep the external LIBERO assets intact. MuJoCo supports uploading changed texture pixels with [`mjr_uploadTexture`](https://mujoco.readthedocs.io/en/2.3.1/APIreference.html#mjr-uploadtexture).

The native atlas is fragmented. Generating an arbitrary square image and replacing the entire atlas would not preserve the surface layout. Mapping and protected-text checks are prerequisites.

## Data split and observation collection

| Role | Initial-state indices | Use |
| --- | --- | --- |
| Training | 0–9 | Collect clean trajectories and optimize texture |
| Appearance calibration | 10–14 | Check mapping, visibility, protected regions, metric implementation, and benign reconstruction |
| Validation | 15–19 | Select one candidate and its corresponding benign control |
| Excluded/reserved | 20–29 | Includes previously evaluated state 22 |
| Final test | 30–49 | Twenty paired episodes per condition, opened after selection |

Before execution, audit existing run manifests for earlier use of test states. If a reserved test state has already influenced this object's texture design or tuning, amend the protocol before training and obtain new held-out initializations. Do not rename familiar states as new test data.

Collect full restorable simulator snapshots from original-texture policy rollouts. Save eight approximately equally spaced frames per trajectory, giving at most 80 training and 40 validation snapshots; deduplicate short trajectories and record actual counts. Save the model/configuration hashes, controller and observation state needed for replay, RNG state, segmentation, and camera settings. An exported RLDS RGB image or robot-only proprioceptive vector is insufficient for rerendering the scene.

Retain unsuccessful clean trajectories and record their status. Allocate training samples evenly across initial states. Use only snapshots with at least 100 target-object pixels in the preprocessed 224 × 224 policy view for the optimization objective; record how many survive and their coverage of approach, grasp, and transport. Freeze this filter before candidate scoring. Final rollouts include every designated test state, regardless of visibility or success.

## Training and candidate selection

Reuse the frozen generator, frozen policy, diffusion anchor, and [existing black-box score estimator](../dura/attack.py). Add a renderer that restores a snapshot, applies a candidate atlas, renders the scene, and obtains the policy action without advancing physics. Native MuJoCo rendering does not supply the current PyTorch overlay's gradient path.

Proposed bounded search:

- 200 DDIM steps, strength 0.5: 100 attack updates per trajectory.
- Anchor weight 0.2; empty prompt and guidance scale 1.0, consistent with the current experiment.
- Step sizes 0.75 and 1.5; optimization/diffusion random seeds 0, 1, and 2: six trajectories.
- 32 sampled latent candidates per update, four training snapshots per candidate, candidate batch size 1, leave-one-out loss centering, and the existing `step` query-alpha convention.
- Evaluate the final PNG of each trajectory. Do not select intermediate checkpoints based on test rollouts.

This budgets `6 × 100 × 32 × 4 = 76,800` optimization policy-frame queries, plus explicitly logged calibration and validation calls. Benchmark a short segment before launching the full search to estimate runtime. These step sizes and sample counts are pilot choices; numerical failures are recorded, and an unsuccessful search does not trigger automatic budget expansion.

Minimize the same decoded-action target loss as the existing patch experiment: target `[0,0,0,0,0,0,0]` in the adapter's action coordinates, with weights `[1,1,1,0.5,0.5,0.5,0.2]`. This encourages little motion and the target gripper state. The existing gripper conversion yields executed target `[0,0,0,0,0,0,1]`. Task failure is the primary outcome; closeness to this action target is a secondary measure.

For each of the three diffusion seeds, also produce a zero-attack diffusion reconstruction using identical generation settings and masking. It is the paired benign control for candidates with that seed. The original label remains the image prior; this pilot introduces no separate animal graphic.

First reject candidates failing the appearance criteria below. Require their corresponding benign control and the original condition each to succeed on at least four of the five validation states. Among eligible candidates, choose the lowest mean validation target-action error, breaking ties by lower texture distortion, then fixed configuration order. Freeze the selected texture, its benign control, masks, and report hashes before opening test results. Record all six outcomes. If none qualifies, report that result without weakening constraints using test feedback.

## Appearance criteria

The following are project-specific pilot bounds, not thresholds from the paper. Their implementations are fixed before adversarial selection. Apply the applicable appearance gates and visual review to both the benign and adversarial textures; reject a candidate if its paired benign texture fails them. For the benign reconstruction, the original render is the SSIM reference; for the adversarial texture, its paired benign render is the reference.

| Check | Acceptance criterion |
| --- | --- |
| Protected atlas pixels | Exact equality to original pixels outside the editable mask |
| Texture distortion | RGB RMSE ≤ 0.075 on editable, used texels, relative to both the original and paired benign texture; RGB normalized to [0,1] |
| Rendered object similarity | SSIM ≥ 0.95 relative to the paired benign render on every eligible calibration crop; use identical object bounding boxes with a 4-pixel border, resized to 128 × 128 |
| Texture continuity | Added seam energy at edited boundaries/UV joins ≤ 25% of the original energy plus `1/255`, using mean absolute RGB differences on corresponding surface edges, with the definition fixed during calibration |
| Surface budget | At most 30% of the object's visual surface area is editable |
| Visual review | Recognizable alphabet-soup packaging, intact product identity, plausible artwork, and no conspicuous noise, text corruption, stretching, or visible seams |

Review a fixed six-view object contact sheet and the calibration trajectory views with candidate identifiers hidden and before viewing adversarial rollout outcomes. Log the reviewer and decisions. Numeric similarity alone does not establish naturalness. Low-visibility crops are marked unavailable rather than counted as passes; require eligible crops spanning at least three calibration trajectories. Also report metrics relative to the original render, the object-mask area, and the visible edited area after policy cropping.

These surface and appearance budgets differ from a fixed 50 × 50 image patch. A direct comparison with prior patch ASR would require matching visible area and evaluation states.

## Evaluation and interpretation

Run the following three conditions on each of the twenty test states, for 60 final rollouts:

| Condition | Appearance |
| --- | --- |
| Original | Unmodified LIBERO texture |
| Benign | Selected candidate's paired zero-attack diffusion texture |
| Adversarial | Selected optimized texture |

Use the same settled simulator state, camera, lighting, horizon, and policy settings for all three conditions. Reset the gripper's cached command to its constructor value: robosuite 1.4's soft reset retains this command, which otherwise makes the settling trajectory depend on the previous episode. Record state hashes and verify a canonical original-texture render before applying the condition. Different condition textures intentionally produce different RGB images, so the existing requirement for identical attacked initial-image hashes must be replaced. Re-render immediately after each texture upload to avoid stale observations. Restore textures between episodes, balance condition order across states, and accept reruns only for documented technical mismatches, independently of outcomes.

Before policy evaluation, run fixed-action replay checks under original and altered textures. Dynamics parameters must match exactly, and state trajectories must match within a predeclared numerical tolerance (maximum absolute state difference `1e-8` on the same simulator/backend). A failure stops the pilot for investigation.

Report counts and rates for:

- Raw failure rate in every condition, using LIBERO's success predicate and 280-step horizon.
- Primary conditional ASR: adversarial failures among states where both original and benign rollouts succeed.
- Paired increase in failure rate over the benign condition and the original condition, including both newly induced failures and recoveries.
- The eligible denominator and a descriptive 95% Wilson interval for conditional ASR; paired outcome tables retain all twenty states.
- Failure categories from recorded rollouts: no approach, failed grasp, dropped object, wrong object, or failed placement; allow ambiguous cases.
- Final appearance metrics, runtime, render count, policy query count, and all candidate selection scores.

A promising pilot requires all appearance criteria, at least 16/20 successes for each baseline, at least 16 states where both baselines succeed, conditional ASR ≥ 50%, and at least a 20-percentage-point increase in failure rate over the benign condition. These are advance/stop criteria, not predicted results. If the controls fail too often, the evidence cannot distinguish the desired attack effect reliably. If appearance passes but the attack effect is small, report a valid negative pilot. Twenty states on one task cannot establish a general 50% attack rate; any follow-up uses a new frozen evaluation set.

## Implementation deliverables and storage

Implemented components:

1. `dura/object_texture.py`: instance/material resolution, atlas mapping, masks, texture upload/restoration, and surface-area accounting.
2. `dura/collect_object_states.py`: trajectory collection and exact state/render replay checks.
3. `dura/train_object_texture.py`: native-rendering query objective, anchored diffusion search, appearance gates, and selection.
4. `dura/evaluate_object_texture.py`: paired three-condition rollouts, physics checks, failure metrics, and provenance.
5. `configs/object_texture_alphabet_soup.yaml`: the frozen task, splits, search budget, and acceptance criteria.

The existing modules remain the reference for policy preprocessing, the action objective, and diffusion updates. `dura/object_texture_pilot.py` provides the stage entry points, and `scripts/submit_object_texture.py` prepares immutable code/configuration snapshots and submits the GPU stages.

The checked-in mapping selects plain blue label artwork at PNG rectangle `[620,1380,900,1570]`, inside panel `[544,1312,1066,2048]`, with a 16-texel boundary taper. Mesh clipping measures an editable surface fraction of approximately 2.71%, below the 30% ceiling. The compiled MuJoCo 2.3.7 mesh UVs and texture-buffer row order were inspected on CPU; this asset uses top-origin texture rows. The original product lettering remains outside this rectangle. `assets/mapping_preview.png` records the editable area for review.

## Running the implementation

Use the environment and external LIBERO checkout described in the root README. On the existing project setup:

```bash
export DURA_SCRATCH=/scratch/jin7/dura
export HF_HOME=/scratch/jin7/huggingface_cache
export DURA_LIBERO_ROOT=/home/jin7/projects/VLA/roboticAttack/LIBERO
export DURA_DATASETS_ROOT=/scratch/jin7/datasets/VLA
source scripts/runtime.sh
export DURA_PYTHON="$DURA_SCRATCH/venv/bin/python"

# Prepare assets, audit historical test-state use, check cached model shards,
# and freeze source/configuration; this command uses no GPU and submits no job.
"$DURA_PYTHON" scripts/submit_object_texture.py

# Submit the training stage after preparation. Add --account if required.
"$DURA_PYTHON" scripts/submit_object_texture.py --submit
```

The default request is one A100 40 GB, four CPU cores, 48 GB system RAM, and six hours. `--exclude` can avoid an occupied node, `--time` sets the walltime, and `--run-root` / `--final-root` select another experiment. The helper calls `sbatch --test-only` before actual submission and refuses duplicate active submissions. It writes `slurm_train.json` and `logs/train-<jobid>.out` / `.err` under the run directory. After a timeout or failure, inspect the logs and use `--resume --submit` with the same configuration; finished episodes and candidates are reused, and interrupted optimization resumes from the last completed update with its random-generator state.

The Slurm script clears the submission shell's environment and obtains its own allocated GPU. The model and code caches stay in scratch, and source files are copied into the run's `source/` directory so later project edits cannot change a queued job. Stable Diffusion loads the VAE, UNet, and text encoder used by the algorithm; the unused pipeline image-checker weights are omitted. All required weights and pinned OpenVLA processing code must be cached before submission; GPU jobs run offline.

The submission helper also executes the frozen source's `prepare` stage on CPU and verifies that its JSON configuration reloads with exactly the same hash. JSON files use the JSON parser so scientific-notation tolerances retain their numeric types. Runtime errors write a `failed` status and error message to `status.json`. When fixing a startup error, use a new `--run-root` (for example `${DURA_SCRATCH}/runs/object_texture_alphabet_soup_v1_retry1`) to preserve the original logs and frozen code; `--resume` intentionally retains the existing source snapshot.

Before rendering, the pilot recomputes camera and geometry transforms at the current integration state and restores the saved solver warm-start vector. MuJoCo's end-of-step cached transforms can otherwise describe an earlier substep and disagree with a restored state. All pilot conditions use this synchronized observation convention. Native preflight schema 2 checks 80 moving-scene snapshots, each with exact RGB and segmentation replay in the original and a freshly created renderer (160 checks), before any policy optimization.

A short isolated rendering check can precede training:

```bash
"$DURA_PYTHON" scripts/submit_object_texture.py --stage native-check \
  --run-root "$DURA_SCRATCH/runs/object_texture_replay_check" --submit
# After obtaining CHECK_JOB_ID, queue training behind its successful completion:
"$DURA_PYTHON" scripts/submit_object_texture.py --stage train \
  --run-root "$DURA_SCRATCH/runs/object_texture_alphabet_soup_v1_retry2" \
  --afterok CHECK_JOB_ID --submit
```

The check requests one A100 40 GB, four CPUs, 8 GB RAM, and fifteen minutes; training retains its six-hour/48 GB limits. The dependent training job becomes eligible only after the check exits successfully. An invalid dependency ends the dependent job automatically, so a failed rendering check cannot consume a second GPU allocation.

Training performs the native upload/restoration and fixed-action physics checks before collection. It then collects original trajectories, runs the six bounded searches, and evaluates benign validation controls. It exits with `awaiting_visual_review` or `no_eligible_candidate`; it does not hold a GPU while waiting for review. Review sheets hide optimization settings and action scores, and test results do not exist at this point.

```bash
export DURA_RUN="$DURA_SCRATCH/runs/object_texture_alphabet_soup_v1"
"$DURA_PYTHON" -m dura.object_texture_pilot --config "$DURA_RUN/config.json" status

# Inspect assets/mapping_preview.png and every sheet in review_queue.json.
# Record an actual decision for each queued candidate. Example for one candidate:
"$DURA_PYTHON" -m dura.object_texture_pilot --config "$DURA_RUN/config.json" review \
  --candidate seed_0_step_0.75 --decision accept \
  --reviewer YOUR_NAME --notes 'Describe the checked packaging and any artifacts.'

# Requires all queued appearance decisions, and freezes the validation-selected winner.
"$DURA_PYTHON" -m dura.object_texture_pilot --config "$DURA_RUN/config.json" select

# Submit 60 held-out rollouts after review; add --account if required.
"$DURA_PYTHON" scripts/submit_object_texture.py \
  --config "$DURA_RUN/config.json" --stage evaluate --submit
```

The review decisions are bound to the exact textures, masks, and sheets by SHA-256. Changing a decision after test access cannot authorize a new selection. Evaluation balances condition order, validates canonical original-texture states/renders, saves videos and executed actions in scratch, and exports compact final results only after all paired episodes finish. Automated reports leave behavioral failure categories for video annotation rather than guessing them from a success flag.

Palmetto resource and submission syntax follow [Job Submission in Slurm](https://docs.rcd.clemson.edu/palmetto/job_management/submit/) (`docs_palmetto/job_management/submit.md`): `--gpus-per-node` allocates the GPU, the VRAM feature filters eligible devices, and `sbatch --test-only` checks a submission without allocating a job.

Use `/scratch/jin7/huggingface_cache/` for model checkpoints and `${DURA_SCRATCH}/runs/object_texture_alphabet_soup_v1/` for snapshots, candidates, logs, and videos (`DURA_SCRATCH` defaults to `/scratch/jin7/dura`). Keep only code, the frozen study specification, selected original/benign/adversarial texture assets, the editable mask/mapping, compact view previews, and final reports in the project. A future final output directory is `final_patches/object_texture_alphabet_soup_v1/`; add it to the publication allowlist only after inspecting the completed artifacts.
