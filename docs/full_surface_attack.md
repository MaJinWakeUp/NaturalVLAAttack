# Full-surface object-texture follow-up

This experiment replaces the complete alphabet-soup object's 2048×2048 RGB atlas,
including the label and lid, on the same LIBERO Object task 0. Every atlas pixel
is editable with opacity one. The surface budget is 100%, the boundary taper is
zero, and no product text or artwork is protected. RMSE, SSIM and seam energy are
reported but never reject a texture. There is no naturalness claim or visual-review
pause. Geometry, collisions, physics, cameras, other objects and the policy remain
fixed. Native texture upload, exact state/render pairing and physics checks remain
mandatory.

The preceding constrained pilot achieved 1/11 (9.1%) ASR among episodes where both
controls succeeded. Its raw adversarial failure rate was 15%, compared with 25%
for each control. That is a negative pilot result.

## Search and selection

- Seven unoptimized whole-object controls: black, white, red, green, blue,
  checkerboard and fixed low-frequency RGB noise.
- Four Stable Diffusion v1.5 candidates: seeds 0/1 × step sizes 2/8, 100 DDIM
  steps with strength 0.8, giving 80 updates per candidate. The original complete
  atlas supplies the initial image. The diffusion anchor weight is zero.
- Each update uses 32 black-box samples over four balanced original-trajectory
  snapshots. Minimize negative weighted squared action difference from the frozen
  policy's original-image actions, encouraging deviation from the clean action
  rather than the previous fixed zero-action target. A gradient RMS ceiling of one
  controls numerical spikes; it does not limit texture distortion. This is an
  experimental modification of the prior DURA update, not an exact paper reproduction.
- The optimization budget is 40,960 policy-frame queries, plus separately recorded
  reference-action, scoring, validation and final-rollout queries.
- Run all five validation episodes for every candidate. Rank candidates by induced
  failures among successful original controls, then longer mean rollout, greater
  action divergence and fixed candidate ID. A simple control can win; the report
  identifies whether the selected texture was diffusion-optimized.
- Freeze one winner before opening final test episodes. For diffusion candidates,
  the paired benign control is the zero-attack reconstruction using the same seed.
  For simple controls, the benign control is the original texture.

## Evaluation states and ASR

Original snapshots from training states 0–9, calibration 10–14 and validation
15–19 can be reused after checking policy, simulator, split and observation settings.
The implementation copies these observations into scratch and checks their hashes
and native replay. It does not copy prior test episodes.

The earlier test states 30–49 have influenced this follow-up and are excluded from
its final test. State 22 was used by an earlier exploratory experiment. The new
test uses the nine remaining states: **20, 21, 23, 24, 25, 26, 27, 28, 29**. The
historical-state audit also recognizes native object-texture ASR reports.

Run original, benign and selected adversarial textures on each new state: 27 final
rollouts with a 280-step horizon. Primary ASR is adversarial failures among states
where the original texture succeeds. The report explicitly sets
`conditional_reference: original` and also reports ASR conditioned on both controls
succeeding, all three raw failure rates, paired differences and a Wilson interval.
This distinguishes general appearance disruption from improvement over the benign
reconstruction. Nine initializations support only a small exploratory result.

Advance criteria require at least seven original successes and eligible states,
at least 50% primary ASR, and at least a 20-percentage-point failure increase over
the benign control. The outcome is measured; no success rate is guaranteed.

## Run on Slurm

Use the environment and checkpoint cache from the main README:

```bash
export DURA_SCRATCH=/scratch/jin7/dura
export HF_HOME=/scratch/jin7/huggingface_cache
export DURA_LIBERO_ROOT=/home/jin7/projects/VLA/roboticAttack/LIBERO
export DURA_DATASETS_ROOT=/scratch/jin7/datasets/VLA
export DURA_PYTHON="$DURA_SCRATCH/venv/bin/python"

# Prepare, freeze code, check cached models and submit the complete pipeline.
"$DURA_PYTHON" scripts/submit_object_texture.py \
  --config configs/object_texture_full_surface.yaml \
  --stage full-run --time 08:00:00 --submit

# Resume after checking a stopped job. Use the frozen JSON configuration.
"$DURA_PYTHON" scripts/submit_object_texture.py \
  --config "$DURA_SCRATCH/runs/object_texture_full_surface_v1/config.json" \
  --stage full-run --time 08:00:00 --resume --submit
```

Add `--account` and `--exclude` as appropriate. The request uses one A100 40 GB,
four CPUs and 48 GB system memory; the eight-hour allowance is based on the
previous roughly 40 seconds per 32×4-query update, plus validation and final
rollouts. The single job automatically proceeds through preflight, search,
validation selection and final ASR. Updates and completed episodes are checkpointed.
There is no separate appearance-review approval step for this profile.

The local default `full_surface.reuse_collection` points to the completed constrained
pilot. Remove that optional setting when its scratch collection is unavailable;
the job will collect the original trajectories again.

Intermediates, candidates, logs, checkpoints and videos stay under
`/scratch/jin7/dura/runs/object_texture_full_surface_v1/`. Model weights remain in
`/scratch/jin7/huggingface_cache/`. After evaluation, final textures, metrics, a
preview and provenance are exported to `final_patches/object_texture_full_surface_v1/`.

## Code

| File | Function |
| --- | --- |
| `configs/object_texture_full_surface.yaml` | Frozen full-atlas mapping, search, state splits and reporting rules. |
| `dura/full_surface_attack.py` | Simple controls, reference actions, diffusion search, closed-loop validation selection and automatic evaluation. |
| `dura/object_texture.py` | Whole-atlas composition, native uploads, unchanged physics and historical-state audit. |
| `dura/train_object_texture.py` | Resumable black-box updates and untargeted action-divergence objective. |
| `dura/evaluate_object_texture.py` | Paired original/benign/adversarial evaluation, both ASR definitions and final export. |
| `scripts/submit_object_texture.py` | Frozen source/configuration, CPU preflight and Slurm submission/resume. |

Sources: [Job Submission in Slurm](https://docs.rcd.clemson.edu/palmetto/job_management/submit/)
(`docs_palmetto/job_management/submit.md`), supporting GPU allocation and walltime flags.
