# Optimize the completed benign diffusion latent

The seed-0 and seed-1 benign reconstructions supply the appearance references for
this follow-up. Generate each reference once with the existing frozen Stable
Diffusion v1.5 model: original atlas → VAE encode → strength-0.8 noise → 80 DDIM
steps. Save the final latent, its decoded 512×512 panel and its 2048×2048 atlas.
Optimize around this completed latent without further diffusion steps or fresh
diffusion noise during optimization.

## Parameterization and objective

Keep the VAE, UNet and OpenVLA weights frozen. The trainable parameter is an
8×8 adjustment grid for each of the four latent channels: 256 variables, interpolated
to the 64×64 latent grid. Project the interpolated adjustment to an RMS radius of
0.05 or 0.15 around the completed benign latent.

Let `z_b` be that latent, `D` the frozen VAE decoder, and `I_b` the saved benign
panel. The optimized panel is:

```text
z = z_b + bilinear_upsample(theta)
I = clip(I_b + D(z) - D(z_b), 0, 1)
```

The small decoder-residual correction makes zero adjustment reproduce the saved
8-bit benign image exactly. There is no repeated denoising operation pushing the
optimization toward a different image.

Use eight antithetic Gaussian direction pairs per update, evaluate both signs
through native texture rendering, and estimate a zeroth-order gradient. Apply
Adam updates with learning rate 0.01, probe standard deviation 0.025 and gradient
norm cap 10. The optimizer, parameter field, random-generator state and query
counters are checkpointed after every update.

The policy loss is **negative teacher-forced cross entropy of the benign policy's
action tokens**. Minimize it to reduce confidence in those benign actions. This
gives a continuous confidence signal even when the decoded, discrete action is
unchanged by a small perturbation. Native rendering remains nondifferentiable;
the optimizer queries scalar losses rather than backpropagating through MuJoCo.
Add `25 × image MSE to I_b + 0.1 × latent-adjustment MSE` to discourage drift.

## Appearance and candidate selection

Every atlas texel remains editable. The appearance reference is the corresponding
benign reconstruction, including its existing changes to product lettering. There
is no requirement to recover the original packaging.

At updates 20, 40 and 60, save a candidate and require:

- RGB RMSE ≤ 0.10 relative to the benign atlas.
- Object-crop SSIM ≥ 0.90 on every eligible calibration crop.
- Seam energy ≤ 1.5 × benign seam energy + `1/255`.
- Unchanged geometry, dynamics and other objects, with exact canonical pairing.

These are numerical proximity checks, not proof of perceptual realism. The
original-label RMSE remains a diagnostic and is not an acceptance limit.

Run all five validation episodes for each numerically accepted candidate. Rank by
new failures where both original and benign controls succeed, then failures
relative to original, smaller benign distortion, longer rollout and fixed ID.
Include the unchanged benign reconstructions as fallback candidates. If no learned
candidate improves on them, report that outcome rather than selecting a damaged
texture or claiming that pre-existing benign failures were optimization successes.

There are four optimizer runs: two seeds × two RMS radii, each 60 updates. At 16
probes × four snapshots per update, the optimization budget is **15,360 policy-frame
queries**, plus reference actions, validation and final rollouts.

## Fresh final initializations

All 50 built-in initial states have now participated in development or evaluation.
The preparation stage therefore freezes 20 new initial states using LIBERO's
native placement sampler, with reset seeds/IDs 1000–1019. It uses no policy queries
or rendered images and rejects nonfinite, duplicate and already-successful states.
The bank, metadata and SHA-256 are saved in scratch before training; a frozen bank
is never silently regenerated. These are new sampler initializations, **not the
original benchmark's pruned test distribution**.

Training, calibration and validation retain the compatible original observations
from states 0–19. Final evaluation opens the new states only after candidate
selection and runs original, benign and selected textures: 60 paired rollouts.
Primary ASR conditions on both controls succeeding; original-conditioned ASR,
raw failures, paired differences and a Wilson interval are also reported.

## Run and inspect

Use the environment variables and runtime from the main README. Preparation needs
the native simulator, so source the evaluation runtime before submitting:

The configuration's optional `full_surface.reuse_collection` points to the original
local pilot's scratch observations. On a fresh installation, remove that setting
or set it to `null`; the GPU job will collect its own original train, calibration
and validation trajectories before optimizing. Set `DURA_LIBERO_ROOT` to your
external checkout and use new output directories for a new experiment.

```bash
export DURA_SCRATCH=/scratch/jin7/dura
export HF_HOME=/scratch/jin7/huggingface_cache
export DURA_LIBERO_ROOT=/home/jin7/projects/VLA/roboticAttack/LIBERO
export DURA_DATASETS_ROOT=/scratch/jin7/datasets/VLA
source scripts/eval_runtime.sh
"$DURA_PYTHON" scripts/submit_object_texture.py \
  --config configs/object_texture_benign_latent.yaml \
  --stage latent-run --time 06:00:00 --submit
```

Add the appropriate `--account` and `--exclude` options. One GPU job handles native
checks, local optimization, validation selection and final ASR, with no separate
manual appearance-review pause. Resume with the frozen scratch `config.json` and
`--stage latent-run --resume --submit`. Interrupted validation checkpoints are
revisited before optimization continues.

Implementation: `dura/benign_latent.py` contains the optimizer and orchestration;
`dura/initial_states.py` freezes the new initializations; the existing renderer,
VAE, native token objective and paired evaluator are reused. Models and large
intermediates stay in scratch. Final textures, metrics, previews and the small
initial-state bank are exported to `final_patches/object_texture_benign_latent_v1/`.

Sources: [Job Submission in Slurm](https://docs.rcd.clemson.edu/palmetto/job_management/submit/)
(`docs_palmetto/job_management/submit.md`), supporting the walltime and submission flags.
