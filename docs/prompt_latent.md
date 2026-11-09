# Prompt or saved-latent object textures

This follow-up generates material textures independently of the original atlas,
then optimizes their completed VAE latent. It uses the same LIBERO Object task 0,
alphabet-soup object, frozen OpenVLA checkpoint, native renderer and paired ASR
evaluator as [benign-latent optimization](benign_latent.md).

## Planned experiment

Configuration: [object_texture_prompt_latent_v1.yaml](../configs/object_texture_prompt_latent_v1.yaml).

| Setting | Previous benign-latent run | This follow-up |
| --- | --- | --- |
| Reference initialization | Original atlas reconstruction | Copper / ceramic material prompts, seeds 0 / 1 |
| Generator | Stable Diffusion v1.5 | Same pinned checkpoint |
| Reference denoising | 80 image-to-image DDIM steps | 50 text-to-image DDIM steps, guidance 7.5 |
| Minimum object-crop SSIM | 0.90 | 0.80 |
| Maximum atlas RGB RMSE | 0.10 | 0.20 |
| Latent adjustment RMS radii | 0.05, 0.15 | 0.25, 0.50 |
| Optimizer updates / learning rate | 60 / 0.01 | 100 / 0.03 |
| RGB MSE / latent MSE penalties | 25 / 0.1 | 2.5 / 0.02 |
| Validation checkpoints | 20, 40, 60 | 25, 50, 75, 100 |
| Fresh final initialization IDs | 1000–1019 | 2000–2029 |

The preceding winner's SSIM 0.993 and RMSE 0.012 were already within the old
limits. This run also increases the permitted latent motion and weakens the
optimization penalties; simply loosening acceptance thresholds would not force
a stronger attack. Several factors change together, so this is a combined
follow-up rather than an isolated ablation of the similarity limits.

All atlas texels remain editable. The seam bound remains 1.5 times the reference
seam energy plus 1/255. Geometry, physics, other objects and canonical pairing
checks remain unchanged. Similarity is measured against each unoptimized material
reference, not against the original packaging.

Two prompts × two radii × 100 updates × 16 probes × four train snapshots gives
**25,600 optimization policy-frame queries**. Four checkpoints per optimizer
produce 16 learned candidates, with two unchanged references as fallbacks.
Candidates passing appearance limits receive five validation rollouts each.
Selection still requires at least four validation states where both original and
reference controls succeed. If neither material meets that gate, the run stops
without opening final test states rather than relaxing the gate after seeing data.

Final evaluation runs 90 paired rollouts: original, selected unoptimized material,
and optimized material on each of 30 new states. The primary ASR counts newly
induced failures among states where both controls succeed. Original-conditioned
ASR, raw failure rates, net failure increase and a Wilson interval are also saved.
The frozen states come from the native placement sampler, not LIBERO's original
pruned test distribution. Do not directly compare their ASR with previous runs
as if they used the same states.

## Initialization modes

`initialization.mode: prompt` samples fixed standard-normal noise with a CPU torch
generator, then performs a complete deterministic DDIM trajectory. No pixel from
the original atlas is encoded into this latent. The original atlas remains a
control condition for evaluation and native rendering checks.

`initialization.prompts` maps string seed IDs to prompts. Alternatively, use one
`initialization.prompt` for all seeds. The supplied prompts describe flat material
scans with even lighting; the negative prompt discourages humans, portraits,
scenes, lettering and watermarks. Prompts cannot guarantee semantic content or
seamlessness. Inspect the generated reference and rendered previews before
describing a texture as natural or free of unwanted content.

For a predefined starting latent, use `mode: saved_latent` and explicitly choose:

- `space: noise`: a standard-normal starting tensor, followed by full DDIM
  denoising under the material prompt. The tensor should have approximately unit
  variance. It is not an intermediate noisy latent from an arbitrary timestep.
- `space: clean`: an already completed, **scaled Stable Diffusion VAE latent**.
  Decode it directly; no noise injection, prompt conditioning or UNet denoising
  is performed. A prompt cannot change a clean latent through decoding alone.

Example override in a copy of the configuration (also choose new output and
final-output directories, bank path and unused final-state IDs):

```yaml
initialization:
  mode: saved_latent
  space: clean
  files:
    '0':
      path: /scratch/jin7/dura/texture_seeds/copper_clean.npy
      sha256: <64-character SHA-256 of that file>
    '1':
      path: /scratch/jin7/dura/texture_seeds/ceramic_clean.npy
      sha256: <64-character SHA-256 of that file>
```

Inputs must be non-pickled `.npy` floating-point arrays of shape `[1,4,64,64]`
for the configured 512×512 working image. For example, an existing benign
reference can be reused by loading its trusted scratch `benign/seed_*/latent.pt`
checkpoint and saving `checkpoint['latent'].numpy()` with `numpy.save`. This
checkpoint's latent is already scaled; do not multiply it by the VAE scale again.
Preparation validates shape, finiteness and file SHA, and freezes copies in the
new run's `assets/initial_latents/`. Resume uses those verified copies. External
input latents are also included in the final export for reproducibility.

`mode: reconstruction` preserves the old image-to-image initialization. Omitting
`initialization` retains that behavior for existing configurations.

Each reference records prompt, negative prompt, seed, guidance, input tensor SHA,
clean latent SHA and whether the original image was encoded. Optimization then
uses the same decoder-residual parameterization and score-based token objective
as the previous run, without further denoising or changes to model weights.

## Prepare, review and submit

Use the environment setup from [the README](../README.md), including
`DURA_LIBERO_ROOT`, `DURA_DATASETS_ROOT`, and the scratch paths. Preparation checks
cached model files, freezes source/configuration and the new initial-state bank,
and reuses only compatible original train/calibration/validation observations.
It does not load model weights or query a policy on the final states.

The supplied `full_surface.reuse_collection` is a local scratch path. For a fresh
checkout without those observations, remove that setting or set it to `null`.
The GPU job then collects the original train/calibration/validation trajectories
before optimization. Set `DURA_LIBERO_ROOT` to the installed LIBERO checkout.

```bash
source scripts/eval_runtime.sh
"$DURA_PYTHON" scripts/submit_object_texture.py \
  --config configs/object_texture_prompt_latent_v1.yaml \
  --stage latent-run --time 06:00:00
```

Review `submission_plan_latent-run.json` under the scratch run directory. The job
requests one A100 40 GB, four CPUs, 48 GB RAM and six hours. After review, rerun
with `--submit` and the account/exclusion options appropriate for the session.
To resume an interrupted job, use its frozen `config.json` with
`--stage latent-run --resume --submit` after checking that no job is active.

Outputs:

- `benign/seed_*/texture.png`, `latent.pt`, `initialization.json`: unoptimized
  reference and provenance.
- `optimizers/*/checkpoint.pt`, `history.json`: resumable parameter search.
- `candidates/*/preview.png`, `report.json`: rendered comparisons and validation.
- `selection.json`: frozen winner, before final-test access.
- `asr_report.json`, `test/`: final metrics, raw paired rollouts and videos.
- `final_patches/object_texture_prompt_latent_v1/`: selected images, preview,
  reports, configuration, provenance and the small frozen state bank.

Checkpoints remain under `/scratch/jin7/huggingface_cache/`. Intermediate files,
logs and videos stay under `/scratch/jin7/dura/`.

Implementation: [texture_initialization.py](../dura/texture_initialization.py)
handles independent references and frozen input latents;
[diffusion.py](../dura/diffusion.py) adds negative prompt conditioning and full
noise-to-latent DDIM generation; [benign_latent.py](../dura/benign_latent.py)
optimizes, validates and selects; [evaluate_object_texture.py](../dura/evaluate_object_texture.py)
computes paired final ASR and exports results. CPU tests cover deterministic
generation, clean versus noise semantics, checksums, prompt guidance, complete
DDIM trajectories and separation from prior evaluation IDs.

Sources:
- [Job Submission in Slurm](https://docs.rcd.clemson.edu/palmetto/job_management/submit/)
  (`docs_palmetto/job_management/submit.md`): resource flags, GPU selection and
  the A100 40 GB feature constraint.
