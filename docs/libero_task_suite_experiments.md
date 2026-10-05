# Experiment protocol and artifact reference

The current implementation targets OpenVLA on LIBERO. The model, generator, placement, objective, and appearance limits are recorded in `configs/libero_relaxed.yaml`. The suite checkpoint revisions are centralized in `dura/task_experiments.py`.

## Optimization and selection

For each task-specific patch, the runner filters the Spatial RLDS export by exact task instruction: 64 training and 16 selection frames, from separate episodes. New suite-wide runs use all ten tasks, with 640 training and 160 selection frames. The published Spatial suite patch was selected in an earlier run on 40 balanced validation frames; its saved configuration retains that setting. The attack optimizer keeps the policy and generator weights frozen, updates the image latent, and mixes each denoising step with a clean diffusion anchor.

Four step sizes are tried independently. After PNG quantization, the trainer measures seed-relative RMSE, rendered TV, added boundary seam, and local SSIM drop. Only candidates passing all four bounds are scored on recorded selection frames. Lowest weighted decoded-action MSE wins, with target cross entropy as the tie-breaker. Rollout ASR is measured after selection and never ranks candidates.

The target in policy action coordinates is `[0,0,0,0,0,0,0]`, with weights `[1,1,1,0.5,0.5,0.5,0.2]`. OpenVLA's gripper value is converted for LIBERO execution, so action precision is measured against executed target `[0,0,0,0,0,0,1]` with tolerance 0.05. ASR measures task failure independently of whether every action closely matches this target.

The RGB seed is an animal photograph placed in the scene. The naturalness bounds constrain changes relative to that seed. They do not require the animal image to blend into the original tabletop. The original scene crop is an additional diagnostic reference. Visual inspection remains necessary.

## Evaluation and matching

The published task-specific patches use five states (22–26); each suite-wide patch uses state 22 for ten tasks. The existing Spatial suite-wide patch is reused from earlier training and evaluated on fresh states. Increasing the episode count or using another seed creates a new experiment.

The evaluator runs clean, benign-seed, and adversarial conditions and hashes the RGB observation before the first policy action. Every matched comparison must have the same episode key and initial hash in all three conditions. A mismatching rollout is rerun with a guard that aborts before policy actions if its hash differs. The first exact match is used regardless of the outcome; at most twenty guarded attempts are made.

Shared multi-task simulator processes can retain history that changes a rendered starting image. When isolated seed and adversarial observations agree, a guarded isolated clean rollout can replace the shared control for that same task/state. The replacement and all guarded attempts are logged in scratch matching provenance.

Raw ASR is the fraction of adversarial task failures. Conditional ASR uses only episodes where clean and seed both succeed. Reports include the denominator, control failure rates, action precision, and per-pair success outcomes. A missing exact match stops evaluation. A conditional rate with zero eligible controls is undefined.

## Final artifacts

The published result set is `final_patches/libero_task_suite_relaxed_v1/`:

| File | Contents |
| --- | --- |
| `summary.json` | Portable links, patch hashes, raw/conditional ASR, counts, and experiment protocol. |
| `README.md` | Result table with links to each patch and ASR report. |
| `visual_review.json` | Visual-review observations for selected patches. |
| `<experiment>/patch.png` | The exact final RGB patch used during evaluation. |
| `<experiment>/preview.png` | A patched validation observation. |
| `<experiment>/config.json` | Scientific settings and pinned model versions, with portable storage paths. |
| `<experiment>/report.json` | Candidate step sizes, gate metrics, action scores, and selected candidate. |
| `<experiment>/asr_report.json` | Clean/seed/adversarial comparison and exact patch/seed hashes. |
| `<experiment>/provenance.json` | Original training code hashes, model versions, and seed provenance. |

One exact prepared seed is shared through `seed_patches/cat_01_soft.png`. Its generation metadata is in `seed_patches/generation_manifest.json`. No latent tensor is needed to apply a final PNG.

New-run histories, candidate tensors, exported data, raw rollouts, and full matching provenance belong under `DURA_SCRATCH`. Saved-patch evaluation writes new reports into that run directory so the published results remain intact. The root README contains the complete commands.
