# Natural diffusion patches for OpenVLA on LIBERO

Independent implementation of the DURA approach from *Hidden in Plain Sight*, focused on the current relaxed cat-seed experiments. It optimizes a patch through frozen Stable Diffusion v1.5 and a frozen OpenVLA-7B policy, then measures task failures in closed-loop LIBERO simulation.

This checkout contains the training and evaluation code, one reproducible seed, and **14 evaluated final patches**: ten task-specific Spatial patches and four suite-wide patches. Model weights, datasets, candidate trajectories, tensor checkpoints, and rollout logs belong in scratch storage.

![Selected natural cat patch](final_patches/libero_task_suite_relaxed_v1/spatial_suite/patch.png)

## Recorded results

ASR means the fraction of episodes where the task fails. **Conditional ASR** counts adversarial failures only among episodes where both the clean policy and unmodified cat-seed control succeed.

| Suite-wide patch | Episodes | Patch ASR | Seed ASR | Clean ASR | Conditional failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spatial | 10 | 70% | 10% | 10% | 5/8 (62.5%) |
| Object | 10 | 30% | 0% | 20% | 1/8 (12.5%) |
| Goal | 10 | 40% | 40% | 50% | 1/5 (20%) |
| Long (`libero_10`) | 10 | 80% | 60% | 40% | 2/3 (66.7%) |

The ten separate Spatial patches achieved 21/50 task failures (**42% raw ASR**) and 15/36 conditional failures (**41.7%**). Per-task raw ASRs, for task IDs 0–9, are `[40, 80, 20, 100, 100, 40, 40, 0, 0, 0]%`.

[Complete result table and patch links](final_patches/libero_task_suite_relaxed_v1/README.md) · [Machine-readable summary](final_patches/libero_task_suite_relaxed_v1/summary.json) · [Visual review](final_patches/libero_task_suite_relaxed_v1/visual_review.json)

Task-specific evaluations use states 22–26; suite-wide evaluations use state 22 for each of ten tasks. The Spatial suite patch was selected in an earlier relaxed experiment using 40 validation frames and re-evaluated on these fresh states. New suite runs use all 160 validation frames. The small episode counts and baseline failures limit how precisely these rates generalize. Candidate selection uses held-out action error before rollout results are observed.

## 1. Configure storage and install

Run commands from the repository root. The tested environment is Python 3.10, CUDA 12.1 PyTorch 2.2.2, and an A100 with 80 GB. Training needs about 36 GB GPU memory. One evaluator uses about 16 GB; increase evaluation workers only when memory permits.

```bash
export DURA_SCRATCH=/scratch/jin7/dura
export HF_HOME=/scratch/jin7/huggingface_cache
source scripts/runtime.sh

# Creates the Python environment and installs the pinned training/simulator dependencies.
bash scripts/setup_env.sh --evaluation
export DURA_PYTHON="$DURA_SCRATCH/venv/bin/python"
```

For another machine, set `DURA_SCRATCH` and `HF_HOME` to suitable writable directories before sourcing the runtime. `DURA_BASE_PYTHON` can select a Python 3.10 executable when creating the environment. The setup script uses the CUDA 12.1 wheel index; adapt that command if your CUDA environment differs.

Install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) outside the checkout. The recorded experiments used commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`:

```bash
export DURA_LIBERO_ROOT="$DURA_SCRATCH/external/LIBERO"
mkdir -p "$DURA_SCRATCH/external"
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$DURA_LIBERO_ROOT"
git -C "$DURA_LIBERO_ROOT" checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
"$DURA_PYTHON" -m pip install -e "$DURA_LIBERO_ROOT" --no-deps
source scripts/eval_runtime.sh
```

To use an existing LIBERO checkout, export its path instead of cloning. `eval_runtime.sh` creates the benchmark configuration and EGL caches in scratch. If reusing the existing local setup, also set `DURA_DATASETS_ROOT=/scratch/jin7/datasets/VLA` so its existing benchmark configuration agrees.

Cache the pinned model revisions before submitting an offline GPU run:

```bash
source scripts/runtime.sh
"$DURA_PYTHON" scripts/cache_models.py
```

The model-cache command downloads Stable Diffusion v1.5, all four suite-specific OpenVLA checkpoints, and the pinned Transformers remote code. To cache only Spatial, add `--suites spatial_suite`. Every model revision is recorded in [configs/libero_relaxed.yaml](configs/libero_relaxed.yaml) and [dura/task_experiments.py](dura/task_experiments.py). Once cached, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` enable offline runs.

## 2. Export training observations

Use the preprocessed [OpenVLA LIBERO RLDS dataset](https://huggingface.co/datasets/openvla/modified_libero_rlds). Place its four `*_no_noops/1.0.0` directories in scratch storage. Export a bounded sample:

```bash
export DURA_DATASETS_ROOT=/scratch/jin7/datasets/VLA
source scripts/runtime.sh
for suite in libero_spatial libero_object libero_goal libero_10; do
  "$DURA_PYTHON" -m dura.prepare_rlds \
    --dataset "$DURA_DATASETS_ROOT/${suite}_no_noops/1.0.0" \
    --output "$DURA_SCRATCH/data/$suite"
done
```

The exporter keeps whole episodes separate: eight training and two validation episodes per instruction, with eight frames per episode. A complete ten-task export has 640 training and 160 validation frames. Each export includes `train.jsonl`, `val.jsonl`, RGB frames, a diagnostic scene crop, and a summary in scratch. A completed export can be reused; a partial export requires a new output directory.

The supplied [cat seed](seed_patches/cat_01_soft.png) is the exact 512-by-512 RGB PNG used in the recorded runs. Its SHA-256 and generation prompt are in [generation_manifest.json](seed_patches/generation_manifest.json).

## 3. Train new patches

Choose a new run name and a new final output directory:

```bash
source scripts/eval_runtime.sh
export DURA_RUN="$DURA_SCRATCH/runs/libero_relaxed_new"
export DURA_FINAL="$PWD/final_patches/libero_relaxed_new"

"$DURA_PYTHON" -m dura.task_experiments prepare \
  --scratch "$DURA_RUN" --final-root "$DURA_FINAL" \
  --libero-root "$DURA_LIBERO_ROOT"

source scripts/runtime.sh
"$DURA_PYTHON" -u -m dura.task_experiments train --scratch "$DURA_RUN"
```

This prepares ten independent Spatial task patches and one suite-wide patch for each of Spatial, Object, Goal, and Long. Use `--experiments` during preparation to run a subset, such as `--experiments spatial_task_03 object_suite`. `--seed` accepts another 512-by-512 animal image; the appearance bounds then measure changes relative to that seed.

Each experiment searches step sizes 0.75, 1.0, 1.5, and 2.0. The anchor weight is 0.2, DDIM uses 200 steps with strength 0.5, and each candidate receives 100 optimization updates. The fixed patch footprint is `[0, 174, 50, 50]` in a 224-by-224 observation. The policy's center crop may clip part of this lower-left footprint.

A candidate must pass every bound before selection by lowest validation decoded-action error, then target-token cross entropy:

| Appearance metric | Relaxed bound |
| --- | ---: |
| RGB RMSE relative to the cat seed | 0.075 |
| TV at the rendered 50-by-50 footprint | 0.05 |
| Added boundary seam | 6.25 |
| Local SSIM drop relative to the seed-rendered scene | 0.0125 |

These bounds are additional safeguards, relaxed by 25% from the earlier project profile; they are not thresholds stated by the paper. The diffusion anchor and frozen generator provide the image prior. Inspect each final PNG and preview as well: numerical gates do not guarantee naturalness. Comparison to the original tabletop crop is diagnostic and does not decide acceptance.

Training writes candidate histories, latents, and tensors to `DURA_RUN`. Only the selected PNG, preview, configuration, and compact reports are exported to `DURA_FINAL`. The trainer refuses to overwrite an existing final patch. Completed experiments are skipped when resuming `train`.

For one suite-wide Spatial patch without a task plan, the base configuration also supports:

```bash
bash scripts/train_natural_patch.sh configs/libero_relaxed.yaml
```

That command requires the Spatial data export and creates `final_patches/single_spatial`.

## 4. Evaluate and generate ASR reports

```bash
source scripts/eval_runtime.sh
"$DURA_PYTHON" -u -m dura.task_experiments evaluate \
  --scratch "$DURA_RUN" --libero-root "$DURA_LIBERO_ROOT" --workers 1
"$DURA_PYTHON" -m dura.task_experiments report --scratch "$DURA_RUN"
```

Use `--workers 4` on an otherwise idle 80 GB A100; one worker is the default. Controls and adversarial rollouts use the same task/state IDs. The evaluator verifies exact pre-action RGB image hashes and performs guarded rechecks when simulator history changes a rendered frame. It accepts the first exact match independently of outcome. If isolated seed and adversarial images agree while the shared clean process differs, it runs a guarded isolated clean episode. Unmatched episodes stop evaluation instead of producing a misleading ASR.

Preparation accepts `--task-trials`, `--suite-trials`, and `--offset` to enlarge or shift the evaluation set. Published settings are five task trials and one suite trial at offset 22. Model actions, raw rollouts, matching provenance, and videos requested through the low-level evaluator stay in scratch. Completed ASR reports and raw batches are reused on resume; incomplete raw batches require a fresh run directory.

## Evaluate the included final patches

Training data is unnecessary for this mode. After environment, LIBERO, and checkpoint setup:

```bash
source scripts/eval_runtime.sh
export DURA_RUN="$DURA_SCRATCH/runs/verify_published_new"
"$DURA_PYTHON" -m dura.task_experiments prepare \
  --scratch "$DURA_RUN" --reuse-final-patches \
  --final-root "$PWD/final_patches/libero_task_suite_relaxed_v1" \
  --libero-root "$DURA_LIBERO_ROOT"
"$DURA_PYTHON" -u -m dura.task_experiments evaluate \
  --scratch "$DURA_RUN" --libero-root "$DURA_LIBERO_ROOT" --workers 1
"$DURA_PYTHON" -m dura.task_experiments report --scratch "$DURA_RUN"
```

This reads the included PNGs and writes the new ASR reports and result table under `DURA_RUN`; the published reports remain intact. Add `--experiments spatial_suite` during preparation to evaluate only that patch. A new state offset, such as `--offset 27`, avoids reusing the published evaluation states.

## Code and tests

| Component | Purpose |
| --- | --- |
| [task_experiments.py](dura/task_experiments.py) | Prepare task/suite plans; orchestrate training, matched controls, ASR evaluation, and summaries. |
| [natural_train.py](dura/natural_train.py) | Optimize anchored diffusion candidates, enforce appearance gates, select and export a patch. |
| [attack.py](dura/attack.py), [diffusion.py](dura/diffusion.py) | Latent attack updates and frozen Stable Diffusion/DDIM operations. |
| [adapters/openvla.py](dura/adapters/openvla.py), [objective.py](dura/objective.py) | Differentiable OpenVLA preprocessing, target-token loss, and decoded-action queries. |
| [render.py](dura/render.py), [naturalness.py](dura/naturalness.py) | Patch placement and measurable appearance constraints. |
| [prepare_rlds.py](dura/prepare_rlds.py), [data.py](dura/data.py) | Episode-disjoint RLDS export and image/manifest loading. |
| [evaluate_libero.py](dura/evaluate_libero.py), [compare_rollouts.py](dura/compare_rollouts.py) | Closed-loop simulation and raw/conditional ASR calculation. |
| [scripts/](scripts/) | Scratch runtime configuration, environment installation, model caching, and single-patch training. |

[Protocol details](docs/libero_task_suite_experiments.md) explain selection, action coordinates, and report files. Final per-patch `config.json` files use environment variables for scratch/cache paths and relative paths for the shared seed. They retain scientific settings and pinned model revisions; archived execution paths belong to the original run backup.

Run the CPU regression tests without downloading models:

```bash
source scripts/runtime.sh
"$DURA_PYTHON" -m unittest discover -s tests -v
```

`.gitignore` excludes environments, caches, datasets, rollout files, model weights, tensor artifacts, and newly generated final directories. The included `libero_task_suite_relaxed_v1` result set is explicitly retained. To publish a new selected result set, review its contents and add a corresponding exception to `.gitignore`.
