"""Prepare, train, evaluate, and summarize relaxed LIBERO task/suite patches.

Large manifests, checkpoints, rollout videos, and logs stay below ``--scratch``.
Only selected patches and compact reports are written below ``--final-root``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from .config import PROJECT_ROOT, read_config, scratch_root
from .compare_rollouts import key, summarize_conditions


RELAXED_LIMITS = {
    "patch_rgb_rmse_max": 0.075,
    "patch_tv_max": 0.05,
    "boundary_increase_max": 6.25,
    "local_ssim_drop_max": 0.0125,
}
STEP_SIZES = [0.75, 1.0, 1.5, 2.0]
SPATIAL_POLICY = {
    "checkpoint": "openvla/openvla-7b-finetuned-libero-spatial",
    "revision": "962318cec55ac10993ff0f5f43eda9a270b4c873",
    "unnorm_key": "libero_spatial",
}
SUITES = {
    "spatial_suite": {"suite": "libero_spatial", **SPATIAL_POLICY},
    "object_suite": {
        "suite": "libero_object",
        "checkpoint": "openvla/openvla-7b-finetuned-libero-object",
        "revision": "287d6cfdf12d07b1449505f66d9bf3550257e9b3",
        "unnorm_key": "libero_object",
    },
    "goal_suite": {
        "suite": "libero_goal",
        "checkpoint": "openvla/openvla-7b-finetuned-libero-goal",
        "revision": "fa5ae1e7509348889295bba8e08621d8b55e9baf",
        "unnorm_key": "libero_goal",
    },
    "long_suite": {
        "suite": "libero_10",
        "checkpoint": "openvla/openvla-7b-finetuned-libero-10",
        "revision": "80970322773f81baa2e22fe495d0487b93a05cfa",
        "unnorm_key": "libero_10",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def absolute_rows(source: Path, instruction: str | None = None) -> list[dict]:
    result = []
    for row in read_rows(source):
        if instruction is not None and row["instruction"] != instruction:
            continue
        row = dict(row)
        for field in ("image", "wrist_image"):
            if field in row:
                row[field] = str((source.parent / row[field]).resolve())
        result.append(row)
    if not result:
        raise ValueError(f"No rows selected from {source}: {instruction}")
    return result


def task_languages(libero_root: str, suite: str) -> list[str]:
    sys.path.insert(0, str(Path(libero_root).resolve()))
    from libero.libero import benchmark

    benchmark_obj = benchmark.get_benchmark_dict()[suite]()
    return [benchmark_obj.get_task(i).language for i in range(benchmark_obj.n_tasks)]


def variant(base: dict, *, manifest: Path, validation: Path, seed: Path,
            scene_seed: Path, output: Path, final: Path, policy: dict) -> dict:
    cfg = json.loads(json.dumps(base))
    cfg.update({
        "manifest": str(manifest),
        "validation_manifest": str(validation),
        "validation_frames": len(read_rows(validation)),
        "seed_patch": str(seed),
        "output": str(output),
        "final_output": str(final),
    })
    cfg["policy"].update(policy)
    cfg["policy"]["local_files_only"] = base.get("policy", {}).get("local_files_only", False)
    cfg["diffusion"]["local_files_only"] = base.get("diffusion", {}).get("local_files_only", False)
    cfg["naturalness"].update({
        "fixed_limits": False,
        "limits": dict(RELAXED_LIMITS),
        "step_sizes": list(STEP_SIZES),
        "scene_reference_patch": str(scene_seed),
        "constraint_profile": "relaxed_v2",
    })
    cfg["naturalness"].pop("comparison_patches", None)
    return cfg


def prepare(args) -> None:
    scratch, final_root = Path(args.scratch).resolve(), Path(args.final_root).resolve()
    if scratch.exists() and any(scratch.iterdir()):
        raise FileExistsError(f"Preparation root is not empty: {scratch}")
    if not args.reuse_final_patches and final_root.exists():
        raise FileExistsError(final_root)
    scratch.mkdir(parents=True, exist_ok=True)
    base = read_config(args.base_config)
    seed = Path(args.seed).resolve()
    if not seed.is_file():
        raise FileNotFoundError(seed)
    entries = []
    names = args.experiments or [f"spatial_task_{i:02d}" for i in range(10)] + list(SUITES)
    if args.reuse_final_patches:
        for name in names:
            final = final_root / name
            source = json.loads((final / "asr_report.json").read_text())
            entries.append({
                "name": name, "kind": "task" if name.startswith("spatial_task_") else "suite",
                "suite": source["suite"], "task_ids": source["task_ids"],
                "instruction": source.get("instruction"),
                "trials": args.task_trials if name.startswith("spatial_task_") else args.suite_trials,
                "initial_state_offset": args.offset, "config": str(final / "config.json"),
                "final": str(final), "report_output": str(scratch / "reports" / name),
            })
    else:
        languages = (task_languages(args.libero_root, "libero_spatial")
                     if any(name.startswith("spatial_task_") for name in names) else [])
        data_root = Path(args.data_root).resolve()
        for name in names:
            task_specific = name.startswith("spatial_task_")
            if task_specific:
                task_id = int(name.rsplit("_", 1)[1])
                language = languages[task_id]
                data = data_root / "libero_spatial"
                manifest_dir = scratch / "manifests" / name
                train, val = manifest_dir / "train.jsonl", manifest_dir / "val.jsonl"
                write_rows(train, absolute_rows(data / "train.jsonl", language))
                write_rows(val, absolute_rows(data / "val.jsonl", language))
                suite, task_ids, policy = "libero_spatial", [task_id], SPATIAL_POLICY
            else:
                settings = SUITES[name]
                suite, task_ids = settings["suite"], list(range(10))
                language = None
                data = data_root / suite
                train, val = data / "train.jsonl", data / "val.jsonl"
                policy = {k: settings[k] for k in ("checkpoint", "revision", "unnorm_key")}
            cfg = variant(base, manifest=train, validation=val, seed=seed,
                          scene_seed=data / "seed.png", output=scratch / name / "training",
                          final=final_root / name, policy=policy)
            config_path = scratch / name / "config.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            entries.append({
                "name": name, "kind": "task" if task_specific else "suite", "suite": suite,
                "task_ids": task_ids, "instruction": language,
                "trials": args.task_trials if task_specific else args.suite_trials,
                "initial_state_offset": args.offset, "config": str(config_path),
                "final": str(final_root / name),
            })
    plan = {
        "schema_version": 2,
        "description": "Relaxed cat patches for individual Spatial tasks and four LIBERO suites",
        "reuse_final_patches": args.reuse_final_patches,
        "final_root": str(scratch / "results" if args.reuse_final_patches else final_root),
        "base_config": str(Path(args.base_config).resolve()),
        "seed": str(seed), "seed_sha256": sha256(seed),
        "limits": RELAXED_LIMITS, "step_sizes": STEP_SIZES,
        "selection": "appearance gate, then lowest held-out decoded-action MSE, then CE",
        "evaluation": {"offset": args.offset, "task_trials": args.task_trials,
                       "suite_trials": args.suite_trials, "tolerance": 0.05},
        "entries": entries,
    }
    (scratch / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"plan": str(scratch / "plan.json"), "experiments": len(entries)}, indent=2))


def train(args) -> None:
    root = Path(args.scratch).resolve()
    plan = json.loads((root / "plan.json").read_text())
    if plan.get("reuse_final_patches"):
        raise ValueError("This plan evaluates saved patches; no training is needed")
    results = []
    for entry in plan["entries"]:
        report = Path(entry["final"]) / "report.json"
        if report.exists():
            results.append({**entry, "status": "already_complete", "report": str(report)})
            continue
        log_path = root / f"{entry['name']}_training.log"
        print(json.dumps({"training": entry["name"]}), flush=True)
        with log_path.open("x") as log:
            result = subprocess.run(
                [sys.executable, "-u", "-m", "dura.natural_train", "--config", entry["config"]],
                stdout=log, stderr=subprocess.STDOUT,
            )
        scratch_report = Path(read_config(entry["config"])["output"]) / "report.json"
        status = json.loads(scratch_report.read_text())["status"] if scratch_report.exists() else "error"
        row = {**entry, "returncode": result.returncode, "status": status,
               "report": str(report if report.exists() else scratch_report), "log": str(log_path)}
        results.append(row)
        (root / "training_results.json").write_text(json.dumps(results, indent=2))
        if result.returncode and status != "no_candidate_passed_appearance_limits":
            raise RuntimeError(f"Training failed: {entry['name']}")
    (root / "training_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"trained": len(results), "results": str(root / "training_results.json")}, indent=2))


def eval_command(entry: dict, condition: str, output: Path, patch: str | None,
                 libero_root: str, task_id: int | None = None,
                 state: int | None = None, expected: str | None = None) -> list[str]:
    task_ids = [task_id] if task_id is not None else entry["task_ids"]
    trials = 1 if task_id is not None else entry["trials"]
    offset = state if state is not None else entry["initial_state_offset"]
    command = [sys.executable, "-u", "-m", "dura.evaluate_libero",
               "--config", entry["config"], "--libero-root", libero_root,
               "--suite", entry["suite"], "--task-ids", *map(str, task_ids),
               "--trials", str(trials), "--initial-state-offset", str(offset),
               "--seed", "0", "--tolerance", ".05", "--output", str(output)]
    if patch:
        command += ["--patch", patch]
    if expected:
        command += ["--expected-initial-sha256", expected]
    return command


def run_job(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def exact_rows(entry: dict, condition: str, raw: Path, clean_rows: list[dict],
               patch: str | None, libero_root: str, root: Path) -> tuple[list[dict], list[dict]]:
    clean = {key(row): row for row in clean_rows}
    all_current = {key(row): row for row in read_rows(raw / "rollouts.jsonl")}
    current = {episode_key: all_current[episode_key] for episode_key in clean
               if episode_key in all_current}
    if set(current) != set(clean):
        raise ValueError(f"Missing episode keys in {raw}")
    provenance = []
    for episode_key in sorted(clean):
        expected = clean[episode_key]["initial_image_sha256"]
        source = raw
        attempts = []
        if current[episode_key]["initial_image_sha256"] != expected:
            prefix = f"{condition}_recheck_t{episode_key[1]}_r{episode_key[2]}_a"
            existing = sorted((path for path in root.glob(prefix + "*") if path.is_dir()),
                              key=lambda path: int(path.name.rsplit("_a", 1)[1]))
            matched = False
            for candidate in existing:
                attempt = int(candidate.name.rsplit("_a", 1)[1])
                attempts.append({"source": str(candidate), "returncode": 0,
                                 "reused": True})
                rollout = candidate / "rollouts.jsonl"
                if not rollout.is_file():
                    continue
                replacement = read_rows(rollout)
                if (len(replacement) == 1 and key(replacement[0]) == episode_key
                        and replacement[0]["initial_image_sha256"] == expected):
                    current[episode_key] = replacement[0]
                    source = candidate
                    matched = True
                    break
            if not matched:
                first_attempt = (int(existing[-1].name.rsplit("_a", 1)[1]) + 1
                                 if existing else 1)
                for attempt in range(first_attempt, 21):
                    source = root / f"{prefix}{attempt}"
                    command = eval_command(entry, condition, source, patch, libero_root,
                                           episode_key[1], episode_key[2], expected)
                    with source.with_suffix(".log").open("x") as log:
                        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                    attempts.append({"source": str(source), "returncode": result.returncode,
                                     "reused": False})
                    if result.returncode:
                        if (source / "initial_mismatch.json").exists():
                            continue
                        raise RuntimeError(f"Recheck failed: {source}")
                    replacement = read_rows(source / "rollouts.jsonl")
                    if len(replacement) != 1 or key(replacement[0]) != episode_key:
                        raise ValueError(f"Unexpected recheck output: {source}")
                    current[episode_key] = replacement[0]
                    matched = True
                    break
            if not matched:
                raise RuntimeError(f"No exact initial image match after 20 attempts: "
                                   f"{condition} {episode_key}")
        provenance.append({"condition": condition, "key": list(episode_key),
                           "source": str(source), "attempts": attempts})
    return [current[k] for k in sorted(current)], provenance


def align_clean_rows(entry: dict, clean_rows: list[dict], seed_raw: Path, adv_raw: Path,
                     libero_root: str, root: Path) -> tuple[list[dict], list[dict]]:
    """Replace shared-process clean rows when isolated seed and attack agree exactly."""
    seed = {key(row): row for row in read_rows(seed_raw / "rollouts.jsonl")}
    adversarial = {key(row): row for row in read_rows(adv_raw / "rollouts.jsonl")}
    aligned, provenance = [], []
    for clean_row in clean_rows:
        episode_key = key(clean_row)
        seed_row, adversarial_row = seed[episode_key], adversarial[episode_key]
        common_hash = seed_row["initial_image_sha256"]
        if (clean_row["initial_image_sha256"] == common_hash
                or adversarial_row["initial_image_sha256"] != common_hash):
            aligned.append(clean_row)
            continue
        replacement = None
        attempts = []
        prefix = f"clean_alignment_t{episode_key[1]}_r{episode_key[2]}_a"
        for attempt in range(1, 21):
            output = root / f"{prefix}{attempt}"
            if output.exists():
                rows = read_rows(output / "rollouts.jsonl") if (output / "rollouts.jsonl").is_file() else []
                if len(rows) == 1 and rows[0]["initial_image_sha256"] == common_hash:
                    replacement = rows[0]
                    attempts.append({"source": str(output), "returncode": 0, "reused": True})
                    break
                attempts.append({"source": str(output), "returncode": 1, "reused": True})
                continue
            command = eval_command(entry, "clean", output, None, libero_root,
                                   episode_key[1], episode_key[2], common_hash)
            with output.with_suffix(".log").open("x") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            attempts.append({"source": str(output), "returncode": result.returncode,
                             "reused": False})
            if result.returncode:
                if (output / "initial_mismatch.json").exists():
                    continue
                raise RuntimeError(f"Clean alignment failed: {output}")
            rows = read_rows(output / "rollouts.jsonl")
            if len(rows) != 1 or key(rows[0]) != episode_key:
                raise ValueError(f"Unexpected clean alignment output: {output}")
            replacement = rows[0]
            break
        if replacement is None:
            raise RuntimeError(f"No isolated clean match after 20 attempts: {episode_key}")
        aligned.append(replacement)
        provenance.append({"condition": "clean", "key": list(episode_key),
                           "source": attempts[-1]["source"], "attempts": attempts,
                           "reason": "shared multi-task control process had a different render hash"})
    return aligned, provenance

def evaluate_one(entry: dict, baseline: Path, root: Path, libero_root: str) -> dict:
    final = Path(entry["final"])
    patch = final / "patch.png"
    if not patch.is_file():
        return {"name": entry["name"], "status": "no_accepted_patch"}
    report_output = Path(entry.get("report_output", final))
    report_output.mkdir(parents=True, exist_ok=True)
    existing_report = report_output / "asr_report.json"
    expected_eval_root = (root / "evaluations" / entry["name"]).resolve()
    report = json.loads(existing_report.read_text()) if existing_report.is_file() else None
    if (report is not None
            and Path(report.get("raw_evaluation", "")).resolve() == expected_eval_root
            and report.get("patch_sha256") == sha256(patch)):
        comparison = report["comparison"]
        return {
            "name": entry["name"], "status": "evaluated",
            "report": str(existing_report),
            "asr": comparison["conditions"]["adversarial"]["asr"],
            "seed_asr": comparison["conditions"]["seed"]["asr"],
            "clean_asr": comparison["conditions"]["clean"]["asr"],
            "conditional_asr": comparison["conditional_on_both_baselines_succeeding"]["rate"],
        }
    cfg = read_config(entry["config"])
    seed = cfg["seed_patch"]
    eval_root = root / "evaluations" / entry["name"]
    adv_raw = eval_root / "adversarial"
    eval_root.mkdir(parents=True, exist_ok=True)
    if not (adv_raw / "metrics.json").is_file():
        if adv_raw.exists():
            raise FileExistsError(f"Incomplete adversarial rollout exists: {adv_raw}")
        run_job(eval_command(entry, "adversarial", adv_raw, str(patch), libero_root),
                eval_root / "adversarial.log")
    first_state = entry["initial_state_offset"]
    last_state = first_state + entry["trials"]
    clean_rows = [row for row in read_rows(baseline / "clean" / "rollouts.jsonl")
                  if row["task_id"] in entry["task_ids"]
                  and first_state <= row["trial"] < last_state]
    clean_rows, clean_prov = align_clean_rows(entry, clean_rows, baseline / "seed", adv_raw,
                                              libero_root, eval_root)
    seed_rows, seed_prov = exact_rows(entry, "seed", baseline / "seed", clean_rows,
                                     seed, libero_root, eval_root)
    adv_rows, adv_prov = exact_rows(entry, "adversarial", adv_raw, clean_rows,
                                   str(patch), libero_root, eval_root)
    conditions = {"clean": clean_rows, "seed": seed_rows, "adversarial": adv_rows}
    comparison = summarize_conditions(conditions, [0, 0, 0, 0, 0, 0, 1], [.05], list(range(7)))
    matched = eval_root / "matched"
    matched.mkdir(exist_ok=True)
    for name, rows in conditions.items():
        write_rows(matched / f"{name}.jsonl", rows)
    provenance = {"rule": "first exact pre-action image match; outcome-independent",
                  "episodes": clean_prov + seed_prov + adv_prov}
    (eval_root / "matching_provenance.json").write_text(json.dumps(provenance, indent=2))
    (eval_root / "comparison.json").write_text(json.dumps(comparison, indent=2))
    report = {
        "name": entry["name"], "suite": entry["suite"], "task_ids": entry["task_ids"],
        "instruction": entry.get("instruction"), "trials_per_task": entry["trials"],
        "initial_state_offset": entry["initial_state_offset"],
        "patch": str(patch), "patch_sha256": sha256(patch),
        "seed": seed, "seed_sha256": sha256(Path(seed)),
        "comparison": comparison, "training_report": str(final / "report.json"),
        "raw_evaluation": str(eval_root),
    }
    (report_output / "asr_report.json").write_text(json.dumps(report, indent=2))
    return {"name": entry["name"], "status": "evaluated", "report": str(report_output / "asr_report.json"),
            "asr": comparison["conditions"]["adversarial"]["asr"],
            "seed_asr": comparison["conditions"]["seed"]["asr"],
            "clean_asr": comparison["conditions"]["clean"]["asr"],
            "conditional_asr": comparison["conditional_on_both_baselines_succeeding"]["rate"]}


def evaluate(args) -> None:
    root = Path(args.scratch).resolve()
    plan = json.loads((root / "plan.json").read_text())
    accepted = [entry for entry in plan["entries"] if (Path(entry["final"]) / "patch.png").is_file()]
    groups: dict[str, list[dict]] = {}
    for entry in accepted:
        group = "spatial_tasks" if entry["suite"] == "libero_spatial" else entry["name"]
        groups.setdefault(group, []).append(entry)
    baseline_paths = {}
    # One clean/seed baseline covers all task-specific Spatial patches. Suite patches
    # each receive their own baseline because their policies and environments differ.
    for group, entries in groups.items():
        template = dict(entries[0])
        if group == "spatial_tasks":
            template.update({"task_ids": list(range(10)), "trials": max(entry["trials"] for entry in entries)})
        baseline = root / "baselines" / group
        baseline.mkdir(parents=True, exist_ok=True)
        cfg = read_config(template["config"])
        requested = [
            ("clean", None),
            ("seed", cfg["seed_patch"]),
        ]
        jobs = []
        for condition, patch in requested:
            output = baseline / condition
            if (output / "metrics.json").is_file():
                continue
            if output.exists():
                raise FileExistsError(f"Incomplete baseline exists: {output}")
            jobs.append((eval_command(template, condition, output, patch, args.libero_root),
                         baseline / f"{condition}.log"))
        with ThreadPoolExecutor(max_workers=min(2, args.workers)) as pool:
            futures = [pool.submit(run_job, command, log) for command, log in jobs]
            for future in as_completed(futures):
                future.result()
        baseline_paths[group] = baseline
    results = []
    # Choose workers according to available GPU memory; one is the portable default.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for entry in accepted:
            group = "spatial_tasks" if entry["suite"] == "libero_spatial" else entry["name"]
            future = pool.submit(evaluate_one, entry, baseline_paths[group], root, args.libero_root)
            futures[future] = entry["name"]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            (root / "evaluation_results.json").write_text(json.dumps(sorted(results, key=lambda x: x["name"]), indent=2))
            print(json.dumps(row), flush=True)
    missing = [entry for entry in plan["entries"] if entry not in accepted]
    results += [{"name": entry["name"], "status": "no_accepted_patch"} for entry in missing]
    (root / "evaluation_results.json").write_text(json.dumps(sorted(results, key=lambda x: x["name"]), indent=2))


def publish_summary(final_root: Path, entries: list[dict], protocol: dict) -> None:
    """Write portable result links and exact ASR denominators."""
    final_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in entries:
        report_path = Path(entry.get("report_output", entry["final"])) / "asr_report.json"
        if not report_path.is_file():
            rows.append({"name": entry["name"], "status": "no_evaluation"})
            continue
        source = json.loads(report_path.read_text())
        comparison = source["comparison"]
        conditional = comparison["conditional_on_both_baselines_succeeding"]
        rows.append({
            "name": entry["name"], "suite": source["suite"],
            "instruction": source.get("instruction"), "status": "evaluated",
            "report": os.path.relpath(report_path, final_root),
            "patch": os.path.relpath(Path(entry["final"]) / "patch.png", final_root),
            "patch_sha256": source["patch_sha256"],
            "episodes": comparison["paired_rollouts_per_condition"],
            "asr": comparison["conditions"]["adversarial"]["asr"],
            "seed_asr": comparison["conditions"]["seed"]["asr"],
            "clean_asr": comparison["conditions"]["clean"]["asr"],
            "conditional_asr": conditional["rate"],
            "conditional_failures": conditional["adversarial_failures"],
            "conditional_episodes": conditional["reference_successes"],
            "initial_observations_match": comparison["initial_observations_match"],
        })
    (final_root / "summary.json").write_text(json.dumps({
        "schema_version": 2, "protocol": protocol, "results": rows,
    }, indent=2) + "\n")
    lines = ["# Relaxed LIBERO patches", "",
             "ASR is task failure. Conditional ASR counts failures among episodes where clean and benign-seed controls both succeed.", "",
             "Task-specific patches use five initial states per task in the published experiment; suite-wide patches use one state per task. Exact counts appear below.", "",
             "| Patch / report | Scope | Episodes | Patch ASR | Seed ASR | Clean ASR | Conditional failures |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        if row["status"] != "evaluated":
            lines.append(f"| {row['name']} | pending | — | — | — | — | — |")
            continue
        scope = row["instruction"] or row["suite"]
        count = f"{row['conditional_failures']}/{row['conditional_episodes']}"
        rate = "undefined" if row["conditional_asr"] is None else f"{row['conditional_asr']:.1%}"
        lines.append(f"| [{row['name']}]({row['report']}) · [PNG]({row['patch']}) | {scope} | {row['episodes']} | {row['asr']:.0%} | {row['seed_asr']:.0%} | {row['clean_asr']:.0%} | {count} ({rate}) |")
    lines += ["", "Each patch directory retains the final PNG, preview, configuration, selected-candidate report, ASR report, and compact provenance.", "",
              "Training uses frozen Stable Diffusion v1.5 and the matching suite-specific OpenVLA checkpoint. Selection uses held-out action error after appearance gates, before rollout ASR is measured.", "",
              "The published Spatial suite row reuses the previously selected relaxed cat patch. Its fresh evaluation uses state 22; the other task-specific rows use states 22–26.", "",
              "These small samples describe the recorded experiment; more states and seeds are needed for a stable success-rate estimate."]
    (final_root / "README.md").write_text("\n".join(lines) + "\n")


def report(args) -> None:
    root = Path(args.scratch).resolve()
    plan = json.loads((root / "plan.json").read_text())
    final_root = Path(args.final_root or plan["final_root"]).resolve()
    protocol = {key: plan[key] for key in ("limits", "step_sizes", "selection", "evaluation", "seed_sha256")}
    publish_summary(final_root, plan["entries"], protocol)


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    default_scratch = str(scratch_root() / "runs/libero_relaxed")
    prep = sub.add_parser("prepare", help="Create a fresh training or saved-patch evaluation plan")
    prep.add_argument("--scratch", default=default_scratch)
    prep.add_argument("--final-root", default=str(PROJECT_ROOT / "final_patches/libero_relaxed_run"))
    prep.add_argument("--base-config", default=str(PROJECT_ROOT / "configs/libero_relaxed.yaml"))
    prep.add_argument("--seed", default=str(PROJECT_ROOT / "seed_patches/cat_01_soft.png"))
    prep.add_argument("--data-root", default=str(scratch_root() / "data"))
    prep.add_argument("--libero-root", required=True)
    prep.add_argument("--offset", type=int, default=22)
    prep.add_argument("--task-trials", type=positive_integer, default=5)
    prep.add_argument("--suite-trials", type=positive_integer, default=1)
    prep.add_argument("--experiments", nargs="+", choices=[f"spatial_task_{i:02d}" for i in range(10)] + list(SUITES))
    prep.add_argument("--reuse-final-patches", action="store_true",
                      help="Evaluate existing final PNGs; keep new reports in scratch")
    prep.set_defaults(function=prepare)
    train_cmd = sub.add_parser("train", help="Optimize all patches in a prepared plan")
    train_cmd.add_argument("--scratch", default=default_scratch)
    train_cmd.set_defaults(function=train)
    evaluate_cmd = sub.add_parser("evaluate", help="Evaluate with clean and benign-seed controls")
    evaluate_cmd.add_argument("--scratch", default=default_scratch)
    evaluate_cmd.add_argument("--libero-root", required=True)
    evaluate_cmd.add_argument("--workers", type=positive_integer, default=1)
    evaluate_cmd.set_defaults(function=evaluate)
    report_cmd = sub.add_parser("report", help="Build the result table and summary JSON")
    report_cmd.add_argument("--scratch", default=default_scratch)
    report_cmd.add_argument("--final-root")
    report_cmd.set_defaults(function=report)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
