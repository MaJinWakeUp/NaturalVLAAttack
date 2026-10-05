"""LIBERO simulation evaluation and clean-frame collection; imports are lazy."""

import argparse
from collections import deque
from contextlib import ExitStack
import json
import hashlib
import time
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

from .adapters import load_policy
from .cli import read_config
from .data import load_image, save_image
from .defenses import transform
from .metrics import rollout_metrics, target_matches
from .render import PatchRenderer

LIMITS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


def policy_image(array, kind, size=224):
    array = np.ascontiguousarray(array[::-1, ::-1])
    if kind == "openvla":
        # Match the official OpenVLA LIBERO JPEG/lanczos3 input path.
        import tensorflow as tf
        image = tf.io.decode_image(tf.image.encode_jpeg(array), expand_animations=False, dtype=tf.uint8)
        image = tf.image.resize(image, (size, size), method="lanczos3", antialias=True)
        return tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
    raise ValueError("Unsupported policy image convention")


def robot_state(observation):
    quat = np.asarray(observation["robot0_eef_quat"], dtype=float)
    w = np.clip(quat[3], -1, 1)
    denominator = np.sqrt(1-w*w)
    axisangle = np.zeros(3) if denominator < 1e-8 else quat[:3] * (2*np.arccos(w)/denominator)
    return np.concatenate((observation["robot0_eef_pos"], axisangle, observation["robot0_gripper_qpos"]))


def executable_action(action, kind):
    action = np.array(action, dtype=float, copy=True)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("LIBERO requires finite seven-dimensional actions")
    if kind == "openvla":
        action[6] = -np.sign(2*action[6]-1)
    return action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--suite", choices=list(LIMITS), default="libero_spatial")
    parser.add_argument("--libero-root", help="Local LIBERO checkout")
    parser.add_argument("--patch", help="PNG patch; omit for no-patch baseline")
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--video", action="store_true", help="Save policy-view rollout videos in the output directory")
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--initial-state-offset", type=int, default=0)
    parser.add_argument("--expected-initial-sha256", help="Require this exact initial-image hash for a single-episode recheck")
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, nargs="+", required=True,
                        help="AP tolerance in executed-action coordinates (scalar or per dimension)")
    parser.add_argument("--defense", choices=["none", "jpeg", "bit_depth", "gaussian"], default="none")
    parser.add_argument("--defense-strength", type=float)
    parser.add_argument("--export-every", type=int, default=0,
                        help="Save a frame every N action steps; 0 disables collection")
    args = parser.parse_args()
    if args.trials < 1 or args.replan_steps < 1 or min(args.initial_state_offset, args.settle_steps, args.export_every) < 0:
        parser.error("Trial/replan counts must be positive; offsets and settle/export counts nonnegative")
    config = read_config(args.config)
    kind = config["policy"]["kind"]
    if kind != "openvla" or config.get("image_size", 224) != 224:
        parser.error("This runner supports OpenVLA with image_size=224")
    target = executable_action(config["objective"]["target"], kind)
    dimensions = config["objective"].get("dimensions")
    target_matches(target[None], target, args.tolerance, dimensions)
    generator = torch.Generator().manual_seed(args.seed)
    # Validate defense arguments before loading models.
    transform(torch.zeros(1, 3, 8, 8), args.defense, args.defense_strength, generator)
    if args.libero_root:
        sys.path.insert(0, str(Path(args.libero_root).resolve()))
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = list(range(suite.n_tasks)) if args.task_ids is None else args.task_ids
    if not task_ids or len(set(task_ids)) != len(task_ids) or any(i < 0 or i >= suite.n_tasks for i in task_ids):
        raise ValueError("Task IDs must be unique, valid suite indices")
    if args.expected_initial_sha256 and (len(task_ids) != 1 or args.trials != 1):
        parser.error("expected-initial-sha256 requires exactly one task and one trial")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy = load_policy(config["policy"])
    renderer = PatchRenderer(config["patch_box"])
    patch = load_image(args.patch)[None] if args.patch else None
    (output / "evaluation_config.json").write_text(json.dumps(
        {"arguments": vars(args), "policy_config": config, "executed_target": target.tolist()}, indent=2))
    records = []
    started = time.monotonic()
    with ExitStack() as stack:
        log = stack.enter_context((output / "rollouts.jsonl").open("x"))
        manifest = stack.enter_context((output / "manifest.jsonl").open("x")) if args.export_every else None
        if manifest:
            (output / "frames").mkdir()
        for task_id in task_ids:
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            if args.initial_state_offset + args.trials > len(initial_states):
                raise ValueError("Not enough initial states for the requested trials")
            bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
            try:
                env.seed(args.seed)
                for trial in range(args.trials):
                    state_index = args.initial_state_offset + trial
                    env.seed(args.seed + state_index)
                    np.random.seed(args.seed + state_index)
                    torch.manual_seed(args.seed + state_index)
                    env.reset()
                    observation = env.set_init_state(initial_states[state_index])
                    for _ in range(args.settle_steps):
                        observation, _, _, _ = env.step([0., 0., 0., 0., 0., 0., -1.])
                    initial_image = policy_image(observation["agentview_image"], kind)
                    initial_image_sha256 = hashlib.sha256(initial_image.tobytes()).hexdigest()
                    if args.expected_initial_sha256 and initial_image_sha256 != args.expected_initial_sha256:
                        (output / "initial_mismatch.json").write_text(json.dumps({
                            "expected": args.expected_initial_sha256, "actual": initial_image_sha256,
                            "policy_actions_taken": 0}, indent=2))
                        raise ValueError("Recheck initial image differs; no policy actions were taken")
                    writer = None
                    if args.video:
                        import imageio.v2 as imageio
                        writer = imageio.get_writer(output / f"rollout_t{task_id}_r{state_index}.mp4", fps=20)
                        stack.callback(writer.close)
                    actions, action_queue, success = [], deque(), False
                    for step in range(LIMITS[args.suite]):
                        image = policy_image(observation["agentview_image"], kind)
                        row = {"state": robot_state(observation).tolist()}
                        if manifest:
                            wrist = policy_image(observation["robot0_eye_in_hand_image"], kind)
                            row["wrist_array"] = wrist
                        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float()[None]/255
                        if manifest and step % args.export_every == 0:
                            stem = f"t{task_id:02d}_r{state_index:03d}_s{step:04d}"
                            Image.fromarray(image).save(output / "frames" / f"{stem}.png")
                            Image.fromarray(wrist).save(output / "frames" / f"{stem}_wrist.png")
                            entry = {"image": f"frames/{stem}.png", "wrist_image": f"frames/{stem}_wrist.png",
                                     "instruction": task.language, "state": row["state"], "suite": args.suite,
                                     "task_id": task_id, "rollout_id": state_index, "step": step}
                            manifest.write(json.dumps(entry)+"\n")
                            manifest.flush()
                        if not action_queue:
                            if patch is not None:
                                tensor = renderer(tensor, patch, [row])[0]
                            tensor = transform(tensor, args.defense, args.defense_strength, generator)
                            if step == 0:
                                save_image(tensor, output / f"preview_t{task_id}_r{state_index}.png")
                            with torch.no_grad():
                                prediction = policy.predict_actions(tensor, [task.language], [row])[0]
                            prediction = torch.as_tensor(prediction).detach().cpu().numpy()
                            if prediction.ndim == 1:
                                prediction = prediction[None]
                            if len(prediction) < args.replan_steps:
                                raise ValueError("replan-steps exceeds the policy action horizon")
                            action_queue.extend(prediction[:args.replan_steps])
                        if writer is not None:
                            video_frame = (tensor[0].detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).round().astype(np.uint8)
                            writer.append_data(video_frame)
                        if step % 50 == 0:
                            print(json.dumps({"task": task_id, "trial": state_index, "step": step,
                                              "elapsed_seconds": time.monotonic()-started}), flush=True)
                        action = executable_action(action_queue.popleft(), kind)
                        observation, _, done, _ = env.step(action.tolist())
                        actions.append(action.tolist())
                        if done:
                            success = bool(env.check_success())
                            break
                    if writer is not None:
                        writer.close()
                    record = {"suite": args.suite, "task_id": task_id, "trial": state_index,
                              "initial_image_sha256": initial_image_sha256,
                              "instruction": task.language, "success": success, "actions": actions}
                    records.append(record)
                    log.write(json.dumps(record)+"\n")
                    log.flush()
                    print(json.dumps({"task": task_id, "trial": state_index, "success": success}), flush=True)
            finally:
                env.close()
    metrics = rollout_metrics(records, target, args.tolerance, dimensions)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
