"""Evaluate ordinary LIBERO or fixed recovery scenes through the offline-steering websocket service."""

import argparse
from collections import deque
import json
import os
from pathlib import Path

import numpy as np

from .protocol import ROOT, build_plan, load_tasks, perturb_xy, policy_observation, summarize


def object_addresses(env):
    model = env.sim.model
    addresses = []
    for name, body_id in env.env.obj_body_id.items():
        joint = int(model.body_jntadr[body_id])
        if joint >= 0 and model.jnt_type[joint] == 0:
            qpos = int(model.jnt_qposadr[joint])
            addresses.append((name, qpos, qpos + 1))
    return addresses


def settle(env, obs, gripper, steps):
    action = [0.0] * 6 + [1.0 if gripper == "closed" else -1.0]
    for _ in range(steps):
        obs, _, _, _ = env.step(action)
    return obs


def reset_trial(env, trial, mode, noise_xy, settle_steps, attempts, randomization_error):
    # Bypass ControlEnv.reset's unbounded placement loop; failures remain in the report.
    for attempt in range(attempts):
        try:
            env.seed((trial["seed"] + attempt) % (2**32))
            env.env.reset()
            break
        except randomization_error:
            if attempt == attempts - 1:
                raise
    if mode == "recovery":
        state = np.load(trial["state_file"])
        current = env.sim.get_state()
        if len(state) != len(current.flatten()) or not np.array_equal(current.flatten()[1:1 + len(current.qpos)], current.qpos):
            raise ValueError("Recovery state does not match this simulator's [time, qpos, qvel] layout")
        state, deltas = perturb_xy(state, object_addresses(env), noise_xy, np.random.default_rng(trial["seed"]))
    else:
        import torch
        state = np.asarray(torch.load(trial["state_file"], weights_only=False)[trial["init_index"]])
        deltas = {}
    obs = env.set_init_state(state)
    for robot in env.robots:
        robot.controller.update()
        robot.controller.update_initial_joints(robot.controller.joint_pos)
        robot.controller.reset_goal()
    return settle(env, obs, trial["gripper"], settle_steps), deltas


def rollout(env, obs, task, client, trial, query_freq, horizon, video_path=None):
    plan = deque()
    noise_rng = np.random.default_rng(np.random.SeedSequence([trial["seed"], 1]))
    writer = None
    if video_path:
        import imageio.v2 as imageio
        writer = imageio.get_writer(video_path, fps=20)
    try:
        for step in range(horizon):
            if writer:
                writer.append_data(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            if not plan:
                request = policy_observation(obs, task, noise_rng.integers(0, 2**32))
                actions = np.asarray(client.infer(request)["actions"], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < query_freq or not np.isfinite(actions).all():
                    raise ValueError("Policy must return finite native LIBERO actions of shape (H, 7), H >= query_freq")
                plan.extend(actions[:query_freq])
            obs, _, _, _ = env.step(plan.popleft().tolist())
            if env.check_success():
                return True, step + 1
        return False, horizon
    finally:
        if writer:
            writer.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/plus_featured")
    p.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--mode", choices=["standard", "recovery"], default="recovery")
    p.add_argument("--scene-root", type=Path, default=ROOT / "benchmark_failure/scenes")
    p.add_argument("--scene-glob", default="*.bddl", help="Use step_*.bddl to exclude l4i_* variants")
    p.add_argument("--num-scenes", type=int, default=30, help="Per task; 0 selects all")
    p.add_argument("--trials", type=int, default=25, help="Standard init states per task")
    p.add_argument("--close-trials", type=int, default=5)
    p.add_argument("--open-trials", type=int, default=5)
    p.add_argument("--noise-xy", type=float, default=0.02)
    p.add_argument("--settle-steps", type=int, default=20)
    p.add_argument("--query-freq", type=int, default=10)
    p.add_argument("--horizon", type=int, default=520)
    p.add_argument("--reset-attempts", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18010)
    p.add_argument("--render-gpu-device-id", type=int, default=-1)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--plan", type=Path, help="Replay a saved plan.json including its protocol settings")
    p.add_argument("--dry-run", action="store_true", help="Write the plan without importing simulator/client")
    p.add_argument("--video", action="store_true")
    p.add_argument("--disable-websocket-pings", action="store_true",
                   help="Allow long local blocking rendering; use an external process deadline")
    p.add_argument("--scene-compatibility", action="store_true",
                   help="Repair known exported BDDL declarations in new output copies; preserve original scenes/states")
    p.add_argument("--state-restore-reset", action="store_true",
                   help="Recovery only: allow overlap in provisional reset, then verify exact saved-state restoration")
    return p


def main():
    args = parser().parse_args()
    settings = {key: getattr(args, key) for key in ("mode", "noise_xy", "settle_steps", "query_freq", "horizon", "reset_attempts")}
    if args.plan:
        saved = json.loads(args.plan.read_text())
        settings, trials = saved["settings"], saved["trials"]
        tasks = load_tasks(args.dataset_root, sorted({t["task_index"] for t in trials}))
        if saved["tasks"] != tasks:
            raise ValueError("Saved plan task definitions differ from the dataset")
    else:
        tasks = load_tasks(args.dataset_root, args.tasks)
        trials = build_plan(tasks, args.mode, args.scene_root, args.seed, args.num_scenes, args.trials,
                            args.close_trials, args.open_trials, args.scene_glob)
    if not trials:
        raise ValueError("No evaluation trials selected")
    if args.scene_compatibility:
        settings["scene_compatibility"] = True
    if args.state_restore_reset:
        settings["state_restore_reset"] = True
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.json").write_text(json.dumps({"tasks": tasks, "settings": settings, "trials": trials}, ensure_ascii=False, indent=2))
    print(f"Planned {len(trials)} trials across {len(tasks)} tasks", flush=True)
    if args.dry_run:
        return

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.errors import RandomizationError
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    client_class = WebsocketClientPolicy
    if args.disable_websocket_pings:
        from .transport import NoPingClientPolicy
        client_class = NoPingClientPolicy
    client = client_class(args.host, args.port)
    metadata = client.get_server_metadata()
    if metadata["protocol"] != "libero_offline_v1" or not set(tasks) <= set(metadata["task_keys"]):
        raise ValueError("Server protocol/tasks do not match this evaluation")
    if not 1 <= settings["query_freq"] <= metadata["action_horizon"]:
        raise ValueError("query_freq must be in [1, model action_horizon]")
    (args.output / "server.json").write_text(json.dumps(metadata, indent=2))
    results = []
    env = None
    current_bddl = None
    try:
        with (args.output / "trials.jsonl").open("w") as log:
            for trial in trials:
                result = dict(trial, success=False, steps=0)
                try:
                    if current_bddl != trial["bddl"]:
                        if env is not None:
                            env.close()
                            env = None
                        np.random.seed(trial["seed"])
                        runtime_bddl = trial["bddl"]
                        if settings.get("scene_compatibility", False):
                            from .scene_compat import prepare_bddl
                            runtime_bddl, repair = prepare_bddl(runtime_bddl, args.output / "scene_bddl")
                            if repair:
                                with (args.output / "scene_repairs.jsonl").open("a") as repairs:
                                    repairs.write(json.dumps(repair) + "\n")
                        factory = OffScreenRenderEnv
                        if settings["mode"] == "recovery" and settings.get("state_restore_reset", False):
                            from .recovery_env import make_state_restore_env
                            factory = make_state_restore_env
                        env = factory(bddl_file_name=runtime_bddl, camera_heights=256, camera_widths=256,
                                                 horizon=settings["horizon"] + settings["settle_steps"],
                                                 render_gpu_device_id=args.render_gpu_device_id)
                        current_bddl = trial["bddl"]
                    obs, deltas = reset_trial(env, trial, settings["mode"], settings["noise_xy"], settings["settle_steps"],
                                              settings["reset_attempts"], RandomizationError)
                except RandomizationError as exc:
                    result.update(status="reset_error", error=str(exc))
                else:
                    result["object_xy_deltas"] = deltas
                    if env.check_success():
                        result["status"] = "initial_success"
                    else:
                        video = str(args.output / f"trial_{trial['id']:05d}.mp4") if args.video else None
                        success, steps = rollout(env, obs, tasks[trial["task_key"]], client, trial,
                                                 settings["query_freq"], settings["horizon"], video)
                        result.update(status="completed", success=bool(success), steps=steps)
                results.append(result)
                log.write(json.dumps(result) + "\n")
                log.flush()
                print(f"trial={trial['id']} task={trial['task_index']} status={result['status']} success={result['success']}", flush=True)
    finally:
        if env is not None:
            env.close()
    (args.output / "summary.json").write_text(json.dumps(summarize(results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
