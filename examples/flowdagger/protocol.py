"""Task mapping, evaluation plans and raw observation conventions (no simulator imports)."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_tasks(dataset_root, indices):
    tasks = json.loads((Path(dataset_root) / "data_pool.json").read_text())["tasks"]
    return {t["task_key"]: t for t in tasks if t["task_index"] in indices}


def build_plan(tasks, mode, scene_root, seed, num_scenes, trials, close_trials, open_trials, scene_glob):
    plan = []
    for key, task in tasks.items():
        task_index = task["task_index"]
        if mode == "recovery":
            directory = Path(scene_root) / task["suite"] / task["task_id"]
            scenes = sorted(directory.rglob(scene_glob))
            if not scenes:
                raise FileNotFoundError(f"No recovery scenes for {key} in {directory}")
            if 0 < num_scenes < len(scenes):
                rng = np.random.default_rng(np.random.SeedSequence([seed, task_index]))
                scenes = [scenes[i] for i in sorted(rng.choice(len(scenes), num_scenes, replace=False))]
            starts = [(p, p.with_name(p.stem + "_sim_state.npy"), None) for p in scenes]
            grips = ["closed"] * close_trials + ["open"] * open_trials
        else:
            base = ROOT / "LIBERO/libero/libero"
            bddl = base / "bddl_files" / task["suite"] / (task["task_id"] + ".bddl")
            state = base / "init_files" / task["suite"] / (task["task_id"] + ".pruned_init")
            starts = [(bddl, state, i) for i in range(trials)]
            grips = ["open"]
        for scene_index, (bddl, state, init_index) in enumerate(starts):
            if not bddl.is_file() or not state.is_file():
                raise FileNotFoundError(f"Missing BDDL/state pair: {bddl}, {state}")
            for trial_index, grip in enumerate(grips):
                trial_seed = int(np.random.SeedSequence([seed, task_index, scene_index, trial_index]).generate_state(1)[0])
                plan.append({"id": len(plan), "task_key": key, "task_index": task_index,
                             "bddl": str(bddl.resolve()), "state_file": str(state.resolve()),
                             "init_index": init_index, "gripper": grip, "seed": trial_seed})
    return plan


def perturb_xy(flat_state, addresses, noise_xy, rng):
    """MjSimState.flatten() stores time at index 0, then qpos, then qvel."""
    state = np.array(flat_state, dtype=np.float64, copy=True)
    deltas = {}
    for name, qpos_x, qpos_y in addresses:
        delta = rng.uniform(-noise_xy, noise_xy, size=2)
        state[1 + qpos_x] += delta[0]
        state[1 + qpos_y] += delta[1]
        deltas[name] = delta.tolist()
    return state, deltas


def quat_to_axis_angle(quat):
    quat = np.asarray(quat, dtype=np.float64)
    w = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - w * w)
    return np.zeros(3) if denominator < 1e-8 else quat[:3] * (2 * np.arccos(w) / denominator)


def policy_observation(obs, task, noise_seed):
    return {
        # Only raw simulator images are rotated. Training cache images are already aligned.
        "observation/image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
        "observation/wrist_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
        "observation/state": np.concatenate((obs["robot0_eef_pos"], quat_to_axis_angle(obs["robot0_eef_quat"]),
                                             obs["robot0_gripper_qpos"])).astype(np.float32),
        "prompt": task["official_task_language"], "task_key": task["task_key"], "noise_seed": int(noise_seed),
    }


def summarize(results):
    def counts(rows):
        valid = [r for r in rows if r["status"] == "completed"]
        successes = sum(r["success"] for r in valid)
        return {"planned": len(rows), "valid": len(valid), "successes": successes,
                "success_rate_planned": successes / len(rows) if rows else None,
                "success_rate_valid": successes / len(valid) if valid else None,
                "reset_errors": sum(r["status"] == "reset_error" for r in rows),
                "initially_successful": sum(r["status"] == "initial_success" for r in rows)}
    by_task = {key: counts([r for r in results if r["task_key"] == key]) for key in sorted({r["task_key"] for r in results})}
    return {"overall": counts(results), "by_task": by_task,
            "macro_success_rate_planned": float(np.mean([v["success_rate_planned"] for v in by_task.values()])) if by_task else None,
            "by_gripper": {g: counts([r for r in results if r["gripper"] == g]) for g in ("open", "closed")}}
