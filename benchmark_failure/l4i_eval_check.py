#!/usr/bin/env python
"""L4-inject full-chain verification: replay the EXACT eval load path of
eval_libero_custom_scene.py for a sample of injected scenes and check the L4
obstruction survives it.

Per scene:
  1. build env from the l4i scene bddl (reseed-retry on RandomizationError,
     mirroring _construct_env)
  2. reset() (reseed-retry, mirroring _reset_env)
  3. set_init_state(injected sim_state npy)   <- perturbation restored here
  4. reset robot controller goals (the OSC workaround from the eval script)
  5. run WAIT_STEPS dummy action steps
  6. checks (HARD — define a usable L4 scene):
     a. build env + reset() succeed (with the eval's reseed retries)
     b. obstruction present at t=0: right after set_init_state, the disturbed
        object physically CONTACTS the task object, or leans on it
        (xy gap <=5cm and |dz| <=5cm)
     c. render at t=0 matches the perturbed.jpg dumped at injection time
        (mean abs diff <= 8/255)
     SOFT (reported, not gating): whether the obstruction SURVIVES the eval's
     20-step gripper-settle phase. Mid-rollout snapshots where the restored arm
     was carrying an object are inherently dynamic (true for the base benchmark
     too), so post-settle motion is informational.

Usage (libero env, see run_l4i.sh): python l4i_eval_check.py [--n 12] [--seed 7]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("STARVLA_ROOT", "/H20_vepfs/liulin/starVLA")
OUT_ROOT = os.environ.get("L4I_OUT_ROOT", os.path.join(HERE, "l4_injected"))

sys.path.insert(0, HERE)
from l4i_inject import SOURCE_SUITE, body_touching, lookup_task, parse_bddl_objects  # noqa: E402

WAIT_STEPS = 10          # (unused now, kept for reference)
GRIPPER_SETTLE_STEPS = 20  # mirrors eval_libero_custom_scene.settle_gripper_to
MAX_BUILD_ATTEMPTS = 5


def bddl_path(r):
    """Prefer a repaired bddl (repair trees) if one exists."""
    fname = os.path.basename(r["bddl_file"])
    rel_scene_dir = os.path.relpath(r["scene_dir"], "benchmark_failure/l4_injected")
    for base in (os.environ.get("L4I_REPAIR_OUT", ""),
                 os.path.join(OUT_ROOT, "repair_bddl")):
        if base:
            cand = os.path.join(base, rel_scene_dir, fname)
            if os.path.exists(cand):
                return cand
    return os.path.join(ROOT, r["bddl_file"])


def build_env(env_args, seed_base):
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.errors import RandomizationError
    for attempt in range(MAX_BUILD_ATTEMPTS):
        np.random.seed(seed_base + attempt)
        try:
            env = OffScreenRenderEnv(**env_args)
            return env, attempt
        except RandomizationError:
            pass
    return None, MAX_BUILD_ATTEMPTS


def reset_env(env, seed_base):
    from robosuite.utils.errors import RandomizationError
    for attempt in range(MAX_BUILD_ATTEMPTS):
        np.random.seed(seed_base + attempt)
        try:
            env.reset()
            return True
        except RandomizationError:
            pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="scenes to check (stratified)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--egl", type=int, default=1)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(args.egl))
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    from libero.libero import benchmark
    from robosuite.utils.binding_utils import MjSimState

    recs, seen = [], set()
    for p in sorted(glob.glob(os.path.join(OUT_ROOT, "index", "injected*.jsonl"))):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if r["scene_id"] not in seen:
                seen.add(r["scene_id"])
                recs.append(r)

    # stratified sample: every (source, strategy) pair gets ceil share
    groups = {}
    for r in recs:
        groups.setdefault((r["original_source"], r["injection"]["perturbed"]["strategy"]),
                          []).append(r)
    rng = np.random.default_rng(args.seed)
    sample = []
    per = max(1, args.n // len(groups))
    for k, v in sorted(groups.items()):
        idx = rng.choice(len(v), size=min(per, len(v)), replace=False)
        sample += [v[i] for i in idx]
    print(f"records: {len(recs)}, groups: {len(groups)}, sampling {len(sample)}")

    suites = {s: benchmark.get_benchmark_dict()[s]() for s in set(SOURCE_SUITE.values())}
    n_pass = n_fail = 0
    for i, r in enumerate(sample):
        sid = r["scene_id"]
        pert = r["injection"]["perturbed"]
        task_dir = os.path.basename(os.path.dirname(r["original_scene_dir"]))
        try:
            env, attempts = build_env(
                {"bddl_file_name": bddl_path(r),
                 "camera_heights": 256, "camera_widths": 256},
                seed_base=args.seed * 1000 + i)
            if env is None:
                raise RuntimeError("env build failed after reseed retries")
            if not reset_env(env, seed_base=args.seed * 1000 + i + 100):
                raise RuntimeError("reset failed after reseed retries")
            init = np.load(os.path.join(ROOT, r["sim_state_file"]))
            env.set_init_state(init)
            for robot in env.robots:
                robot.controller.update()
                robot.controller.update_initial_joints(robot.controller.joint_pos)
                robot.controller.reset_goal()
            inner = env.env
            sim = inner.sim
            bid = inner.obj_body_id
            a, b = pert["object"], pert["onto"]

            def obstruction():
                gap = float(np.hypot(*(sim.data.body_xpos[bid[a]][:2]
                                       - sim.data.body_xpos[bid[b]][:2])))
                dz = float(sim.data.body_xpos[bid[a]][2] - sim.data.body_xpos[bid[b]][2])
                return body_touching(sim, bid[a], bid[b]), gap, dz

            ok, msgs = True, []
            if attempts:
                msgs.append(f"build retries={attempts}")
            # --- hard checks at t=0 (the state the annotation describes) ---
            contact, gap, dz = obstruction()
            if not (contact or (gap <= 0.05 and abs(dz) <= 0.05)):
                ok = False
                msgs.append(f"no obstruction at t=0: contact={contact} "
                            f"gap={gap:.3f} dz={dz:.3f}")
            fdir = os.path.join(OUT_ROOT, "frames", sid)
            pjpgs = sorted(glob.glob(os.path.join(fdir, "*_perturbed.jpg")))
            if pjpgs:
                img = sim.render(256, 256, camera_name="agentview")[::-1, ::-1]
                ref = np.asarray(Image.open(pjpgs[0]).convert("RGB"), dtype=np.float32)
                diff = float(np.abs(img.astype(np.float32) - ref).mean())
                if diff > 8.0:
                    ok = False
                    msgs.append(f"render mismatch at t=0: mean abs diff {diff:.1f}")
            # --- soft: eval's 20-step gripper settle (close), like the real run ---
            for _ in range(GRIPPER_SETTLE_STEPS):
                env.step([0.0] * 6 + [-1.0])
            c2, gap2, dz2 = obstruction()
            survived = c2 or (gap2 <= 0.05 and abs(dz2) <= 0.05)
            if ok:
                n_pass += 1
                print(f"PASS {sid[:95]} ({pert['strategy']}, t0 contact={contact} "
                      f"gap={gap*100:.0f}cm, settle_survives={survived}) {'; '.join(msgs)}")
            else:
                n_fail += 1
                print(f"FAIL {sid[:95]} ({pert['strategy']}) :: {'; '.join(msgs)}")
            env.close()
        except Exception as e:
            n_fail += 1
            print(f"FAIL {sid[:95]} :: {type(e).__name__}: {str(e)[:80]}")
    print(f"\neval-check: {n_pass} PASS, {n_fail} FAIL of {len(sample)}")


if __name__ == "__main__":
    main()
