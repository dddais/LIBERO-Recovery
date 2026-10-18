#!/usr/bin/env python
"""L4-inject repair: regenerate every l4i_step_XXXX.bddl from the
PRE-perturbation state with the fixed workspace detection.

Why: the original bddls were dumped from the PERTURBED state, so
  - onto scenes: object regions overlap -> RandomizationError at reset
  - tasks without a table fixture (libero_object, ...): workspace defaulted to
    a nonexistent "main_table" -> KeyError at env construction
The eval only needs the bddl to build the env + sample a collision-free reset
layout; set_init_state(npy) restores the perturbed state. So the bddl is now
dumped from the original state (regions never overlap, layout identical to the
proven-loadable original scene dumps).

Writes the regenerated bddls to l4_injected/repair_bddl/ (mirroring the scenes/
tree; run as a user that may not own the root-run scene files). Install with:
  cp -a <OUT_ROOT>/repair_bddl/. <OUT_ROOT>/scenes/
The DONE list defaults to index/repair_bddl.done, override with L4I_REPAIR_DONE
when index/ is not writable. npy / frames / annotations / index are untouched.
Usage (libero env, see run_l4i.sh): python l4i_repair_bddl.py [--limit N]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("STARVLA_ROOT", "/H20_vepfs/liulin/starVLA")
OUT_ROOT = os.environ.get("L4I_OUT_ROOT", os.path.join(HERE, "l4_injected"))
DONE = os.environ.get("L4I_REPAIR_DONE", os.path.join(OUT_ROOT, "index/repair_bddl.done"))
# output tree: defaults inside l4_injected/, override when that dir is not
# writable (root-run) — e.g. benchmark_failure/l4i_repair_out
REPAIR_OUT = os.environ.get("L4I_REPAIR_OUT", os.path.join(OUT_ROOT, "repair_bddl"))

sys.path.insert(0, HERE)
from l4i_inject import SOURCE_SUITE, generate_scene_bddl, lookup_task  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--egl", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(args.egl))
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.binding_utils import MjSimState
    import pathlib

    recs, seen = [], set()
    for p in sorted(glob.glob(os.path.join(OUT_ROOT, "index", "injected*.jsonl"))):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if r["scene_id"] not in seen:
                seen.add(r["scene_id"])
                recs.append(r)
    done_ids = set()
    if os.path.exists(DONE):
        done_ids = {l.strip() for l in open(DONE, encoding="utf-8")}
    todo = [r for r in recs if r["scene_id"] not in done_ids]
    print(f"records: {len(recs)}, already repaired: {len(done_ids)}, todo: {len(todo)}")
    if args.limit:
        todo = todo[:args.limit]

    # reuse ONE suite benchmark object per suite and ONE env per task
    from libero.libero import benchmark
    suites = {s: benchmark.get_benchmark_dict()[s]() for s in set(SOURCE_SUITE.values())}
    # sort by (suite, task_dir) so each env is built once
    todo.sort(key=lambda r: (SOURCE_SUITE[r["original_source"]],
                             os.path.basename(os.path.dirname(r["original_scene_dir"]))))
    env = None
    env_key = None
    n_ok = n_err = 0
    with open(DONE, "a", encoding="utf-8") as df:
        for i, r in enumerate(todo):
            task_dir = os.path.basename(os.path.dirname(r["original_scene_dir"]))
            suite_name = SOURCE_SUITE[r["original_source"]]
            try:
                task = lookup_task(suites[suite_name], task_dir, r["task"])
                if task is None:
                    raise RuntimeError("task lookup failed")
                key = (suite_name, task_dir)
                if env_key != key:
                    if env is not None:
                        env.close()
                    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                    env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                                             camera_heights=256, camera_widths=256)
                    env.seed(7)
                    env.reset()
                    env_key = key
                # restore the ORIGINAL (pre-perturbation) state of this scene
                orig_npy = os.path.join(ROOT, r["original_scene_dir"],
                                        f"step_{r['step']:04d}_sim_state.npy")
                arr = np.load(orig_npy)
                env.env.sim.set_state(MjSimState.from_flattened(arr, env.env.sim))
                env.env.sim.forward()
                text = generate_scene_bddl(env)
                # scene dirs may be owned by another uid (full run as root), so
                # write to a parallel tree we own: l4_injected/repair_bddl/
                # mirroring scenes/<source>/<task>/<ep>/; afterwards one root
                # `cp -a repair_bddl/. scenes/` installs them over the originals
                fname = os.path.basename(r["bddl_file"])
                rel_scene_dir = os.path.relpath(r["scene_dir"],
                                                "benchmark_failure/l4_injected")
                out = os.path.join(REPAIR_OUT, rel_scene_dir, fname)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "w", encoding="utf-8") as f:
                    f.write(text)
                df.write(r["scene_id"] + "\n")
                n_ok += 1
            except Exception as e:
                print(f"[ERROR] {r['scene_id']}: {e}", file=sys.stderr)
                n_err += 1
                continue
            if (i + 1) % 50 == 0:
                print(f"[{i + 1}/{len(todo)}] repaired={n_ok} err={n_err}")
    if env is not None:
        env.close()
    print(f"repair: {n_ok} regenerated, {n_err} errors, {len(todo)} attempted")
    print(f"done list: {DONE}")


if __name__ == "__main__":
    main()
