#!/usr/bin/env python
"""Step 1: build a unified scene index over the 5 benchmark sources.

Sources (ROOT = /H20_vepfs/liulin/starVLA):
  - libero_10    : assets_times/bddl_scenes_10    + actionhead/Qwen3-VL-PI-LIBERO-4in1/libero_10
  - libero_goal  : assets_times/bddl_scenes_goal  + .../libero_goal
  - libero_object: assets_times/bddl_scenes_object+ .../libero_object
  - libero_spatial: assets_times/bddl_scenes_spatial + .../libero_spatial
  - libero_90    : actionhead_rerun/Qwen3-VL-PI-V2-LIBERO-4in1/libero_90 (clean rerun,
                   manifest rows carry run_id/npz_file/episode_idx_in_task/bddl_scene_dir)

Matching rule:
  - old PI suites: scene dir "<...>SCENE<n>_<task_us>" or plain "<task_us>"; episode dir
    episode_%03d is the trial index within the task -> manifest row with
    (task == extracted task) and (episode_index % 50 == trial).
  - libero_90 rerun: direct field lookup in the manifest (no guessing).

Each step_XXXX.bddl + step_XXXX_sim_state.npy pair = ONE scene.
Output: scenes_index.jsonl (one line per scene) + printed summary.

Usage: python build_index.py [--out index/scenes_index.jsonl]
"""
import argparse
import dataclasses
import json
import os
import re
import sys
from collections import defaultdict

ROOT = "/H20_vepfs/liulin/starVLA"
PI_MODEL = "Qwen3-VL-PI-LIBERO-4in1"
RERUN_MODEL = "Qwen3-VL-PI-V2-LIBERO-4in1"

OLD_SOURCES = {
    "libero_10": "assets_times/bddl_scenes_10",
    "libero_goal": "assets_times/bddl_scenes_goal",
    "libero_object": "assets_times/bddl_scenes_object",
    "libero_spatial": "assets_times/bddl_scenes_spatial",
}

_SCENE_PREFIX = re.compile(r"^(?:[A-Z]+_)*SCENE\d+_(.+)$")
FAIL_STEP_RE = re.compile(r"step_(\d+)")


def task_from_dir(task_dir: str) -> str:
    m = _SCENE_PREFIX.match(task_dir)
    us = m.group(1) if m else task_dir
    return us.replace("_", " ")


def load_manifest(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def rel(p: str) -> str:
    return os.path.relpath(p, ROOT)


def iter_old_source(source: str, scenes_rel: str):
    """Yield (task, trial_idx, episode_dir_abs, manifest_row) for an old PI suite."""
    man_path = os.path.join(ROOT, "LIBERO/eval_result/actionhead", PI_MODEL, source,
                            "libero_goal/raw/manifest.jsonl")
    by_task = defaultdict(list)
    for r in load_manifest(man_path):
        by_task[r["task"]].append(r)

    scenes_abs = os.path.join(ROOT, scenes_rel)
    for task_dir in sorted(os.listdir(scenes_abs)):
        task = task_from_dir(task_dir)
        rows = by_task.get(task)
        if not rows:
            print(f"[WARN] {source}: no manifest task for dir {task_dir}", file=sys.stderr)
            continue
        tdir_abs = os.path.join(scenes_abs, task_dir)
        for ep_dir in sorted(os.listdir(tdir_abs)):
            trial = int(ep_dir.rsplit("_", 1)[1])
            hits = [r for r in rows if r["episode_index"] % 50 == trial]
            if len(hits) != 1 or hits[0]["success"]:
                print(f"[WARN] {source}/{task_dir}/{ep_dir}: {len(hits)} manifest hits", file=sys.stderr)
                continue
            yield task, task_dir, trial, os.path.join(tdir_abs, ep_dir), hits[0]


def iter_rerun_source():
    """Yield (task, task_dir, trial, episode_dir_abs, manifest_row) for the libero_90 rerun."""
    suite_abs = os.path.join(ROOT, "LIBERO/eval_result/actionhead_rerun", RERUN_MODEL, "libero_90")
    man_path = os.path.join(suite_abs, "libero_goal/raw/manifest.jsonl")
    scenes_abs = os.path.join(suite_abs, "bddl_scenes")
    for r in load_manifest(man_path):
        if r["success"] or not r.get("bddl_scene_dir"):
            continue
        ep_abs = os.path.join(scenes_abs, r["bddl_scene_dir"])
        if not os.path.isdir(ep_abs):
            print(f"[WARN] missing scene dir {r['bddl_scene_dir']}", file=sys.stderr)
            continue
        task_dir = r["bddl_scene_dir"].split("/")[0]
        yield r["task"], task_dir, r["episode_idx_in_task"], ep_abs, r


def scene_row(source, model, task, task_dir, trial, ep_dir, row, npz_abs, run_id):
    scene_dir_rel = rel(ep_dir)
    ep_name = os.path.basename(ep_dir)
    return {
        "scene_id": f"{source}__{task_dir}__{ep_name}",
        "source": source,
        "model": model,
        "task": task,
        "episode_index": row["episode_index"],        # global index inside the manifest
        "episode_idx_in_task": trial,
        "run_id": run_id,
        "npz_path": rel(npz_abs),
        "scene_dir": scene_dir_rel,
        "episode_length": row["length"],
        "first_contact_step": row.get("first_contact_step"),
        "first_gripper_change_step": row.get("first_gripper_change_step"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "index/scenes_index.jsonl"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_scenes = defaultdict(int)
    n_eps = defaultdict(int)
    with open(args.out, "w", encoding="utf-8") as out:
        # ---- old PI suites -------------------------------------------------
        for source, scenes_rel in OLD_SOURCES.items():
            npz_dir = os.path.join(ROOT, "LIBERO/eval_result/actionhead", PI_MODEL, source,
                                   "libero_goal/raw")
            for task, task_dir, trial, ep_dir, row in iter_old_source(source, scenes_rel):
                npz_abs = os.path.join(npz_dir, f"episode_{row['episode_index']:06d}.npz")
                if not os.path.exists(npz_abs):
                    print(f"[WARN] missing npz {npz_abs}", file=sys.stderr)
                    continue
                base = scene_row(source, PI_MODEL, task, task_dir, trial, ep_dir, row, npz_abs, None)
                n_eps[source] += 1
                for fname in sorted(os.listdir(ep_dir)):
                    if not fname.endswith(".bddl"):
                        continue
                    step = int(FAIL_STEP_RE.match(fname).group(1))
                    r = dict(base)
                    r["step"] = step
                    r["timestamp_sec"] = round(step / 20.0, 3)
                    r["scene_id"] = f"{base['scene_id']}__step{step:04d}"
                    r["bddl_file"] = f"{base['scene_dir']}/{fname}"
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n_scenes[source] += 1

        # ---- libero_90 clean rerun ------------------------------------------
        source = "libero_90"
        npz_dir = os.path.join(ROOT, "LIBERO/eval_result/actionhead_rerun", RERUN_MODEL,
                               "libero_90/libero_goal/raw")
        for task, task_dir, trial, ep_dir, row in iter_rerun_source():
            npz_abs = os.path.join(npz_dir, row["npz_file"])
            if not os.path.exists(npz_abs):
                print(f"[WARN] missing npz {npz_abs}", file=sys.stderr)
                continue
            base = scene_row(source, RERUN_MODEL, task, task_dir, trial, ep_dir, row,
                             npz_abs, row["run_id"])
            n_eps[source] += 1
            for fname in sorted(os.listdir(ep_dir)):
                if not fname.endswith(".bddl"):
                    continue
                step = int(FAIL_STEP_RE.match(fname).group(1))
                r = dict(base)
                r["step"] = step
                r["timestamp_sec"] = round(step / 20.0, 3)
                r["scene_id"] = f"{base['scene_id']}__step{step:04d}"
                r["bddl_file"] = f"{base['scene_dir']}/{fname}"
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_scenes[source] += 1

    print("\n=== scene index summary ===")
    for s in list(OLD_SOURCES) + ["libero_90"]:
        print(f"{s:15s} episodes={n_eps[s]:5d} scenes={n_scenes[s]:6d}")
    print(f"total scenes: {sum(n_scenes.values())}")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
