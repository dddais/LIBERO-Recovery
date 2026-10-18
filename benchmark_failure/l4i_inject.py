#!/usr/bin/env python
"""L4-inject step 2: OBSTRUCTIVE perturbation in real scenes -> L4 twins.

L4 definition (aligned): a task-IRRELEVANT object is disturbed such that it
BLOCKS task completion and must be restored/removed before the task can
continue. Two obstruction strategies, chosen randomly per scene:

  onto   drop the unrelated object ON TOP of a task object (blocks it directly)
  beside place the unrelated object immediately NEXT TO a task object
         (6-12cm gap, toppling onto its side -> blocks the grasp approach)

For every selected scene (base = L1/L2, task objects basically intact):
  1. build env from the ORIGINAL task bddl (suite lookup), restore the scene's
     real sim_state (robot pose = real pre-fail history, untouched)
  2. pick the (unrelated object, task object) pair with the SMALLEST distance
     (most plausible "robot knocked it"); pairs where both objects share the
     same bddl category (e.g. two identical black bowls) are avoided — the VLM
     cannot tell identical-looking objects apart and misjudges the scene
  3. teleport the unrelated object per strategy (random 90-180deg rotation),
     freeze robot joint velocities, let physics settle
  4. verify the task object itself barely moved (retry with a safer parameter
     otherwise), then export the perturbed state (new sim_state .npy +
     regenerated scene bddl) into l4_injected/scenes/, and dump verification
     frames into l4_injected/frames/<scene_id>/: ORIGINAL npz frames
     (step-48..step, real robot actions) + the final perturbed render as the
     last frame.

Resumable: scenes with an existing DONE marker or already present in any
index/injected*.jsonl are skipped. Parallel: --shard k --num-shards n handles
rows[k::n] (each shard must use its OWN --egl device).
Paths from env: STARVLA_ROOT (data root, default /H20_vepfs/liulin/starVLA),
L4I_OUT_ROOT (output root, default <here>/l4_injected).
Usage (libero env, see run_l4i.sh):
  python l4i_inject.py [--egl 1] [--limit N] [--shard k --num-shards n]
"""
import argparse
import glob
import json
import os
import random
import re
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("STARVLA_ROOT", "/H20_vepfs/liulin/starVLA")
OUT_ROOT = os.environ.get("L4I_OUT_ROOT", os.path.join(HERE, "l4_injected"))
SEL = os.path.join(OUT_ROOT, "index/selection.jsonl")

SOURCE_SUITE = {"libero_10": "libero_10", "libero_goal": "libero_goal",
                "libero_object": "libero_object", "libero_spatial": "libero_spatial",
                "libero_90": "libero_90"}
CONTEXT_BEFORE = 48   # real npz frames: step-48 .. step, stride 8 (as in pass-2)
STRIDE = 8


# ---- bddl generation (copied from examples/LIBERO/eval_files/eval_libero_bddl.py) ----

def _format_goal_state(goal_state: list) -> str:
    if not goal_state:
        return "()"
    parts = []
    for item in goal_state:
        if isinstance(item, (list, tuple)):
            parts.append(f"({' '.join(str(x) for x in item)})")
        else:
            parts.append(str(item))
    if len(parts) == 1:
        return f"(And {parts[0]})"
    return "(And " + " ".join(parts) + ")"


def generate_scene_bddl(env) -> str:
    inner_env = env.env
    parsed = inner_env.parsed_problem
    skip_names = {"main_table", "kitchen_table", "floor", "countertop", "coffee_table",
                  "living_room_table", "study_table"}
    table_types = {"living_room_table", "kitchen_table", "main_table", "coffee_table",
                   "study_table", "table"}
    workspace_name = "main_table"
    for cat, names_list in parsed["fixtures"].items():
        if cat in table_types:
            workspace_name = names_list[0] if names_list else cat
            break
    else:
        # no table-type fixture (e.g. libero_object has only `floor`): fall back
        # to the first fixture — targets like a nonexistent "main_table" crash
        # env construction with KeyError at eval time
        for names_list in parsed["fixtures"].values():
            workspace_name = names_list[0]
            break
    obj_states = {}
    for name, body_id in inner_env.obj_body_id.items():
        if name in skip_names:
            continue
        pos = inner_env.sim.data.body_xpos[body_id].copy()
        quat = inner_env.sim.data.body_xquat[body_id].copy()
        yaw = np.arctan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                         1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2))
        obj_states[name] = {"pos": pos, "yaw": float(yaw)}

    # de-overlap init regions: scenes whose ORIGINAL layout stacks objects
    # (bowl on bowl / bowl on the ramekin / objects in the basket) produce
    # overlapping +-1cm regions that the reset sampler can NEVER place
    # collision-free (RandomizationError at build/reset for every seed).
    # set_init_state(npy) overwrites all poses at eval time, so region xy only
    # feeds the throwaway sampler layout: push overlapping pairs apart by
    # their real geom radii + margin (the smaller object moves).
    sim = inner_env.sim
    radii = {}
    for name in obj_states:
        body_id = inner_env.obj_body_id[name]
        r = 0.0
        g0, gn = sim.model.body_geomadr[body_id], sim.model.body_geomnum[body_id]
        for g in range(g0, g0 + gn):
            off = float(np.linalg.norm(sim.model.geom_pos[g][:2]))
            ext = float(np.linalg.norm(sim.model.geom_size[g][:2]))
            r = max(r, off + ext)
        radii[name] = r
    names = list(obj_states)
    for _ in range(8):
        moved_any = False
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                pa, pb = obj_states[a]["pos"], obj_states[b]["pos"]
                need = radii[a] + radii[b] + 0.02
                d = pb[:2] - pa[:2]
                dist = float(np.hypot(*d))
                if dist >= need:
                    continue
                if dist < 1e-6:
                    d, dist = np.array([1.0, 0.0]), 1e-6
                d = d / dist
                if radii[a] <= radii[b]:   # push a away from b
                    xy = pb[:2] - d * need
                    obj_states[a]["pos"] = np.array([xy[0], xy[1], pa[2]])
                else:                      # push b away from a
                    xy = pa[:2] + d * need
                    obj_states[b]["pos"] = np.array([xy[0], xy[1], pb[2]])
                moved_any = True
        if not moved_any:
            break
    half_len = 0.01
    # region ranges are interpreted RELATIVE to workspace_offset (see
    # bddl_base_domain._add_placement_initializer: reference_pos=workspace_offset;
    # e.g. study_table_offset=(-0.2,0,·)), while body_xpos is world-absolute.
    # Without this conversion every placement shifts by the table offset at
    # reset (STUDY scenes: caddy drifted -0.2m in x, killing the obstruction:
    # fixtures without free joints are NOT restored by set_init_state).
    woff = getattr(inner_env, "workspace_offset", (0.0, 0.0, 0.0))
    lines = [f"(define (problem {parsed['problem_name']})",
             "  (:domain robosuite)"]
    lang = parsed["language_instruction"]
    if isinstance(lang, list):
        lang = " ".join(lang)
    lines.append(f"  (:language {lang})")
    lines.append("    (:regions")
    for name, state in obj_states.items():
        x = float(state["pos"][0]) - float(woff[0])
        y = float(state["pos"][1]) - float(woff[1])
        yaw = state["yaw"]
        region_name = f"{name}_init_region"
        lines += [f"      ({region_name}",
                  f"          (:target {workspace_name})",
                  "          (:ranges (",
                  f"              ({x - half_len} {y - half_len} {x + half_len} {y + half_len})",
                  "            )",
                  "          )"]
        if abs(yaw) > 0.01:
            lines += ["          (:yaw_rotation (",
                      f"              ({yaw} {yaw})",
                      "            )",
                      "          )"]
        lines.append("      )")
    # re-emit ALL original regions verbatim (ranges + yaw): goal predicates
    # reference them (e.g. "push the plate to the stove-front REGION"), and a
    # dropped region makes _check_success KeyError at eval step time. Our own
    # per-object init regions above are ADDITIONAL, not replacements.
    for region_full_name, region_info in parsed["regions"].items():
        target = region_info["target"]
        prefix = target + "_"
        region_short = (region_full_name[len(prefix):]
                        if region_full_name.startswith(prefix) else region_full_name)
        lines += [f"      ({region_short}", f"          (:target {target})"]
        if region_info.get("ranges"):
            lines.append("          (:ranges (")
            for rng in region_info["ranges"]:
                lines.append(f"              ({' '.join(str(x) for x in rng)})")
            lines += ["            )", "          )"]
        yaw = region_info.get("yaw_rotation") or [0, 0]
        if abs(float(yaw[0])) > 0.01 or abs(float(yaw[1])) > 0.01:
            lines += ["          (:yaw_rotation (",
                      f"              ({yaw[0]} {yaw[1]})",
                      "            )",
                      "          )"]
        lines.append("      )")
    lines += ["    )\n", "  (:fixtures"]
    for cat, names_list in parsed["fixtures"].items():
        # all instances of a category on ONE line: the bddl parser OVERWRITES
        # (not merges) per-category entries, so one-line-per-instance silently
        # drops all but the last instance (KeyError at load for the others)
        lines.append("    " + " ".join(names_list) + f" - {cat}")
    lines += ["  )\n", "  (:objects"]
    for cat, names_list in parsed["objects"].items():
        lines.append("    " + " ".join(names_list) + f" - {cat}")
    lines += ["  )\n", "  (:obj_of_interest"]
    for o in parsed["obj_of_interest"]:
        lines.append(f"    {o}")
    lines += ["  )\n", "  (:init"]
    for name in obj_states:
        lines.append(f"    (On {name} {workspace_name}_{name}_init_region)")
    lines += ["  )\n", "  (:goal",
              f"    {_format_goal_state(parsed['goal_state'])}",
              "  )\n",
              ")"]
    return "\n".join(lines)


# ---- helpers ----

def parse_bddl_objects(bddl_path):
    """Return (object names, obj_of_interest names, name->category map)."""
    s = open(bddl_path, encoding="utf-8").read()
    m = re.search(r"\(:obj_of_interest(.*?)\)", s, re.S)
    ooi = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", m.group(1))) if m else set()
    m = re.search(r"\(:objects(.*?)\)", s, re.S)
    cats = dict(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s+-\s+([A-Za-z_][A-Za-z0-9_]*)",
                           m.group(1))) if m else {}
    return set(cats), ooi, cats


def rand_quat(nrng):
    axis = nrng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = np.deg2rad(nrng.uniform(90, 180))
    return np.array([np.cos(ang / 2), *(np.sin(ang / 2) * axis)])  # w x y z


def lookup_task(suite, task_dir, language):
    for pred in (lambda t: t.bddl_file == task_dir + ".bddl",
                 lambda t: t.language.replace(" ", "_") == task_dir,
                 lambda t: t.language == language):
        hits = [t for t in suite.tasks if pred(t)]
        if len(hits) == 1:
            return hits[0]
    return None


def free_joint_of(sim, body_id):
    """Return (qposadr, dofadr) if the body has a free root joint, else None."""
    jnt = sim.model.body_jntadr[body_id]
    if jnt < 0 or sim.model.jnt_type[jnt] != 0:  # 0 = free
        return None
    return int(sim.model.jnt_qposadr[jnt]), int(sim.model.jnt_dofadr[jnt])


def body_touching(sim, body_a, body_b):
    """True if the two bodies have at least one active MuJoCo contact."""
    want = {int(body_a), int(body_b)}
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        if {int(sim.model.geom_bodyid[c.geom1]), int(sim.model.geom_bodyid[c.geom2])} == want:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--egl", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--settle-steps", type=int, default=20)
    ap.add_argument("--shard", type=int, default=0, help="this process handles rows[shard::num_shards]")
    ap.add_argument("--num-shards", type=int, default=1, help="total parallel inject processes")
    ap.add_argument("--index-out", type=str, default="",
                    help="jsonl this process appends records to "
                         "(default index/injected.jsonl; parallel shards use "
                         "index/injected.shard<k>.jsonl)")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(args.egl))
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.binding_utils import MjSimState
    import pathlib

    selection = [json.loads(l) for l in open(SEL, encoding="utf-8")]
    selection = selection[args.shard::args.num_shards]
    # group by (suite, task_dir) so each env is built once
    selection.sort(key=lambda r: (SOURCE_SUITE[r["source"]],
                                  os.path.basename(os.path.dirname(r["scene_dir"]))))
    print(f"selected scenes: {len(selection)} (shard {args.shard}/{args.num_shards})")

    suites = {s: benchmark.get_benchmark_dict()[s]() for s in set(SOURCE_SUITE.values())}
    rng = random.Random(args.seed + args.shard)
    nrng = np.random.default_rng(args.seed + args.shard)
    env = None
    env_key = None
    inj_path = (args.index_out or os.path.join(OUT_ROOT, "index/injected.jsonl"))
    os.makedirs(os.path.dirname(inj_path), exist_ok=True)
    injected = open(inj_path, "a", encoding="utf-8")
    # resumable: skip every scene already recorded by ANY shard file
    done_ids = set()
    for p in glob.glob(os.path.join(OUT_ROOT, "index", "injected*.jsonl")):
        done_ids |= {json.loads(l)["scene_id"] for l in open(p, encoding="utf-8")}
    n_ok = n_skip = 0
    for i, r in enumerate(selection):
        new_id = f"l4_injected__{r['scene_id']}"
        if new_id in done_ids:
            n_skip += 1
            continue
        if args.limit and n_ok >= args.limit:
            break
        fdir = os.path.join(OUT_ROOT, "frames", new_id)
        if os.path.exists(os.path.join(fdir, "DONE")):
            n_skip += 1
            continue
        task_dir = os.path.basename(os.path.dirname(r["scene_dir"]))
        suite_name = SOURCE_SUITE[r["source"]]
        task = lookup_task(suites[suite_name], task_dir, r["task"])
        if task is None:
            print(f"[NO-TASK] {r['scene_id']}", file=sys.stderr)
            continue
        key = (suite_name, task_dir)
        if env_key != key:
            if env is not None:
                env.close()
            bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
            env.seed(7)
            env.reset()
            env_key = key
        inner = env.env
        sim = inner.sim

        state_file = os.path.join(ROOT, r["sim_state_file"])
        orig_bddl = os.path.join(ROOT, r["bddl_file"])
        arr = np.load(state_file)
        sim.set_state(MjSimState.from_flattened(arr, sim))
        sim.forward()

        # scene bddl is dumped from the PRE-perturbation state: the eval only
        # needs it to build the env and sample a collision-free reset layout
        # (regions never overlap), while set_init_state(npy) below restores the
        # exact perturbed state — the perturbation lives ONLY in the npy
        scene_bddl_text = generate_scene_bddl(env)

        # unrelated movable objects and task objects (obj_of_interest)
        objs, ooi, cats = parse_bddl_objects(orig_bddl)
        ooi_names = sorted(ooi & set(inner.obj_body_id.keys()))
        watch = [n for n in sorted((objs - ooi) & set(inner.obj_body_id.keys()))
                 if free_joint_of(sim, inner.obj_body_id[n]) is not None]
        if not watch or not ooi_names:
            print(f"[NO-WATCH] {r['scene_id']}", file=sys.stderr)
            continue

        # closest (unrelated, task-object) pairs = most plausible knocks; avoid
        # same-category pairs (identical-looking objects confuse the VLM judge)
        all_pairs = sorted(
            (float(np.linalg.norm(sim.data.body_xpos[inner.obj_body_id[n]]
                                  - sim.data.body_xpos[inner.obj_body_id[m]])), n, m)
            for n in watch for m in ooi_names)
        pairs = [p for p in all_pairs
                 if cats.get(p[1]) != cats.get(p[2])] or all_pairs

        # robot dof indices (to freeze during settle)
        robot_dofs = [int(sim.model.jnt_dofadr[j]) for j in range(sim.model.njnt)
                      if str(sim.model.joint_names[j]).startswith("panda")
                      and sim.model.jnt_dofadr[j] >= 0]
        ooi_before = {m: sim.data.body_xpos[inner.obj_body_id[m]].copy() for m in ooi_names}

        # obstruction strategy: 'onto' = drop ON the task object,
        #                      'beside' = place right NEXT TO it (blocks approach)
        # a scene is VALID L4 only if, after settle:
        #   (a) every task object moved <=3cm (task state intact), AND
        #   (b) the disturbed object PHYSICALLY TOUCHES a task object (contact)
        #       or leans right against one (xy gap <=5cm and |dz| <=5cm)
        # retry with safer params, then with the next-closest pair; a scene
        # that cannot be made valid is skipped (no invalid L4 ever written)
        strategy = rng.choice(["onto", "beside"])
        attempts = (0.10, 0.05, 0.02) if strategy == "onto" else (0.06, 0.09, 0.12)
        valid = None
        for dist0, name, oname in pairs[:3]:
            for param in attempts:
                sim.set_state(MjSimState.from_flattened(arr, sim))
                sim.forward()
                adr, dof = free_joint_of(sim, inner.obj_body_id[name])
                tpos = sim.data.body_xpos[inner.obj_body_id[oname]]
                q = sim.data.qpos
                if strategy == "onto":
                    q[adr] = tpos[0] + rng.uniform(-0.02, 0.02)
                    q[adr + 1] = tpos[1] + rng.uniform(-0.02, 0.02)
                    q[adr + 2] = tpos[2] + param
                else:  # beside: param = xy gap to keep from the task object
                    ang = rng.uniform(0, 2 * np.pi)
                    q[adr] = tpos[0] + param * np.cos(ang)
                    q[adr + 1] = tpos[1] + param * np.sin(ang)
                    q[adr + 2] = tpos[2]
                q[adr + 3:adr + 7] = rand_quat(nrng)
                sim.data.qvel[dof:dof + 6] = 0.0
                for _ in range(args.settle_steps):
                    sim.data.qvel[robot_dofs] = 0.0
                    sim.step()
                # recompute collision at the SETTLED qpos: data.contact after
                # step() reflects the pre-step configuration, so a contact that
                # broke during the last step would otherwise be a false positive
                sim.forward()
                moved = max(float(np.linalg.norm(sim.data.body_xpos[inner.obj_body_id[m]]
                                                 - ooi_before[m])) for m in ooi_names)
                if moved > 0.03:      # (a) violated: task object disturbed
                    continue
                a = sim.data.body_xpos[inner.obj_body_id[name]]
                b = sim.data.body_xpos[inner.obj_body_id[oname]]
                gap = float(np.hypot(*(a[:2] - b[:2])))
                dz = float(a[2] - b[2])
                contact = body_touching(sim, inner.obj_body_id[name],
                                        inner.obj_body_id[oname])
                if contact or (gap <= 0.05 and abs(dz) <= 0.05):   # (b) obstruction real
                    valid = {"strategy": strategy, "object": name, "onto": oname,
                             "param_m": param, "knocked_from_m": round(dist0, 3),
                             "ooi_moved_m": round(moved, 3),
                             "validity": {"contact": contact, "xy_gap_m": round(gap, 3),
                                          "dz_m": round(dz, 3)}}
                    break
            if valid:
                break
        if not valid:
            print(f"[INVALID] {r['scene_id']} (no valid obstruction found)", file=sys.stderr)
            continue

        # export perturbed scene files
        dst_dir = os.path.join(OUT_ROOT, "scenes", r["source"], task_dir,
                               os.path.basename(r["scene_dir"]))
        os.makedirs(dst_dir, exist_ok=True)
        step = r["step"]
        sim_name = f"l4i_step_{step:04d}_sim_state.npy"
        bddl_name = f"l4i_step_{step:04d}.bddl"
        np.save(os.path.join(dst_dir, sim_name), sim.get_state().flatten())
        open(os.path.join(dst_dir, bddl_name), "w", encoding="utf-8").write(scene_bddl_text)

        # verification frames: real npz history + final perturbed render
        os.makedirs(fdir, exist_ok=True)
        frames_meta = []
        npz = np.load(os.path.join(ROOT, r["npz_path"]))["images"]
        steps = [s for s in range(max(0, step - CONTEXT_BEFORE), step + 1, STRIDE)]
        if steps[-1] != step:
            steps.append(step)
        for j, s in enumerate(steps):
            p = os.path.join(fdir, f"{j:02d}_real{s:04d}.jpg")
            Image.fromarray(npz[s]).save(p, quality=90)
            frames_meta.append({"kind": "real", "step": s})
        img = sim.render(256, 256, camera_name="agentview")[::-1, ::-1]  # matches npz frames (180deg)
        p = os.path.join(fdir, f"{len(steps):02d}_perturbed.jpg")
        Image.fromarray(img).save(p, quality=92)
        frames_meta.append({"kind": "perturbed"})
        open(os.path.join(fdir, "DONE"), "w").write(f"{len(frames_meta)} frames")

        record = {
            "scene_id": new_id, "source": "l4_injected",
            "original_scene_id": r["scene_id"], "original_source": r["source"],
            "model": r["model"], "task": r["task"],
            "episode_index": r.get("episode_index"),
            "episode_idx_in_task": r.get("episode_idx_in_task"),
            "run_id": r.get("run_id"),
            "npz_path": r["npz_path"],
            "original_scene_dir": r["scene_dir"],
            "scene_dir": os.path.relpath(dst_dir, ROOT),
            "episode_length": r["episode_length"], "step": step,
            "timestamp_sec": r["timestamp_sec"],
            "bddl_file": os.path.relpath(os.path.join(dst_dir, bddl_name), ROOT),
            "sim_state_file": os.path.relpath(os.path.join(dst_dir, sim_name), ROOT),
            "stage": r["stage"], "fail_steps": r["fail_steps"],   # inherited: real annotation
            "level": "L4",   # validity guaranteed by construction (see injection.perturbed.validity)
            "level_reason": (f"constructed: {valid['object']} disturbed "
                             f"({'contact' if valid['validity']['contact'] else 'leaning'}, "
                             f"gap {valid['validity']['xy_gap_m']}m) onto {valid['onto']}"),
            "level_confidence": None,
            "frame_index_in_npz": step,
            "vlm_model": None,
            "injection": {"seed": args.seed, "settle_steps": args.settle_steps,
                          "perturbed": valid},
        }
        injected.write(json.dumps(record, ensure_ascii=False) + "\n")
        injected.flush()
        # the annotation IS the record (validity by construction, no VLM gate)
        out = os.path.join(OUT_ROOT, "annotations", f"{new_id}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(record, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        n_ok += 1
        if n_ok % 20 == 0:
            print(f"[{n_ok}/{len(selection) - n_skip}] {new_id}")

    injected.close()
    if env is not None:
        env.close()
    print(f"inject: {n_ok} newly injected, {n_skip} skipped (already done)")


if __name__ == "__main__":
    main()
