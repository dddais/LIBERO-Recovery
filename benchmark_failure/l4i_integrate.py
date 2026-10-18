#!/usr/bin/env python
"""Integrate the L4-injected scenes INTO their original LIBERO suite folders.

Before: a separate `l4_injected/` source tree (source="l4_injected") — readers
cannot tell which suite a scene belongs to.
After:  L4 scene files live beside the original dumps of their suite:

    scenes/<suite>/<task>/<episode>/l4i_step_XXXX.bddl + _sim_state.npy
    annotations/scenes/<suite>/<scene_id>.json

Per record the `source` field becomes the real suite (libero_10/...); L4
identity stays discoverable via `level == "L4"`, the `l4_injected__` scene_id
prefix, and the `injection` provenance block. `original_scene_dir` etc. keep
pointing at the pre-injection snapshot.

Actions (all inside the liulin-writable trees; the root-owned l4_injected/
tree is only READ and stays as the raw provenance):
  1. copy per scene: repaired bddl (l4i_repair_out) + npy -> scenes/<suite>/...
  2. rewrite record paths + source; write annotation json to
     annotations/scenes/<suite>/<scene_id>.json; delete the old
     annotations/scenes/l4_injected/ copies
  3. rewrite index/selection.jsonl (465 L4 rows replaced; backup first) and
     index/selection_l4.jsonl; write index/injection_provenance.jsonl

Usage: python l4i_integrate.py [--dry-run]
"""
import argparse
import glob
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("STARVLA_ROOT", "/H20_vepfs/liulin/starVLA")
OUT_ROOT = os.path.join(HERE, "l4_injected")
REPAIR = os.path.join(HERE, "l4i_repair_out")
DROPPED = os.path.join(HERE, "l4i_dropped.txt")
SEL = os.path.join(HERE, "index", "selection.jsonl")
SEL_L4 = os.path.join(HERE, "index", "selection_l4.jsonl")


def load_records():
    dropped = {l.strip() for l in open(DROPPED, encoding="utf-8")
               if l.strip() and not l.startswith("#")}
    recs, seen = [], set()
    for p in sorted(glob.glob(os.path.join(OUT_ROOT, "index", "injected.shard*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r["scene_id"] not in seen:
                seen.add(r["scene_id"])
                recs.append(r)
    return [r for r in recs if r["scene_id"] not in dropped], dropped


def integrated(r):
    """Record with suite-integrated paths/source (pure dict transform)."""
    r = dict(r)
    task_dir = os.path.basename(os.path.dirname(r["original_scene_dir"]))
    ep_dir = os.path.basename(r["original_scene_dir"])
    suite = r["original_source"]
    r["source"] = suite
    r["scene_dir"] = f"benchmark_failure/scenes/{suite}/{task_dir}/{ep_dir}"
    r["bddl_file"] = f"{r['scene_dir']}/{os.path.basename(r['bddl_file'])}"
    r["sim_state_file"] = f"{r['scene_dir']}/{os.path.basename(r['sim_state_file'])}"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    keep, dropped = load_records()
    print(f"keep: {len(keep)}, dropped: {len(dropped)}")

    # 1+2. files + annotations
    n_files = 0
    for r in keep:
        r2 = integrated(r)
        rel_dir = os.path.relpath(r["scene_dir"], "benchmark_failure/l4_injected")
        repaired = os.path.join(REPAIR, rel_dir, os.path.basename(r["bddl_file"]))
        bddl_src = repaired if os.path.exists(repaired) else os.path.join(ROOT, r["bddl_file"])
        npy_src = os.path.join(ROOT, r["sim_state_file"])
        dst_dir = os.path.join(ROOT, r2["scene_dir"])
        if not args.dry_run:
            assert os.path.isdir(dst_dir), f"episode dir missing: {dst_dir}"
            shutil.copy2(bddl_src, os.path.join(ROOT, r2["bddl_file"]))
            shutil.copy2(npy_src, os.path.join(ROOT, r2["sim_state_file"]))
            ann_src = os.path.join(OUT_ROOT, "annotations", r["scene_id"] + ".json")
            with open(ann_src, encoding="utf-8") as f:
                ann = integrated(json.load(f))
            with open(os.path.join(HERE, "annotations", "scenes",
                                   r2["source"], r["scene_id"] + ".json"), "w",
                      encoding="utf-8") as f:
                json.dump(ann, f, ensure_ascii=False, indent=2)
        n_files += 2
    print(f"scene files copied: {n_files}, annotations written: {len(keep)}")

    # 3. index rewrite
    rows = [json.loads(l) for l in open(SEL, encoding="utf-8")]
    by_sid = {r["scene_id"]: r for r in rows}
    n_repl = 0
    for r in keep:
        r2 = integrated(r)
        if r["scene_id"] in by_sid:
            by_sid[r["scene_id"]] = r2
            n_repl += 1
        else:
            by_sid[r["scene_id"]] = r2
    out_rows = list(by_sid.values())
    print(f"selection.jsonl: {len(rows)} -> {len(out_rows)} rows ({n_repl} L4 replaced)")
    if args.dry_run:
        return
    shutil.copy2(SEL, SEL + ".pre_integrate.bak")
    with open(SEL, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(SEL_L4, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(integrated(r), ensure_ascii=False) + "\n")
    with open(os.path.join(HERE, "index", "injection_provenance.jsonl"), "w",
              encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(integrated(r), ensure_ascii=False) + "\n")
    old_ann_dir = os.path.join(HERE, "annotations", "scenes", "l4_injected")
    if os.path.isdir(old_ann_dir):
        shutil.rmtree(old_ann_dir)
        print("removed annotations/scenes/l4_injected/")
    print("done: selection.pre_integrate.bak / selection_l4.jsonl / "
          "injection_provenance.jsonl")


if __name__ == "__main__":
    main()
