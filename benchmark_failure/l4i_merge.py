#!/usr/bin/env python
"""Merge the verified L4-injected scenes into the main benchmark tree.

What it does (idempotent-ish: backups made before any mutation):
  1. annotations: copy the kept L4 annotation jsons to
     annotations/scenes/l4_injected/   (11 dropped scenes excluded, see
     l4i_dropped.txt)
  2. index: append the kept L4 records to index/selection.jsonl
     (backup -> index/selection.pre_l4.jsonl.bak), and write the standalone
     index/selection_l4.jsonl (465 rows) for L4-only eval runs
  3. scene FILES are NOT moved: records already point at
     benchmark_failure/l4_injected/scenes/...; the repaired bddls still need
     the one-time root install (printed at the end if pending):
       cp -a <STARVLA>/benchmark_failure/l4i_repair_out/. \\
             <STARVLA>/benchmark_failure/l4_injected/scenes/

Usage: python l4i_merge.py [--dry-run]
"""
import argparse
import glob
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("STARVLA_ROOT", "/H20_vepfs/liulin/starVLA")
OUT_ROOT = os.path.join(HERE, "l4_injected")
DROPPED = os.path.join(HERE, "l4i_dropped.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dropped = {l.strip() for l in open(DROPPED, encoding="utf-8")
               if l.strip() and not l.startswith("#")}
    recs, seen = [], set()
    for p in sorted(glob.glob(os.path.join(OUT_ROOT, "index", "injected.shard*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r["scene_id"] not in seen:
                seen.add(r["scene_id"])
                recs.append(r)
    keep = [r for r in recs if r["scene_id"] not in dropped]
    print(f"records: {len(recs)}, dropped: {len(dropped)}, keep: {len(keep)}")

    # 1. annotations -> annotations/scenes/l4_injected/
    ann_src = os.path.join(OUT_ROOT, "annotations")
    ann_dst = os.path.join(HERE, "annotations", "scenes", "l4_injected")
    os.makedirs(ann_dst, exist_ok=True)
    n_ann = 0
    for r in keep:
        f = os.path.join(ann_src, r["scene_id"] + ".json")
        assert os.path.exists(f), f"missing annotation {f}"
        if args.dry_run:
            continue
        shutil.copy2(f, os.path.join(ann_dst, r["scene_id"] + ".json"))
        n_ann += 1
    print(f"annotations copied: {n_ann} -> annotations/scenes/l4_injected/"
          if not args.dry_run else "annotations would be copied "
          f"(dry-run): {len(keep)}")

    # 2a. standalone L4 index
    if not args.dry_run:
        with open(os.path.join(HERE, "index", "selection_l4.jsonl"), "w",
                  encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote index/selection_l4.jsonl ({len(keep)} rows)")

    # 2b. append to the main selection (backup first)
    sel = os.path.join(HERE, "index", "selection.jsonl")
    bak = os.path.join(HERE, "index", "selection.pre_l4.jsonl.bak")
    already = set()
    for line in open(sel, encoding="utf-8"):
        already.add(json.loads(line)["scene_id"])
    new = [r for r in keep if r["scene_id"] not in already]
    print(f"selection: {len(already)} existing, {len(new)} new L4 rows to append")
    if args.dry_run:
        return
    if not os.path.exists(bak):
        shutil.copy2(sel, bak)
        print(f"backup: index/selection.pre_l4.jsonl.bak")
    with open(sel, "a", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"selection.jsonl now: {len(already) + len(new)} rows")

    # 3. repaired-bddl install check
    rep_root = os.path.join(HERE, "l4i_repair_out")
    scn_root = os.path.join(OUT_ROOT, "scenes")
    pending = 0
    for r in keep:
        rel = os.path.relpath(r["scene_dir"], "benchmark_failure/l4_injected")
        fixed = os.path.join(rep_root, rel, os.path.basename(r["bddl_file"]))
        live = os.path.join(ROOT, r["bddl_file"])
        if os.path.exists(fixed) and (not os.path.exists(live)
                                      or open(fixed, "rb").read() != open(live, "rb").read()):
            pending += 1
    if pending:
        print(f"\n[ACTION REQUIRED] {pending} live bddls differ from the repaired "
              f"versions — run as root:")
        print("  cp -a /H20_vepfs/liulin/starVLA/benchmark_failure/l4i_repair_out/. "
              "/H20_vepfs/liulin/starVLA/benchmark_failure/l4_injected/scenes/")
    else:
        print("repaired bddls: all installed")


if __name__ == "__main__":
    main()
