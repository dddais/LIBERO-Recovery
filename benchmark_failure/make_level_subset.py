#!/usr/bin/env python
"""Build an eval-ready scene subset filtered by suite and recovery level.

The eval client discovers scenes by globbing `*.bddl` under `--scene_dir`; it
does not read the annotation index. Use this helper to materialize a subset
(e.g. "all L4 scenes of libero_90") as a symlink tree you can pass directly:

  python benchmark_failure/make_level_subset.py \
      --suite libero_90 --level L4 \
      --out $STARVLA/benchmark_failure/subsets/libero_90_L4

  bash examples/LIBERO/eval_files/eval_libero_custom_scene.sh \
      --scene_dir $STARVLA/benchmark_failure/subsets/libero_90_L4

Symlinks (not copies) keep it near-free on disk. `--level` accepts a comma
list (L1,L2) or `any`; `--suite` likewise (or `any`).
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("STARVLA_ROOT", "/H20_vepfs/liulin/starVLA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="any",
                    help="suite or comma list (libero_10,libero_90,...) or any")
    ap.add_argument("--level", default="any",
                    help="level or comma list (L1,L2,L3,L4) or any")
    ap.add_argument("--out", required=True, help="output subset directory")
    args = ap.parse_args()

    suites = None if args.suite == "any" else set(args.suite.split(","))
    levels = None if args.level == "any" else set(args.level.split(","))
    n = 0
    for f in sorted(glob.glob(os.path.join(HERE, "annotations", "scenes", "*", "*.json"))):
        ann = json.load(open(f, encoding="utf-8"))
        if suites and ann["source"] not in suites:
            continue
        if levels and ann["level"] not in levels:
            continue
        for key in ("bddl_file", "sim_state_file"):
            src = os.path.join(ROOT, ann[key])
            rel = os.path.relpath(ann[key], "benchmark_failure")
            dst = os.path.join(args.out, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.lexists(dst):
                os.unlink(dst)
            os.symlink(src, dst)
        n += 1
    print(f"{n} scenes -> {args.out}")
    print(f"pass to the launcher:  --scene_dir {args.out}")


if __name__ == "__main__":
    main()
