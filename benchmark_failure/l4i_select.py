#!/usr/bin/env python
"""L4-inject step 1: sample scenes from the existing annotations as injection bases.

L1 bases first (task objects intact -> injected scene is PURE environmental
recovery), then fill from L2 up to --n. Sampled rows ARE the original
annotation JSONs (all fields: task, stage, fail_steps, npz_path, ...), so
injected twins inherit them.

Usage: python l4i_select.py [--n 620] [--seed 20261013] [--pools L1,L2]
"""
import argparse
import glob
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ANN = os.environ.get("L4I_ANN_ROOT", os.path.join(HERE, "annotations/scenes"))
OUT_DIR = os.path.join(os.environ.get("L4I_OUT_ROOT", os.path.join(HERE, "l4_injected")), "index")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=620,
                    help="sample size (slight buffer over 600: VLM verify may drop a few)")
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--pools", type=str, default="L1,L2",
                    help="base levels, in priority order (comma separated)")
    args = ap.parse_args()

    pools = args.pools.split(",")
    by_level = {p: [] for p in pools}
    for p in glob.glob(os.path.join(ANN, "*", "*.json")):
        r = json.load(open(p, encoding="utf-8"))
        if r.get("level") in by_level:
            by_level[r["level"]].append(r)

    rng = random.Random(args.seed)
    sample = []
    for p in pools:  # earlier pools have priority, later pools fill the remainder
        take = min(args.n - len(sample), len(by_level[p]))
        if take > 0:
            sample += rng.sample(by_level[p], take)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "selection.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    print("pools:", {p: len(v) for p, v in by_level.items()},
          f" sampled: {len(sample)}  (seed {args.seed})")
    print("base level:", dict(Counter(r["level"] for r in sample)))
    print("by source:", dict(Counter(r["source"] for r in sample)))
    print("by stage :", dict(Counter(r["stage"] for r in sample)))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
