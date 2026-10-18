#!/usr/bin/env python
"""Stage the two ModelScope release datasets for the failure-recovery benchmark
(outside the git repo so nothing large can be committed accidentally).

  1. LIBERO_Recovery_FailureScenes  (~40 MB)
       scenes/            4338 L1-L3 dumps (bddl + npy)
       l4_injected/       465 kept L4 scenes (npy from the live tree, bddl from
                          the REPAIRED tree l4i_repair_out — works whether or
                          not the root `cp -a` install has been run), plus
                          annotations/ and index/
       annotations/       episodes + scenes (L1-L3 + merged l4_injected)
       index/             selection.jsonl (4803), selection_l4.jsonl,
                          scenes_index.jsonl (26710), l4_verdicts.jsonl
       l4i_dropped.txt    the 11 dropped scenes
     NOT included: l4_injected/frames (per decision), npz rollouts.

  2. LIBERO_Recovery_Rollouts  (80 GB, HARDLINKS — same filesystem, ~no cost)
       LIBERO/eval_result/...  exactly the 1446 npz files referenced by
                               selection.jsonl, at their starVLA-relative paths
                               so npz_path fields resolve against $STARVLA.

Usage: python make_release_datasets.py [--out /H20_vepfs/liulin/release]
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
    return recs, dropped


def stage_failure(dst, recs, dropped, dry):
    # suite-integrated layout (see l4i_integrate.py): L4 scene files and
    # annotations already live inside their LIBERO suite trees, so staging is
    # a plain copy of scenes/ + annotations/ + index/
    def cp(src, rel):
        if dry:
            return
        t = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(t), exist_ok=True)
        shutil.copy2(src, t)

    for sub in ("scenes", "annotations"):
        n = 0
        for f in glob.glob(os.path.join(HERE, sub, "**", "*.*"), recursive=True):
            if sub == "annotations":
                sid = os.path.basename(f)[:-5]
                if sid in dropped:
                    continue
            cp(f, os.path.relpath(f, HERE)); n += 1
        print(f"[FailureScenes] {sub}/: {n} files")
    for name in ("selection.jsonl", "selection_l4.jsonl", "scenes_index.jsonl",
                 "l4_verdicts.jsonl", "injection_provenance.jsonl"):
        cp(os.path.join(HERE, "index", name), os.path.join("index", name))
    cp(DROPPED, "l4i_dropped.txt")
    print("[FailureScenes] index/ + l4i_dropped.txt: done")


def stage_rollouts(dst, dry):
    npzs = sorted({json.loads(l)["npz_path"]
                   for l in open(os.path.join(HERE, "index", "selection.jsonl"),
                                 encoding="utf-8")})
    total = 0
    for i, rel in enumerate(npzs):
        src = os.path.join(ROOT, rel)
        assert os.path.exists(src), f"missing {src}"
        total += os.path.getsize(src)
        if dry:
            continue
        t = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(t), exist_ok=True)
        if os.path.exists(t):
            os.unlink(t)
        try:
            os.link(src, t)          # same filesystem: zero-cost
        except OSError:
            shutil.copy2(src, t)     # fallback across filesystems
        if (i + 1) % 300 == 0:
            print(f"  [Rollouts] {i+1}/{len(npzs)}")
    print(f"[Rollouts] {len(npzs)} npz, {total/1e9:.1f} GB "
          f"({'hardlinked' if not dry else 'dry-run'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/H20_vepfs/liulin/release")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip", choices=("failure", "rollouts"), action="append", default=[])
    args = ap.parse_args()

    recs, dropped = load_records()
    print(f"records: {len(recs)}, dropped: {len(dropped)}")
    if "failure" not in args.skip:
        stage_failure(os.path.join(args.out, "LIBERO_Recovery_FailureScenes"),
                      recs, dropped, args.dry_run)
    if "rollouts" not in args.skip:
        stage_rollouts(os.path.join(args.out, "LIBERO_Recovery_Rollouts"), args.dry_run)


if __name__ == "__main__":
    main()
