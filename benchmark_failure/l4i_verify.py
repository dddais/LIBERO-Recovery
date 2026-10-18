#!/usr/bin/env python
"""L4-inject step 3: VLM audit — annotations stay L4 regardless of verdict.

Validity of constructed L4 scenes is guaranteed BY CONSTRUCTION in
l4i_inject.py. This step runs the VLM with the UNCHANGED LEVEL_PROMPT and
records its opinion into l4_injected/index/verdicts.jsonl (audit trail only).
It NEVER modifies annotation levels: every injected scene's annotation JSON
keeps level="L4" and all other fields exactly as written at injection time;
if an annotation file is missing (e.g. interrupted run), it is written from
the injected.jsonl record verbatim.

Usage (qwen env, see run_l4i.sh): python l4i_verify.py [--limit N]
Paths come from env: L4I_OUT_ROOT (default <here>/l4_injected),
L4I_MODEL_PATH / STARVLA_ROOT (see vlm_annotate.py).
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.environ.get("L4I_OUT_ROOT", os.path.join(HERE, "l4_injected"))
VERDICTS = os.path.join(OUT_ROOT, "index/verdicts.jsonl")

sys.path.insert(0, HERE)
from vlm_annotate import LEVEL_PROMPT, MODEL_PATH, generate, load_model, parse_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    # merge view over ALL injected*.jsonl (single run + parallel shard files)
    rows, seen = [], set()
    for p in sorted(glob.glob(os.path.join(OUT_ROOT, "index", "injected*.jsonl"))):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if r["scene_id"] not in seen:
                seen.add(r["scene_id"])
                rows.append(r)
    done = set()
    if os.path.exists(VERDICTS):
        done = {json.loads(l)["scene_id"] for l in open(VERDICTS, encoding="utf-8")}
    print(f"injected: {len(rows)}, already verified: {len(done)}")

    model, processor = load_model()
    n_l4 = n_checked = 0
    with open(VERDICTS, "a", encoding="utf-8") as vf:
        for r in rows:
            if r["scene_id"] in done:
                continue
            if args.limit and n_checked >= args.limit:
                break
            frames = sorted(glob.glob(os.path.join(OUT_ROOT, "frames", r["scene_id"], "*.jpg")))
            if not frames:
                print(f"[NO-FRAMES] {r['scene_id']}", file=sys.stderr)
                continue
            prompt = LEVEL_PROMPT.format(task=r["task"], stride=8,
                                         n=len(frames), step=r["step"])
            try:
                raw = generate(model, processor, frames, prompt, args.max_new_tokens)
                parsed, err = parse_json(raw)
                if parsed is None or parsed.get("level") not in (1, 2, 3, 4, "1", "2", "3", "4"):
                    print(f"[PARSE-FAIL] {r['scene_id']}: {err}", file=sys.stderr)
                    continue
                level = int(parsed["level"])
                vf.write(json.dumps({"scene_id": r["scene_id"], "level": level,
                                     "reason": parsed.get("reason", ""),
                                     "confidence": parsed.get("confidence")},
                                    ensure_ascii=False) + "\n")
                vf.flush()
                n_checked += 1
                if level == 4:
                    n_l4 += 1
                # annotation always L4, never overwritten by the VLM verdict;
                # write it verbatim from the injected record if missing
                out = os.path.join(OUT_ROOT, "annotations", f"{r['scene_id']}.json")
                if not os.path.exists(out):
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    json.dump(r, open(out, "w", encoding="utf-8"),
                              ensure_ascii=False, indent=1)
                if n_checked % 20 == 0:
                    print(f"[{n_checked}/{len(rows) - len(done)}] vlm_l4={n_l4} "
                          f"({100.0 * n_l4 / max(1, n_checked):.1f}%)")
            except Exception as e:
                print(f"[ERROR] {r['scene_id']}: {e}", file=sys.stderr)
    print(f"verify: checked {n_checked}, L4 confirmed {n_l4} "
          f"({100.0 * n_l4 / max(1, n_checked):.1f}%)")
    print(f"annotations: {OUT_ROOT}/annotations/")


if __name__ == "__main__":
    main()
