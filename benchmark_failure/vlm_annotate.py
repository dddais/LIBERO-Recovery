#!/usr/bin/env python
"""Step 3+4: VLM annotation with the local Qwen3-VL-27B (env: miniconda3/envs/qwen).

Two passes (run `episodes` first, then `scenes`):

  episodes  one call per selected EPISODE: watch downsampled frames, output the step
            indices of failure-causing moments (grasp/place failure onset, collision,
            drop, wrong object...). -> annotations/episodes/<source>/<ep_key>.json
  scenes    one call per selected SCENE: frames around the scene step, judge the
            recovery level L1-L4. pre/mid/post-fail is then computed deterministically
            from (scene step vs fail steps):  |step-fail|<=20 -> mid-fail;
            step < first_fail-20 -> pre-fail;  otherwise -> post-fail.
            -> annotations/scenes/<source>/<scene_id>.json  (one JSON per scene)

Resumable: existing output files are skipped. Run with:
  /H20_vepfs/liulin/miniconda3/envs/qwen/bin/python vlm_annotate.py episodes --limit 5
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("STARVLA_ROOT", "/H20_vepfs/liulin/starVLA")
MODEL_PATH = os.environ.get("L4I_MODEL_PATH", "/H20_vepfs/liulin/Qwen3VL-27B")

EP_PROMPT = """你是一位机器人操作视频分析专家。

下面的图像序列是机械臂执行任务 "{task}" 的一段失败 rollout，按时间顺序排列。
图像是每隔 {stride} 个控制步采样一帧（每帧间隔 {dt:.2f} 秒，共 20 步/秒），
第 1 帧对应 step 0，第 k 帧对应 step (k-1)*{stride}。

请找出所有"失败性交互"发生的时刻（fail 时刻），包括：抓取失败或放置失败的最开始时刻、
物体掉落、碰撞、抓错物体、把物体放错位置等。如果同一物体多次失败，每次都单独列出。

要求：
- 以原始 step 序号为单位返回（等于帧号乘以 {stride}，允许合理估计，不要返回 null）。
- 只列出关键的失败性交互时刻（通常 1~6 个），不要把每个采样帧都列为失败。
- 输出紧凑的单行 JSON（不要换行、不要 markdown 代码围栏），格式严格如下：
{{ "fail_steps": [40, 180] }}"""

LEVEL_PROMPT = """你是一位机器人操作失败分析专家。

机械臂正在执行任务 "{task}"，但执行失败了。
下面的图像序列来自该失败 rollout，时间顺序排列，每隔 {stride} 个控制步采样一帧，
最后一帧（第 {n} 帧）是当前时刻的场景状态（step {step}），是你要评判的状态。

请根据当前时刻的场景状态，判断从该状态恢复任务所需的最低恢复等级：

- L1 Action Retry：失败未实质改变任务构型，物体位置/姿态基本如预期，重新执行原动作即可（如再抓一次）。
- L2 Action Adaptation：物体构型发生了有限变化（位置/朝向偏移），但物体仍可直接使用，需要调整动作而非机械重复。
- L3 Object State Recovery：任务相关物体状态被显著改变（掉落、翻倒、被移出工作区等），必须先恢复物体状态才能继续任务。
- L4 Environmental Recovery：任务无关物体被扰动（被打翻、移位、堆叠到任务物体或目标区域上），
  其当前状态阻碍任务完成，必须先恢复或移开这些无关物体才能继续任务。

要求：
- 只输出 JSON，格式严格如下（confidence 为 0~1 的小数）：
{{ "level": 1, "reason": "简短原因", "confidence": 0.9 }}"""


def load_model():
    from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return model, processor


def build_inputs(processor, image_paths, prompt):
    from PIL import Image
    content = [{"type": "image", "image": p} for p in image_paths]
    content.append({"type": "text", "text": prompt})
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    # This checkpoint's chat template always opens a <think> block and ignores the
    # enable_thinking kwarg; close the block empty so the model answers directly
    # (much faster, and output parses as JSON).
    if text.rstrip().endswith("<think>"):
        text = text.rstrip() + "\n\n</think>\n\n"
    images = [Image.open(p).convert("RGB") for p in image_paths]
    return processor(text=[text], images=images, return_tensors="pt")


def generate(model, processor, image_paths, prompt, max_new_tokens):
    inputs = build_inputs(processor, image_paths, prompt).to(model.device)
    ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    out = processor.batch_decode(
        [o[len(i):] for i, o in zip(inputs.input_ids, ids)],
        skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    if "</think>" in out:
        out = out.split("</think>")[-1].strip()
    return out


def parse_json(text):
    # strip markdown code fences (```json ... ```) and leading/trailing noise
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t).strip()
    try:
        return json.loads(t), None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]*\}", t, re.S)
    if m:
        try:
            return json.loads(m.group()), None
        except json.JSONDecodeError:
            pass
    # truncated output repair: cut at the last complete number, close open brackets
    for cut in range(len(t) - 1, 0, -1):
        if t[cut] in ",]\"}0123456789":
            cand = t[: cut + 1]
            closing = cand.count("[") - cand.count("]")
            cand += "]" * max(0, closing) + "}" * max(0, cand.count("{") - cand.count("}"))
            try:
                return json.loads(cand), None
            except json.JSONDecodeError:
                continue
    return None, text[:300]


def dump_frames(npz_path, steps, out_dir, key="images"):
    """Save the requested frames of an npz as jpgs; returns {step: path}."""
    import numpy as np
    from PIL import Image
    imgs = np.load(npz_path)[key]
    paths = {}
    for s in steps:
        if not (0 <= s < len(imgs)):
            continue
        p = os.path.join(out_dir, f"f{s:05d}.jpg")
        Image.fromarray(imgs[s]).save(p, quality=90)
        paths[s] = p
    return paths


def ep_key_of(sc):
    return sc["scene_id"].rsplit("__step", 1)[0]


def cmd_episodes(args, model, processor):
    selection = [json.loads(l) for l in open(args.selection, encoding="utf-8")]
    eps = {}
    for sc in selection:
        eps[ep_key_of(sc)] = sc
    items = sorted(eps.items())
    if args.limit:
        items = items[:args.limit]
    out_dir_root = os.path.join(HERE, "annotations/episodes")
    tmp = tempfile.mkdtemp(prefix="vlm_frames_")
    done = fail = 0
    for i, (ep_key, sc) in enumerate(items):
        out_path = os.path.join(out_dir_root, sc["source"], f"{ep_key}.json")
        if os.path.exists(out_path):
            done += 1
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        import numpy as np
        n = len(np.load(os.path.join(ROOT, sc["npz_path"]))["images"]) \
            if not args.frame_count else args.frame_count
        steps = list(range(0, n, args.stride))
        try:
            paths = dump_frames(os.path.join(ROOT, sc["npz_path"]), steps, tmp)
            ordered = [paths[s] for s in sorted(paths)]
            prompt = EP_PROMPT.format(task=sc["task"], stride=args.stride, dt=args.stride / 20.0)
            raw = generate(model, processor, ordered, prompt, args.max_new_tokens)
            parsed, err = parse_json(raw)
            if parsed is None or not isinstance(parsed.get("fail_steps"), list):
                print(f"[PARSE-FAIL] {ep_key}: {err}", file=sys.stderr)
                fail += 1
                continue
            fs = sorted({int(round(f)) for f in parsed["fail_steps"]})
            json.dump({"ep_key": ep_key, "task": sc["task"], "source": sc["source"],
                       "npz_path": sc["npz_path"], "episode_length": sc["episode_length"],
                       "stride": args.stride, "fail_steps": fs, "raw": raw},
                      open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            done += 1
            print(f"[{i+1}/{len(items)}] {ep_key} -> fail_steps={fs}")
        except Exception as e:
            print(f"[ERROR] {ep_key}: {e}", file=sys.stderr)
            fail += 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            tmp = tempfile.mkdtemp(prefix="vlm_frames_")
    print(f"episodes done={done} fail={fail}")


def compute_stage(step, fails, window=20):
    if not fails:
        return "unknown"
    if any(abs(step - f) <= window for f in fails):
        return "mid-fail"
    if step < fails[0] - window:
        return "pre-fail"
    return "post-fail"


def cmd_scenes(args, model, processor):
    selection = [json.loads(l) for l in open(args.selection, encoding="utf-8")]
    if args.limit:
        selection = selection[:args.limit]
    ep_dir_root = os.path.join(HERE, "annotations/episodes")
    out_dir_root = os.path.join(HERE, "annotations/scenes")
    tmp = tempfile.mkdtemp(prefix="vlm_frames_")
    done = fail = missing_ep = 0
    for i, sc in enumerate(selection):
        out_path = os.path.join(out_dir_root, sc["source"], f"{sc['scene_id']}.json")
        if os.path.exists(out_path):
            done += 1
            continue
        ep_path = os.path.join(ep_dir_root, sc["source"], f"{ep_key_of(sc)}.json")
        if not os.path.exists(ep_path):
            missing_ep += 1
            continue
        ep = json.load(open(ep_path, encoding="utf-8"))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        step = sc["step"]
        steps = sorted({s for s in range(max(0, step - args.context_before),
                                         step + args.context_after + 1, args.stride)} | {step})
        try:
            paths = dump_frames(os.path.join(ROOT, sc["npz_path"]), steps, tmp)
            ordered = [paths[s] for s in sorted(paths)]
            prompt = LEVEL_PROMPT.format(task=sc["task"], stride=args.stride,
                                         n=len(ordered), step=step)
            raw = generate(model, processor, ordered, prompt, args.max_new_tokens)
            parsed, err = parse_json(raw)
            if parsed is None or parsed.get("level") not in (1, 2, 3, 4, "1", "2", "3", "4"):
                print(f"[PARSE-FAIL] {sc['scene_id']}: {err}", file=sys.stderr)
                fail += 1
                continue
            result = dict(sc)  # all index fields: task, episode_index, npz_path, step, ...
            result["stage"] = compute_stage(step, ep["fail_steps"])
            result["fail_steps"] = ep["fail_steps"]
            result["level"] = f"L{int(parsed['level'])}"
            result["level_reason"] = parsed.get("reason", "")
            result["level_confidence"] = parsed.get("confidence")
            result["frame_index_in_npz"] = step
            result["vlm_model"] = MODEL_PATH
            json.dump(result, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            done += 1
            if (i + 1) % 20 == 0 or i + 1 == len(selection):
                print(f"[{i+1}/{len(selection)}] {sc['scene_id']} -> {result['stage']} {result['level']}")
        except Exception as e:
            print(f"[ERROR] {sc['scene_id']}: {e}", file=sys.stderr)
            fail += 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            tmp = tempfile.mkdtemp(prefix="vlm_frames_")
    print(f"scenes done={done} fail={fail} missing_episode_annotation={missing_ep}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("episodes", "scenes"):
        p = sub.add_parser(name)
        p.add_argument("--selection", default=os.path.join(HERE, "index/selection.jsonl"))
        p.add_argument("--stride", type=int, default=8)
        p.add_argument("--limit", type=int, default=0, help="only annotate first N items (smoke test)")
        p.add_argument("--max-new-tokens", type=int, default=512)
        if name == "episodes":
            p.add_argument("--frame-count", type=int, default=0,
                           help="override episode frame count (debug only)")
        else:
            p.add_argument("--context-before", type=int, default=48)
            p.add_argument("--context-after", type=int, default=16)
    args = ap.parse_args()
    model, processor = load_model()
    (cmd_episodes if args.cmd == "episodes" else cmd_scenes)(args, model, processor)


if __name__ == "__main__":
    main()
