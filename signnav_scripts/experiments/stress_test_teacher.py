#!/usr/bin/env python3
"""
stress_test_teacher.py  —  Stress-test the teacher sign-reader on a hard frame.

Probes robustness by running MANY goals against the SAME image and checking:
  - multi-arrow disambiguation (does goal X get the correct arrow group?)
  - spelling/synonym robustness (restroom vs restrooms vs bathroom)
  - the arrow-vs-position trap (sign on the left of frame, arrow points right)
  - distractor resistance (a big faculty directory board nearby)
  - timing per query

Each goal can carry an EXPECTED direction; the script flags pass/fail so you
see at a glance where the teacher breaks.

Usage:
    python stress_test_teacher.py --frame path/to/frame.jpg
    python stress_test_teacher.py --frame path/to/frame.jpg --box X Y W H   # tight crop
"""

import argparse
import json
import sys
import time

import torch
from PIL import Image

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# (goal, expected_direction or None if unknown/just-observe)
# EDIT these to match the sign in your test frame.
STRESS_GOALS = [
    # --- exact wording from the sign ---
    ("Restrooms",                 "right"),
    ("Staff Elevators",           "right"),
    ("Conference Hall 3-180",     "left"),
    ("Plaza & Church St. Entrances", "left"),
    # --- spelling / synonym robustness (should still resolve) ---
    ("restroom",                  "right"),   # singular
    ("bathroom",                  "right"),   # synonym
    ("restrooms please",          "right"),   # noisy phrasing
    # --- range matching ---
    ("classroom 3-220",           "left"),    # falls in "3-210 & 3-230" group
    ("room 3-140",                "right"),   # falls in "3-130 to 3-147" group
    # --- the arrow-vs-position trap (sign is on LEFT of frame; arrow says right) ---
    ("Stairs",                    "right"),   # if it says 'left' it's reading position not arrow
    # --- distractor / not-present goal (should say unknown, NOT hallucinate) ---
    ("cafeteria",                 "unknown"), # not on this sign at all
]


def goal_prompt(goal):
    return (
        "This is a photo taken by a robot in a building hallway. There may be a "
        "DIRECTIONAL/NAVIGATION sign (with arrows) and also unrelated boards "
        "(faculty directories, maps) — focus ONLY on the directional sign with arrows.\n"
        f"The robot's goal is to reach: \"{goal}\".\n"
        "Find the line on the directional sign that best matches this goal BY MEANING "
        "(treat singular/plural and synonyms as the same; match room numbers to ranges). "
        "Report the DIRECTION of the ARROW on that line: left, right, straight, or unknown. "
        "Use the ARROW, not where the sign sits in the image. "
        "If the goal is NOT on the directional sign, answer direction 'unknown'.\n"
        "Answer ONLY compact JSON:\n"
        '{"matched_sign_line": "...", "direction": "left|right|straight|unknown", '
        '"route_fact": "..."}'
    )


def load_model():
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
        from transformers import AutoProcessor
    except ImportError:
        sys.exit('Need Qwen2.5-VL: pip install "git+https://github.com/huggingface/transformers" accelerate')
    print(f"Loading {MODEL_ID} ...")
    t0 = time.perf_counter()
    model = VLModel.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"(load: {time.perf_counter()-t0:.1f}s)\n")
    return model, processor


def ask(model, processor, image, goal):
    conv = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": goal_prompt(goal)}]}]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    t0 = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    dt = time.perf_counter() - t0
    resp = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    try:
        s, e = resp.index("{"), resp.rindex("}") + 1
        return json.loads(resp[s:e]), dt
    except (ValueError, json.JSONDecodeError):
        return {"_raw": resp}, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--box", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    args = ap.parse_args()

    img = Image.open(args.frame).convert("RGB")
    if args.box:
        x, y, w, h = args.box
        img = img.crop((x, y, x + w, y + h))
        print(f"Cropped to {args.box} -> {img.size}")
    print(f"Frame: {args.frame}  size={img.size}\n")

    model, processor = load_model()

    print(f"=== Stress test: {len(STRESS_GOALS)} goals on one image ===\n")
    npass = nfail = 0
    times = []
    for goal, expected in STRESS_GOALS:
        result, dt = ask(model, processor, img, goal)
        times.append(dt)
        got = result.get("direction", "?")
        line = result.get("matched_sign_line", result.get("_raw", "?"))
        if expected is None:
            mark = "  ·"
        elif got == expected:
            mark = "  ✓"; npass += 1
        else:
            mark = "  ✗"; nfail += 1
        print(f"{mark} goal={goal!r:34s} expected={expected!s:8s} got={got!s:8s} [{dt:.2f}s]")
        print(f"      matched: {line}")
    print(f"\nPASS {npass} / FAIL {nfail} "
          f"(of {npass+nfail} with expected answers)")
    if len(times) > 1:
        warm = times[1:]
        print(f"Per-image inference: first={times[0]:.2f}s, "
              f"warm-avg={sum(warm)/len(warm):.2f}s")


if __name__ == "__main__":
    main()