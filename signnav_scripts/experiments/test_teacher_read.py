#!/usr/bin/env python3
"""
test_teacher_read.py  —  TEST 1: can a VLM read destination + direction from our sign crops?

This is the GATE for the whole distillation approach. If a strong VLM can't read
the destination AND the arrow direction from our real sign crops, the information
isn't recoverable and no crop-pathway/training will fix it. If it CAN, the question
becomes "route that info into OmniVLA" (trainable). So run this first.

Uses Qwen2.5-VL-7B-Instruct (strong document/text reading; dynamic resolution, so it
reads the crop at high res — a fair test of legibility). This also doubles as the
spec for Yehor's sign-reader: same input (crop), same output (route-fact string).

Input: either a pre-cropped sign image, OR a full frame + a crop box (x y w h).
Output (per image): the model's read in the route-fact format the distillation targets:
    {"destination": "...", "direction": "left|right|straight|unknown", "route_fact": "..."}

Setup (in the OmniVLA env or a fresh one with a GPU):
    pip install "transformers>=4.49" accelerate torchvision qwen-vl-utils
    # (build transformers from source if the version is too old for Qwen2.5-VL)

Usage:
    # whole crop images already saved:
    python test_teacher_read.py --crops sign1.jpg sign2.jpg

    # or full frames + a box to crop first:
    python test_teacher_read.py --frame frames/<ts>.jpg --box 900 200 200 400
"""

import argparse
import json
import sys
import time

import torch
from PIL import Image

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Ungated prompt: no goal given, picks the most prominent destination.
PROMPT = (
    "This is a photo of a navigational/directional sign in a building. "
    "Read it and tell me, for navigation:\n"
    "1) the DESTINATION it points to (e.g. 'restroom', 'cafeteria', 'classroom 3-114'),\n"
    "2) the DIRECTION its arrow indicates: one of left, right, straight, or unknown.\n"
    "If there are multiple destinations, pick the single most prominent one.\n"
    "Answer ONLY as compact JSON, no other text, like:\n"
    '{"destination": "restroom", "direction": "right", '
    '"route_fact": "the restroom is to the right"}'
)

# Goal-aware prompt: given a target, find THAT destination's row and report ITS arrow.
def goal_prompt(goal):
    return (
        "This is a photo of a navigational/directional sign in a building. "
        f"The robot's goal is to reach: \"{goal}\".\n"
        "Find the line on the sign that best matches this goal BY MEANING, not by "
        "exact spelling. Treat singular/plural and synonyms as the same destination "
        "(e.g. 'restroom' = 'restrooms' = 'bathroom' = 'toilet'; "
        "'classroom 3-120' matches a range like 'Classrooms 3-111 to 3-125'). "
        "Then report the DIRECTION of the arrow associated with THAT line: "
        "one of left, right, straight, or unknown.\n"
        "Answer ONLY as compact JSON, no other text, like:\n"
        '{"destination": "restrooms", "direction": "right", '
        '"matched_sign_line": "Restrooms", '
        '"route_fact": "the restrooms are to the right"}'
    )


def load_model():
    # Qwen2.5-VL uses its own class; fall back to Qwen2-VL if needed.
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
        from transformers import AutoProcessor
    except ImportError:
        print("Qwen2.5-VL not in this transformers; install/upgrade:")
        print('  pip install "git+https://github.com/huggingface/transformers" accelerate')
        sys.exit(1)
    print(f"Loading {MODEL_ID} ... (downloads to HF cache on first run)")
    model = VLModel.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def read_sign(model, processor, image: Image.Image, prompt: str = PROMPT):
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    t0 = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    elapsed = time.perf_counter() - t0
    trimmed = out[0][inputs.input_ids.shape[1]:]
    resp = processor.decode(trimmed, skip_special_tokens=True).strip()
    # try to parse JSON; if the model added stray text, extract the braces
    try:
        start, end = resp.index("{"), resp.rindex("}") + 1
        return json.loads(resp[start:end]), resp, elapsed
    except (ValueError, json.JSONDecodeError):
        return None, resp, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", nargs="*", help="Pre-cropped sign images")
    ap.add_argument("--frame", help="Full frame to crop from")
    ap.add_argument("--box", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                    help="Crop box on --frame")
    ap.add_argument("--goal", help="Target destination; if set, asks the direction "
                    "for THIS goal (disambiguates multi-destination signs)")
    args = ap.parse_args()

    prompt = goal_prompt(args.goal) if args.goal else PROMPT
    if args.goal:
        print(f"GOAL-CONDITIONED read for goal: \"{args.goal}\"\n")

    images = []
    if args.frame:
        frame = Image.open(args.frame).convert("RGB")
        if args.box:
            x, y, w, h = args.box
            crop = frame.crop((x, y, x + w, y + h))
            print(f"Cropped {args.frame} to box {args.box} -> {crop.size}")
            images.append((f"{args.frame}[crop]", crop))
        else:
            images.append((args.frame, frame))   # whole frame; tests if it finds the sign
    for c in (args.crops or []):
        images.append((c, Image.open(c).convert("RGB")))

    if not images:
        sys.exit("Give --crops <files> and/or --frame <file> [--box x y w h]")

    t_load0 = time.perf_counter()
    model, processor = load_model()
    load_time = time.perf_counter() - t_load0
    print(f"\n(model load took {load_time:.1f}s — one-time cost, not per-image)\n")
    print("=== Test 1: teacher sign-reading on real crops ===\n")
    times = []
    for name, img in images:
        parsed, raw, elapsed = read_sign(model, processor, img, prompt)
        times.append(elapsed)
        print(f"[{name}]  (size {img.size})  [inference: {elapsed:.2f}s]")
        if parsed:
            print(f"  destination: {parsed.get('destination')}")
            print(f"  direction  : {parsed.get('direction')}")
            if parsed.get("matched_sign_line"):
                print(f"  matched line: {parsed.get('matched_sign_line')}")
            print(f"  route_fact : {parsed.get('route_fact')}")
        else:
            print(f"  (couldn't parse JSON) raw: {raw}")
        print()
    if times:
        # first image is often slower (CUDA warmup); report both raw and warm avg
        print(f"Per-image inference: first={times[0]:.2f}s, "
              f"avg={sum(times)/len(times):.2f}s", end="")
        if len(times) > 1:
            warm = times[1:]
            print(f", warm-avg={sum(warm)/len(warm):.2f}s (excl. first)")
        else:
            print(" (run multiple images for a warm average)")
    print("Verdict to judge yourself: did it get BOTH destination and arrow direction "
          "right, on crops at your real resolution? That's the gate.")


if __name__ == "__main__":
    main()