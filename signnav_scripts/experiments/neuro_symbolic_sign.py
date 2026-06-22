#!/usr/bin/env python3
"""
neuro_symbolic_sign.py  —  Neuro-symbolic sign reasoning (SignScene-style parse + code logic).

WHY: end-to-end VLM reasoning failed on our signs (arrow-association errors,
hallucinated directions for absent goals). The neuro-symbolic split fixes this:

  NEURAL  (Qwen-32B): only PARSE the sign into structured data — a mapping from
          each destination to its direction. This is perception, which VLMs do well.
          Output format mirrors SignScene (AdaCompNUS/Sign-Understanding):
            {"text_labels": {"Vending Services": "left", ...},
             "symbol_labels": {"Restrooms": "right", ...}}
          directions are one of: left | right | straight | locational

  SYMBOLIC (this code): given the parse + the goal, resolve the action with EXACT
          logic — fuzzy-match the goal to a destination, map its direction to an
          action, REFUSE (unknown) if the goal isn't on the sign, and track maneuver
          state across frames. Code never hallucinates and never mis-associates.

This is the Phase-2 reasoner. Run it per frame; the symbolic layer carries memory.

Usage:
    python neuro_symbolic_sign.py --frame path/to/frame.jpg --goal "Vending Services"
    # or sequential over a trip:
    python neuro_symbolic_sign.py --trip <dir> --goal "Vending Services" --every 15
"""

import argparse
import csv
import difflib
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"

# --- the NEURAL half: parse the sign into structured directions (SignScene format) ---
PARSE_PROMPT = (
    "You are reading a wall-mounted navigational/directory sign for a robot. "
    "Parse the sign into structured data. Directory signs group destinations under "
    "arrows (up, left, right); every destination listed below an arrow — until the "
    "next arrow — takes that arrow's direction.\n"
    "Map the arrow to a direction word: up -> 'straight', left -> 'left', "
    "right -> 'right'. A destination that only marks the current place (no travel) "
    "is 'locational'.\n"
    "List EVERY destination you can read, each with its direction. Separate text "
    "destinations (text_labels) from symbol/icon destinations like restroom or stair "
    "icons (symbol_labels). Do NOT decide what the robot should do — only transcribe "
    "the sign's structure. If you cannot read the sign, return empty objects.\n"
    "Answer ONLY as compact JSON:\n"
    '{"text_labels": {"DEST": "left|right|straight|locational", ...}, '
    '"symbol_labels": {"DEST": "left|right|straight|locational", ...}}'
)

ARROW_TO_ACTION = {"left": "turn left", "right": "turn right",
                   "straight": "go straight", "locational": "arrived"}


def load_model():
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
    from transformers import AutoProcessor
    print(f"Loading {MODEL_ID} (bf16) ...")
    t0 = time.perf_counter()
    model = VLModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"(load: {time.perf_counter()-t0:.1f}s)\n")
    return model, processor


def parse_sign(model, processor, image):
    """NEURAL half: image -> {text_labels, symbol_labels} structured dict."""
    conv = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": PARSE_PROMPT}]}]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    t0 = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=400, do_sample=False)
    dt = time.perf_counter() - t0
    resp = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    try:
        s, e = resp.index("{"), resp.rindex("}") + 1
        return json.loads(resp[s:e]), dt
    except (ValueError, json.JSONDecodeError):
        return {"text_labels": {}, "symbol_labels": {}, "_raw": resp}, dt


# --- the SYMBOLIC half: deterministic resolution + maneuver memory ---
def resolve_direction(parsed, goal):
    """Given the parsed sign and the goal, return (direction, matched_dest) or (None, None)."""
    all_dests = {}
    all_dests.update(parsed.get("text_labels", {}) or {})
    all_dests.update(parsed.get("symbol_labels", {}) or {})
    if not all_dests:
        return None, None
    goal_l = goal.lower().strip()
    # 1) exact / substring match
    for dest, direction in all_dests.items():
        d = dest.lower()
        if goal_l == d or goal_l in d or d in goal_l:
            return direction, dest
    # 2) fuzzy match (handles spelling/synonym drift)
    names = list(all_dests.keys())
    close = difflib.get_close_matches(goal, names, n=1, cutoff=0.6)
    if close:
        return all_dests[close[0]], close[0]
    # 3) room-number range match e.g. "2-130" within "2-111 to 2-140"
    import re
    gm = re.search(r"(\d+)-(\d+)", goal)
    if gm:
        gfloor, groom = gm.group(1), int(gm.group(2))
        for dest, direction in all_dests.items():
            rng = re.findall(rf"{gfloor}-(\d+)", dest)
            if len(rng) >= 2 and int(rng[0]) <= groom <= int(rng[-1]):
                return direction, dest
    return None, None


class ManeuverState:
    """SYMBOLIC memory: tracks the committed maneuver across frames."""
    def __init__(self):
        self.committed_direction = None     # 'left'/'right'/'straight'
        self.turning = False

    def update(self, sign_direction):
        """Return the action to take this frame, given this frame's sign reading (or None)."""
        if sign_direction in ("left", "right"):
            # a directional sign for our goal: commit to the turn
            self.committed_direction = sign_direction
            self.turning = True
            return f"turn {sign_direction}"
        if sign_direction == "straight":
            self.committed_direction = "straight"
            return "go straight"
        if sign_direction == "locational":
            return "arrived"
        # no sign reading this frame:
        if self.turning:
            # mid-maneuver: keep completing the committed turn
            return f"continue turning {self.committed_direction}"
        if self.committed_direction:
            return "continue straight"
        return "unknown"

    def note_straightened(self):
        """Call when the hallway has clearly straightened — turn is done."""
        self.turning = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", help="Single frame")
    ap.add_argument("--trip", help="Extracted trip dir (sequential mode)")
    ap.add_argument("--goal", required=True)
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--out")
    args = ap.parse_args()

    model, processor = load_model()
    state = ManeuverState()
    lines = [f"# Neuro-symbolic sign reasoning — goal '{args.goal}'\n"]

    if args.frame:
        frames = [(0, Path(args.frame))]
    else:
        trip = Path(args.trip)
        fi = trip / "frame_index.csv"
        if fi.exists():
            with open(fi) as f:
                rows = [(int(r["timestamp_ns"]), trip / "frames" / r["filename"])
                        for r in csv.DictReader(f)]
        else:
            rows = sorted((int(p.stem), p) for p in (trip / "frames").glob("*.jpg"))
        rows.sort()
        frames = rows[::args.every]

    for i, (ts, path) in enumerate(frames):
        img = Image.open(path).convert("RGB")
        parsed, dt = parse_sign(model, processor, img)        # NEURAL
        direction, matched = resolve_direction(parsed, args.goal)  # SYMBOLIC lookup
        action = state.update(direction)                       # SYMBOLIC memory

        merged = {**parsed.get("text_labels", {}), **parsed.get("symbol_labels", {})}
        merged_str = json.dumps(merged)[:300]
        block = (f"\n=== step {i+1}/{len(frames)} (frame {ts}) [{dt:.1f}s] ===\n"
                 f"  parsed sign : {merged_str}\n"
                 f"  goal match  : {matched or '(goal not on this sign)'}"
                 f"  -> direction: {direction or 'none'}\n"
                 f"  ACTION      : {action}")
        print(block)
        lines.append(block)

    if args.out:
        Path(args.out).write_text("\n".join(lines), encoding="utf-8")
        print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()