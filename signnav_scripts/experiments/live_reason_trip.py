#!/usr/bin/env python3
"""
live_reason_trip.py  —  TEST alpha: sequential "live" reasoning over a real trip.

Alpamayo-style: walk the trip's frames in order, carrying MEMORY of prior reasoning,
and at each step produce a four-part trace:
    OBSERVE  - what's in view now (sign? text? hazard? hallway changed?)
    RECALL   - what did I commit to earlier (goal + maneuver in progress)
    REASON   - given observation + memory, what should I do and why
    DECIDE   - the concrete action: straight / turn left / turn right / continue / stop

This tests whether the REASONER (the Phase-2 brain / future teacher) can produce
coherent, sign-grounded, trajectory-level navigation reasoning on real data --
including maintaining the goal after the sign leaves view. The transcript IS the
result to show Ajay, and the traces preview what Phase-3 distillation would target.

No OmniVLA, no Isaac Sim. Just frames + a big VLM.

Model: Qwen2.5-VL-32B in 4-bit (fits one A100-40GB). Falls back to 7B if needed.

Setup (GPU env with recent transformers):
    pip install "git+https://github.com/huggingface/transformers" accelerate bitsandbytes pillow

Usage:
    python live_reason_trip.py \
        --trip signnav_scripts/datasets/extracted/rosbag2_keller_29 \
        --goal "the classroom" \
        --every 15
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# Big reasoner options. On a single H100-80GB: 72B fits in 4-bit, 32B fits in bf16.
MODELS = {
    "72b": "Qwen/Qwen2.5-VL-72B-Instruct",
    "32b": "Qwen/Qwen2.5-VL-32B-Instruct",
    "7b":  "Qwen/Qwen2.5-VL-7B-Instruct",
}


def load_frames(trip_dir: Path, every: int):
    """Return [(timestamp_ns, path), ...] subsampled every `every` frames, in order."""
    fi = trip_dir / "frame_index.csv"
    if fi.exists():
        with open(fi) as f:
            rows = [(int(r["timestamp_ns"]), trip_dir / "frames" / r["filename"])
                    for r in csv.DictReader(f)]
    else:
        rows = sorted((int(p.stem), p) for p in (trip_dir / "frames").glob("*.jpg"))
    rows.sort(key=lambda r: r[0])
    return rows[::every]


def build_system_prompt(goal):
    return (
        "You are the reasoning module of an indoor delivery robot driving through a "
        "building hallway, navigating by reading signs. You receive one camera frame "
        f"at a time, in order. Your destination goal is: \"{goal}\".\n\n"
        "At EACH frame, think like a careful driver and answer in four parts:\n"
        "OBSERVE: what you see now (a directional sign? what does it say and which way "
        "does its arrow point? stairs or a hazard you must avoid? has the hallway turned?).\n"
        "RECALL: what you already decided in earlier frames (which way the sign told you "
        "to go, whether you are mid-turn or have finished it).\n"
        "REASON: combine OBSERVE + RECALL. If a sign is visible, read the direction for "
        "your goal. If no sign is visible, decide whether you are still completing a turn "
        "or should continue straight toward the goal. If you see stairs or a 'staff only' "
        "type restriction on your path, reason that you must STOP or avoid it.\n"
        "DECIDE: exactly one of: go straight | turn left | turn right | continue turning | "
        "continue straight | stop.\n\n"
        "Keep each part to 1-2 sentences. Be concrete. Answer ONLY in this format:\n"
        "OBSERVE: ...\nRECALL: ...\nREASON: ...\nDECIDE: ..."
    )


def load_model(model_id, precision):
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
        from transformers import AutoProcessor, BitsAndBytesConfig
    except ImportError:
        sys.exit('Need: pip install "git+https://github.com/huggingface/transformers" accelerate bitsandbytes')
    print(f"Loading {model_id} ({precision}) ...")
    t0 = time.perf_counter()
    if precision == "4bit":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        model = VLModel.from_pretrained(model_id, quantization_config=bnb,
                                        device_map="auto")
    else:  # bf16
        model = VLModel.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                        device_map="auto")
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"(load: {time.perf_counter()-t0:.1f}s)\n")
    return model, processor


def step_reason(model, processor, system_prompt, memory, image):
    """One reasoning step. `memory` is a short text summary of prior DECIDE/REASON."""
    user_text = (
        (f"So far: {memory}\n\n" if memory else "This is the first frame.\n\n")
        + "Here is the current camera frame. Reason about what the robot should do now."
    )
    conv = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_text}]},
    ]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    t0 = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=220, do_sample=False)
    dt = time.perf_counter() - t0
    resp = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return resp, dt


def parse_decide(trace):
    for line in trace.splitlines():
        if line.strip().upper().startswith("DECIDE"):
            return line.split(":", 1)[-1].strip()
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip", required=True, help="Extracted trip dir")
    ap.add_argument("--goal", required=True, help="Destination goal for this trip")
    ap.add_argument("--every", type=int, default=15, help="Subsample: 1 frame every N")
    ap.add_argument("--model", choices=["72b", "32b", "7b"], default="72b")
    ap.add_argument("--precision", choices=["4bit", "bf16"], default="4bit",
                    help="4bit fits 72B on one H100; bf16 for 32B/7B if you have room")
    ap.add_argument("--out", help="Save transcript to this file (optional)")
    args = ap.parse_args()

    trip = Path(args.trip)
    frames = load_frames(trip, args.every)
    print(f"Trip: {trip.name}  goal: {args.goal!r}  "
          f"frames: {len(frames)} (every {args.every})\n")

    model_id = MODELS[args.model]
    model, processor = load_model(model_id, args.precision)
    system_prompt = build_system_prompt(args.goal)

    transcript = [f"# Live reasoning — trip {trip.name}, goal '{args.goal}'\n"]
    memory = ""
    times = []
    for i, (ts, path) in enumerate(frames):
        img = Image.open(path).convert("RGB")
        trace, dt = step_reason(model, processor, system_prompt, memory, img)
        times.append(dt)
        decide = parse_decide(trace)

        header = f"\n===== step {i+1}/{len(frames)}  (frame {ts})  [{dt:.1f}s] ====="
        print(header)
        print(trace)
        transcript.append(header + "\n" + trace)

        # update memory: a compact running summary threaded into the next step
        memory = (memory + f" Step {i+1}: decided '{decide}'.").strip()
        # keep memory from growing unbounded — last ~6 decisions is plenty of context
        if len(memory) > 600:
            memory = "... " + memory[-560:]

    avg = sum(times) / len(times) if times else 0
    summary = (f"\n\n# Summary: {len(frames)} steps, "
               f"avg {avg:.1f}s/step, model {model_id.split('/')[-1]}")
    print(summary)
    transcript.append(summary)

    if args.out:
        Path(args.out).write_text("\n".join(transcript))
        print(f"\nTranscript saved -> {args.out}")


if __name__ == "__main__":
    main()