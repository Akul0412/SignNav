#!/usr/bin/env python3
"""
debug_one_read.py - read ONE image with Qwen and print the RAW output.

This bypasses the whole pipeline to answer one question: when we hand Qwen a
clear sign crop, what does it actually say? Run on MSI:

    python3 debug_one_read.py --image debug_crops/frame0010_crop.jpg
    # or point at any full frame / sign photo:
    python3 debug_one_read.py --image /path/to/sign.png

It prints Qwen's raw response BEFORE any parsing, so we can see whether:
  (a) Qwen returns prose (reads the sign but not as JSON) -> fix the prompt/parser
  (b) Qwen returns empty / a refusal / "I can't see an image" -> image not reaching it
  (c) Qwen returns JSON that the parser should handle -> parser bug
"""
import argparse
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
args = ap.parse_args()

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

print(f"loading {args.model} ...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained(args.model)
print("loaded.\n")

crop = Image.open(args.image).convert("RGB")
print(f"image: {args.image}  size={crop.size}\n")

PROMPT = (
    "Read this building directional sign. List each destination with its arrow "
    "direction (up=straight, left, right). Answer ONLY compact JSON: "
    '{"labels": {"DEST": "left|right|straight", ...}}'
)

conv = [{"role": "user", "content": [
    {"type": "image", "image": crop},
    {"type": "text", "text": PROMPT}]}]
text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)

# show how many image placeholders ended up in the templated text (sanity check)
print(f"--- templated prompt (first 300 chars) ---\n{text[:300]}\n")
print(f"image-pad token count in prompt: {text.count('<|image_pad|>') or text.count('<|vision_start|>')}")

inputs = processor(text=[text], images=[crop], return_tensors="pt").to(model.device)
print(f"input keys: {list(inputs.keys())}")
print(f"pixel_values present: {'pixel_values' in inputs}  "
      f"shape={getattr(inputs.get('pixel_values'), 'shape', None)}\n")

out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
resp = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("=" * 60)
print("RAW QWEN OUTPUT:")
print("=" * 60)
print(repr(resp))
print("=" * 60)