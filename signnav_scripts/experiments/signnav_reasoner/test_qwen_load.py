#!/usr/bin/env python3
"""
test_qwen_load.py - verify Qwen2.5-VL loads WITHOUT torchvision, in isolation.

Run this BEFORE the full live node to confirm the processor loads. Much faster
to debug than waiting for the whole pipeline.
    python3 test_qwen_load.py
"""
import sys

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

print("=== testing Qwen2.5-VL processor load (no torchvision) ===")

# Route 1: build processor from image processor + tokenizer (no video proc)
try:
    from transformers import Qwen2_5_VLProcessor, AutoImageProcessor, AutoTokenizer
    ip = AutoImageProcessor.from_pretrained(MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    proc = Qwen2_5_VLProcessor(image_processor=ip, tokenizer=tok)
    print("ROUTE 1 OK: Qwen2_5_VLProcessor(image_processor, tokenizer) works")
    print("=> use route 1 in reader.py")
    sys.exit(0)
except Exception as e:
    print(f"route 1 failed: {e}\n")

# Route 2: AutoProcessor use_fast
try:
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL, use_fast=True)
    print("ROUTE 2 OK: AutoProcessor(use_fast=True) works")
    sys.exit(0)
except Exception as e:
    print(f"route 2 failed: {e}\n")

print("Both routes failed. Qwen processor needs torchvision on this transformers version.")
print("Options: (a) install a working torchvision, (b) downgrade transformers to a")
print("version whose Qwen processor doesn't require torchvision for images.")