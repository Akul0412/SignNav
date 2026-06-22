#!/usr/bin/env python3
"""
test_qwen_load.py - verify Qwen2.5-VL processor loads with PIL backend (no torchvision).
Run BEFORE the full live node.  python3 test_qwen_load.py
"""
import sys
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
print("=== testing Qwen2.5-VL processor load with PIL backend (no torchvision) ===")

from transformers import AutoTokenizer, Qwen2_5_VLProcessor
tokenizer = AutoTokenizer.from_pretrained(MODEL)
print("tokenizer OK")

image_processor = None
# A: backend="pil"
try:
    from transformers import AutoImageProcessor
    image_processor = AutoImageProcessor.from_pretrained(MODEL, backend="pil")
    print("image processor OK via backend='pil'")
except Exception as e:
    print(f"route A failed: {e}")
# B: direct PIL class
if image_processor is None:
    try:
        from transformers import Qwen2VLImageProcessorPil
        image_processor = Qwen2VLImageProcessorPil.from_pretrained(MODEL)
        print("image processor OK via Qwen2VLImageProcessorPil")
    except Exception as e:
        print(f"route B failed: {e}")
# C: use_fast=False
if image_processor is None:
    try:
        from transformers import AutoImageProcessor
        image_processor = AutoImageProcessor.from_pretrained(MODEL, use_fast=False)
        print("image processor OK via use_fast=False")
    except Exception as e:
        print(f"route C failed: {e}")

if image_processor is None:
    print("\nALL image-processor routes failed. Paste transformers version:")
    import transformers; print("transformers", transformers.__version__)
    sys.exit(1)

proc = Qwen2_5_VLProcessor(image_processor=image_processor, tokenizer=tokenizer)
print("\nSUCCESS: Qwen2_5_VLProcessor built with PIL backend, no torchvision, no video proc.")
print("=> the live node should now load Qwen. Re-run it.")