#!/usr/bin/env python3
"""test_qwen_load.py - verify Qwen2.5-VL processor loads with PIL backend, no torchvision."""
import sys
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
print("=== testing Qwen2.5-VL processor load (PIL backend, dummy video proc) ===")

from transformers import AutoTokenizer, Qwen2_5_VLProcessor
tokenizer = AutoTokenizer.from_pretrained(MODEL)
print("tokenizer OK")

# image processor: PIL backend (no torchvision)
image_processor = None
try:
    from transformers import Qwen2VLImageProcessorPil
    image_processor = Qwen2VLImageProcessorPil.from_pretrained(MODEL)
    print("image processor OK via Qwen2VLImageProcessorPil")
except Exception as e:
    print(f"PIL image proc failed: {e}")
    try:
        from transformers import AutoImageProcessor
        image_processor = AutoImageProcessor.from_pretrained(MODEL, use_fast=False)
        print("image processor OK via use_fast=False")
    except Exception as e2:
        print(f"all image proc routes failed: {e2}"); sys.exit(1)

# video processor: real one, or dummy to satisfy the type check
video_processor = None
try:
    from transformers import Qwen2_5_VLVideoProcessor
    video_processor = Qwen2_5_VLVideoProcessor.from_pretrained(MODEL)
    print("video processor OK via Qwen2_5_VLVideoProcessor (real)")
except Exception as e:
    print(f"real video proc failed (expected): trying dummy")
    try:
        from transformers.video_processing_utils import BaseVideoProcessor
        class _Dummy(BaseVideoProcessor):
            def __init__(self):
                try: super().__init__()
                except Exception: pass
            def preprocess(self, *a, **k): raise RuntimeError("no video")
        video_processor = _Dummy()
        print("dummy video processor created")
    except Exception as e2:
        print(f"dummy video proc failed: {e2}")

# build processor
try:
    proc = Qwen2_5_VLProcessor(image_processor=image_processor, tokenizer=tokenizer,
                               video_processor=video_processor)
    print("\nSUCCESS: Qwen2_5_VLProcessor built (PIL image, dummy video). Re-run live node.")
except Exception as e:
    print(f"\nprocessor build failed: {e}")
    import transformers; print("transformers", transformers.__version__)
    sys.exit(1)