"""
Model interfaces — thin abstractions so Reader and Reasoner are independent of
which vision-language model backs them.

Adding a new backend (e.g. GeminiVLM) means implementing VLMInterface and
registering it in make_vlm; the rest of the pipeline is unchanged.
"""

from typing import Optional
from .types import Config


class VLMInterface:
    """A vision-language model: prompt + image → text."""
    def generate(self, prompt: str, image, sample: bool = False,
                 max_new_tokens: int = 400) -> str:
        raise NotImplementedError


class QwenVLM(VLMInterface):
    """Qwen2.5-VL running locally via transformers (fp16 or 4-bit)."""

    def __init__(self, config: Config):
        self.cfg = config
        self._model = None
        self._processor = None
        self._diag_done = False
        self._load()

    def _load(self):
        try:
            import torch
            from transformers import (Qwen2_5_VLForConditionalGeneration,
                                      AutoProcessor, BitsAndBytesConfig)
            kw = {"device_map": "auto"}
            if self.cfg.reasoner_4bit:
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True)
            else:
                kw["torch_dtype"] = torch.float16
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.cfg.reasoner_model, **kw)
            self._processor = AutoProcessor.from_pretrained(self.cfg.reasoner_model)
            print(f"[QwenVLM] {self.cfg.reasoner_model} loaded "
                  f"(4bit={self.cfg.reasoner_4bit})")
        except Exception as e:
            import traceback
            print("\n" + "!" * 72)
            print(f"[QwenVLM] FAILED TO LOAD: {e}")
            traceback.print_exc()
            print("!" * 72)
            if self.cfg.allow_stub_fallback:
                print("[QwenVLM] allow_stub_fallback=True -> model is None "
                      "(stub reads will be used).")
                self.cfg.stub_reader = True
            else:
                raise RuntimeError(
                    f"QwenVLM failed to load and stub fallback is disabled: {e}")

    def generate(self, prompt: str, image, sample: bool = False,
                 max_new_tokens: int = 400) -> str:
        """Run one VLM call.  Returns '' if the model failed to load."""
        if self._model is None:
            return ""
        # Bare {"type": "image"} placeholder — apply_chat_template inserts
        # <|image_pad|> vision tokens here; actual pixel_values come from
        # images=[image] in the processor call below.
        # Do NOT use inline {"type": "image", "image": ...}: on transformers 4.x
        # that form skips vision-token insertion (effectively a text-only prompt).
        conv = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False)
        inputs = self._processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self._model.device)
        if not self._diag_done:
            pad_count = text.count("<|image_pad|>")
            has_pv = "pixel_values" in inputs
            print(f"  [QwenVLM diag] <|image_pad|> tokens in template: {pad_count}  "
                  f"pixel_values present: {has_pv}")
            self._diag_done = True
        gen = {"max_new_tokens": max_new_tokens, "do_sample": sample}
        if sample:
            gen["temperature"] = 0.7
        out = self._model.generate(**inputs, **gen)
        return self._processor.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def make_vlm(config: Config) -> VLMInterface:
    """Return the right VLMInterface for config.vlm_backend.

    Only 'qwen' is implemented; future backends (e.g. 'gemini') register here.
    """
    if config.vlm_backend == "qwen":
        return QwenVLM(config)
    raise ValueError(
        f"Unknown vlm_backend: {config.vlm_backend!r}. Supported: 'qwen'.")
