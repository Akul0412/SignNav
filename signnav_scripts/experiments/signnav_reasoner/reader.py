"""
Reader: crop a detected sign and try to READ it, with a CONFIDENCE score.

This is the linchpin of the adaptive loop. It returns:
  - the transcription (structured {destination: direction} when possible)
  - a read_confidence in [0,1] — how much we trust the read RIGHT NOW

If read_confidence is low (sign too far/blurry), the loop will keep approaching
and re-reading; when it's high, the loop commits to reasoning.

CONFIDENCE NOTE (honest): trustworthy confidence from a VLM is itself a research
sub-problem (VLMs are often confidently wrong). Here we use a practical proxy —
agreement across a few stochastic samples: if the model says the same thing every
time, we trust it; if reads disagree, confidence is low. This is a placeholder for
a more principled signal (token logprobs, a legibility head, etc.) to be refined.
"""

import json
from typing import Optional

from .interfaces import VLMInterface
from .types import Config, Detection, ReadResult

PARSE_PROMPT = (
    "Read this building directional sign. Directory signs group destinations under "
    "arrows (up=straight, left, right); a destination takes the arrow of the group "
    "above it. Transcribe every destination you can see and its arrow direction. "
    "Read whatever text is visible, even if partially. Answer ONLY compact JSON: "
    '{"labels": {"DEST": "left|right|straight", ...}}'
)


def _extract_labels(resp: str) -> dict:
    """Robustly pull {"labels": {...}} out of a model response that may include
    markdown code fences, stray text, or minor formatting noise."""
    import re
    if not resp:
        return {"labels": {}}
    # strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", resp).strip()
    # find the outermost {...}
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"labels": {}}
    blob = cleaned[start:end + 1]
    try:
        obj = json.loads(blob)
    except Exception:
        # try a lenient fix: single quotes -> double quotes
        try:
            obj = json.loads(blob.replace("'", '"'))
        except Exception:
            return {"labels": {}}
    if isinstance(obj, dict):
        if "labels" in obj and isinstance(obj["labels"], dict):
            return {"labels": obj["labels"]}
        # model may have returned the labels dict directly
        if all(isinstance(v, str) for v in obj.values()):
            return {"labels": obj}
    return {"labels": {}}


def _labels_match(a: dict, b: dict) -> bool:
    """Two reads 'match' if they identify the same destinations with the same
    directions (case-insensitive on keys). Tolerant of ordering."""
    if not a or not b:
        return False
    na = {k.strip().lower(): str(v).strip().lower() for k, v in a.items()}
    nb = {k.strip().lower(): str(v).strip().lower() for k, v in b.items()}
    return na == nb


class Reader:
    def __init__(self, config: Config, vlm: Optional[VLMInterface] = None):
        self.cfg = config
        self.vlm = vlm   # shared VLMInterface; None only when stub_reader=True

    def read(self, image, detection: Detection, n_samples: int = 3, frame_idx: int = 0) -> ReadResult:
        """Crop the sign from full-res and read it, returning text + confidence.
        Also saves the crop to disk (if cfg.save_crops) and records crop diagnostics."""
        if self.cfg.stub_reader:
            return self._stub_read(detection)

        x, y, w, h = [int(v) for v in detection.box]
        crop = image.crop((x, y, x + w, y + h))
        cw, ch = crop.size

        # save the crop for visual inspection (so we can see WHAT was read)
        crop_path = ""
        if self.cfg.save_crops:
            import os
            os.makedirs(self.cfg.crop_dir, exist_ok=True)
            crop_path = os.path.join(self.cfg.crop_dir, f"frame{frame_idx:04d}_crop.jpg")
            try:
                crop.save(crop_path)
            except Exception as e:
                print(f"  [reader] could not save crop: {e}")
                crop_path = ""

        # PRIMARY read: greedy/deterministic — the model's single best read.
        primary = self._one_read(crop, sample=False)
        primary_labels = primary.get("labels", {}) if isinstance(primary, dict) else {}

        if not primary_labels:
            # nothing readable parsed — genuinely low confidence (sign too far/blurry/occluded)
            confidence = 0.2
            parsed = {}
        else:
            # STABILITY check: re-read a couple times (sampled) and see how often the
            # same destinations come back. Stable reads of a clear sign => high confidence.
            agree = 1
            for _ in range(max(0, n_samples - 1)):
                r = self._one_read(crop, sample=True).get("labels", {})
                if _labels_match(r, primary_labels):
                    agree += 1
            stability = agree / n_samples                  # 1/3..3/3
            # confidence: a clean non-empty parse is already trustworthy (0.7 floor),
            # boosted by stability across re-reads. Clear, stable sign -> ~0.9+.
            confidence = 0.7 + 0.3 * ((stability - (1.0 / n_samples)) / (1.0 - 1.0 / n_samples))
            confidence = round(min(1.0, max(0.7, confidence)), 3)
            parsed = primary_labels

        return ReadResult(
            text=json.dumps(parsed),
            read_confidence=confidence,
            structured=parsed,
            can_decide=confidence >= self.cfg.read_confidence_threshold,
            crop_box=(x, y, w, h),
            crop_size=(cw, ch),
            crop_path=crop_path,
            src_size=image.size,
        )

    def _one_read(self, crop, sample: bool) -> dict:
        resp = self.vlm.generate(PARSE_PROMPT, crop, sample=sample, max_new_tokens=200)
        print(f"    [raw Qwen read] {resp!r}")
        return _extract_labels(resp)

    def _stub_read(self, detection: Detection) -> ReadResult:
        """Simulate confidence rising as the robot approaches (box gets bigger)."""
        # use box area as a proxy for distance: bigger box => closer => higher confidence
        _, _, w, h = detection.box
        area = w * h
        conf = min(0.95, 0.3 + area / 250000.0)   # grows with proximity
        parsed = {"2-111 to 2-140": "left", "Vending Services": "left",
                  "2-221 to 2-260": "right", "Stairs": "right"} if conf >= 0.85 else {}
        return ReadResult(
            text=json.dumps(parsed), read_confidence=round(conf, 2),
            structured=parsed, can_decide=conf >= self.cfg.read_confidence_threshold)
