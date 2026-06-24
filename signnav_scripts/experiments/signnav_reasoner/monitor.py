"""
Monitor: the cheap, always-on perceptual trigger - with TWO detectors.

  SIGNS   -> Yehor's YOLO pipeline (via shared convert/sign_detection.py)
  HAZARDS -> GroundingDINO (open-vocab), prompted for stairs/steps/obstacle

detect_all() returns the RAW output of BOTH detectors plus the chosen detection,
so the debug logger can show everything (this is how we caught GroundingDINO
firing on the printed word 'stairs').

KNOWN ISSUES surfaced by testing (see analysis):
  - GroundingDINO grounds TEXT queries to image regions, so it can fire on the
    WORD 'stairs' printed on a directory sign (false hazard). Mitigations below.
  - YOLO COCO classes do NOT include indoor directory signs ('stop sign' only),
    so YOLO alone misses these signs. Yehor's batch pipeline relied on the
    dark-panel heuristic for that; the live monitor needs an equivalent.
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

from .types import Config, Detection, ObjectClass

_CONVERT = Path(__file__).resolve().parents[2] / "convert"
if str(_CONVERT) not in sys.path:
    sys.path.insert(0, str(_CONVERT))


@dataclass
class DetectionBundle:
    """Everything the monitor saw this frame (for debugging + decision)."""
    chosen: Detection
    sign_dets: List[dict]       # raw YOLO detections (dicts)
    hazard_dets: List[dict]     # raw GroundingDINO detections (dicts)


class Monitor:
    def __init__(self, config: Config):
        self.cfg = config
        self._yolo = None
        self._gdino = None
        self._cv2 = None              # for heuristic-only sign detection
        self._heuristic_only = not config.use_yolo
        self._last_sign_t: float = 0.0    # wall-clock duration of last sign detection
        self._last_hazard_t: float = 0.0  # wall-clock duration of last hazard detection
        if not config.stub_detector:
            self._load_sign_detector()
            self._load_hazard_detector()

    def _load_sign_detector(self):
        # heuristic-only path: no YOLO, no torchvision needed
        if self._heuristic_only:
            try:
                import cv2
                self._cv2 = cv2
                print("[Monitor] sign detection: OpenCV dark-panel HEURISTIC only "
                      "(YOLO disabled; no torchvision needed)")
            except Exception as e:
                print(f"[Monitor] OpenCV unavailable for heuristic ({e}).")
                if not self.cfg.allow_stub_fallback:
                    raise RuntimeError(f"heuristic sign detector needs opencv: {e}")
            return
        # YOLO path (needs ultralytics + torchvision)
        try:
            from sign_detection import LiveSignDetector
            self._yolo = LiveSignDetector(
                model_path=self.cfg.yolo_model_path, classes="auto",
                conf=self.cfg.detect_confidence_threshold, imgsz=self.cfg.yolo_imgsz,
                device=self.cfg.device, margin=self.cfg.crop_margin)
            print(f"[Monitor] YOLO sign detector loaded ({self.cfg.yolo_model_path})")
            print(f"          YOLO sign-relevant classes: {self._yolo.allowed}")
        except Exception as e:
            print(f"[Monitor] YOLO sign detector FAILED to load ({e}).")
            if not self.cfg.allow_stub_fallback:
                raise RuntimeError(f"YOLO detector failed to load: {e}")
            print("[Monitor] allow_stub_fallback=True -> signs STUBBED (fake).")
            self._yolo = None

    def _load_hazard_detector(self):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            model_id = "IDEA-Research/grounding-dino-tiny"
            self._gdino_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._gdino_proc = AutoProcessor.from_pretrained(model_id)
            self._gdino = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self._gdino_device)
            print(f"[Monitor] GroundingDINO hazard detector loaded on {self._gdino_device}")
        except Exception as e:
            print(f"[Monitor] GroundingDINO FAILED to load ({e}).")
            if not self.cfg.allow_stub_fallback:
                raise RuntimeError(f"GroundingDINO failed to load: {e}")
            print("[Monitor] allow_stub_fallback=True -> hazards STUBBED (fake).")
            self._gdino = None

    def detect_all(self, image, step: int = 0) -> DetectionBundle:
        """Run BOTH detectors, return raw results + the chosen detection."""
        if self.cfg.stub_detector:
            d = self._stub_detect(step)
            return DetectionBundle(chosen=d, sign_dets=[], hazard_dets=[])

        _t0 = time.perf_counter()
        sign_dets = self._detect_signs_raw(image)
        self._last_sign_t = time.perf_counter() - _t0

        _t0 = time.perf_counter()
        hazard_dets = self._detect_hazards_raw(image)
        self._last_hazard_t = time.perf_counter() - _t0

        # choose: hazard outranks sign (safety) — but ONLY above the higher hazard bar.
        # Below that bar, a "hazard" is treated as a false positive and ignored
        # (GroundingDINO hallucinates weak stairs from handrails/trim/whole-frame).
        chosen = Detection(ObjectClass.NONE, 0.0, (0, 0, 0, 0))
        hazard_best = hazard_dets[0] if hazard_dets else None
        sign_best = sign_dets[0] if sign_dets else None

        hazard_ok = hazard_best and hazard_best["confidence"] >= self.cfg.hazard_confidence_threshold
        sign_ok = sign_best and sign_best["confidence"] >= self.cfg.sign_confidence_threshold
        if hazard_ok:
            chosen = Detection(ObjectClass.STAIRS, hazard_best["confidence"],
                               hazard_best["box_xywh"], hazard_best["label"])
        elif sign_ok:
            cx1, cy1, cx2, cy2 = sign_best["crop_box"]
            chosen = Detection(ObjectClass.SIGN, sign_best["confidence"],
                               (cx1, cy1, cx2 - cx1, cy2 - cy1), sign_best["class_name"])
        # NOTE: a below-threshold hazard with no sign => NONE (ignored as false positive)

        return DetectionBundle(chosen=chosen, sign_dets=sign_dets, hazard_dets=hazard_dets)

    # backward-compatible single-detection entry
    def detect(self, image, step: int = 0) -> Detection:
        return self.detect_all(image, step).chosen

    def _detect_signs_raw(self, image) -> List[dict]:
        # heuristic-only mode: run the OpenCV dark-panel detector directly
        if self._heuristic_only:
            if self._cv2 is None:
                return []
            import numpy as np
            from sign_detection import choose_heuristic_sign, expand_bbox
            arr = np.array(image)[:, :, ::-1]   # PIL RGB -> BGR
            h, w = arr.shape[:2]
            hc = choose_heuristic_sign(self._cv2, arr)
            if hc is None or hc["score"] < 0.30:
                return []
            x1, y1, x2, y2 = hc["bbox"]
            d = {"bbox": (x1, y1, x2, y2), "confidence": float(hc["score"]),
                 "class_id": -1, "class_name": f"heuristic_{hc['source']}",
                 "score": float(hc["score"]), "box_area_ratio": hc["box_area_ratio"]}
            d["crop_box"] = expand_bbox(d["bbox"], w, h, self.cfg.crop_margin)
            return [d]
        # YOLO mode
        if self._yolo is None:
            return []
        return self._yolo.detect(image)

    def _detect_hazards_raw(self, image) -> List[dict]:
        if self._gdino is None:
            return []
        import torch
        inputs = self._gdino_proc(images=image, text=self.cfg.hazard_prompt,
                                  return_tensors="pt").to(self._gdino_device)
        with torch.no_grad():
            outputs = self._gdino(**inputs)
        res = self._gdino_proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.cfg.detect_confidence_threshold,
            text_threshold=0.25, target_sizes=[(image.height, image.width)])[0]
        labels = res.get("text_labels", res.get("labels", []))
        out = []
        for box, score, label in zip(res["boxes"], res["scores"], labels):
            x0, y0, x1, y1 = [float(v) for v in box]
            label_str = label.lower() if isinstance(label, str) else str(label)
            out.append({"label": label_str, "confidence": float(score),
                        "box_xywh": (x0, y0, x1 - x0, y1 - y0)})
        out.sort(key=lambda d: d["confidence"], reverse=True)
        return out

    def _stub_detect(self, step: int) -> Detection:
        if step < 3:
            return Detection(ObjectClass.NONE, 0.0, (0, 0, 0, 0))
        if 3 <= step < 9:
            return Detection(ObjectClass.SIGN, 0.7, (900, 200, 240, 420), "sign")
        if step == 9:
            return Detection(ObjectClass.STAIRS, 0.8, (700, 300, 500, 400), "stairs")
        return Detection(ObjectClass.NONE, 0.0, (0, 0, 0, 0))