#!/usr/bin/env python3
"""
sign_detection.py - Shared sign-detection logic (single source of truth).

Yehor's batch pipeline (detect_sign_crops.py) and the live Jetson monitor BOTH
import from here, so the actual YOLO detection logic exists in exactly one place.

Yehor's original detect_sign_crops.py keeps its batch orchestration (folder walking,
manifests, best-per-bag selection) but delegates the per-frame detection to the
functions here. The live loop calls detect_signs_in_frame() on one frame at a time.

This file contains the detection PRIMITIVES, not the batch machinery.
"""

from typing import Dict, List, Optional, Sequence, Set, Tuple


AUTO_CLASS_KEYWORDS = (
    "sign", "traffic light", "wayfinding", "marker", "arrow", "direction",
)


def normalize_model_names(names: object) -> Dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def resolve_allowed_classes(names: Dict[int, str], class_spec: str) -> Optional[Set[int]]:
    spec = class_spec.strip()
    if not spec or spec.lower() == "all":
        return None
    if spec.lower() == "auto":
        allowed = {cid for cid, cname in names.items()
                   if any(k in cname.lower() for k in AUTO_CLASS_KEYWORDS)}
        return allowed or None
    by_name = {cname.lower(): cid for cid, cname in names.items()}
    allowed, missing = set(), []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.isdigit():
            allowed.add(int(tok))
        elif tok.lower() in by_name:
            allowed.add(by_name[tok.lower()])
        else:
            missing.append(tok)
    if missing:
        raise SystemExit(f"Could not resolve class name(s): {', '.join(missing)}")
    return allowed or None


def expand_bbox(bbox, image_width, image_height, margin) -> Tuple[int, int, int, int]:
    """Yehor's crop expansion: pad the box by `margin` on each side, clamp to frame."""
    x1, y1, x2, y2 = bbox
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    dx, dy = box_w * margin, box_h * margin
    cx1 = max(0, int(x1 - dx))
    cy1 = max(0, int(y1 - dy))
    cx2 = min(image_width, int(x2 + dx + 0.9999))
    cy2 = min(image_height, int(y2 + dy + 0.9999))
    return cx1, cy1, cx2, cy2


def detections_from_result(
    result: object,
    names: Dict[int, str],
    allowed_classes: Optional[Set[int]],
    image_width: int,
    image_height: int,
) -> List[Dict[str, object]]:
    """Yehor's per-detection scoring, but returns ALL candidates (live loop wants
    every sign in the frame, not just the single best). Sorted best-first."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)

    image_area = max(1, image_width * image_height)
    out = []
    for box, conf, cid in zip(xyxy, confs, classes):
        if allowed_classes is not None and int(cid) not in allowed_classes:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1) / image_area)
        area_bonus = min(area_ratio / 0.08, 1.0) * 0.12
        out.append({
            "bbox": (x1, y1, x2, y2),
            "confidence": float(conf),
            "class_id": int(cid),
            "class_name": names.get(int(cid), str(cid)),
            "score": float(conf) + area_bonus,
            "box_area_ratio": area_ratio,
        })
    out.sort(key=lambda d: d["score"], reverse=True)
    return out


def choose_detection(result, names, allowed_classes, image_width, image_height):
    """Single best detection (Yehor's batch behavior) — kept for the batch script."""
    cands = detections_from_result(result, names, allowed_classes, image_width, image_height)
    return cands[0] if cands else None


class LiveSignDetector:
    """Per-frame YOLO sign detector for the live loop. Loads the model once,
    then detect() runs on a single image and returns all sign detections."""

    def __init__(self, model_path: str = "yolov8n.pt", classes: str = "auto",
                 conf: float = 0.15, iou: float = 0.70, imgsz: int = 640,
                 device: str = "", margin: float = 0.10, use_heuristic: bool = True,
                 heuristic_min_score: float = 0.30):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.names = normalize_model_names(getattr(self.model, "names", {}))
        self.allowed = resolve_allowed_classes(self.names, classes)
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.device, self.margin = device, margin
        self.use_heuristic = use_heuristic
        self.heuristic_min_score = heuristic_min_score
        self._cv2 = None
        if use_heuristic:
            try:
                import cv2
                self._cv2 = cv2
            except ImportError:
                print("[LiveSignDetector] opencv unavailable; heuristic disabled.")
                self.use_heuristic = False

    def detect(self, image) -> List[Dict[str, object]]:
        """image: a PIL.Image or np array. Returns sign detections, best-first,
        each with an expanded crop box ready to slice.
        Falls back to the dark-panel HEURISTIC when YOLO finds nothing — indoor
        directory signs are NOT in YOLO's COCO classes, so YOLO always misses them."""
        import numpy as np
        arr = np.array(image)[:, :, ::-1] if hasattr(image, "size") else image  # PIL RGB -> BGR
        h, w = arr.shape[:2]
        kw = dict(source=arr, verbose=False, conf=self.conf, iou=self.iou, imgsz=self.imgsz)
        if self.device:
            kw["device"] = self.device
        results = self.model.predict(**kw)
        dets = []
        if results:
            dets = detections_from_result(results[0], self.names, self.allowed, w, h)
        # heuristic fallback when YOLO found no sign (the usual case for indoor signs)
        if not dets and self.use_heuristic and self._cv2 is not None:
            hc = choose_heuristic_sign(self._cv2, arr)
            if hc is not None and hc["score"] >= self.heuristic_min_score:
                x1, y1, x2, y2 = hc["bbox"]
                dets = [{
                    "bbox": (x1, y1, x2, y2), "confidence": float(hc["score"]),
                    "class_id": -1, "class_name": f"heuristic_{hc['source']}",
                    "score": float(hc["score"]), "box_area_ratio": hc["box_area_ratio"],
                }]
        for d in dets:
            d["crop_box"] = expand_bbox(d["bbox"], w, h, self.margin)
        return dets


# ============================================================================
# Heuristic sign detection (ported from Yehor's detect_sign_crops.py).
# Detects dark wayfinding panels (indoor directory signs) WITHOUT YOLO classes.
# This is what makes the live loop actually see indoor signs, since YOLO's COCO
# classes (traffic light, stop sign) cannot.
# ============================================================================

import math


def choose_heuristic_sign(cv2_module, image):
    """Find a likely indoor wayfinding sign (dark panel). Returns dict with
    'bbox' (x1,y1,x2,y2 in full-res), 'score', 'source', 'box_area_ratio', or None.
    Ported from Yehor's batch detector for live per-frame use."""
    image_height, image_width = image.shape[:2]
    scale = min(1.0, 960.0 / max(image_width, image_height))
    if scale < 1.0:
        small = cv2_module.resize(image, (int(image_width * scale), int(image_height * scale)),
                                  interpolation=cv2_module.INTER_AREA)
    else:
        small = image

    small_height, small_width = small.shape[:2]
    gray = cv2_module.cvtColor(small, cv2_module.COLOR_BGR2GRAY)
    blur = cv2_module.GaussianBlur(gray, (3, 3), 0)
    edges = cv2_module.Canny(blur, 50, 150)
    image_area = max(1, small_width * small_height)

    def component_boxes(mask):
        labels, _, stats, _ = cv2_module.connectedComponentsWithStats(mask, 8)
        boxes = []
        for label_id in range(1, labels):
            x, y, w, h, area = stats[label_id]
            boxes.append((int(x), int(y), int(w), int(h), float(area)))
        return boxes

    def contour_boxes(mask):
        found = cv2_module.findContours(mask, cv2_module.RETR_EXTERNAL, cv2_module.CHAIN_APPROX_SIMPLE)
        contours = found[0] if len(found) == 2 else found[1]
        boxes = []
        for contour in contours:
            x, y, w, h = cv2_module.boundingRect(contour)
            boxes.append((x, y, w, h, float(cv2_module.contourArea(contour))))
        return boxes

    def area_preference(area_ratio):
        small_ok = min(math.sqrt(area_ratio / 0.035), 1.0)
        large_ok = min(math.sqrt(0.18 / max(area_ratio, 0.001)), 1.0)
        return small_ok * large_ok

    def aspect_preference(aspect):
        if 0.28 <= aspect <= 0.95:
            return 1.0
        if 0.95 < aspect <= 3.2:
            return 0.82
        return max(0.0, 1.0 - abs(math.log(max(aspect, 0.01))) / 1.8)

    def border_touch_count(x, y, w, h):
        t = 0
        t += int(x <= 2); t += int(y <= 2)
        t += int(x + w >= small_width - 2); t += int(y + h >= small_height - 2)
        return t

    def border_preference(x, y, w, h):
        return max(0.35, 1.0 - 0.22 * border_touch_count(x, y, w, h))

    def position_preference(y, h):
        center_y = (y + h / 2.0) / max(1, small_height)
        if 0.08 <= center_y <= 0.74:
            return 1.0
        if center_y < 0.08:
            return max(0.0, center_y / 0.08)
        return max(0.0, 1.0 - (center_y - 0.74) / 0.26)

    def candidate_from_box(x, y, w, h, contour_area, source, dark_mask_roi=None):
        if w < 32 or h < 32:
            return None
        area = w * h
        area_ratio = area / image_area
        if area_ratio < 0.0025 or area_ratio > 0.42:
            return None
        aspect = w / max(1, h)
        if aspect < 0.12 or aspect > 5.8:
            return None
        crop_gray = gray[y:y + h, x:x + w]
        crop_edges = edges[y:y + h, x:x + w]
        if crop_gray.size == 0:
            return None
        contrast = float(crop_gray.std())
        mean_brightness = float(crop_gray.mean())
        edge_density = float(cv2_module.countNonZero(crop_edges)) / max(1, area)
        if contrast < 16.0 and edge_density < 0.012:
            return None
        if dark_mask_roi is None:
            dark_fraction = float((crop_gray < 105).sum()) / max(1, area)
        else:
            dark_roi = dark_mask_roi[y:y + h, x:x + w]
            dark_fraction = float(cv2_module.countNonZero(dark_roi)) / max(1, area)
        text_edge_score = min(edge_density / 0.075, 1.0)
        contrast_score = min(contrast / 68.0, 1.0)
        dark_score = min(dark_fraction / 0.62, 1.0)
        darkness_score = max(0.0, min((108.0 - mean_brightness) / 72.0, 1.0))
        rectangularity = min(contour_area / max(1.0, float(area)), 1.0)
        area_score = area_preference(area_ratio)
        aspect_score = aspect_preference(aspect)
        border_score = border_preference(x, y, w, h)
        border_touches = border_touch_count(x, y, w, h)
        position_score = position_preference(y, h)
        if source == "dark-panel":
            if dark_fraction < 0.52:
                return None
            if mean_brightness > 92.0 and dark_fraction < 0.72:
                return None
            score = (0.31 * dark_score + 0.15 * darkness_score + 0.17 * text_edge_score
                     + 0.13 * contrast_score + 0.11 * area_score + 0.06 * rectangularity
                     + 0.04 * aspect_score + 0.02 * border_score + 0.01 * position_score)
            if border_touches >= 2 and area_ratio > 0.08:
                score *= 0.38
            elif border_touches >= 1 and area_ratio > 0.16:
                score *= 0.62
        else:
            if edge_density < 0.018 or area_ratio > 0.22:
                return None
            score = (0.30 * text_edge_score + 0.22 * contrast_score + 0.18 * area_score
                     + 0.11 * aspect_score + 0.09 * rectangularity + 0.06 * border_score
                     + 0.04 * position_score)
        inv = 1.0 / scale
        return {"bbox": (x * inv, y * inv, (x + w) * inv, (y + h) * inv),
                "score": min(score, 1.0), "source": source, "box_area_ratio": area_ratio}

    # Pass 1: dark wayfinding panels
    _, dark_mask = cv2_module.threshold(blur, 80, 255, cv2_module.THRESH_BINARY_INV)
    dark_mask = cv2_module.morphologyEx(
        dark_mask, cv2_module.MORPH_OPEN,
        cv2_module.getStructuringElement(cv2_module.MORPH_RECT, (3, 3)), iterations=1)
    dark_candidates = []
    for x, y, w, h, ca in component_boxes(dark_mask):
        c = candidate_from_box(x, y, w, h, ca, "dark-panel", dark_mask)
        if c is not None:
            dark_candidates.append(c)
    if dark_candidates:
        return max(dark_candidates, key=lambda d: d["score"])

    # Pass 2: non-black signs (round plaques etc.)
    edge_mask = cv2_module.morphologyEx(
        edges, cv2_module.MORPH_CLOSE,
        cv2_module.getStructuringElement(cv2_module.MORPH_RECT, (11, 11)), iterations=2)
    edge_candidates = []
    for x, y, w, h, ca in contour_boxes(edge_mask):
        c = candidate_from_box(x, y, w, h, ca, "edge-panel")
        if c is not None:
            edge_candidates.append(c)
    if not edge_candidates:
        return None
    return max(edge_candidates, key=lambda d: d["score"])