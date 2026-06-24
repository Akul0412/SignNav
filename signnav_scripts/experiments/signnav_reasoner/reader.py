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

READ SCHEMA (grouped): directory signs assign ONE arrow to a GROUP of destinations,
and every destination in that group goes that way. We ask the model for groups
({"dir","dests"}) rather than per-line directions, then expand each group so every
destination inherits its group's single direction. This removes the failure where
trailing lines under a turn arrow get defaulted to "straight". _extract_labels still
returns the SAME flat {destination: direction} dict the rest of the pipeline expects,
so nothing downstream changes; flat legacy shapes are still parsed for robustness.

NOTE: notices/advisories (e.g. "Restroom Closed") are NOT read here. The monitor's
notice channel only DETECTS that something notice-like is present (a cheap trigger);
the Reasoner then examines the whole frame and interprets it (see reasoner.py
scene_alert). We deliberately do not OCR/parse notices into a schema — the VLM reads
and interprets the scene directly, which is more general.
"""

import json
import time
from typing import List, Optional

from .interfaces import VLMInterface
from .types import Config, Detection, ObjectClass, ReadResult

PARSE_PROMPT = (
    "Read this building directional sign. The sign is organized into GROUPS: each "
    "group begins with ONE arrow (up=straight, left, right) followed by one or more "
    "destination lines, and EVERY destination in that group goes in the arrow's "
    "direction. Do NOT assign directions line-by-line — assign ONE direction per "
    "arrow group and list all destinations under that arrow. Read whatever text is "
    "visible, even if partial. Answer ONLY compact JSON: "
    '{"groups": [{"dir": "left|right|straight", "dests": ["<line>", "<line>"]}, ...]}'
)


_DIRECTION_WORDS = {"straight", "left", "right", "up", "down", "forward", "back",
                    "backward", "ahead", "behind", "unknown", "none", ""}
_PLACEHOLDER_KEYS = {"dest", "destination", "destinations", "label", "labels",
                     "dir", "direction", "key", "name", "value", "sign", "text", "line"}
# map arrow words / glyphs to the three directions the pipeline uses
_DIR_MAP = {"up": "straight", "\u2191": "straight", "forward": "straight", "ahead": "straight",
            "straight": "straight", "left": "left", "\u2190": "left",
            "right": "right", "\u2192": "right",
            "down": "down", "\u2193": "down", "back": "back", "behind": "back"}


def _norm_dir(d) -> str:
    """Normalize an arrow word/glyph to left|right|straight (pass through unknowns)."""
    s = str(d).strip().lower()
    return _DIR_MAP.get(s, s)


def _is_placeholder_key(k) -> bool:
    """True for schema scaffolding the model sometimes echoes instead of real text:
    'DEST', 'labels', 'DEST1'.., or a literal template token like '<line>'."""
    k = str(k).strip().lower()
    if "<" in k and ">" in k:
        return True
    import re
    return k in _PLACEHOLDER_KEYS or re.fullmatch(r"dest\d+", k) is not None


def _is_bare_direction(v) -> bool:
    return str(v).strip().lower() in _DIRECTION_WORDS


def _clean_labels(d: dict) -> dict:
    """A dict whose keys are all schema placeholders mapping to bare directions
    (e.g. {'DEST': 'straight'}) carries no sign content -> treat as empty so it
    can't out-vote a real read."""
    if not isinstance(d, dict) or not d:
        return {}
    if all(_is_placeholder_key(k) for k in d) and all(_is_bare_direction(v) for v in d.values()):
        return {}
    return d


def _expand_group(g) -> list:
    """One arrow group -> [(dest, dir), ...]. Every destination in the group inherits
    the group's single direction — that's the whole point (no per-line defaulting).
    Tolerant of {"dir","dests"} and the single-key {"<direction>": [dests]} shape."""
    if not isinstance(g, dict) or not g:
        return []
    d = g.get("dir") or g.get("direction") or g.get("arrow")
    dests = g.get("dests") or g.get("destinations") or g.get("labels")
    if d is not None and dests is not None:
        if isinstance(dests, str):
            dests = [dests]
        if isinstance(dests, dict):
            dests = list(dests.keys())
        if isinstance(dests, (list, tuple)):
            return [(str(x).strip(), _norm_dir(d)) for x in dests if str(x).strip()]
    if len(g) == 1:                                   # tolerate {"right": [dests]}
        (k, v), = g.items()
        if _norm_dir(k) in {"left", "right", "straight"}:
            if isinstance(v, str):
                v = [v]
            if isinstance(v, (list, tuple)):
                return [(str(x).strip(), _norm_dir(k)) for x in v if str(x).strip()]
    return []


def _extract_labels(resp: str) -> dict:
    import re
    if not resp:
        return {"labels": {}}
    cleaned = re.sub(r"```(?:json)?", "", resp).strip()
    start = cleaned.find("{"); end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"labels": {}}
    blob = cleaned[start:end + 1]
    try:
        obj = json.loads(blob)
    except Exception:
        try:
            obj = json.loads(blob.replace("'", '"'))
        except Exception:
            return {"labels": {}}
    if not isinstance(obj, dict):
        return {"labels": {}}
    # NEW grouped schema: {"groups": [{"dir","dests"}, ...]} -> expand, one dir per arrow
    groups = obj.get("groups")
    if isinstance(groups, list):
        flat = {}
        for g in groups:
            for dest, d in _expand_group(g):
                flat[dest] = d
        return {"labels": _clean_labels(flat)}
    # --- backward-compatible flat shapes (legacy / fallback if model emits flat) ---
    # unwrap a wrapper key ("labels", "DEST", ...) that maps to a dict of entries
    for k, v in obj.items():
        if _is_placeholder_key(k) and isinstance(v, dict):
            return {"labels": _clean_labels(v)}
    # otherwise treat the object itself as the flat {destination: direction} dict
    if all(isinstance(v, str) for v in obj.values()):
        return {"labels": _clean_labels(obj)}
    return {"labels": {}}


def _labels_match(a: dict, b: dict) -> bool:
    """Two reads 'match' if they identify the same destinations with the same
    directions (case-insensitive on keys). Tolerant of ordering."""
    if not a or not b:
        return False
    na = {k.strip().lower(): str(v).strip().lower() for k, v in a.items()}
    nb = {k.strip().lower(): str(v).strip().lower() for k, v in b.items()}
    return na == nb


def _goal_room(goal: str) -> str:
    """'room 2-101' -> '2-101'. Falls back to the lowercased goal if no room token."""
    import re
    m = re.search(r"[A-Za-z]?\d+[-–]\d+", goal or "")
    return (m.group(0) if m else (goal or "")).strip().lower()


def _goal_in_parsed(parsed: dict, goal_room: str) -> bool:
    """True if any destination key references the goal room. Substring match on the
    room token (handles goal '2-101' vs key '2-101 to 2-129'). NOTE: substring, not
    range arithmetic — if a goal can fall strictly INSIDE a printed range (goal 2-115
    under '2-101 to 2-129'), add range-containment here. For the current data the goal
    is the range start, so substring suffices."""
    if not parsed or not goal_room:
        return False
    return any(goal_room in str(k).lower() for k in parsed.keys())


class Reader:
    def __init__(self, config: Config, vlm: Optional[VLMInterface] = None):
        self.cfg = config
        self.vlm = vlm   # shared VLMInterface; None only when stub_reader=True
        self._last_read_times: List[float] = []   # per-call durations from last read()

    def read(self, image, detection: Detection, n_samples: int = 3, frame_idx: int = 0) -> ReadResult:
        """Crop the sign from full-res and read it, returning text + confidence.
        Also saves the crop to disk (if cfg.save_crops) and records crop diagnostics."""
        self._last_read_times = []   # reset per-call timing for this invocation
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

    def read_best_for_goal(self, image, detections: List[dict], goal: str,
                           n_samples: int = 3, frame_idx: int = 0) -> ReadResult:
        """MULTI-SIGN read. Reads candidates in score order and SELECTS the one whose
        entries reference the goal. Cost-aware: stops at the first confident goal-match,
        so single-sign frames cost what they did before. Size gate (cfg.min_read_area_ratio)
        skips reads on tiny spurious panels. Emits a MULTI-SIGN trace when >1 candidate."""
        self._last_read_times = []
        if self.cfg.stub_reader:
            d0 = detections[0] if detections else {"confidence": 0.0, "class_name": "sign"}
            return self._stub_read(Detection(ObjectClass.SIGN, float(d0["confidence"]),
                                             (0, 0, 240, 420), d0.get("class_name", "sign")))
        if not detections:
            return self._empty_read(image)
        goal_room = _goal_room(goal)
        gate = getattr(self.cfg, "min_read_area_ratio", 0.003)
        all_times = []
        records = []
        top_rec = fallback_rec = selected_rec = None
        for rank, det_dict in enumerate(detections):
            area_ratio = float(det_dict.get("box_area_ratio", 1.0))
            score = float(det_dict.get("score", det_dict.get("confidence", 0.0)))
            rec = {"rank": rank, "score": score, "area": area_ratio, "skipped": False,
                   "parsed": {}, "conf": 0.0, "can_decide": False, "goal": False,
                   "selected": False, "_res": None}
            records.append(rec)
            if area_ratio < gate:
                rec["skipped"] = True
                continue
            cx1, cy1, cx2, cy2 = det_dict["crop_box"]
            det = Detection(ObjectClass.SIGN, float(det_dict["confidence"]),
                            (cx1, cy1, cx2 - cx1, cy2 - cy1), det_dict.get("class_name", "sign"))
            res = self.read(image, det, n_samples=n_samples, frame_idx=frame_idx)
            all_times.extend(self._last_read_times)
            rec["parsed"] = res.structured
            rec["conf"] = res.read_confidence
            rec["can_decide"] = res.can_decide
            rec["goal"] = bool(res.can_decide and _goal_in_parsed(res.structured, goal_room))
            rec["_res"] = res
            if top_rec is None:
                top_rec = rec
            if res.can_decide:
                if rec["goal"]:
                    selected_rec = rec
                    break
                if fallback_rec is None:
                    fallback_rec = rec
        if selected_rec is None:
            selected_rec = fallback_rec or top_rec
        if selected_rec is not None:
            selected_rec["selected"] = True
        self._last_read_times = all_times
        self._log_multi_sign(records, len(detections))
        if selected_rec is not None:
            return selected_rec["_res"]
        return self._empty_read(image)

    def _log_multi_sign(self, records, n_detected):
        if len(records) < 2:
            return
        n_read = sum(1 for r in records if not r["skipped"])
        print(f"  MULTI-SIGN: {n_detected} candidate panel(s), {n_read} read")
        for r in records:
            mark = "  <- SELECTED" if r["selected"] else ""
            if r["skipped"]:
                print(f"    cand {r['rank']}  score={r['score']:.3f}  "
                      f"area={r['area'] * 100:.1f}%  SKIPPED (below size gate){mark}")
            else:
                goal = "YES" if r["goal"] else "NO"
                print(f"    cand {r['rank']}  score={r['score']:.3f}  "
                      f"area={r['area'] * 100:.1f}%  conf={r['conf']:.3f}  "
                      f"goal={goal}  read={r['parsed']}{mark}")
        not_read = n_detected - len(records)
        if not_read > 0:
            print(f"    (+{not_read} more not read - goal already found, reads saved)")

    def _one_read(self, crop, sample: bool) -> dict:
        _t0 = time.perf_counter()
        resp = self.vlm.generate(PARSE_PROMPT, crop, sample=sample, max_new_tokens=200)
        self._last_read_times.append(time.perf_counter() - _t0)
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

    def _empty_read(self, image) -> ReadResult:
        """No readable candidate (e.g. all below the size gate). Low confidence so the
        loop keeps approaching until a panel is large enough to read."""
        try:
            src = image.size
        except Exception:
            src = (0, 0)
        return ReadResult(text="{}", read_confidence=0.0, structured={},
                          can_decide=False, crop_box=(0, 0, 0, 0), crop_size=(0, 0),
                          crop_path="", src_size=src)