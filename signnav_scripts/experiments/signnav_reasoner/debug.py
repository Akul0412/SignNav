"""
debug.py - structured, readable logging for the adaptive-reasoning loop.

The goal: make it OBVIOUS what ran, what each detector saw (with raw scores),
which branch fired and why, and what the reasoning + action were. So problems
like "GroundingDINO fired on the word 'stairs' printed on a sign" or "YOLO saw
no sign" jump out instead of hiding.

Toggle verbosity with Config.debug_level:
  0 = quiet (one line per frame)
  1 = normal (branch + decision + key facts)
  2 = verbose (ALL raw detections from BOTH detectors, every score)
"""

SEP = "─" * 72


class DebugLogger:
    def __init__(self, level: int = 2):
        self.level = level

    def frame_header(self, idx: int, total: int, ts: str = ""):
        print(f"\n{SEP}")
        print(f"FRAME {idx+1}/{total}" + (f"   ({ts})" if ts else ""))
        print(SEP)

    def detectors(self, sign_dets: list, hazard_dets: list):
        """Show raw output from BOTH detectors, even the ones not chosen."""
        if self.level >= 1:
            print("  DETECTORS:")
            # YOLO signs
            if sign_dets:
                print(f"    [YOLO/signs]   {len(sign_dets)} detection(s):")
                for d in sign_dets[:5]:
                    print(f"        - {d.get('class_name','?'):20s} conf={d.get('confidence',0):.3f} "
                          f"box={_fmt_box(d.get('bbox'))}")
            else:
                print(f"    [YOLO/signs]   none")
            # GroundingDINO hazards
            if hazard_dets:
                print(f"    [GDINO/hazards] {len(hazard_dets)} detection(s):")
                for d in hazard_dets[:5]:
                    print(f"        - {d['label']:20s} conf={d['confidence']:.3f} "
                          f"box={_fmt_box(d['box_xywh'])}")
            else:
                print(f"    [GDINO/hazards] none")

    def chosen(self, det):
        from .types import ObjectClass
        if det.cls == ObjectClass.NONE:
            print("  CHOSEN: nothing relevant  ->  branch=CONTINUE")
        else:
            print(f"  CHOSEN: {det.cls.value} (conf={det.confidence:.3f}, "
                  f"label='{det.label}')  ->  branch={det.cls.value.upper()}")

    def read(self, read):
        print(f"  READ: confidence={read.read_confidence:.3f}  "
              f"can_decide={read.can_decide}")
        print(f"        parsed sign = {read.structured}")

    def crop(self, box_xywh, crop_size, saved_path=None, src_size=None):
        """Log what region was cropped and fed to the reader — so you can tell a
        bad-crop failure from a bad-read failure."""
        x, y, w, h = [int(v) for v in box_xywh]
        line = (f"  CROP: region=({x},{y},{w}x{h})  crop_size={crop_size[0]}x{crop_size[1]}")
        if src_size:
            # how much of the frame the crop covers (tiny crop => sign was far/small)
            frac = (w * h) / max(1, src_size[0] * src_size[1])
            line += f"  ({frac*100:.1f}% of frame)"
        print(line)
        if saved_path:
            print(f"        saved crop -> {saved_path}")
        # warn on likely-bad crops
        if crop_size[0] < 80 or crop_size[1] < 80:
            print(f"        !! crop is small ({crop_size[0]}x{crop_size[1]}) — sign may be "
                  f"too far/low-res to read")

    def approach(self, step, cap):
        print(f"  ACTION: read confidence too low -> APPROACH to re-read "
              f"[{step}/{cap}]")

    def reasoning(self, rationale: str):
        if self.level >= 1:
            print("  REASONING (VLM chain-of-thought):")
            for line in rationale.splitlines():
                if line.strip():
                    print(f"      {line}")

    def action(self, action_value: str, note: str = ""):
        print(f"  >>> FINAL ACTION: {action_value.upper()}" + (f"  ({note})" if note else ""))

    def quiet_line(self, idx, det, action_value):
        from .types import ObjectClass
        d = "none" if det.cls == ObjectClass.NONE else f"{det.cls.value}:{det.confidence:.2f}"
        print(f"[{idx}] detect={d:16s} action={action_value}")


def _fmt_box(box):
    if not box:
        return "?"
    return "(" + ",".join(f"{int(v)}" for v in box) + ")"