"""
The adaptive-reasoning closed loop — the heart of the system.

Per frame:
  1. Monitor.detect()  (cheap, always on)
  2. branch on what was detected:
       NONE   -> Controller.continue_previous()        (keep moving)
       STAIRS -> Reasoner.reason_hazard() -> STOP       (no reading needed)
       SIGN   -> Reader.read() -> confidence gate:
                   low  -> approach (FORWARD) and re-read next frame
                   high -> Reasoner.reason_sign(goal) -> action
  3. Controller.execute(decision)

This is "adaptive / dynamic reasoning": heavy reasoning fires only when something
relevant appears AND (for signs) only once the read is confident. Otherwise the
robot just keeps going — cheaply.

Run modes:
  - on a folder of frames (a recorded trip) for testing
  - (later) on a live camera / ROS image stream on the Jetson
"""

import argparse
import time
from pathlib import Path

from typing import Optional
from .interfaces import make_vlm
from .types import ActionType, Config, Decision, ObjectClass
from .monitor import Monitor
from .reader import Reader
from .reasoner import Reasoner
from .controller import Controller


def _indent(text: str, pad: str = "            ") -> str:
    """Indent a multi-line reasoning trace for readable logging."""
    return "\n".join(pad + line for line in text.splitlines()) if text else pad + "(no reasoning)"


class AdaptiveReasoningLoop:
    def __init__(self, config: Config):
        self.cfg = config
        print("=== initializing adaptive-reasoning loop ===")
        self.monitor = Monitor(config)
        # one VLM instance shared by Reader and Reasoner — model loads once
        vlm = make_vlm(config) if not config.stub_reader else None
        self.reader = Reader(config, vlm=vlm)
        self.reasoner = Reasoner(config, vlm=vlm)
        self.controller = Controller()
        self._approach_steps = 0
        self._read_cooldown = 0            # frames left to coast before re-reading a sign
        self._memory = ""                  # running summary of decisions (chain-of-thought memory)
        from .debug import DebugLogger
        self.dbg = DebugLogger(level=config.debug_level)
        print(f"=== ready. goal: '{config.goal}' ===\n")

    def _log_timing(self, t_frame_start: float, t_detect: float,
                    t_read: float = 0.0, t_reason: float = 0.0) -> None:
        if not self.cfg.log_timing:
            return
        t_total = time.perf_counter() - t_frame_start
        sign_t   = getattr(self.monitor, "_last_sign_t",   0.0)
        hazard_t = getattr(self.monitor, "_last_hazard_t", 0.0)
        parts = [f"detect {t_detect:.2f}s (sign {sign_t:.2f}s gdino {hazard_t:.2f}s)"]
        if t_read > 0:
            calls = getattr(self.reader, "_last_read_times", [])
            per_call = "+".join(f"{t:.2f}s" for t in calls)
            parts.append(f"read×{len(calls)} {t_read:.2f}s ({per_call})")
        if t_reason > 0:
            parts.append(f"reason {t_reason:.2f}s")
        parts.append(f"frame {t_total:.2f}s")
        print("[timing] " + " | ".join(parts))

    def step(self, image, idx: int, total: int = 0, ts: str = "",
             memory_override: Optional[str] = None) -> Optional[Decision]:
        """Process one frame through the full loop.

        Returns the committed Decision when one is made this frame, else None.
        memory_override: when provided (e.g. by JourneyLoop), used as the memory
        context for VLM calls instead of the internal _memory string.  The caller
        is then responsible for memory bookkeeping.
        """
        t_frame = time.perf_counter()
        dbg = self.dbg
        dbg.frame_header(idx, total, ts)

        _t0 = time.perf_counter()
        bundle = self.monitor.detect_all(image, step=idx)
        t_detect = time.perf_counter() - _t0
        det = bundle.chosen
        dbg.detectors(bundle.sign_dets, bundle.hazard_dets)
        dbg.chosen(det)

        # pick which memory string the VLM sees
        mem = memory_override if memory_override is not None else self._memory

        # --- nothing relevant: keep going (no timing print — nothing expensive ran) ---
        if det.cls == ObjectClass.NONE:
            self.controller.continue_previous()
            dbg.action(self.controller.current_action.value, "continuing previous")
            self._approach_steps = 0
            return None

        # --- hazard: VLM reasons about what it means ---
        if det.cls in (ObjectClass.STAIRS, ObjectClass.OBSTACLE):
            _t0 = time.perf_counter()
            decision = self.reasoner.reason_hazard(image, det, mem)
            t_reason = time.perf_counter() - _t0
            dbg.reasoning(decision.rationale)
            self.controller.execute(decision)
            dbg.action(decision.action.value)
            if memory_override is None:
                self._memory = (self._memory + f" | {decision.action.value}").strip(" |")
            self._approach_steps = 0
            self._log_timing(t_frame, t_detect, t_reason=t_reason)
            return decision

        # --- sign: read with confidence gate ---
        _t0 = time.perf_counter()
        # re-read cooldown: after a sub-threshold read, coast forward K frames
        # without spending VLM reads — the robot's own motion brings the sign
        # nearer so the next read is bigger. Cooldown frames do NOT count against
        # max_approach_steps (that budget counts read attempts).
        if getattr(self, "_read_cooldown", 0) > 0:
            self._read_cooldown -= 1
            print(f"  COOLDOWN: skip re-read ({self._read_cooldown} left) — approaching")
            self.controller.execute_forward_to_read()
            dbg.action(self.controller.current_action.value)
            self._log_timing(t_frame, t_detect)
            return None
        read = self.reader.read_best_for_goal(image, bundle.sign_dets, self.cfg.goal, frame_idx=idx)
        t_read = time.perf_counter() - _t0
        dbg.crop(read.crop_box, read.crop_size, read.crop_path, read.src_size)
        dbg.read(read)

        if not read.can_decide:
            self._approach_steps += 1
            self._read_cooldown = getattr(self.cfg, "reread_cooldown_frames", 4)
            if self._approach_steps <= self.cfg.max_approach_steps:
                dbg.approach(self._approach_steps, self.cfg.max_approach_steps)
                self.controller.execute_forward_to_read()
            else:
                print("  ACTION: approach cap reached; continuing forward")
                self.controller.continue_previous()
            dbg.action(self.controller.current_action.value)
            self._log_timing(t_frame, t_detect, t_read=t_read)
            return None

        # high confidence -> commit to chain-of-thought reasoning
        self._read_cooldown = 0
        self._approach_steps = 0

        _t0 = time.perf_counter()
        decision = self.reasoner.reason_sign(image, read, self.cfg.goal, mem)
        t_reason = time.perf_counter() - _t0
        dbg.reasoning(decision.rationale)
        self.controller.execute(decision)
        dbg.action(decision.action.value)
        if memory_override is None:
            self._memory = (self._memory + f" | {decision.action.value}").strip(" |")
        self._log_timing(t_frame, t_detect, t_read=t_read, t_reason=t_reason)
        return decision

    def run_on_folder(self, folder: str):
        from PIL import Image
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"ERROR: frames folder does not exist: {folder}")
            return
        frames = sorted(folder_path.glob("*.jpg"))[:: self.cfg.every_n_frames]
        if not frames:
            print(f"ERROR: no .jpg frames found in {folder}")
            return
        print(f"running on {len(frames)} frames from {folder}\n")
        errors = []
        for i, p in enumerate(frames):
            try:
                img = Image.open(p).convert("RGB")
                self.step(img, i, total=len(frames), ts=p.stem)
            except Exception as e:
                import traceback
                errors.append((p.name, repr(e)))
                print(f"[frame {i}] ERROR on {p.name}: {e}")
                traceback.print_exc()
        print("\n=== run complete ===")
        if errors:
            print(f"\n{len(errors)} frame(s) errored:")
            for name, err in errors[:10]:
                print(f"  {name}: {err}")
        else:
            print("no per-frame errors.")


# small helper added to Controller via monkeypatch-free method:
def _execute_forward_to_read(self):
    from .types import ActionType
    self.current_action = ActionType.FORWARD
    self._emit(ActionType.FORWARD, "approaching sign to improve read confidence")
Controller.execute_forward_to_read = _execute_forward_to_read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames",  help="Folder of .jpg frames — short-horizon inner-loop mode")
    ap.add_argument("--journey", help="Trip directory (frames/, odom.csv, frame_index.csv) "
                                      "— long-horizon JourneyLoop mode")
    ap.add_argument("--goal", default="room 2-130")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--stub-all", action="store_true",
                    help="Run with ALL components stubbed (no models) — tests loop logic only")
    ap.add_argument("--stub-detector", action="store_true")
    ap.add_argument("--stub-reader",   action="store_true")
    ap.add_argument("--stub-reasoner", action="store_true")
    args = ap.parse_args()

    cfg = Config(goal=args.goal, every_n_frames=args.every)
    if args.stub_all:
        cfg.stub_detector = cfg.stub_reader = cfg.stub_reasoner = True
    cfg.stub_detector = cfg.stub_detector or args.stub_detector
    cfg.stub_reader   = cfg.stub_reader   or args.stub_reader
    cfg.stub_reasoner = cfg.stub_reasoner or args.stub_reasoner

    inner = AdaptiveReasoningLoop(cfg)

    if args.journey:
        from .journey import JourneyLoop
        jloop = JourneyLoop(cfg, inner)
        jloop.run_on_trip(args.journey)
    elif args.frames:
        inner.run_on_folder(args.frames)
    else:
        # no source given: run the fully-stubbed scripted scenario to demo the inner loop
        print("(no --frames or --journey given; running scripted STUB scenario)\n")
        cfg.stub_detector = cfg.stub_reader = True
        inner = AdaptiveReasoningLoop(cfg)
        from PIL import Image
        blank = Image.new("RGB", (1920, 1280))
        for i in range(11):
            inner.step(blank, i)
            time.sleep(0.05)


if __name__ == "__main__":
    main()