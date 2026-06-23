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
        self._memory = ""                  # running summary of decisions (chain-of-thought memory)
        from .debug import DebugLogger
        self.dbg = DebugLogger(level=config.debug_level)
        print(f"=== ready. goal: '{config.goal}' ===\n")

    def step(self, image, idx: int, total: int = 0, ts: str = "",
             memory_override: Optional[str] = None) -> Optional[Decision]:
        """Process one frame through the full loop.

        Returns the committed Decision when one is made this frame, else None.
        memory_override: when provided (e.g. by JourneyLoop), used as the memory
        context for VLM calls instead of the internal _memory string.  The caller
        is then responsible for memory bookkeeping.
        """
        dbg = self.dbg
        dbg.frame_header(idx, total, ts)

        bundle = self.monitor.detect_all(image, step=idx)
        det = bundle.chosen
        dbg.detectors(bundle.sign_dets, bundle.hazard_dets)
        dbg.chosen(det)

        # pick which memory string the VLM sees
        mem = memory_override if memory_override is not None else self._memory

        # --- nothing relevant: keep going ---
        if det.cls == ObjectClass.NONE:
            self.controller.continue_previous()
            dbg.action(self.controller.current_action.value, "continuing previous")
            self._approach_steps = 0
            return None

        # --- hazard: VLM reasons about what it means ---
        if det.cls in (ObjectClass.STAIRS, ObjectClass.OBSTACLE):
            decision = self.reasoner.reason_hazard(image, det, mem)
            dbg.reasoning(decision.rationale)
            self.controller.execute(decision)
            dbg.action(decision.action.value)
            if memory_override is None:
                self._memory = (self._memory + f" | {decision.action.value}").strip(" |")
            self._approach_steps = 0
            return decision

        # --- sign: read with confidence gate ---
        read = self.reader.read(image, det, frame_idx=idx)
        dbg.crop(read.crop_box, read.crop_size, read.crop_path, read.src_size)
        dbg.read(read)

        if not read.can_decide:
            self._approach_steps += 1
            if self._approach_steps <= self.cfg.max_approach_steps:
                dbg.approach(self._approach_steps, self.cfg.max_approach_steps)
                self.controller.execute_forward_to_read()
            else:
                print("  ACTION: approach cap reached; continuing forward")
                self.controller.continue_previous()
            dbg.action(self.controller.current_action.value)
            return None

        # high confidence -> commit to chain-of-thought reasoning
        self._approach_steps = 0
        decision = self.reasoner.reason_sign(image, read, self.cfg.goal, mem)
        dbg.reasoning(decision.rationale)
        self.controller.execute(decision)
        dbg.action(decision.action.value)
        if memory_override is None:
            self._memory = (self._memory + f" | {decision.action.value}").strip(" |")
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
    ap.add_argument("--frames", help="Folder of .jpg frames (a recorded trip) to run on")
    ap.add_argument("--goal", default="room 2-130")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--stub-all", action="store_true",
                    help="Run with ALL components stubbed (no models) — tests the loop logic only")
    ap.add_argument("--stub-detector", action="store_true")
    ap.add_argument("--stub-reader", action="store_true")
    ap.add_argument("--stub-reasoner", action="store_true")
    args = ap.parse_args()

    cfg = Config(goal=args.goal, every_n_frames=args.every)
    if args.stub_all:
        cfg.stub_detector = cfg.stub_reader = cfg.stub_reasoner = True
    cfg.stub_detector = cfg.stub_detector or args.stub_detector
    cfg.stub_reader = cfg.stub_reader or args.stub_reader
    cfg.stub_reasoner = cfg.stub_reasoner or args.stub_reasoner

    loop = AdaptiveReasoningLoop(cfg)

    if args.frames:
        loop.run_on_folder(args.frames)
    else:
        # no frames given: run the fully-stubbed scripted scenario to show the loop
        print("(no --frames given; running scripted STUB scenario to demo the loop)\n")
        cfg.stub_detector = cfg.stub_reader = True
        loop = AdaptiveReasoningLoop(cfg)
        from PIL import Image
        blank = Image.new("RGB", (1920, 1280))
        for i in range(11):
            loop.step(blank, i)
            time.sleep(0.05)


if __name__ == "__main__":
    main()