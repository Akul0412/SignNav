"""
Core data types passed between components of the adaptive-reasoning loop.

The whole system is a pipeline:
    frame -> Monitor (detect) -> [branch] -> Reader (read+confidence) -> Reasoner -> Decision -> Controller

Keeping the interfaces as small dataclasses lets each component be developed,
tested, and swapped (real model <-> stub) independently.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ObjectClass(str, Enum):
    """What the cheap monitor can detect. Drives the branch."""
    SIGN = "sign"
    STAIRS = "stairs"
    OBSTACLE = "obstacle"
    NONE = "none"


@dataclass
class Detection:
    """One thing the monitor found in a frame."""
    cls: ObjectClass
    confidence: float                 # detector's confidence the object is there
    box: tuple                        # (x, y, w, h) in the full-res frame
    label: str = ""                   # raw detector label text


@dataclass
class ReadResult:
    """Result of trying to READ a detected sign."""
    text: str                         # transcribed sign content (may be partial)
    read_confidence: float            # how confident we are we read it correctly (0-1)
    structured: dict = field(default_factory=dict)  # {destination: direction} if parsed
    can_decide: bool = False          # read_confidence >= threshold
    # crop diagnostics (for the debug analyzer)
    crop_box: tuple = (0, 0, 0, 0)    # (x, y, w, h) region cropped from the full frame
    crop_size: tuple = (0, 0)         # (w, h) pixel size of the crop fed to the reader
    crop_path: str = ""               # where the crop image was saved (if saving on)
    src_size: tuple = (0, 0)          # (w, h) of the source frame (for % coverage)


class ActionType(str, Enum):
    """High-level actions the controller can execute. (Placeholder for OmniVLA mapping.)"""
    FORWARD = "forward"               # continue / approach to read better
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    STOP = "stop"
    REROUTE = "reroute"               # back off and seek alternative
    CONTINUE = "continue"             # keep doing the previous action


@dataclass
class Decision:
    """The output of a reasoning step: an action plus the rationale (the trace)."""
    action: ActionType
    rationale: str                    # human-readable reasoning trace
    triggered_by: ObjectClass = ObjectClass.NONE
    read: Optional[ReadResult] = None


@dataclass
class DecisionRecord:
    """One committed decision in the journey — appended to history, not per frame."""
    leg: int
    sign_text: dict        # what the sign said, e.g. {"6-201 to 6-225": "left"}
    action: ActionType
    rationale: str         # full reasoning trace from the VLM
    ts: str = ""           # frame timestamp when the decision was made
    turn_magnitude_deg: float = 0.0   # actual turn recorded from odom (updated after leg completes; logging only)


@dataclass
class JourneyState:
    """Persistent memory for a full multi-leg journey."""
    goal: str
    current_leg: int = 0
    history: List[DecisionRecord] = field(default_factory=list)
    last_sign_text: dict = field(default_factory=dict)  # sign we most recently acted on (dedup)
    progress_note: str = ""   # crude heuristic note updated after each leg
    arrived: bool = False

    def summary(self, max_legs: int = 5) -> str:
        """Compact history string for the VLM memory context."""
        if not self.history:
            return "(just started)"
        recent = self.history[-max_legs:]
        parts = [f"leg {r.leg}: {r.action.value} (saw {list(r.sign_text.keys())})"
                 for r in recent]
        note = f" | {self.progress_note}" if self.progress_note else ""
        return "; ".join(parts) + note


@dataclass
class Config:
    """Tunables + stub switches so the loop runs even when models can't load."""
    # confidence gating
    read_confidence_threshold: float = 0.87   # >= this => commit; a clean parse scores 0.7,
                                              # stable re-reads push higher (see reader.py)
    detect_confidence_threshold: float = 0.35 # detector min confidence to count as a detection
    hazard_confidence_threshold: float = 0.60   #catches real stairs (0.47-0.53 in testing) # HIGHER bar for hazards (reduces false stairs
                                              # from the word 'stairs' printed on signs)

    # debug
    debug_level: int = 2                       # 0=quiet, 1=normal, 2=verbose (all raw detections)
    save_crops: bool = True                    # save each sign crop to disk for inspection
    crop_dir: str = "debug_crops"              # where to save crops

    # detector prompts (open-vocab GroundingDINO text queries)
    sign_prompt: str = "directional sign . room number sign . wall sign"
    hazard_prompt: str = "stairs . staircase . steps . obstacle"

    # YOLO sign detector settings
    yolo_model_path: str = "yolov8n.pt"
    yolo_imgsz: int = 640
    crop_margin: float = 0.10
    sign_confidence_threshold: float = 0.85
    device: str = ""
    # multi-crop: skip a VLM read on candidate panels smaller than this fraction of the
    # frame (filters tiny spurious dark regions surfaced when returning all candidates).
    # Keep BELOW your smallest real sign — c11's hard sign read at ~0.42%, and the
    # heuristic already floors candidates at 0.25%, so the useful band is ~0.0030–0.0040.
    min_read_area_ratio: float = 0.003

    # model selection / resources (Jetson-friendly defaults)
    reasoner_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    reasoner_4bit: bool = False               # fp16 on Jetson (no bitsandbytes needed)
    use_yolo: bool = False                     # if False, skip YOLO (no torchvision needed);
                                              # signs come from the OpenCV dark-panel heuristic

    # STUB SWITCHES — flip to True if a model can't load (memory/hardware),
    stub_detector: bool = False
    stub_reader: bool = False
    stub_reasoner: bool = False
    allow_stub_fallback: bool = False

    # backend selection (swap for evaluation)
    vlm_backend: str = "qwen"              # "qwen" (local) — future: "gemini"

    # goal for this run (a concrete destination on the signs)
    goal: str = "room 2-130"

    # loop behavior
    every_n_frames: int = 1           # process every Nth frame from the source
    max_approach_steps: int = 8       # safety cap on "keep approaching to read"

    log_timing: bool = True           # print per-frame and per-leg wall-clock timing

    # odom-based completion detection — motion-settled (all tunable against a recording)
    # a leg is done when the robot was moving and then its displacement goes quiet
    odom_sample_ms: float = 75.0          # cadence at which is_done() samples odom (ms)
    motion_window_ms: float = 500.0       # recent window over which progress is measured (ms)
    motion_active_thresh: float = 0.10    # m — recent-window disp > this => robot is moving
    motion_settle_thresh: float = 0.03    # m — recent-window disp < this => stopped making progress
    settle_debounce_ms: float = 1000.0    # ms — must stay settled this long before declaring done
    leg_timeout_sec: float = 30.0         # s  — generous fallback if motion never settles

    reread_cooldown_frames: int = 4