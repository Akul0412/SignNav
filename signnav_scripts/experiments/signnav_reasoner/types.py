"""
Core data types passed between components of the adaptive-reasoning loop.

The whole system is a pipeline:
    frame -> Monitor (detect) -> [branch] -> Reader (read+confidence) -> Reasoner -> Decision -> Controller

Keeping the interfaces as small dataclasses lets each component be developed,
tested, and swapped (real model <-> stub) independently.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


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
class Config:
    """Tunables + stub switches so the loop runs even when models can't load."""
    # confidence gating
    read_confidence_threshold: float = 0.70   # >= this => commit; a clean parse scores 0.7,
                                              # stable re-reads push higher (see reader.py)
    detect_confidence_threshold: float = 0.35 # detector min confidence to count as a detection
    hazard_confidence_threshold: float = 0.55 # HIGHER bar for hazards (reduces false stairs
                                              # from the word 'stairs' printed on signs)

    # debug
    debug_level: int = 2                       # 0=quiet, 1=normal, 2=verbose (all raw detections)
    save_crops: bool = True                    # save each sign crop to disk for inspection
    crop_dir: str = "debug_crops"              # where to save crops

    # detector prompts (open-vocab GroundingDINO text queries)
    sign_prompt: str = "directional sign . room number sign . wall sign"
    hazard_prompt: str = "stairs . staircase . steps . obstacle"

    # YOLO sign detector (Yehor's pipeline) settings
    yolo_model_path: str = "yolov8n.pt"
    yolo_imgsz: int = 640
    crop_margin: float = 0.10
    device: str = ""                          # "" -> auto; "cpu" or "0" to force

    # model selection / resources (Jetson-friendly defaults)
    reasoner_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    reasoner_4bit: bool = False               # fp16 on Jetson (no bitsandbytes needed)
    use_yolo: bool = False                     # if False, skip YOLO (no torchvision needed);
                                              # signs come from the OpenCV dark-panel heuristic

    # STUB SWITCHES — flip to True if a model can't load (memory/hardware),
    # so the architecture/loop still runs end-to-end with mocks.
    stub_detector: bool = False
    stub_reader: bool = False
    stub_reasoner: bool = False
    # If a REAL model fails to load, do NOT silently serve fake data unless this
    # is explicitly True. On live/real runs keep this False so failures are loud.
    allow_stub_fallback: bool = False

    # goal for this run (a concrete destination on the signs)
    goal: str = "room 2-130"

    # loop behavior
    every_n_frames: int = 1           # process every Nth frame from the source
    max_approach_steps: int = 8       # safety cap on "keep approaching to read"