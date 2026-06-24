"""
JourneyLoop — the outer state machine that chains many per-frame inner-loop
decisions into a full multi-leg journey.

States:
  REASONING        — run the inner loop frame-by-frame; wait for a committed Decision
  ACTING           — decision committed; hand to VLA (placeholder); immediately → WAITING
  WAITING          — action is executing; poll odom for completion; NO reasoning here

The single most important invariant: VLM reasoning happens ONLY in REASONING.
ACTING and WAITING consume frames without any model calls.

Arrival is NOT autonomously detected here — the run ends when frames are exhausted
or the caller stops it.  The researcher checks arrival manually.
"""

import math
import time
from enum import Enum
from pathlib import Path
from typing import List, Optional

from .odom_utils import OdomRow, displacement_between, load_odom, pose_at
from .types import ActionType, Config, DecisionRecord, JourneyState


# ─────────────────────────────────────────────────────────────────────────────
# State enum
# ─────────────────────────────────────────────────────────────────────────────

class JourneyPhase(str, Enum):
    REASONING = "reasoning"
    ACTING    = "acting"     # instant in offline mode (no real VLA); transitions → WAITING
    WAITING   = "waiting"    # action executing; odom polled for completion


# ─────────────────────────────────────────────────────────────────────────────
# CompletionDetector interface
# ─────────────────────────────────────────────────────────────────────────────

class CompletionDetector:
    """Interface for detecting when a committed maneuver has finished.

    The default implementation uses recorded odometry (angle-agnostic settling).
    A live version (or an instrumented-OmniVLA signal) can swap in by subclassing.
    """
    def start(self, decision, t_start_ns: int) -> None:
        """Called when an action commits.  Record start state."""
        raise NotImplementedError

    def is_done(self, t_now_ns: int) -> bool:
        """Return True once the maneuver has settled."""
        raise NotImplementedError

    def turn_magnitude_deg(self) -> float:
        """Integrated yaw change since start (degrees).  Logging only — NOT the trigger."""
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Real odom-based implementation
# ─────────────────────────────────────────────────────────────────────────────

class OdomCompletionDetector(CompletionDetector):
    """Detects maneuver completion from recorded odometry — motion-settled approach.

    A leg is complete when the robot was making progress and has now stopped:
      Phase 1 — moving gate:  recent-window displacement exceeds motion_active_thresh
                              (confirms motion actually began; guards against pre-leg rest)
      Phase 2 — settled:      after moving, recent-window displacement drops below
                              motion_settle_thresh and stays there for settle_debounce_ms
                              of continuous odom time

    Applies to ALL action types identically — turns and forward legs both complete
    when motion goes quiet.  No angle targets, no action-type special-casing.

    Timeout fallback: if leg_timeout_sec elapses without settling, declare done and
    log "TIMEOUT" so the settle-vs-timeout distinction is visible in the output.

    Yaw accumulation for turn_magnitude_deg() is unchanged — integrated for logging,
    NOT used as the completion trigger.
    """

    def __init__(self, odom: List[OdomRow], config: Config):
        self._odom = odom
        self.cfg = config
        self._reset()

    def _reset(self) -> None:
        self._t_start_ns: int = 0
        # motion-settled state
        self._has_moved: bool = False            # True once recent-window disp > motion_active_thresh
        self._settle_start_ns: Optional[int] = None  # odom time when settling began; None if not settling
        self._done: bool = False
        self._done_via_timeout: bool = False     # True when timeout fired instead of detected settle
        self._total_disp_m: float = 0.0         # straight-line displacement start→completion (logging)
        # yaw accumulation (turn_magnitude logging; NOT the completion trigger)
        self._cum_yaw: float = 0.0
        self._last_yaw: Optional[float] = None
        self._last_sample_t: Optional[int] = None
        # incremental sampling cursor
        self._next_sample_t: int = 0

    def start(self, decision, t_start_ns: int) -> None:
        self._reset()
        self._t_start_ns = t_start_ns
        self._next_sample_t = t_start_ns

    def is_done(self, t_now_ns: int) -> bool:
        if self._done:
            return True

        step_ns = int(self.cfg.odom_sample_ms * 1_000_000)
        while self._next_sample_t <= t_now_ns:
            self._process_sample(self._next_sample_t)
            if self._done:
                break
            self._next_sample_t += step_ns

        # timeout safety: fires at real frame time, not odom-sample time
        if not self._done:
            elapsed_s = (t_now_ns - self._t_start_ns) / 1e9
            if elapsed_s >= self.cfg.leg_timeout_sec:
                self._done = True
                self._done_via_timeout = True
                self._total_disp_m = displacement_between(
                    self._odom, self._t_start_ns, t_now_ns)

        return self._done

    def _process_sample(self, t_ns: int) -> None:
        pose = pose_at(self._odom, t_ns)
        if pose is None:
            return
        _, _, yaw = pose

        # ── yaw accumulation for turn_magnitude_deg (logging only) ───────────
        if self._last_yaw is not None and self._last_sample_t is not None:
            dt_s = (t_ns - self._last_sample_t) / 1e9
            if dt_s > 0:
                dyaw = math.atan2(
                    math.sin(yaw - self._last_yaw),
                    math.cos(yaw - self._last_yaw))
                self._cum_yaw += dyaw
        self._last_yaw = yaw
        self._last_sample_t = t_ns

        # ── motion-settled completion detection ───────────────────────────────
        window_ns = int(self.cfg.motion_window_ms * 1_000_000)
        recent_progress = displacement_between(self._odom, t_ns - window_ns, t_ns)

        # phase 1: confirm motion began
        if not self._has_moved and recent_progress >= self.cfg.motion_active_thresh:
            self._has_moved = True

        # phase 2: after moving, detect settling
        if self._has_moved:
            if recent_progress < self.cfg.motion_settle_thresh:
                if self._settle_start_ns is None:
                    self._settle_start_ns = t_ns   # entered settle state
                # else: still settling — keep the earlier start time
            else:
                self._settle_start_ns = None       # progress resumed; reset debounce

            debounce_ns = int(self.cfg.settle_debounce_ms * 1_000_000)
            if (self._settle_start_ns is not None
                    and (t_ns - self._settle_start_ns) >= debounce_ns):
                self._done = True
                self._done_via_timeout = False
                self._total_disp_m = displacement_between(
                    self._odom, self._t_start_ns, t_ns)

    def turn_magnitude_deg(self) -> float:
        return math.degrees(abs(self._cum_yaw))


# ─────────────────────────────────────────────────────────────────────────────
# JourneyLoop — the outer state machine
# ─────────────────────────────────────────────────────────────────────────────

class JourneyLoop:
    """Outer state machine that chains inner-loop decisions into a full journey.

    Constructed with a Config and an already-initialised AdaptiveReasoningLoop.
    Call run_on_trip(trip_dir) to process a recorded trip.
    """

    def __init__(self, config: Config, inner_loop):
        self.cfg = config
        self._inner = inner_loop
        self._state = JourneyState(goal=config.goal)
        self._phase = JourneyPhase.REASONING
        self._detector: Optional[CompletionDetector] = None
        self._pending_rec: Optional[DecisionRecord] = None   # record for the leg in flight
        # journey-level wall-clock timing
        self._journey_start_wall: Optional[float] = None
        self._phase_start_wall: Optional[float] = None   # when current phase began
        self._leg_reasoning_wall_s: float = 0.0          # reasoning wall-time for leg in flight
        self._total_odom_s: float = 0.0                  # cumulative odom-time across all legs

    # ── per-frame tick ────────────────────────────────────────────────────────

    def step(self, image, frame_t_ns: int, idx: int,
             total: int = 0, ts: str = "") -> None:
        """Process one frame.  State machine transitions happen here."""

        if self._phase == JourneyPhase.REASONING:
            self._tick_reasoning(image, frame_t_ns, idx, total, ts)

        elif self._phase in (JourneyPhase.ACTING, JourneyPhase.WAITING):
            self._tick_waiting(frame_t_ns, idx)

    def _tick_reasoning(self, image, frame_t_ns: int, idx: int,
                        total: int, ts: str) -> None:
        decision = self._inner.step(
            image, idx, total=total, ts=ts,
            memory_override=self._state.summary()
        )

        if decision is None:
            return   # nothing committed this frame; stay in REASONING

        # ── dedup: don't re-decide the same sign we already acted on ──────────
        if (decision.read is not None
                and decision.read.structured
                and decision.read.structured == self._state.last_sign_text):
            print(f"  [Journey] frame {idx} | REASONING | dedup: same sign as last leg, skipping")
            return

        # ── commit the decision ───────────────────────────────────────────────
        rec = DecisionRecord(
            leg=self._state.current_leg,
            sign_text=decision.read.structured if decision.read else {},
            action=decision.action,
            rationale=decision.rationale,
            ts=ts,
        )
        self._state.history.append(rec)
        self._state.last_sign_text = rec.sign_text
        self._state.current_leg += 1
        self._pending_rec = rec

        print(f"\n[Journey] ══════════════════════════════════════════════")
        print(f"[Journey] LEG {rec.leg} COMMITTED  (frame {idx}, t={ts})")
        print(f"[Journey]   action : {decision.action.value}")
        print(f"[Journey]   saw    : {rec.sign_text}")
        print(f"[Journey]   goal   : {self.cfg.goal}")
        print(f"[Journey]   memory : {self._state.summary()}")
        print(f"[Journey] → ACTING → WAITING")
        print(f"[Journey] ══════════════════════════════════════════════\n")

        # capture reasoning wall-time for this leg before transitioning
        if self._phase_start_wall is not None:
            self._leg_reasoning_wall_s = time.perf_counter() - self._phase_start_wall

        # hand off to detector (ACTING is instant in offline mode — no real VLA)
        self._phase = JourneyPhase.ACTING
        if self._detector is not None:
            self._detector.start(decision, frame_t_ns)
        self._phase = JourneyPhase.WAITING
        self._phase_start_wall = time.perf_counter()   # WAITING phase begins now

    def _tick_waiting(self, frame_t_ns: int, idx: int) -> None:
        if self._detector is None:
            # no odom detector set up — shouldn't happen in a full run
            print(f"  [Journey] frame {idx} | WAITING | (no detector; staying)")
            return

        print(f"  [Journey] frame {idx} | WAITING (leg {self._state.current_leg - 1})")

        if self._detector.is_done(frame_t_ns):
            mag = self._detector.turn_magnitude_deg()
            if self._pending_rec is not None:
                self._pending_rec.turn_magnitude_deg = mag

            odom_elapsed_s = ((frame_t_ns - self._detector._t_start_ns) / 1e9
                              if hasattr(self._detector, "_t_start_ns") else 0.0)
            via_timeout = getattr(self._detector, "_done_via_timeout", False)
            total_disp  = getattr(self._detector, "_total_disp_m", 0.0)
            completion  = "TIMEOUT (motion never settled)" if via_timeout else "SETTLED (motion went quiet)"

            # wall-clock timing for this leg
            waiting_wall_s  = (time.perf_counter() - self._phase_start_wall
                               if self._phase_start_wall is not None else 0.0)
            journey_wall_s  = (time.perf_counter() - self._journey_start_wall
                               if self._journey_start_wall is not None else 0.0)
            self._total_odom_s += odom_elapsed_s

            leg_num = self._pending_rec.leg if self._pending_rec else "?"
            print(f"\n[Journey] ══════════════════════════════════════════════")
            print(f"[Journey] LEG {leg_num} COMPLETE  ({completion})")
            print(f"[Journey]   turn magnitude     : {mag:.1f}°")
            print(f"[Journey]   total disp         : {total_disp:.2f} m")
            print(f"[Journey]   odom elapsed       : {odom_elapsed_s:.1f} s  "
                  f"(robot motion duration — recorded)")
            if self.cfg.log_timing:
                print(f"[Journey]   reasoning wall     : {self._leg_reasoning_wall_s:.1f} s  "
                      f"(wall-clock: VLM calls for this leg)")
                print(f"[Journey]   waiting wall       : {waiting_wall_s:.1f} s  "
                      f"(wall-clock: consuming WAITING frames)")
                print(f"[Journey]   journey wall total : {journey_wall_s:.1f} s  "
                      f"(wall-clock since run started)")
                print(f"[Journey]   journey odom total : {self._total_odom_s:.1f} s  "
                      f"(odom-time across all completed legs)")
            print(f"[Journey] → REASONING (leg {self._state.current_leg})")
            print(f"[Journey] ══════════════════════════════════════════════\n")

            self._pending_rec = None
            self._phase = JourneyPhase.REASONING
            self._phase_start_wall = time.perf_counter()   # REASONING phase begins now
            self._leg_reasoning_wall_s = 0.0

    # ── batch entry point ─────────────────────────────────────────────────────

    def run_on_trip(self, trip_dir: str) -> None:
        """Run the journey loop over a recorded trip directory.

        Expected layout:
          {trip_dir}/frames/         — .jpg files (named by timestamp_ns or numbered)
          {trip_dir}/odom.csv        — columns: timestamp_ns, x, y, yaw
          {trip_dir}/frame_index.csv — one row per frame: timestamp_ns
        """
        import csv as csv_mod
        from PIL import Image

        trip = Path(trip_dir)
        frames_dir = trip / "frames"
        odom_csv = trip / "odom.csv"
        frame_idx_csv = trip / "frame_index.csv"

        if not frames_dir.exists():
            print(f"ERROR: {frames_dir} does not exist")
            return
        if not odom_csv.exists():
            print(f"ERROR: {odom_csv} does not exist")
            return

        # load odom and wire up the completion detector
        odom = load_odom(str(odom_csv))
        print(f"[Journey] odom loaded: {len(odom)} readings  "
              f"t=[{odom[0][0]}, {odom[-1][0]}] ns")
        self._detector = OdomCompletionDetector(odom, self.cfg)

        # load frame list and timestamps
        frames = sorted(frames_dir.glob("*.jpg"))[:: self.cfg.every_n_frames]
        if not frames:
            print(f"ERROR: no .jpg frames in {frames_dir}")
            return

        frame_t_ns: List[int] = []
        if frame_idx_csv.exists():
            with open(frame_idx_csv, newline="") as f:
                for row in csv_mod.DictReader(f):
                    frame_t_ns.append(int(row["timestamp_ns"]))
            # apply every_n_frames to timestamps too
            frame_t_ns = frame_t_ns[:: self.cfg.every_n_frames]
        else:
            # fall back to parsing filenames as timestamps
            for p in frames:
                try:
                    frame_t_ns.append(int(p.stem))
                except ValueError:
                    frame_t_ns.append(0)

        # align lengths (frame list and timestamp list must match)
        n = min(len(frames), len(frame_t_ns))
        if n != len(frames) or n != len(frame_t_ns):
            print(f"[Journey] WARNING: {len(frames)} frames but {len(frame_t_ns)} timestamps; "
                  f"truncating to {n}")
        frames = frames[:n]
        frame_t_ns = frame_t_ns[:n]

        print(f"[Journey] {n} frames  |  goal: '{self.cfg.goal}'\n")

        self._journey_start_wall = time.perf_counter()
        self._phase_start_wall   = time.perf_counter()   # first REASONING phase begins now

        errors = []
        for idx, (p, t_ns) in enumerate(zip(frames, frame_t_ns)):
            try:
                img = Image.open(p).convert("RGB")
                self.step(img, t_ns, idx, total=n, ts=p.stem)
            except Exception as e:
                import traceback
                errors.append((p.name, repr(e)))
                print(f"[Journey] ERROR frame {idx} ({p.name}): {e}")
                traceback.print_exc()

        # ── end-of-run summary ────────────────────────────────────────────────
        print(f"\n[Journey] ══════════════════════════ JOURNEY COMPLETE ══")
        print(f"[Journey] {len(self._state.history)} leg(s) committed  |  "
              f"goal: '{self.cfg.goal}'")
        for rec in self._state.history:
            mag_str = (f"  turn {rec.turn_magnitude_deg:.1f}°"
                       if rec.action in (ActionType.TURN_LEFT, ActionType.TURN_RIGHT)
                       else "")
            print(f"  leg {rec.leg}: {rec.action.value:12s} | "
                  f"saw {list(rec.sign_text.keys())}{mag_str}")
        if errors:
            print(f"\n{len(errors)} frame(s) errored:")
            for name, err in errors[:10]:
                print(f"  {name}: {err}")
        print(f"[Journey] ════════════════════════════════════════════════\n")
