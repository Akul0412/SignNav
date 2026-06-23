# SignNav — Long-Horizon Architecture Spec (for implementation)

This spec describes the components to BUILD to extend the existing short-horizon
SignNav pipeline into a long-horizon, multi-leg navigation system. It references
the existing code so you implement *against* it rather than rewriting it.

**Context for the implementer:** SignNav is a reasoning-based indoor sign-navigation
system. A robot reads directional signs ("Elevators ←", "6-201 to 6-225 ←") and
navigates a building to a distant goal (e.g. "room 6-202") across many turns. The
current system handles ONE sign decision at a time (short-horizon). This spec adds
the layer that chains many decisions into a full journey (long-horizon), which is
the project's main research contribution.

---

## 0. What exists today (KEEP — do not rewrite)

Package: `signnav_scripts/experiments/signnav_reasoner/`

- `types.py` — `Config`, and dataclasses: `Detection`, `ReadResult`, `Decision`,
  enums `ObjectClass` (SIGN/STAIRS/OBSTACLE/NONE) and `ActionType`
  (FORWARD/TURN_LEFT/TURN_RIGHT/STOP/REROUTE/CONTINUE).
- `monitor.py` — `Monitor.detect_all(image) -> DetectionBundle`. Runs sign detection
  (OpenCV dark-panel heuristic) + hazard detection (GroundingDINO), returns raw
  detections plus a chosen one. Choosing rule today: strong hazard (≥0.55) > sign >
  weak hazard (ignored).
- `reader.py` — `Reader.read(image, detection) -> ReadResult`. Crops the sign,
  reads it with Qwen2.5-VL, returns parsed `{destination: direction}` + confidence.
- `reasoner.py` — `Reasoner.reason_sign(...)` and `.reason_hazard(...)` -> `Decision`.
  VLM chain-of-thought.
- `controller.py` — placeholder; prints actions, no motion.
- `loop.py` — `AdaptiveReasoningLoop.step(image, idx)`. The per-frame pass with the
  3 branches (NONE / HAZARD / SIGN). This is the "inner loop".
- `visualizer.py` — HTML visual log.

**The inner loop (`step`) is correct and stays.** The long-horizon work WRAPS it;
it does not replace it.

---

## 1. The core change: an outer state machine ("journey loop")

Today `step()` is stateless per frame — every frame is judged fresh, and the only
memory is a thin action string. Long-horizon requires an OUTER loop with explicit
states. Reasoning must only happen in one state; while an action executes, frames
must NOT trigger new reasoning.

Build a new class, e.g. `JourneyLoop` (new file `journey.py`), with these states:

```
class JourneyPhase(str, Enum):
    REASONING = "reasoning"            # run the inner loop, produce a Decision
    ACTING = "acting"                  # VLA is executing the chosen leg
    WAITING_COMPLETION = "waiting"     # poll odom until the action is done
    CHECK_ARRIVAL = "check_arrival"    # has the goal been reached?
    DONE = "done"                      # goal reached; stop
```

### State transitions
- **REASONING**: run the existing inner loop on the current frame. If it produces a
  committed `Decision` (sign read with confidence ≥ threshold, or a hazard decision),
  transition to ACTING. If nothing actionable (NONE / still approaching), stay in
  REASONING (keep consuming frames).
- **ACTING**: hand the `Decision` to the VLA bridge (§3), which sends a prompt to the
  VLA. Immediately transition to WAITING_COMPLETION. **Do NOT reason on frames here.**
- **WAITING_COMPLETION**: ignore camera reasoning; poll ODOM to detect when the
  maneuver finished (see §1a). When done, transition to CHECK_ARRIVAL.
- **CHECK_ARRIVAL**: run the arrival check (§4). If arrived, DONE. Else, back to
  REASONING for the next leg.
- **DONE**: terminal. Stop the journey.

### 1a. Action-completion detection (odom, NOT a VLA signal)
OmniVLA likely does NOT emit a "maneuver complete" event. Detect completion from
ODOMETRY instead: in WAITING_COMPLETION, watch yaw/position and consider the action
done when the target delta is reached and motion has stabilized (e.g. for a turn:
yaw changed ~90° and angular velocity ≈ 0 for N frames). Implement this behind a
small interface `CompletionDetector` so it can be stubbed offline:

```
class CompletionDetector:
    def start(self, decision): ...          # record start odom + expected delta
    def is_done(self, odom) -> bool: ...     # True when maneuver complete
```
For OFFLINE rosbag testing, provide a stub that returns done after K frames, so the
state machine can be tested without real odom.

NOTE: the project forked OmniVLA and CAN instrument it to emit a completion signal.
If that is done later, add an alternate `CompletionDetector` that uses it. Keep odom
as the default/fallback.

---

## 2. Memory: the JourneyState object

Replace the thin `_memory` string with a structured object (new, in `types.py` or
`journey.py`):

```
@dataclass
class DecisionRecord:
    leg: int
    sign_text: dict           # what the sign said, e.g. {"6-201 to 6-225": "left"}
    action: ActionType
    rationale: str            # the reasoning trace
    ts: str = ""

@dataclass
class JourneyState:
    goal: str                              # fixed target, e.g. "room 6-202"
    current_leg: int = 0
    history: list = field(default_factory=list)   # list[DecisionRecord]
    last_sign_text: dict = field(default_factory=dict)  # to avoid re-reasoning same sign
    progress_note: str = ""                # crude progress signal (see below)
    arrived: bool = False
```

### Memory rules
- **Append on DECISION, not on frame.** Only write a `DecisionRecord` to `history`
  when a real decision is committed (sign read+reasoned, or hazard decided). Do NOT
  append on every frame — that floods memory with "continue, continue".
- The reasoner should receive a compact summary of `history` (last few decisions +
  the goal) as context, so it reasons WITH journey awareness ("I came from the
  elevators, I already turned left at the 6-100s sign, now heading to 6-202").
- `last_sign_text` prevents re-reasoning the same sign on every approach frame: if
  the current read matches `last_sign_text`, and we already acted on it, don't
  re-decide — wait for the action/odom cycle.
- `progress_note` can start crude: e.g. "goal number 6-202 is in range 6-201..6-225
  which the last sign pointed left → on track". A simple heuristic is fine for v1;
  it does not need to be a learned model.

---

## 3. The VLA bridge: turning a Decision into a VLA prompt

The inner loop outputs a `Decision` (an `ActionType` + rationale). The VLA
(OmniVLA / TickVLA) does NOT take "turn_left" — it takes a natural-language prompt.
Build a translation layer (new file `vla_bridge.py`):

```
class VLABridge:
    def __init__(self, vlm, vla):   # both are interfaces (see §5)
        ...
    def decision_to_prompt(self, decision, image, journey_state) -> str:
        # Tier 1: if a visual reference exists, make a referring prompt
        # Tier 2: else use the direction->prompt map
        ...
    def execute(self, decision, image, journey_state):
        prompt = self.decision_to_prompt(...)
        return self.vla.execute(prompt)   # via the VLA interface
```

### Two-tier prompt scheme
- **Tier 1 (preferred): referring prompt.** OmniVLA performs better when the prompt
  references a visible object ("go to that sign", "head toward that door"). If the
  scene has a usable reference (the detected sign, a door, a landmark), make an extra
  VLM call: given the image + the decision, produce a short referring instruction.
- **Tier 2 (fallback): direction map.** If no usable reference, map the `ActionType`
  to a canned prompt the VLA understands: e.g.
  `{TURN_LEFT: "turn left and proceed down the hallway", FORWARD: "go straight", ...}`.

**Risk note for implementer:** the exact Tier-1/Tier-2 behavior depends on OmniVLA's
real prompt-following, which is still being probed. Build the structure and
interfaces now; expect the internals (especially the referring-prompt wording) to be
tuned later against real OmniVLA results. Keep `decision_to_prompt` easy to edit.

---

## 4. Arrival detection (goal-completion reasoning)

Short-horizon ends when an action finishes. Long-horizon ends when the GOAL is
reached, and the robot must recognize it. Add a small reasoning step (in `reasoner.py`
or a new `arrival.py`):

```
def check_arrival(self, image, read_result_or_none, goal) -> bool:
    # VLM: given the goal and the current view/sign, have we arrived?
    # e.g. goal "room 6-202" + a visible door/plate reading "6-202" => arrived
```
This is a distinct VLM call from sign-reading. It can reuse the latest sign read
(if the current sign IS the goal) or look at the current frame for a matching
door/room plate. Testable in isolation on rosbag frames near a goal.

---

## 5. Swappable model interfaces (REQUIRED for evaluation)

The evaluation methodology swaps components and compares: OmniVLA vs TickVLA, and
Qwen-local vs Gemini-API. This REQUIRES every model behind a clean interface so a
swap is a config change, not a code change. Define two interfaces (new file
`interfaces.py`):

```
class VLMInterface:
    def generate(self, prompt: str, image) -> str: ...
# Implementations: QwenVLM (local, current code), GeminiVLM (API)

class VLAInterface:
    def execute(self, prompt: str): ...        # send nav prompt to the VLA
    # may also expose odom access for the CompletionDetector
# Implementations: OmniVLA, TickVLA
```

- Refactor `reader.py` and `reasoner.py` to call a `VLMInterface` instead of Qwen
  directly. The existing Qwen code becomes `QwenVLM(VLMInterface)`.
- The VLA bridge (§3) and controller call a `VLAInterface`, never a concrete VLA.
- Add to `Config`: `vlm_backend: str = "qwen"` and `vla_backend: str = "omnivla"`,
  and a factory that instantiates the right implementations. Evaluation then runs the
  same system with different `{vlm_backend, vla_backend}` and logs metrics.

Keep these interfaces THIN — just `generate` / `execute`. Don't over-design.

---

## 6. Build order (dependency-ordered; each step testable on rosbags, no robot)

1. **JourneyState + DecisionRecord** (§2). Pure data. Wire the reasoner to accept a
   journey summary as context. Test: history appends on decision only.
2. **VLMInterface + refactor Qwen behind it** (§5, VLM half). No behavior change;
   just decoupling. Test: existing rosbag run still works via `QwenVLM`.
3. **Arrival check** (§4). Test in isolation on frames near a goal: does it correctly
   fire "arrived" only at the goal?
4. **JourneyLoop state machine** (§1) wrapping the inner loop, with a STUB
   `CompletionDetector` (done after K frames) and the controller still a placeholder.
   Test on a FULL-JOURNEY rosbag: does it chain decisions across many signs, maintain
   memory, and reach CHECK_ARRIVAL → DONE? **This validates the core contribution
   with no VLA and no robot.**
5. **VLA bridge + VLAInterface** (§3, §5 VLA half). Start with the direction-map
   (Tier 2) only; add Tier-1 referring prompts once OmniVLA behavior is known.
6. **Real odom CompletionDetector** (§1a) — when running on the robot/with odom data.
7. **Evaluation harness** — run permutations {OmniVLA/Tick} × {Qwen/Gemini} over an
   eval rosbag set, log metrics (e.g. correct turns, goal reached, # interventions).

Steps 1–4 need NO working VLA and NO Jetson — they are the research core and can be
done on MSI/local against recorded rosbags. Steps 5–7 are integration, deferred until
OmniVLA findings land.

---

## 7. Notes / constraints

- Keep the inner loop (`loop.py` `step`) intact; `JourneyLoop` calls into it.
- Config currently has: `goal`, `read_confidence_threshold` (0.70),
  `hazard_confidence_threshold` (0.55), `max_approach_steps` (8), `use_yolo` (False,
  heuristic-only), `reasoner_4bit` (False, fp16), `reasoner_model`
  (Qwen2.5-VL-7B-Instruct). Add the new backend-selection fields here.
- A separate fix (from a collaborator) will improve hazard detection by masking the
  detected sign region out of the image before GroundingDINO runs (so DINO stops
  firing "stairs" on the printed word). This lands in `monitor.py` and makes sign
  detection run BEFORE hazard detection. Leave room for that ordering in the Monitor.
- "Both fire" today picks one detection upstream. A future improvement: when both a
  sign and a hazard are present, pass BOTH to the reasoner so it can reason over the
  full scene ("stairs ahead but my goal is left per the sign"). Not required for v1.
- Offline-first: everything should be testable on extracted rosbag frames via a
  batch entry point, mirroring the existing `python -m signnav_reasoner.loop --frames
  <dir> --goal <goal>`. Add an equivalent journey-level entry point.
