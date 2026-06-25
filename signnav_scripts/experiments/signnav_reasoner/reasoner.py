"""
Reasoner: the high-level decision-maker — VLM CHAIN-OF-THOUGHT, not hardcoded rules.

This does NOT contain prohibition word-lists or if-then decision logic. The
intelligence lives in the VLM. Given:
  - the structured sign facts the Reader already extracted (perception),
  - the goal,
  - the robot's constraints (wheeled; cannot take stairs; is a delivery robot, not staff),
the Reasoner prompts the VLM to reason step-by-step (Alpamayo-style chain of
causation) and produce a decision + rationale.

Why this generalizes: the VLM UNDERSTANDS what "Staff Only" / "Authorized Personnel"
/ "Construction Ahead" mean — it isn't matching a fixed list. So it handles
restrictions and situations nobody enumerated. The rationale it emits is also the
reasoning trace you would later DISTILL into OmniVLA.

Two-call design (chosen): Reader extracts facts (reliable perception) -> Reasoner
reasons over those facts (open reasoning). Keeps perception trustworthy while
letting reasoning be general.

SCENE ALERT (notice/hazard awareness): when the monitor's notice channel flags a
posted notice / floor sign / placard / obstacle in view, reason_sign is called with
scene_alert=True, which adds a clause telling the VLM to examine the WHOLE frame —
not just the directory sign — read any such notice itself, and treat a closed /
blocked / relocated goal as a route that does NOT achieve the goal (-> reroute/stop).
There is NO notice OCR or schema here: the VLM reads and interprets the scene
directly. The detector only decides WHEN to look hard (cheap trigger -> expensive
reasoning); the VLM does all the interpreting.
"""

import json
import re
from typing import Optional

from .interfaces import VLMInterface
from .types import ActionType, Config, Decision, Detection, ObjectClass, ReadResult


# The chain-of-thought reasoning prompt. The model thinks step by step, then commits.
# Note: NO hardcoded prohibitions — the model reasons about what it sees.
def build_reasoning_prompt(goal: str, sign_facts: dict, memory_summary: str,
                           scene_alert: bool = False) -> str:
    # When something notice-like was detected in the scene, tell the VLM to look at the
    # whole image and interpret it. No fixed rules — the VLM decides what it means.
    alert_block = ""
    if scene_alert:
        alert_block = (
            "IMPORTANT — look at the ENTIRE image, not only the directory sign. A posted "
            "notice, floor sign, placard, cone, or obstacle has been detected in the scene. "
            "Read any such notice yourself and judge what it means for you. If it indicates "
            "that your GOAL, or the path you would take to reach it, is closed, blocked, out "
            "of service, relocated, or otherwise unavailable, then following the directory "
            "sign does NOT achieve your goal — do not commit to that route. Describe what you "
            "see and choose reroute (or stop) instead. If the notice is unrelated to your "
            "goal and your path, note it and proceed normally.\n\n"
        )
    return (
        "You are the reasoning module of an indoor wheeled delivery robot. You reason "
        "step by step about what to do next, like a careful driver.\n\n"
        f"YOUR GOAL: reach \"{goal}\".\n"
        "YOUR CONSTRAINTS: you are a wheeled robot (you CANNOT use stairs). You are a "
        "delivery robot, not staff (you must respect access restrictions). Safety first.\n\n"
        f"WHAT YOU JUST READ ON THE SIGN(S): {json.dumps(sign_facts)}\n"
        f"WHAT YOU'VE DONE SO FAR: {memory_summary or '(just started)'}\n\n"
        f"{alert_block}"
        "Reason through this ONE STEP AT A TIME. Think about:\n"
        "  - Which entries on the sign are relevant to your goal, and what direction they imply.\n"
        "  - Whether any entry is a RESTRICTION or WARNING that affects you (think about what "
        "each sign actually means for a delivery robot — do not assume, reason about it).\n"
        "  - Whether your goal's direction conflicts with any restriction or physical limit.\n"
        "  - If something conflicts, what the right alternative is.\n\n"
        "Write your reasoning as numbered steps, then end with TWO lines:\n"
        "DECISION: <one of: go_straight | turn_left | turn_right | stop | reroute | "
        "approach_to_read | continue>\n"
        "TARGET: <in 6 words or fewer, the single visible thing or corridor to head "
        "toward to carry out the decision — e.g. 'the left hallway', 'the glass doors "
        "ahead', 'the elevators'. If you are approaching a sign to read it, name the "
        "sign (e.g. 'the directory sign ahead'). If nothing salient is visible, write "
        "'the hallway ahead'. For stop, write 'none'.>\n\n"
        "Format:\n"
        "REASONING:\n1. ...\n2. ...\n3. ...\nDECISION: <action>\nTARGET: <short phrase>"
    )


# map the VLM's decision word to the controller's ActionType (this is just plumbing,
# not decision logic — the VLM already decided)
_DECISION_WORDS = {
    "go_straight": ActionType.FORWARD,
    "turn_left": ActionType.TURN_LEFT,
    "turn_right": ActionType.TURN_RIGHT,
    "stop": ActionType.STOP,
    "reroute": ActionType.REROUTE,
    "approach_to_read": ActionType.FORWARD,
    "continue": ActionType.CONTINUE,
}


class Reasoner:
    def __init__(self, config: Config, vlm: Optional[VLMInterface] = None):
        """vlm: shared VLMInterface (same instance as Reader's); None only when stubbing."""
        self.cfg = config
        self._vlm = vlm

    def set_vlm(self, vlm: VLMInterface):
        self._vlm = vlm

    # ---------- hazard branch ----------
    # Even here we let the VLM reason about WHAT the hazard means, rather than
    # hardcoding "stairs -> stop". We give it the detection and ask it to reason.
    def reason_hazard(self, image, detection: Detection, memory_summary: str = "") -> Decision:
        if self.cfg.stub_reasoner or self._vlm is None:
            # minimal non-VLM fallback ONLY when reasoning model is unavailable
            return Decision(
                action=ActionType.STOP,
                rationale=f"(stub) hazard '{detection.label}' detected; stopping.",
                triggered_by=detection.cls)
        prompt = (
            "You are a wheeled indoor delivery robot. LOOK CAREFULLY AT THIS IMAGE.\n"
            "A rough obstacle detector flagged a possible hazard, but it is frequently "
            "WRONG — it often mistakes handrails, wall trim, doorframes, or printed words "
            "for stairs. Do not trust it. Judge from the image yourself.\n"
            f"(For reference only, the detector guessed: '{detection.label}', "
            f"confidence {detection.confidence:.2f}.)\n\n"
            "Looking at the actual image: are there REAL stairs, steps, or a drop-off that "
            "you (a wheeled robot) physically cannot cross? Or is the path actually clear "
            "(just a hallway, handrail, door, or sign)?\n"
            f"What you've done so far: {memory_summary or '(just started)'}.\n\n"
            "Reason step by step about WHAT YOU ACTUALLY SEE, then decide. End with:\n"
            "DECISION: <go_straight | turn_left | turn_right | stop | reroute | continue>\n"
            "Format:\nREASONING:\n1. (describe what you actually see in the image)\n2. ...\n"
            "DECISION: <action>"
        )
        text = self._vlm.generate(prompt, image)
        return self._parse(text, triggered_by=detection.cls)

    # ---------- sign branch: chain-of-thought reasoning over the read facts ----------
    def reason_sign(self, image, read: ReadResult, goal: str, memory_summary: str = "",
                    scene_alert: bool = False) -> Decision:
        """scene_alert: set True when the monitor flagged a notice/obstacle in the scene.
        It adds a clause that makes the VLM examine the whole frame and treat a closed /
        blocked / relocated goal as a route that does NOT achieve the goal."""
        if self.cfg.stub_reasoner or self._vlm is None:
            return Decision(
                action=ActionType.FORWARD,
                rationale=f"(stub) read {read.structured}; would reason here.",
                triggered_by=ObjectClass.SIGN, read=read)
        prompt = build_reasoning_prompt(goal, read.structured, memory_summary,
                                        scene_alert=scene_alert)
        text = self._vlm.generate(prompt, image)      # VLM reasons step by step
        decision = self._parse(text, triggered_by=ObjectClass.SIGN)
        decision.read = read
        return decision

    # ---------- arrival check: have we reached the goal? ----------
    def check_arrival(self, image, read_or_none: Optional[ReadResult], goal: str) -> bool:
        """Return True only when the scene shows clear evidence of having reached the goal.

        Conservative by design: ambiguous or unparseable responses → False.
        Not called anywhere yet; the JourneyLoop will call this in Step 4.
        """
        if self.cfg.stub_reasoner or self._vlm is None:
            return False   # conservative: don't declare arrival without real reasoning

        sign_context = (
            f"LATEST SIGN READ (for context): {json.dumps(read_or_none.structured)}\n"
            if read_or_none and read_or_none.structured else ""
        )
        prompt = (
            f"You are a wheeled indoor delivery robot. Your goal is: \"{goal}\".\n\n"
            "CRITICAL DISTINCTION — read this carefully before answering:\n"
            "  NOT ARRIVED: a directional sign saying '{goal} → right' or 'this way to {goal}' "
            "means the goal is AHEAD in that direction. You are still navigating toward it.\n"
            "  ARRIVED: a door plate, room number plate, or entrance sign mounted directly "
            "ON or immediately beside the destination itself — the kind placed on the actual "
            "room or office door. You can see the goal's own entrance, not a pointer to it.\n\n"
            f"{sign_context}"
            f"Look at this scene. Is there clear evidence that \"{goal}\" is HERE — "
            "specifically, can you see a door plate, room marker, or entrance that directly "
            "identifies this location as the goal?\n\n"
            "Reason briefly (1-3 sentences), then end with exactly one of:\n"
            "ARRIVED: yes\n"
            "ARRIVED: no\n\n"
            "Answer 'yes' ONLY if you can clearly see the goal's own door/plate/marker "
            "immediately in front of you. If the scene shows a directory sign pointing "
            "toward the goal, or if you are uncertain, answer 'no'."
        )
        text = self._vlm.generate(prompt, image)
        return self._parse_arrival(text)

    @staticmethod
    def _parse_arrival(text: str) -> bool:
        """Extract the ARRIVED line; conservative — only True on explicit 'yes'."""
        for line in reversed(text.splitlines()):   # scan from the end (answer is last)
            stripped = line.strip().upper()
            if stripped.startswith("ARRIVED:"):
                answer = stripped.split(":", 1)[-1].strip()
                return answer == "YES"
        return False   # no parseable line found → not arrived

    def _parse(self, text: str, triggered_by: ObjectClass) -> Decision:
        """Pull the DECISION line out; keep the full chain-of-thought as the rationale.
        Also scan for an optional TARGET line -> nav_target (sign branch). Conservative:
        missing/unusable TARGET leaves nav_target='' and never blocks the decision.
        Hazard responses have no TARGET line, so nav_target stays '' there (correct)."""
        action = ActionType.CONTINUE
        for line in text.splitlines():
            if line.strip().upper().startswith("DECISION"):
                word = line.split(":", 1)[-1].strip().lower().replace(" ", "_")
                action = _DECISION_WORDS.get(word, ActionType.CONTINUE)
                break
        nav_target = ""
        for line in text.splitlines():
            if line.strip().upper().startswith("TARGET"):
                raw = line.split(":", 1)[-1].strip()
                if raw and raw.lower() not in ("none", "n/a", "-", ""):
                    raw = re.sub(r'^\s*move toward\s+', '', raw, flags=re.I).strip().strip('."\'')
                    nav_target = " ".join(raw.split()[:6])   # clamp to 6 words
                break
        return Decision(action=action, rationale=text.strip(),
                        triggered_by=triggered_by, nav_target=nav_target)