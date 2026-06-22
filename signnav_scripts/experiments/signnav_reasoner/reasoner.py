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
"""

import json
from typing import Optional

from .types import ActionType, Config, Decision, Detection, ObjectClass, ReadResult


# The chain-of-thought reasoning prompt. The model thinks step by step, then commits.
# Note: NO hardcoded prohibitions — the model reasons about what it sees.
def build_reasoning_prompt(goal: str, sign_facts: dict, memory_summary: str) -> str:
    return (
        "You are the reasoning module of an indoor wheeled delivery robot. You reason "
        "step by step about what to do next, like a careful driver.\n\n"
        f"YOUR GOAL: reach \"{goal}\".\n"
        "YOUR CONSTRAINTS: you are a wheeled robot (you CANNOT use stairs). You are a "
        "delivery robot, not staff (you must respect access restrictions). Safety first.\n\n"
        f"WHAT YOU JUST READ ON THE SIGN(S): {json.dumps(sign_facts)}\n"
        f"WHAT YOU'VE DONE SO FAR: {memory_summary or '(just started)'}\n\n"
        "Reason through this ONE STEP AT A TIME. Think about:\n"
        "  - Which entries on the sign are relevant to your goal, and what direction they imply.\n"
        "  - Whether any entry is a RESTRICTION or WARNING that affects you (think about what "
        "each sign actually means for a delivery robot — do not assume, reason about it).\n"
        "  - Whether your goal's direction conflicts with any restriction or physical limit.\n"
        "  - If something conflicts, what the right alternative is.\n\n"
        "Write your reasoning as numbered steps, then end with a single line:\n"
        "DECISION: <one of: go_straight | turn_left | turn_right | stop | reroute | "
        "approach_to_read | continue>\n\n"
        "Format:\n"
        "REASONING:\n1. ...\n2. ...\n3. ...\nDECISION: <action>"
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
    def __init__(self, config: Config, vlm=None):
        """vlm: a callable (prompt, image)->text, or None to lazy-share the Reader's model.
        In the loop we pass the Reader's loaded VLM so we don't load a second model."""
        self.cfg = config
        self._vlm = vlm

    def set_vlm(self, vlm):
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
        text = self._vlm(prompt, image)
        return self._parse(text, triggered_by=detection.cls)

    # ---------- sign branch: chain-of-thought reasoning over the read facts ----------
    def reason_sign(self, image, read: ReadResult, goal: str, memory_summary: str = "") -> Decision:
        if self.cfg.stub_reasoner or self._vlm is None:
            return Decision(
                action=ActionType.FORWARD,
                rationale=f"(stub) read {read.structured}; would reason here.",
                triggered_by=ObjectClass.SIGN, read=read)
        prompt = build_reasoning_prompt(goal, read.structured, memory_summary)
        text = self._vlm(prompt, image)              # VLM reasons step by step
        decision = self._parse(text, triggered_by=ObjectClass.SIGN)
        decision.read = read
        return decision

    def _parse(self, text: str, triggered_by: ObjectClass) -> Decision:
        """Pull the DECISION line out; keep the full chain-of-thought as the rationale."""
        action = ActionType.CONTINUE
        for line in text.splitlines():
            if line.strip().upper().startswith("DECISION"):
                word = line.split(":", 1)[-1].strip().lower().replace(" ", "_")
                action = _DECISION_WORDS.get(word, ActionType.CONTINUE)
                break
        return Decision(action=action, rationale=text.strip(), triggered_by=triggered_by)