"""
VLA bridge — turn a SignNav `Decision` into OmniVLA's language instruction (spec v1).

LANGUAGE channel only (§1-§4). The pose/waypoint channel is parked pending team
alignment and has been removed.

A deterministic, model-free formatter maps a Decision to OmniVLA's `lang` slot,
always of the form  "move toward {target}".  The call site wraps it as
"What action should the robot take to {lang}?" (lowercased), matching OmniVLA's
verified training template.  `stop` → no prompt (halting is the orchestrator's job).

Everything here is PURE and model-free (no torch, no VLM) → unit-testable offline.
The literal OmniVLA inference call is NOT here (it lives in the forked OmniVLA repo);
this module only produces the prompt a VLAInterface would send (Step E).
"""

from typing import Optional

from .types import ActionType, Decision


# ─────────────────────────────────────────────────────────────────────────────
# Constants / sentinels
# ─────────────────────────────────────────────────────────────────────────────

# OmniVLA's verified training/inference wrapper (prismatic/models/vlas/openvla.py).
# Capital "W" per §1/§4; only the inner instruction is lowercased. (The §6 example
# shows it fully lowercase — cosmetic; we follow the §4 formula.)
OMNIVLA_WRAPPER = "What action should the robot take to {lang}?"

# Pose/image-goal mode: the exact no-language sentinel OmniVLA expects (§1, §4 rule 7).
NO_LANGUAGE = "No language instruction"

# Per-action default visual target when nav_target is empty (spec §4 rule 4), keyed
# on ActionType ONLY. NOTE: go_straight AND approach_to_read both arrive here as
# ActionType.FORWARD (the reasoner-word→ActionType merge in reasoner._DECISION_WORDS),
# so FORWARD's default is the go_straight phrase; the "directory sign ahead" phrasing
# for an approach is realized by the reasoner setting nav_target explicitly.
_ACTION_DEFAULT_TARGET = {
    ActionType.FORWARD:    "the hallway ahead",
    ActionType.TURN_LEFT:  "the hallway on the left",
    ActionType.TURN_RIGHT: "the hallway on the right",
    ActionType.REROUTE:    "the alternative route",
    ActionType.CONTINUE:   "the hallway ahead",
}

# Length-reducer vocabulary (safety net only — the reasoner is meant to emit ≤6 words).
_HEAD_NOUNS = ["intersection", "corridor", "hallway", "elevators", "elevator",
               "stairs", "doors", "door", "exit", "lobby", "sign", "room", "ramp"]
_DESCRIPTOR_PHRASES = ["on the left", "on the right", "to the left", "to the right",
                       "straight ahead", "ahead"]
_DESCRIPTOR_WORDS = ["left", "right", "ahead"]

_MAX_TARGET_WORDS = 6   # spec §4 rule 2


# ─────────────────────────────────────────────────────────────────────────────
# Language channel (§1-§4)
# ─────────────────────────────────────────────────────────────────────────────

def _simplify_target(target: str) -> str:
    """Reduce an over-long target to head-noun + one descriptor (spec §4 rule 2).

    Best-effort safety net: the reasoner is the primary length controller. If the
    target already fits, it is returned unchanged.
    """
    words = target.split()
    if len(words) <= _MAX_TARGET_WORDS:
        return target

    lower = target.lower()
    # first head noun occurring in the phrase (scan the phrase, not the vocab order)
    noun = next((w for w in (x.strip(".,") for x in lower.split())
                 if w in _HEAD_NOUNS), None)
    if noun is None:
        return " ".join(words[:_MAX_TARGET_WORDS])   # no known noun → plain truncate

    # first descriptor (multi-word phrase preferred over a bare word)
    descriptor = next((p for p in _DESCRIPTOR_PHRASES if p in lower), None)
    if descriptor is None:
        descriptor = next((w for w in _DESCRIPTOR_WORDS if w in lower.split()), None)

    parts = ["the", noun] + ([descriptor] if descriptor else [])
    return " ".join(parts)


def _clean_target(raw: str) -> str:
    """Normalize a nav_target: strip whitespace and any accidental leading verb the
    reasoner may have prepended ('move toward the doors' → 'the doors')."""
    t = (raw or "").strip()
    low = t.lower()
    for lead in ("move toward ", "go to ", "head toward ", "head to ", "toward ", "to "):
        if low.startswith(lead):
            t = t[len(lead):].strip()
            break
    return t


def decision_to_omni_lang(decision: Decision, language_enabled: bool = True) -> Optional[str]:
    """Map a Decision to OmniVLA's inner `lang` phrase (spec §4). Keyed on action only.

    Returns:
      - "move toward {target}"  for a driving action,
      - None                    for STOP (do NOT drive OmniVLA; orchestrator halts),
      - NO_LANGUAGE             when language is deliberately disabled (pose-only mode).
    The caller wraps a non-None, non-sentinel result via wrap_for_omnivla().
    """
    if decision.action == ActionType.STOP:
        return None   # STOP sentinel — never a prompt (§4 rule 4)

    if not language_enabled:
        return NO_LANGUAGE   # §4 rule 7 — rely on pose/image goal

    target = _clean_target(decision.nav_target)
    if not target:
        target = _ACTION_DEFAULT_TARGET.get(decision.action, "the hallway ahead")  # §4 rule 6
    target = _simplify_target(target)
    return f"move toward {target}"


def wrap_for_omnivla(omni_lang: str) -> str:
    """Wrap the inner phrase in OmniVLA's exact template, lowercased (§1/§4)."""
    return OMNIVLA_WRAPPER.format(lang=omni_lang.lower())
