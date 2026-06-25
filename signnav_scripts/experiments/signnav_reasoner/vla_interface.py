"""
VLAInterface — turn a committed Decision into an OmniVLA prompt and record it.

This is the SEAM where OmniVLA inference will later plug in. For now it only
generates the prompt (via the pure vla_bridge formatter) and appends a record to a
JSONL sidecar — it does NOT run OmniVLA. The sidecar is SEPARATE from the .log:
logging here never touches the dbg logger or the .log format, and never crashes the
loop (all writes are guarded).
"""

import json
from typing import Optional

from .types import Config, Decision
from .vla_bridge import decision_to_omni_lang, wrap_for_omnivla


class VLAInterface:
    """Committed Decision -> OmniVLA prompt, logged to a JSONL sidecar.
    Does NOT run OmniVLA yet; the inference call is the seam in process()."""

    def __init__(self, config: Config, sidecar_path: Optional[str] = None):
        self.cfg = config
        self.path = sidecar_path

    def process(self, decision: Decision, frame_idx: int, goal: str) -> Optional[str]:
        lang = decision_to_omni_lang(decision)              # "move toward ..." | None (STOP) | sentinel
        wrapped = wrap_for_omnivla(lang) if lang else None
        self._log(frame_idx, goal, decision, lang, wrapped)
        # --- future seam: send `wrapped` to OmniVLA and return its action. Not wired yet. ---
        return wrapped

    def _log(self, frame_idx, goal, decision, lang, wrapped):
        if not self.path:
            return
        try:
            rec = {"frame": frame_idx, "goal": goal,
                   "action": decision.action.value,
                   "nav_target": getattr(decision, "nav_target", "") or "",
                   "omni_lang": lang, "wrapped": wrapped,
                   "rationale": decision.rationale}
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:
            print(f"[vla] sidecar log skipped: {e}")
