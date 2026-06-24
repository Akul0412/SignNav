#!/usr/bin/env python
"""
Mask-the-sign experiment: does blanking the directory panel BEFORE the notice
detector remove the directory false-positives WITHOUT losing the real floor A-frame?

Runs the SAME trip twice in ONE process (Qwen + GroundingDINO load once),
force_scene_alert OFF both times so the notice channel itself is what gates reasoning:

  RUN 1 : mask_signs_for_notices = False   (baseline — current behavior)
  RUN 2 : mask_signs_for_notices = True    (blank sign panels before the notice pass)

WHAT TO COMPARE in the log / HTML player:
  - [GDINO/notices] per frame: with masking ON, the HIGH boxes (the wall directory,
    roughly y < 650) should disappear, while LOW boxes near the floor (the 'no entry'
    A-frame, roughly y > 900) should survive. That is the win — the gate stops firing
    on the directory we're already reading, but still catches the real hazard.
  - Frame-fire rate: count frames whose notice line is non-empty. Masking ON should
    drop it well below the ~every-frame baseline.
  - End-to-end recall: does SCENE ALERT still fire and the decision still become
    reroute / stop with masking ON? If yes, masking didn't cost us the catch.

Quick counts from a finished log:
  grep -c "GDINO/notices] [1-9]" run1_half      # frames where the gate fired

Usage:  python run_mask_test.py <TRIP_DIR> [goal]   (run from .../experiments)
"""
import sys

from signnav_reasoner.types import Config
from signnav_reasoner.loop import AdaptiveReasoningLoop
from signnav_reasoner.journey import JourneyLoop


def main():
    if len(sys.argv) < 2:
        print("usage: python run_mask_test.py <TRIP_DIR> [goal]")
        sys.exit(1)
    trip = sys.argv[1]
    goal = sys.argv[2] if len(sys.argv) > 2 else "Restrooms"

    cfg = Config(goal=goal)
    cfg.force_scene_alert = False                 # the notice channel gates reasoning
    inner = AdaptiveReasoningLoop(cfg)            # loads Qwen + GroundingDINO ONCE

    for label, mask in [
        ("RUN 1  —  mask_signs_for_notices = OFF  (baseline)", False),
        ("RUN 2  —  mask_signs_for_notices = ON   (blank sign panels before notice pass)", True),
    ]:
        print("\n\n" + "#" * 78)
        print(f"#  {label}")
        print(f"#  goal = {goal!r}    trip = {trip}")
        print("#" * 78 + "\n")
        cfg.mask_signs_for_notices = mask
        inner._approach_steps = 0
        inner._read_cooldown = 0
        inner._memory = ""
        JourneyLoop(cfg, inner).run_on_trip(trip)

    print("\n\n=== mask experiment complete ===")


if __name__ == "__main__":
    main()