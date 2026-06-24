#!/usr/bin/env python
"""
Notice / closure challenge test — runs the SAME trip twice in ONE process,
so Qwen2.5-VL + GroundingDINO load only ONCE (model load is the slow part):

  TEST A : cfg.force_scene_alert = True
           The scene-alert clause is ON for every decision, regardless of whether the
           DINO notice channel fires. This isolates one question:
           "Can the VLM read the 'Restroom Closed' notice off the FULL frame and
            choose reroute/stop instead of following the directory sign?"

  TEST B : cfg.force_scene_alert = False
           The clause is gated on the DINO notice trigger (bundle.notice_dets). This
           tests the END-TO-END path: does GroundingDINO actually fire on the yellow
           floor A-frame, and does that flip the decision on its own?

WHAT TO LOOK FOR in the .log:
  TEST A, at the leg-0 decision frame:
     - the REASONING should mention the restroom is closed / not usable, and
     - the DECISION line should be `reroute` (or `stop`), NOT `turn_left`.
  TEST A control (run separately on a CLEAN trip, no notice):
     - it should still decide normally — i.e. it does NOT invent a closure.
  TEST B:
     - `[GDINO/notices] N detection(s)` on the A-frame frames, and
     - `SCENE ALERT: N notice/obstacle region(s) detected`.
     If those lines never appear, the notice prompt did not fire (sweep a shorter
     prompt: "safety sign . floor sign").

Usage:
    python run_notice_test.py <TRIP_DIR> [goal]
Run it from  .../signnav_scripts/experiments  so that `signnav_reasoner` imports.
TRIP_DIR is the extracted trip directory (contains frames/, odom.csv, frame_index.csv).
"""

import sys

from signnav_reasoner.types import Config
from signnav_reasoner.loop import AdaptiveReasoningLoop
from signnav_reasoner.journey import JourneyLoop


def main():
    if len(sys.argv) < 2:
        print("usage: python run_notice_test.py <TRIP_DIR> [goal]")
        sys.exit(1)
    trip = sys.argv[1]
    goal = sys.argv[2] if len(sys.argv) > 2 else "Restrooms"

    cfg = Config(goal=goal)
    inner = AdaptiveReasoningLoop(cfg)          # loads Qwen2.5-VL + GroundingDINO ONCE

    runs = [
        ("TEST A  —  force_scene_alert = ON   (isolates: can the VLM handle the closure?)", True),
        ("TEST B  —  force_scene_alert = OFF  (gated on the DINO notice trigger; end-to-end)", False),
    ]
    for label, force in runs:
        print("\n\n" + "#" * 78)
        print(f"#  {label}")
        print(f"#  goal = {goal!r}    trip = {trip}")
        print("#" * 78 + "\n")
        cfg.force_scene_alert = force
        # reset transient per-run state on the shared inner loop so run B starts clean
        inner._approach_steps = 0
        inner._read_cooldown = 0
        inner._memory = ""
        JourneyLoop(cfg, inner).run_on_trip(trip)

    print("\n\n=== both tests complete ===")


if __name__ == "__main__":
    main()