#!/usr/bin/env python
"""
Decouple + position-gate test — ONE run on the warning-sign trip, normal settings.

What changed in loop.py (the thing under test):
  * POSITION-GATE (_path_hazard): a notice/obstacle detection counts as an in-PATH
    floor hazard only if its box centre sits below cfg.path_hazard_y_frac (=0.55) of
    the frame height. Wall directories / elevator panels sit high -> ignored; the
    floor A-frame sits low -> counts.
  * DECOUPLE: when such a low hazard is in view AND the directory read is NOT yet
    confident, the reasoner is run on the WHOLE frame (scene_alert) right then,
    instead of waiting for the directory read to firm up. It only COMMITS early if
    the reasoner concludes the goal is blocked (stop/reroute). Any other conclusion
    (turn / go-straight / approach) is ignored, so the confident-read path stays the
    only place a direction is committed — no risk of an early wrong turn.

This run uses default config:
    force_scene_alert = OFF   (we want the path-hazard trigger, not a forced alert)
    mask_signs_for_notices = OFF   (masking hurt; it's off)
    path_hazard_y_frac = 0.55  (tunable via getattr; raise it if a wall sign ever trips)

WHAT TO LOOK FOR in the .log:
  * `PATH-HAZARD: low in-frame notice/obstacle — reasoning now (decoupled ...)` lines
    starting EARLY (the floor caution box is in view from frame 1), each followed by a
    REASONING trace.
  * an EARLY `reroute` / `stop` commit — well before the frame-85 commit the un-forced
    baseline produced — once the VLM can read "Restroom Closed" off the full frame.
  * if instead it keeps approaching (reasoner said turn/go), that's the SAFE fallback:
    it did NOT turn into the closure, it just didn't catch it early on that frame.
  * cost: each early PATH-HAZARD frame adds one ~3-4s reason on top of the read; the
    read cooldown bounds it to roughly one reason per ~5 sign-frames.

Usage (run from .../signnav_scripts/experiments so signnav_reasoner imports):
    python run_decouple_test.py <TRIP_DIR> [goal]
TRIP_DIR is the extracted trip directory (frames/, odom.csv, frame_index.csv).
"""

import sys

from signnav_reasoner.types import Config
from signnav_reasoner.loop import AdaptiveReasoningLoop
from signnav_reasoner.journey import JourneyLoop


def main():
    if len(sys.argv) < 2:
        print("usage: python run_decouple_test.py <TRIP_DIR> [goal]")
        sys.exit(1)
    trip = sys.argv[1]
    goal = sys.argv[2] if len(sys.argv) > 2 else "Restrooms"

    cfg = Config(goal=goal)
    # defaults already give us what we want; set explicitly so the run is unambiguous
    cfg.force_scene_alert = False
    # path_hazard_y_frac is read via getattr (default 0.55). Uncomment to tune:
    # cfg.path_hazard_y_frac = 0.55

    print("\n" + "#" * 78)
    print("#  DECOUPLE + POSITION-GATE test  (single run, normal settings)")
    print(f"#  goal = {goal!r}    trip = {trip}")
    print(f"#  force_scene_alert = {cfg.force_scene_alert}   "
          f"path_hazard_y_frac = {getattr(cfg, 'path_hazard_y_frac', 0.55)}")
    print("#" * 78 + "\n")

    inner = AdaptiveReasoningLoop(cfg)          # loads Qwen2.5-VL + GroundingDINO once
    JourneyLoop(cfg, inner).run_on_trip(trip)

    print("\n\n=== decouple test complete ===")


if __name__ == "__main__":
    main()
