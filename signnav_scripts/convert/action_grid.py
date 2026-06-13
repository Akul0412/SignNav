#!/usr/bin/env python3
"""
action_grid.py  —  Build the 8x4 action grid for a frame, from odom.

THE CORE OF THE SIGNNAV DATASET. For a frame at time T, the action grid is the
robot's next 8 waypoints expressed IN THE ROBOT'S OWN FRAME at time T:
    each row = (dx_forward, dy_left, cos(dyaw), sin(dyaw))

Mirrors OmniVLA's action convention so the grid is drop-in compatible with its
training loop (NUM_ACTIONS_CHUNK=8, ACTION_DIM=4). Normalization (BOUNDS_Q99) is
applied by OmniVLA at train time — here we output real metric values.

Design decisions (deliberate, see SignNav schema spec):
  - Waypoints are spaced over a MEANINGFUL HORIZON, not consecutive 50Hz samples
    (8 consecutive samples = ~0.16s = near-zero motion = useless). We space them
    so the chunk covers ~HORIZON_SECONDS of future motion.
  - END-OF-TRIP EDGE CASE: a frame is only a valid example if it has a full
    horizon of future poses. Frames too close to the trip end are DROPPED
    (no fabricated labels). build_grid returns None for those.

Usage (standalone test / verification on a real trip):
    python action_grid.py datasets/extracted/keller_22
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

# --- OmniVLA-matched constants ---
NUM_ACTIONS_CHUNK = 8        # 8 future waypoints (OmniVLA NUM_ACTIONS_CHUNK)
ACTION_DIM = 4               # (dx_forward, dy_left, cos_dyaw, sin_dyaw)
HORIZON_SECONDS = 2.0        # the 8 waypoints span ~2s of future motion
# (tune HORIZON_SECONDS to match the lookahead OmniVLA expects for your robot speed)


def load_odom(odom_csv: Path):
    """Load odom.csv -> sorted lists of (t_ns, x, y, yaw)."""
    rows = []
    with open(odom_csv) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["timestamp_ns"]),
                         float(r["x"]), float(r["y"]), float(r["yaw"])))
    rows.sort(key=lambda r: r[0])
    return rows


def pose_at(odom, t_ns):
    """Linearly interpolate pose (x, y, yaw) at time t_ns. None if out of range."""
    if t_ns < odom[0][0] or t_ns > odom[-1][0]:
        return None
    # binary search for the bracketing samples
    lo, hi = 0, len(odom) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if odom[mid][0] <= t_ns:
            lo = mid
        else:
            hi = mid
    t0, x0, y0, yaw0 = odom[lo]
    t1, x1, y1, yaw1 = odom[hi]
    if t1 == t0:
        return x0, y0, yaw0
    a = (t_ns - t0) / (t1 - t0)
    # interpolate yaw via shortest angular path
    dyaw = math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0))
    return (x0 + a * (x1 - x0),
            y0 + a * (y1 - y0),
            yaw0 + a * dyaw)


def build_grid(odom, t_ns):
    """
    Build the 8x4 action grid for the frame at t_ns.
    Returns np.ndarray (8,4) of (dx_fwd, dy_left, cos_dyaw, sin_dyaw) in robot frame,
    or None if there isn't a full HORIZON of future poses (end-of-trip frame).
    """
    cur = pose_at(odom, t_ns)
    if cur is None:
        return None
    x0, y0, yaw0 = cur

    horizon_ns = int(HORIZON_SECONDS * 1e9)
    step_ns = horizon_ns // NUM_ACTIONS_CHUNK

    # EDGE CASE: need the full horizon of future poses, else drop this frame.
    if t_ns + horizon_ns > odom[-1][0]:
        return None

    grid = np.zeros((NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)
    for i in range(NUM_ACTIONS_CHUNK):
        tf = t_ns + (i + 1) * step_ns
        fut = pose_at(odom, tf)
        if fut is None:
            return None
        xf, yf, yawf = fut
        # world-frame delta
        dx_w = xf - x0
        dy_w = yf - y0
        # rotate into robot frame at T (heading yaw0): forward = +x along heading
        cos0, sin0 = math.cos(-yaw0), math.sin(-yaw0)
        dx_fwd = cos0 * dx_w - sin0 * dy_w
        dy_left = sin0 * dx_w + cos0 * dy_w
        # heading change, as cos/sin (shortest path)
        dyaw = math.atan2(math.sin(yawf - yaw0), math.cos(yawf - yaw0))
        grid[i] = [dx_fwd, dy_left, math.cos(dyaw), math.sin(dyaw)]
    return grid


def describe_grid(grid):
    """Human-readable summary so you can verify a grid is physically correct."""
    if grid is None:
        return "  (dropped: insufficient future horizon — end-of-trip frame)"
    # net motion at the final waypoint
    fwd = grid[-1, 0]
    lat = grid[-1, 1]
    final_dyaw_deg = math.degrees(math.atan2(grid[-1, 3], grid[-1, 2]))
    turn = "straight"
    if final_dyaw_deg > 8:
        turn = f"turning LEFT {final_dyaw_deg:.0f}deg"
    elif final_dyaw_deg < -8:
        turn = f"turning RIGHT {abs(final_dyaw_deg):.0f}deg"
    total = math.hypot(fwd, lat)
    flag = "  <-- near-zero motion (robot ~stationary here)" if total < 0.1 else ""
    return (f"  net over horizon: forward {fwd:+.2f}m, lateral {lat:+.2f}m, "
            f"{turn}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trip_dir", help="Extracted trip dir (has odom.csv, frame_index.csv)")
    ap.add_argument("--n", type=int, default=10, help="How many sample frames to describe")
    args = ap.parse_args()

    trip = Path(args.trip_dir)
    odom = load_odom(trip / "odom.csv")
    print(f"Loaded {len(odom)} odom poses "
          f"({(odom[-1][0]-odom[0][0])/1e9:.1f}s, "
          f"horizon={HORIZON_SECONDS}s, {NUM_ACTIONS_CHUNK} steps)\n")

    # load frame timestamps
    fi = trip / "frame_index.csv"
    if fi.exists():
        with open(fi) as f:
            frame_ts = [int(r["timestamp_ns"]) for r in csv.DictReader(f)]
    else:
        # fall back to frame filenames
        frame_ts = sorted(int(p.stem) for p in (trip / "frames").glob("*.jpg"))
    print(f"{len(frame_ts)} frames total.\n")

    # describe N frames spread across the trip
    valid = dropped = 0
    idxs = np.linspace(0, len(frame_ts) - 1, args.n).astype(int)
    for k, idx in enumerate(idxs):
        t = frame_ts[idx]
        g = build_grid(odom, t)
        if g is None:
            dropped += 1
        else:
            valid += 1
        print(f"frame {idx:4d} (t={t}):")
        print(describe_grid(g))
        if g is not None and k == 0:
            print("  full 8x4 grid (fwd, left, cos, sin):")
            for row in g:
                print(f"    [{row[0]:+.3f} {row[1]:+.3f} {row[2]:+.3f} {row[3]:+.3f}]")
    # full sweep counts
    all_valid = sum(1 for t in frame_ts if build_grid(odom, t) is not None)
    print(f"\nAcross ALL {len(frame_ts)} frames: "
          f"{all_valid} valid grids, {len(frame_ts)-all_valid} dropped "
          f"(end-of-trip frames without full horizon).")


if __name__ == "__main__":
    main()