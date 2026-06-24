"""
Odom utilities for the journey loop.

Three self-contained functions that load and query recorded odometry data.
They are independent of Config and have no side effects, so they can be
unit-tested directly on an odom.csv file.

Expected CSV columns: timestamp_ns, x, y, yaw
All timestamps are nanoseconds (int). Yaw is in radians.
"""

import csv
import math
from typing import List, Optional, Tuple

OdomRow = Tuple[int, float, float, float]   # (timestamp_ns, x, y, yaw)


def load_odom(odom_csv_path: str) -> List[OdomRow]:
    """Load odom.csv and return a list of (timestamp_ns, x, y, yaw) sorted by time."""
    rows: List[OdomRow] = []
    with open(odom_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append((
                int(row["timestamp_ns"]),
                float(row["x"]),
                float(row["y"]),
                float(row["yaw"]),
            ))
    rows.sort(key=lambda r: r[0])
    return rows


def _right_bracket(odom: List[OdomRow], t_ns: int) -> int:
    """Return index i such that odom[i-1][0] <= t_ns < odom[i][0].

    Equivalent to bisect_right on the timestamp column, but avoids building
    a temporary list.  Requires odom to be non-empty and t_ns strictly inside
    the odom time range (caller must check bounds first).
    """
    lo, hi = 0, len(odom)
    while lo < hi:
        mid = (lo + hi) // 2
        if odom[mid][0] <= t_ns:
            lo = mid + 1
        else:
            hi = mid
    return lo


def pose_at(odom: List[OdomRow], t_ns: int) -> Optional[Tuple[float, float, float]]:
    """Return interpolated (x, y, yaw) at t_ns, or None if out of range.

    Linearly interpolates x and y; interpolates yaw via the shortest angular
    path to handle wrap-around correctly.
    """
    if not odom:
        return None
    if t_ns < odom[0][0] or t_ns > odom[-1][0]:
        return None

    # exact endpoints
    if t_ns == odom[0][0]:
        return odom[0][1], odom[0][2], odom[0][3]
    if t_ns == odom[-1][0]:
        return odom[-1][1], odom[-1][2], odom[-1][3]

    i = _right_bracket(odom, t_ns)   # odom[i-1][0] <= t_ns < odom[i][0]
    t0, x0, y0, yaw0 = odom[i - 1]
    t1, x1, y1, yaw1 = odom[i]

    a = (t_ns - t0) / (t1 - t0)
    x = x0 + a * (x1 - x0)
    y = y0 + a * (y1 - y0)
    # shortest-path yaw interpolation — handles the ±π wrap correctly
    dyaw = math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0))
    yaw = yaw0 + a * dyaw

    return x, y, yaw


def displacement_between(odom: List[OdomRow], t_a_ns: int, t_b_ns: int) -> float:
    """Straight-line distance (metres) between interpolated poses at t_a_ns and t_b_ns.

    Returns 0.0 if either timestamp is out of range.  Used by the completion
    detector to measure recent-window progress without accumulating path-integral
    noise from incremental steps.
    """
    pa = pose_at(odom, t_a_ns)
    pb = pose_at(odom, t_b_ns)
    if pa is None or pb is None:
        return 0.0
    return math.sqrt((pb[0] - pa[0]) ** 2 + (pb[1] - pa[1]) ** 2)


def yaw_rate_at(odom: List[OdomRow], t_ns: int, dt_ns: int) -> float:
    """Estimate angular velocity (rad/s) at t_ns via backward finite difference.

    Queries pose_at at t_ns and t_ns - dt_ns and returns the shortest-path
    yaw difference divided by dt.  Returns 0.0 if either pose is unavailable.
    """
    p1 = pose_at(odom, t_ns)
    p0 = pose_at(odom, t_ns - dt_ns)
    if p1 is None or p0 is None:
        return 0.0
    dyaw = math.atan2(math.sin(p1[2] - p0[2]), math.cos(p1[2] - p0[2]))
    dt_s = dt_ns / 1e9
    if dt_s <= 0:
        return 0.0
    return dyaw / dt_s
