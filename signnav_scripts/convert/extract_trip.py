#!/usr/bin/env python3
"""
extract_trip.py  —  STAGE 1 of the SignNav converter.

Reads ONE ROS2 bag (one trip) and writes:
    <out>/frames/<timestamp_ns>.jpg     each camera frame, named by its timestamp
    <out>/odom.csv                      every odom pose: timestamp_ns, x, y, yaw
    <out>/frame_index.csv               timestamp_ns -> frame filename (for fast lookup)

Why timestamps matter: in Stage 2 we build the 8x4 action grid by asking
"for the frame at time T, where did the robot go over the next 8 poses?"
That lookup needs frames and odom on the SAME clock, so we name frames by
their nanosecond timestamp and store odom timestamps alongside.

No ROS install needed (uses the `rosbags` library).
    pip install rosbags opencv-python numpy

Usage:
    python extract_trip.py /path/to/rosbag2_keller_20.1 --out ./extracted/keller_20
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import cv2
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

IMAGE_TOPIC = "/c1/image_raw"   # raw sensor_msgs/Image (no compressed topic recorded)
ODOM_TOPIC = "/odom"

# ROS2 distro typestore — bags lack embedded type defs, so we supply standard ones.
# Swap to ROS2_JAZZY / ROS2_IRON / ROS2_FOXY if your robot used a different distro.
TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)


def quat_to_yaw(x, y, z, w):
    """Quaternion -> yaw (heading about the vertical axis)."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def decode_raw_image(msg):
    """
    Decode a raw sensor_msgs/Image into a BGR uint8 array (ready for cv2.imwrite).

    Handles the encodings a typical robot camera publishes. If your camera uses a
    different encoding, print(msg.encoding) once and add a branch here.
    """
    h, w = msg.height, msg.width
    enc = msg.encoding.lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(h, w, 3)
        if enc == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    if enc in ("yuv422", "yuv422_yuy2", "yuyv"):
        # packed YUV 4:2:2, 2 bytes/pixel.
        # This camera's "yuv422" is UYVY byte order (YUYV gave a green/magenta tint).
        # If a future camera looks wrong, try the other COLOR_YUV2BGR_* 422 codes.
        yuv = buf.reshape(h, w, 2)
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_UYVY)
    if enc == "uyvy":
        yuv = buf.reshape(h, w, 2)
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_UYVY)
    if enc == "mono8":
        return cv2.cvtColor(buf.reshape(h, w), cv2.COLOR_GRAY2BGR)
    if enc.startswith("bayer"):
        # common debayer; bayer_rggb8 is the most frequent. Adjust if colors look wrong.
        return cv2.cvtColor(buf.reshape(h, w), cv2.COLOR_BAYER_RG2BGR)
    if enc in ("rgba8", "bgra8"):
        img = buf.reshape(h, w, 4)
        code = cv2.COLOR_RGBA2BGR if enc == "rgba8" else cv2.COLOR_BGRA2BGR
        return cv2.cvtColor(img, code)

    print(f"  !! Unhandled image encoding '{msg.encoding}' "
          f"({w}x{h}) — add a branch in decode_raw_image(). Skipping frame.")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", help="Path to ONE ROS2 bag folder (one trip)")
    ap.add_argument("--out", required=True, help="Output directory for this trip")
    ap.add_argument("--image-topic", default=IMAGE_TOPIC)
    ap.add_argument("--odom-topic", default=ODOM_TOPIC)
    args = ap.parse_args()

    bag_path = Path(args.bag)
    out_dir = Path(args.out)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    odom_rows = []          # timestamp_ns, x, y, yaw
    frame_index = []        # timestamp_ns, filename
    n_frames = 0

    with AnyReader([bag_path], default_typestore=TYPESTORE) as reader:
        print(f"Opened: {bag_path}")
        print("Topics present:")
        for c in reader.connections:
            print(f"  {c.topic:42s} {c.msgtype:32s} ({c.msgcount} msgs)")
        print()

        wanted = [c for c in reader.connections
                  if c.topic in (args.image_topic, args.odom_topic)]
        if not any(c.topic == args.odom_topic for c in wanted):
            print(f"!! WARNING: odom topic '{args.odom_topic}' not found — "
                  f"action labels cannot be built from this bag.")
        if not any(c.topic == args.image_topic for c in wanted):
            print(f"!! WARNING: image topic '{args.image_topic}' not found.")

        for conn, timestamp, rawdata in reader.messages(connections=wanted):
            msg = reader.deserialize(rawdata, conn.msgtype)

            if conn.topic == args.image_topic:
                # raw sensor_msgs/Image: msg.data is uncompressed pixel bytes.
                # Reshape by height/width/channels, then convert to BGR for cv2.imwrite.
                img = decode_raw_image(msg)
                if img is not None:
                    fname = f"{timestamp}.jpg"
                    cv2.imwrite(str(frames_dir / fname), img)
                    frame_index.append({"timestamp_ns": timestamp, "filename": fname})
                    n_frames += 1

            elif conn.topic == args.odom_topic:
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                odom_rows.append({
                    "timestamp_ns": timestamp,
                    "x": p.x,
                    "y": p.y,
                    "yaw": quat_to_yaw(q.x, q.y, q.z, q.w),
                })

    # Write odom.csv
    if odom_rows:
        with open(out_dir / "odom.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp_ns", "x", "y", "yaw"])
            w.writeheader()
            w.writerows(odom_rows)

    # Write frame_index.csv
    if frame_index:
        with open(out_dir / "frame_index.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp_ns", "filename"])
            w.writeheader()
            w.writerows(frame_index)

    # Quick sanity summary — catches the "odom frozen" problem early
    print(f"Frames extracted : {n_frames}")
    print(f"Odom poses       : {len(odom_rows)}")
    if odom_rows:
        xs = [r["x"] for r in odom_rows]
        ys = [r["y"] for r in odom_rows]
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        print(f"Odom X range     : {dx:.3f} m   Y range: {dy:.3f} m")
        if dx < 0.05 and dy < 0.05:
            print("  !! WARNING: odom barely moved (<5cm range). "
                  "Either a stationary trip or odom not integrating — check before using.")
        else:
            print("  OK: odom shows real motion.")
    print(f"\nWrote -> {out_dir}/  (frames/, odom.csv, frame_index.csv)")


if __name__ == "__main__":
    main()