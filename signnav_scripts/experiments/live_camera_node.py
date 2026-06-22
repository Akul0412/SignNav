#!/usr/bin/env python3
"""
live_camera_node.py - ROS2 bridge: camera topic -> live adaptive reasoning.

Subscribes to the robot's image topic, decodes each frame to a PIL image, and
feeds the LATEST frame into AdaptiveReasoningLoop.step(). While the reasoner is
busy (it takes seconds per triggered frame), incoming frames are dropped so we
always reason on the freshest view instead of building a lag backlog.

For this first live test:
  - controller is the placeholder (prints actions; NO motion)
  - all per-frame debug output is mirrored to a timestamped log file for review

Run on the Jetson (inside your ROS2 + qwen_test environment):
    python3 live_camera_node.py --topic /c1/image_raw --goal "elevator"

Notes:
  - --encoding handles the decode. Default 'uyvy' matches the robot's yuv422 feed
    used in extract_trip.py. If frames look wrong, try 'bgr8','rgb8','mono8'.
  - --reason-min-interval throttles how often we even ATTEMPT a step (seconds),
    so a fast camera doesn't spam the loop. The loop's own adaptive trigger still
    decides whether to run the heavy reasoner.
"""

import argparse
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ROS2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from PIL import Image as PILImage

# the reasoning loop (same package used in offline tests)
from signnav_reasoner.types import Config
from signnav_reasoner.loop import AdaptiveReasoningLoop


# ----------------------------------------------------------------------------
# Image decoding: ROS sensor_msgs/Image -> PIL.Image (RGB)
# ----------------------------------------------------------------------------
def decode_image(msg: Image, encoding_hint: str) -> PILImage.Image:
    """Convert a ROS Image message to a PIL RGB image.
    Handles the robot's UYVY/yuv422 feed plus common encodings."""
    h, w = msg.height, msg.width
    enc = (encoding_hint or msg.encoding or "").lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("uyvy", "yuv422", "yuv422_uyvy"):
        # UYVY: 2 bytes per pixel, packed U Y0 V Y1
        yuv = buf.reshape(h, w, 2)
        # convert UYVY -> BGR using OpenCV (most robust)
        import cv2
        bgr = cv2.cvtColor(buf.reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(rgb)

    if enc in ("yuyv", "yuv422_yuy2"):
        import cv2
        bgr = cv2.cvtColor(buf.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(rgb)

    if enc in ("rgb8",):
        return PILImage.fromarray(buf.reshape(h, w, 3))

    if enc in ("bgr8",):
        return PILImage.fromarray(buf.reshape(h, w, 3)[:, :, ::-1])

    if enc in ("mono8", "8uc1"):
        gray = buf.reshape(h, w)
        return PILImage.fromarray(gray).convert("RGB")

    # fallback: assume bgr8
    print(f"[bridge] unknown encoding '{enc}', assuming bgr8")
    return PILImage.fromarray(buf.reshape(h, w, 3)[:, :, ::-1])


# ----------------------------------------------------------------------------
# Tee logger: mirror stdout to a file so the live run is saved for review
# ----------------------------------------------------------------------------
class Tee:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "w")
    def write(self, m):
        self.terminal.write(m)
        self.log.write(m)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()


class LiveReasoningNode(Node):
    def __init__(self, loop: AdaptiveReasoningLoop, topic: str,
                 encoding: str, reason_min_interval: float):
        super().__init__("signnav_live_reasoning")
        self.loop = loop
        self.encoding = encoding
        self.reason_min_interval = reason_min_interval

        self._latest = None              # most recent decoded frame
        self._latest_lock = threading.Lock()
        self._busy = False               # reasoner running?
        self._frame_count = 0
        self._last_attempt = 0.0

        # subscribe; keep queue small so we don't buffer stale frames
        self.sub = self.create_subscription(Image, topic, self._on_image, 1)

        # worker thread runs the (slow) reasoning so the ROS callback never blocks
        self._worker = threading.Thread(target=self._reason_loop, daemon=True)
        self._worker.start()
        self.get_logger().info(f"subscribed to {topic} (encoding={encoding}); reasoning live.")

    def _on_image(self, msg: Image):
        """ROS callback: just decode + stash the latest frame. Never blocks on reasoning."""
        try:
            img = decode_image(msg, self.encoding)
        except Exception as e:
            self.get_logger().warn(f"decode failed: {e}")
            return
        with self._latest_lock:
            self._latest = img

    def _reason_loop(self):
        """Worker: repeatedly grab the latest frame and run one step(), dropping
        any frames that arrived while we were busy (always freshest view)."""
        while rclpy.ok():
            now = time.time()
            if now - self._last_attempt < self.reason_min_interval:
                time.sleep(0.01)
                continue
            with self._latest_lock:
                img = self._latest
                self._latest = None
            if img is None:
                time.sleep(0.01)
                continue
            self._last_attempt = now
            self._frame_count += 1
            try:
                self.loop.step(img, self._frame_count, total=0,
                               ts=datetime.now().strftime("%H:%M:%S"))
            except Exception as e:
                import traceback
                print(f"[bridge] step error: {e}")
                traceback.print_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/c1/image_raw")
    ap.add_argument("--encoding", default="uyvy",
                    help="image encoding: uyvy|yuyv|rgb8|bgr8|mono8")
    ap.add_argument("--goal", default="elevator")
    ap.add_argument("--reason-min-interval", type=float, default=0.5,
                    help="min seconds between step() attempts (throttle)")
    ap.add_argument("--log-dir", default="live_logs")
    args = ap.parse_args()

    # log file for review
    Path(args.log_dir).mkdir(exist_ok=True)
    log_path = Path(args.log_dir) / f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = Tee(str(log_path))
    print(f"=== live reasoning log -> {log_path} ===")
    print(f"=== topic={args.topic} encoding={args.encoding} goal='{args.goal}' ===")

    # build the loop (real models; controller is placeholder = print only)
    cfg = Config(goal=args.goal)
    loop = AdaptiveReasoningLoop(cfg)

    rclpy.init()
    node = LiveReasoningNode(loop, args.topic, args.encoding, args.reason_min_interval)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n=== stopped by user ===")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()