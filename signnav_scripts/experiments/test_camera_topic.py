#!/usr/bin/env python3
"""
test_camera_topic.py - verify the camera feed BEFORE running full reasoning.

Subscribes to the image topic, grabs ONE frame, prints its dimensions/encoding,
and saves it as test_frame.jpg so you can eyeball whether the decode is correct.

This isolates camera problems from reasoning problems. Run this FIRST:
    python3 test_camera_topic.py --topic /c1/image_raw --encoding uyvy
Then open test_frame.jpg. If it looks right, the full pipeline will get good frames.
If it's garbled, the --encoding is wrong (try yuyv, bgr8, rgb8).
"""

import argparse
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from PIL import Image as PILImage


def decode_image(msg, encoding_hint):
    h, w = msg.height, msg.width
    enc = (encoding_hint or msg.encoding or "").lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    print(f"  raw: {len(buf)} bytes for {w}x{h} (={len(buf)/(w*h):.2f} bytes/pixel)")
    if enc in ("uyvy", "yuv422", "yuv422_uyvy"):
        import cv2
        bgr = cv2.cvtColor(buf.reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
        return PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if enc in ("yuyv", "yuv422_yuy2"):
        import cv2
        bgr = cv2.cvtColor(buf.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
        return PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if enc == "rgb8":
        return PILImage.fromarray(buf.reshape(h, w, 3))
    if enc == "bgr8":
        return PILImage.fromarray(buf.reshape(h, w, 3)[:, :, ::-1])
    if enc in ("mono8", "8uc1"):
        return PILImage.fromarray(buf.reshape(h, w)).convert("RGB")
    print(f"  unknown encoding '{enc}', assuming bgr8")
    return PILImage.fromarray(buf.reshape(h, w, 3)[:, :, ::-1])


class OneShot(Node):
    def __init__(self, topic, encoding):
        super().__init__("signnav_camera_test")
        self.encoding = encoding
        self.got = False
        self.sub = self.create_subscription(Image, topic, self._cb, 1)
        self.get_logger().info(f"waiting for one frame on {topic} ...")

    def _cb(self, msg):
        if self.got:
            return
        self.got = True
        print(f"\nGOT FRAME:")
        print(f"  topic encoding field: '{msg.encoding}'")
        print(f"  dimensions: {msg.width}x{msg.height}")
        print(f"  step (bytes/row): {msg.step}")
        try:
            img = decode_image(msg, self.encoding)
            img.save("test_frame.jpg")
            print(f"  decoded OK -> saved test_frame.jpg ({img.size[0]}x{img.size[1]})")
            print(f"  >>> OPEN test_frame.jpg to verify it looks correct <<<")
        except Exception as e:
            import traceback
            print(f"  DECODE FAILED: {e}")
            traceback.print_exc()
            print(f"  try a different --encoding (yuyv, bgr8, rgb8, mono8)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/c1/image_raw")
    ap.add_argument("--encoding", default="uyvy")
    args = ap.parse_args()
    rclpy.init()
    node = OneShot(args.topic, args.encoding)
    # spin until we get one frame (or Ctrl-C)
    while rclpy.ok() and not node.got:
        rclpy.spin_once(node, timeout_sec=1.0)
    node.destroy_node()
    rclpy.shutdown()
    if not node.got:
        print("No frame received. Is the camera publishing? Check: ros2 topic list")


if __name__ == "__main__":
    main()