#!/usr/bin/env python3
"""Save one frame from the external camera as PNG.

Usage:
    # While the sim is running and the bridge is up:
    python3 scripts/paper_figures/capture_external_camera_frame.py \
        --out /home/joostleliveld/Thesis/thesis-report/figures/external_camera.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2  # type: ignore
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


class FrameGrabber(Node):
    def __init__(self, topic: str, out_path: Path) -> None:
        super().__init__("external_camera_frame_grabber")
        self.bridge = CvBridge()
        self.out_path = out_path
        self.received = False
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub = self.create_subscription(Image, topic, self._cb, qos)
        self.get_logger().info(f"waiting for one frame on {topic} -> {out_path}")

    def _cb(self, msg: Image) -> None:
        if self.received:
            return
        self.received = True
        try:
            cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"cv_bridge failed: {exc}")
            return
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(self.out_path), cv)
        if ok:
            self.get_logger().info(
                f"saved {self.out_path} ({msg.width}x{msg.height}, encoding={msg.encoding})"
            )
        else:
            self.get_logger().error(f"cv2.imwrite failed for {self.out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--topic", default="/external_camera/image_raw",
        help="ROS image topic to subscribe to.",
    )
    p.add_argument(
        "--out", type=Path, required=True,
        help="Output PNG/JPG path.",
    )
    p.add_argument(
        "--timeout", type=float, default=20.0,
        help="Seconds to wait for one frame before giving up.",
    )
    args = p.parse_args()

    rclpy.init()
    node = FrameGrabber(args.topic, args.out)
    try:
        end = node.get_clock().now().nanoseconds + int(args.timeout * 1e9)
        while rclpy.ok() and not node.received:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.get_clock().now().nanoseconds > end:
                node.get_logger().error(
                    f"no frame received within {args.timeout:.1f}s on {args.topic}"
                )
                return 2
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0 if node.received else 2


if __name__ == "__main__":
    raise SystemExit(main())
