#!/usr/bin/env python3
"""Stream-record the external camera to PNG frames named by sim-time stamp.

Runs until killed. Subscribes to /external_camera/image_raw and writes every
frame to <out-dir>/frame_<stamp>.png plus an index.csv (stamp, file). The stamp
is the message header stamp (Gazebo sim time), so frames align with the
experiment.csv `stamp` column for later replay.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2  # type: ignore
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image


class Recorder(Node):
    def __init__(self, topic: str, out_dir: Path, every: int) -> None:
        super().__init__("external_camera_stream_recorder")
        self.bridge = CvBridge()
        self.out_dir = out_dir
        self.every = max(1, every)
        self.count = 0
        self.saved = 0
        self.index = open(out_dir / "index.csv", "w", newline="")
        self.writer = csv.writer(self.index)
        self.writer.writerow(["stamp", "file"])
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(Image, topic, self._cb, qos)
        self.get_logger().info(f"recording {topic} -> {out_dir}")

    def _cb(self, msg: Image) -> None:
        self.count += 1
        if self.count % self.every:
            return
        try:
            cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa
            self.get_logger().warn(f"cv_bridge failed: {exc}")
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        name = f"frame_{stamp:010.3f}.png"
        cv2.imwrite(str(self.out_dir / name), cv)
        self.writer.writerow([f"{stamp:.3f}", name])
        self.index.flush()
        self.saved += 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--topic", default="/external_camera/image_raw")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--every", type=int, default=1, help="save every Nth frame")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = Recorder(args.topic, args.out_dir, args.every)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"recorded {node.saved} frames to {args.out_dir}")
        node.index.close()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
