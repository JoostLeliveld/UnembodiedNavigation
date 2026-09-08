#!/usr/bin/env python3
"""Capture one frame from an overhead camera in the warehouse world.

The world carries two purpose-built overhead cameras: presentation_overview_camera at 19 m,
which frames the hall, and plan_view_camera at 42 m, which flattens the racks. This grabs a
single frame from one of them and writes it as a PNG, so the thesis can show the simulation
itself rather than a schematic of it.

Assumes the simulation is already running. Launch it first, e.g.

    ros2 launch sim bringup_sim.launch.py world:=warehouse_v2.world.sdf

then run this and stop the simulation afterwards.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import cv2  # type: ignore
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

TOPICS = {
    'overview': '/presentation_overview_camera/image_raw',
    'plan': '/plan_view_camera/image_raw',
}


class OneShot(Node):
    def __init__(self, topic: str) -> None:
        super().__init__('overview_frame_capture')
        self.frame = None
        self.bridge = CvBridge()
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.create_subscription(Image, topic, self._on_image, qos)

    def _on_image(self, message: Image) -> None:
        if self.frame is None:
            self.frame = self.bridge.imgmsg_to_cv2(message, 'bgr8')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--view', choices=sorted(TOPICS), default='overview')
    parser.add_argument('--out', type=pathlib.Path, required=True)
    parser.add_argument('--timeout', type=float, default=60.0)
    args = parser.parse_args()

    rclpy.init()
    node = OneShot(TOPICS[args.view])
    deadline = node.get_clock().now().nanoseconds + int(args.timeout * 1e9)
    while node.frame is None and node.get_clock().now().nanoseconds < deadline:
        rclpy.spin_once(node, timeout_sec=0.5)

    frame = node.frame
    node.destroy_node()
    rclpy.shutdown()

    if frame is None:
        print(f'no frame on {TOPICS[args.view]} within {args.timeout:.0f} s; '
              'is the simulation running?', file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), frame)
    print(f'wrote {args.out} ({frame.shape[1]}x{frame.shape[0]})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
