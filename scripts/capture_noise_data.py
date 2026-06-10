#!/usr/bin/env python3
# [DEPRECATED_LEGACY_CLEANUP] Legacy/exploratory/diagnostic script or module. Distracting from paper-facing F85-F88 runtime.
"""Capture odom/cmd/pixel streams for offline Q/R calibration."""

from __future__ import annotations

import argparse
import csv
import math
import os
import pathlib
import time
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class NoiseCaptureNode(Node):
    def __init__(self, output_csv: pathlib.Path):
        super().__init__("noise_capture_node")
        self.output_csv = output_csv
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        self.file = self.output_csv.open("w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            [
                "stamp",
                "source",
                "odom_x",
                "odom_y",
                "odom_yaw",
                "cmd_v",
                "cmd_w",
                "pix_u",
                "pix_v",
                "pix_yaw",
            ]
        )

        # Keep latest command so odom rows carry the command actually applied nearby.
        self._last_cmd: Tuple[float, float] = (0.0, 0.0)
        self.odom_count = 0
        self.pixel_count = 0
        self.cmd_count = 0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )
        self.create_subscription(Odometry, "/odom", self._odom_cb, qos)
        self.create_subscription(Twist, "/cmd_vel", self._cmd_cb, qos)
        self.create_subscription(PoseStamped, "/perception/pixel_pose", self._pixel_cb, qos)

        self.get_logger().info(f"Capturing noise data to {self.output_csv}")

    def _cmd_cb(self, msg: Twist) -> None:
        self._last_cmd = (float(msg.linear.x), float(msg.angular.z))
        self.cmd_count += 1

    def _odom_cb(self, msg: Odometry) -> None:
        t = _stamp_to_sec(msg.header.stamp)
        q = msg.pose.pose.orientation
        yaw = _yaw_from_quat(q.x, q.y, q.z, q.w)
        self.writer.writerow(
            [
                t,
                "odom",
                float(msg.pose.pose.position.x),
                float(msg.pose.pose.position.y),
                yaw,
                self._last_cmd[0],
                self._last_cmd[1],
                "",
                "",
                "",
            ]
        )
        self.odom_count += 1

    def _pixel_cb(self, msg: PoseStamped) -> None:
        t = _stamp_to_sec(msg.header.stamp)
        q = msg.pose.orientation
        yaw = _yaw_from_quat(q.x, q.y, q.z, q.w)
        self.writer.writerow(
            [
                t,
                "pixel",
                "",
                "",
                "",
                "",
                "",
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                yaw,
            ]
        )
        self.pixel_count += 1

    def close(self) -> None:
        try:
            self.file.flush()
            self.file.close()
        except OSError:
            pass


def _default_output_path() -> pathlib.Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return pathlib.Path("logs") / "calibration" / f"noise_capture_{stamp}.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture streams for Q/R calibration.")
    parser.add_argument("--duration-s", type=float, default=60.0, help="Capture duration in wall seconds.")
    parser.add_argument("--output", type=str, default=str(_default_output_path()), help="Output CSV path.")
    args = parser.parse_args()

    output = pathlib.Path(args.output).resolve()
    os.makedirs(output.parent, exist_ok=True)

    rclpy.init()
    node = NoiseCaptureNode(output)
    t0 = time.monotonic()
    try:
        while rclpy.ok() and (time.monotonic() - t0) < args.duration_s:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()

    print(f"output_csv={output}")
    print(f"odom_samples={node.odom_count}")
    print(f"pixel_samples={node.pixel_count}")
    print(f"cmd_samples={node.cmd_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
