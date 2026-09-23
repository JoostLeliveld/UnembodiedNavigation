#!/usr/bin/env python3
"""Measure simultaneous delivered camera cadence using steady wall time."""

import argparse
import json
import math
import statistics
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


TOPICS = {
    "camera_A": "/external_camera/image_raw",
    "camera_B": "/external_camera_b/image_raw",
    "camera_C": "/external_camera_c/image_raw",
    "camera_D": "/external_camera_d/image_raw",
    "camera_E": "/external_camera_e/image_raw",
}


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = fraction * (len(values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    weight = index - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


class Probe(Node):
    def __init__(self):
        super().__init__("camera_rate_probe")
        self.rows = {camera: [] for camera in TOPICS}
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )
        self.subscriptions_ = []
        for camera, topic in TOPICS.items():
            self.subscriptions_.append(
                self.create_subscription(Image, topic, self.callback(camera), qos)
            )

    def callback(self, camera):
        def receive(message):
            source = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            self.rows[camera].append((time.monotonic(), source))
        return receive


def cadence(values, index):
    stamps = [row[index] for row in values]
    gaps = [right - left for left, right in zip(stamps, stamps[1:]) if right > left]
    span = stamps[-1] - stamps[0] if len(stamps) > 1 else 0.0
    return {
        "count": len(stamps),
        "rate_hz": (len(stamps) - 1) / span if span > 0 else None,
        "median_gap_s": statistics.median(gaps) if gaps else None,
        "p95_gap_s": percentile(gaps, 0.95),
        "max_gap_s": max(gaps) if gaps else None,
        "backward_or_duplicate_count": sum(
            right <= left for left, right in zip(stamps, stamps[1:])
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rclpy.init()
    node = Probe()
    try:
        warmup_end = time.monotonic() + args.warmup
        while time.monotonic() < warmup_end:
            rclpy.spin_once(node, timeout_sec=0.1)
        for values in node.rows.values():
            values.clear()
        start = time.monotonic()
        while time.monotonic() - start < args.duration:
            rclpy.spin_once(node, timeout_sec=0.1)
        result = {
            "schema": "camera_rate_probe.v1",
            "duration_wall_s": time.monotonic() - start,
            "cameras": {
                camera: {
                    "wall_delivery": cadence(values, 0),
                    "source_stamp": cadence(values, 1),
                }
                for camera, values in node.rows.items()
            },
        }
        with open(args.output, "x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
