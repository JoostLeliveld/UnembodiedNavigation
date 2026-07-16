#!/usr/bin/env python3
"""Capture one RGB + one depth frame from the fixed external camera into
logs/geometry_visibility_prior/frames/ (persistent, unlike session scratchpads).

Expects a running sim with /external_camera/image_raw bridged (bringup does this)
and /external_camera/depth bridged separately (depth sensor is always_on=0 but
publishes once subscribed via the bridge).

Usage: python3 capture_camera_frames.py [--out DIR] [--timeout 60]
"""
from __future__ import annotations
import argparse, pathlib, sys, time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

_HERE = pathlib.Path(__file__).resolve().parent
REPO = _HERE.parents[1]


class FrameSaver(Node):
    def __init__(self, out: pathlib.Path, rgb_topic: str, depth_topic: str):
        super().__init__("frame_saver")
        self.out = out
        self.rgb = None
        self.depth = None
        self.create_subscription(Image, rgb_topic, self._on_rgb, 1)
        self.create_subscription(Image, depth_topic, self._on_depth, 1)

    def _on_rgb(self, msg: Image):
        if self.rgb is not None:
            return
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        self.rgb = arr[..., :3].copy()  # rgb8
        self.get_logger().info(f"got RGB {self.rgb.shape} encoding={msg.encoding}")

    def _on_depth(self, msg: Image):
        if self.depth is not None:
            return
        if msg.encoding != "32FC1":
            self.get_logger().warn(f"unexpected depth encoding {msg.encoding}")
        self.depth = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width).copy()
        self.get_logger().info(f"got depth {self.depth.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "logs/geometry_visibility_prior/frames"))
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--tag", default="stack")
    ap.add_argument("--rgb-topic", default="/external_camera/image_raw")
    ap.add_argument("--depth-topic", default="/external_camera/depth")
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = FrameSaver(out, args.rgb_topic, args.depth_topic)
    t0 = time.time()
    while rclpy.ok() and time.time() - t0 < args.timeout and (node.rgb is None or node.depth is None):
        rclpy.spin_once(node, timeout_sec=0.5)

    ok = node.rgb is not None and node.depth is not None
    if node.rgb is not None:
        np.save(out / f"rgb_{args.tag}.npy", node.rgb)
        try:
            import cv2
            cv2.imwrite(str(out / f"rgb_{args.tag}.png"), node.rgb[..., ::-1])
        except Exception as e:  # png is a convenience; npy is canonical
            print(f"png save skipped: {e}")
    if node.depth is not None:
        np.save(out / f"depth_{args.tag}.npy", node.depth)
        d = node.depth[np.isfinite(node.depth)]
        print(f"depth stats: {d.min():.2f}-{d.max():.2f} m, {100*len(d)/node.depth.size:.0f}% finite")
    node.destroy_node(); rclpy.shutdown()
    print("CAPTURE_OK" if ok else f"CAPTURE_INCOMPLETE rgb={node.rgb is not None} depth={node.depth is not None}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
