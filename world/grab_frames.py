#!/usr/bin/env python3
"""Grab one frame from every warehouse_v2 camera and build a contact sheet."""
from __future__ import annotations
import argparse, pathlib, sys, time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

CAMS = [("A", "/external_camera/image_raw"),
        ("B", "/external_camera_b/image_raw"),
        ("C", "/external_camera_c/image_raw"),
        ("D", "/external_camera_d/image_raw"),
        ("E", "/external_camera_e/image_raw"),
        ("TOP", "/presentation_overview_camera/image_raw"),
        ("PLAN", "/plan_view_camera/image_raw")]


class Grab(Node):
    def __init__(self, want):
        super().__init__("wh_v2_grab")
        self.frames = {}
        self.want = want
        for name, topic in CAMS:
            if name in want:
                self.create_subscription(Image, topic,
                                         lambda m, n=name: self._on(n, m), 1)

    def _on(self, name, msg):
        if name in self.frames:
            return
        a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        self.frames[name] = a[..., :3].copy()
        self.get_logger().info(f"{name}: {a.shape} {msg.encoding}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="frames")
    ap.add_argument("--tag", default="iter")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--cams", default="A,B,C,D,E,TOP,PLAN")
    a = ap.parse_args()
    want = set(a.cams.split(","))
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    n = Grab(want)
    t0 = time.time()
    while rclpy.ok() and time.time() - t0 < a.timeout and len(n.frames) < len(want):
        rclpy.spin_once(n, timeout_sec=0.5)
    missing = sorted(want - set(n.frames))
    from PIL import Image as PImage
    for k, v in n.frames.items():
        PImage.fromarray(v).save(out / f"{a.tag}_{k}.png")
    n.destroy_node(); rclpy.shutdown()
    print(f"saved {sorted(n.frames)} to {out}" + (f"  MISSING {missing}" if missing else ""))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
