#!/usr/bin/env python3
"""M0.1a config audit: per-camera YOLO detection rate over a teleport grid.

Measures whether the CURRENT detector is out-of-distribution on the four
warehouse_full_4cam viewpoints, at whatever imgsz the running stack was launched
with. This is a RATE audit only: it reads the batched YOLO node's per-camera
observations (`/perception/camera_observation/{camera_A..D}`, field
`detection_valid`) while teleporting the robot over a coarse drivable grid. It
does NOT save labels or train anything.

Run it against a live stack:
  ros2 launch experiments warehouse_full4cam_commissioning.launch.py \
      world:=warehouse_full_4cam.world.sdf \
      yolo_model:=logs/perception_models/warehouse_yolo_detector_v1/model.pt \
      yolo_runtime_backend:=native yolo_imgsz:=640 headless:=true
then, in a second shell (env sourced):
  python3 experiments/multicamera_commissioning_bigwarehouse/tools/audit_detection_rate.py \
      --imgsz 640 --nx 8 --ny 7 --out logs/studies/fourcam_detector_audit/imgsz640

No ground truth is used (positions are commanded teleport poses; det from YOLO).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from std_msgs.msg import String

CAMERAS = ["camera_A", "camera_B", "camera_C", "camera_D"]
# approximate wall-mounted camera positions (m) for range binning only
CAM_XYZ = {"camera_A": (-6.0, -10.0, 6.1), "camera_B": (-6.0, 10.0, 6.1),
           "camera_C": (6.0, -10.0, 6.1), "camera_D": (6.0, 10.0, 6.1)}
TRAV = dict(xmin=-11.20, xmax=11.50, ymin=-8.60, ymax=8.60)
OBST = [  # xmin, xmax, ymin, ymax  (from world_profiles known_2d_regions)
    (-11.91, -11.36, -6.50, 6.90), (-0.25, 0.25, -1.15, -0.65),
    (-6.125, -5.225, -4.90, -4.30), (9.75, 10.65, 0.45, 1.15)]


def in_obstacle(x, y, clear=0.45):
    return any(xm - clear <= x <= xM + clear and ym - clear <= y <= yM + clear
               for xm, xM, ym, yM in OBST)


def build_grid(nx, ny, margin=0.7):
    xs = np.linspace(TRAV["xmin"] + margin, TRAV["xmax"] - margin, nx)
    ys = np.linspace(TRAV["ymin"] + margin, TRAV["ymax"] - margin, ny)
    return [(float(x), float(y)) for x in xs for y in ys if not in_obstacle(x, y)]


def nearest_cam_range(x, y):
    return min(math.dist((x, y, 0.05), c) for c in CAM_XYZ.values())


def as_detected(payload):
    for k in ("detection_valid", "detected_after_threshold", "detected"):
        if k in payload:
            return bool(payload[k])
    return False


def as_score(payload):
    for k in ("detector_score_raw", "detector_score", "detection_confidence",
              "selected_score", "raw_best_score", "confidence"):
        if k in payload and payload[k] is not None:
            try:
                return float(payload[k])
            except (TypeError, ValueError):
                pass
    return 0.0


class Probe(Node):
    def __init__(self, world):
        super().__init__("detection_rate_audit")
        self.count = {c: 0 for c in CAMERAS}
        self.latest = {c: None for c in CAMERAS}
        for c in CAMERAS:
            self.create_subscription(String, f"/perception/camera_observation/{c}",
                                     self._cb(c), 20)
        self.cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.cli = self.create_client(SetEntityPose, f"/world/{world}/set_pose")

    def _cb(self, cam):
        def handler(msg):
            try:
                d = json.loads(msg.data)
            except (ValueError, TypeError):
                return
            self.count[cam] += 1
            self.latest[cam] = (self.count[cam], as_detected(d), as_score(d))
        return handler

    def wait_ready(self, timeout=60.0):
        t0 = time.time()
        while time.time() - t0 < timeout and not self.cli.service_is_ready():
            rclpy.spin_once(self, timeout_sec=0.2)
        got = 0
        t0 = time.time()
        while time.time() - t0 < timeout and got < len(CAMERAS):
            rclpy.spin_once(self, timeout_sec=0.2)
            got = sum(1 for c in CAMERAS if self.count[c] > 0)
        return self.cli.service_is_ready() and got == len(CAMERAS)

    def teleport(self, x, y, yaw=1.5708):
        req = SetEntityPose.Request()
        req.entity = Entity(name="turtlebot3", type=Entity.MODEL)
        req.pose.position.x, req.pose.position.y, req.pose.position.z = x, y, 0.05
        req.pose.orientation.z = math.sin(yaw / 2.0)
        req.pose.orientation.w = math.cos(yaw / 2.0)
        fut = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)

    def sample(self, settle_frames=2, timeout=8.0):
        base = {c: self.count[c] for c in CAMERAS}
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.cmd.publish(Twist())  # hold still
            rclpy.spin_once(self, timeout_sec=0.1)
            if all(self.count[c] - base[c] >= settle_frames for c in CAMERAS):
                break
        return {c: (self.latest[c][1:] if self.count[c] > base[c] and self.latest[c] else None)
                for c in CAMERAS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="warehouse_full_4cam")
    ap.add_argument("--imgsz", type=int, default=0, help="label only (stack sets the real imgsz)")
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--ny", type=int, default=7)
    ap.add_argument("--settle-frames", type=int, default=2)
    ap.add_argument("--sample-timeout", type=float, default=8.0)
    ap.add_argument("--out", default="logs/studies/fourcam_detector_audit/run")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Probe(args.world)
    try:
        if not node.wait_ready():
            print("ERROR: stack not ready (set_pose service or all 4 camera observations missing).")
            return 2
        grid = build_grid(args.nx, args.ny)
        print(f"probing {len(grid)} drivable poses on {args.world} (imgsz label {args.imgsz})")
        rows = []
        for i, (x, y) in enumerate(grid):
            node.teleport(x, y)
            res = node.sample(settle_frames=args.settle_frames, timeout=args.sample_timeout)
            rng = nearest_cam_range(x, y)
            for c in CAMERAS:
                det, sc, stale = (res[c][0], res[c][1], False) if res[c] else (False, 0.0, True)
                rows.append(dict(i=i, x=round(x, 3), y=round(y, 3), camera=c,
                                 detected=int(det), score=round(sc, 4),
                                 nearest_cam_range_m=round(rng, 2), stale=int(stale)))
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(grid)} poses")
    finally:
        node.destroy_node()
        rclpy.shutdown()

    with open(out / "detections.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # per-camera + range-binned detection rate
    summary = {"imgsz": args.imgsz, "world": args.world, "n_poses": len(rows) // len(CAMERAS),
               "per_camera": {}, "by_range": {}}
    print(f"\n=== detection rate (imgsz {args.imgsz}), {summary['n_poses']} poses ===")
    for c in CAMERAS:
        cr = [r for r in rows if r["camera"] == c]
        rate = float(np.mean([r["detected"] for r in cr]))
        msc = float(np.mean([r["score"] for r in cr if r["detected"]])) if any(r["detected"] for r in cr) else 0.0
        summary["per_camera"][c] = dict(detection_rate=round(rate, 3), mean_score_when_detected=round(msc, 3),
                                        stale=sum(r["stale"] for r in cr))
        print(f"  {c}: det_rate={rate:.3f}  mean_score={msc:.3f}  stale={sum(r['stale'] for r in cr)}")
    for lo, hi in [(0, 12), (12, 16), (16, 99)]:
        rr = [r for r in rows if lo <= r["nearest_cam_range_m"] < hi]
        if rr:
            rate = float(np.mean([r["detected"] for r in rr]))
            summary["by_range"][f"{lo}-{hi}m"] = dict(detection_rate=round(rate, 3), n=len(rr))
            print(f"  range {lo}-{hi}m (nearest cam): det_rate={rate:.3f}  n={len(rr)}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}/detections.csv and summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
