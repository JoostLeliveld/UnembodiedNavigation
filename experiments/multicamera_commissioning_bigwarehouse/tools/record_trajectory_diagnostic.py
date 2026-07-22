#!/usr/bin/env python3
"""Lightweight trajectory recorder for M1/M2 (diagnostic bypass of the frozen campaign).

Records, over time while the robot drives, each of the 4 cameras' observations
(detection + selected pixel + covariance + score) alongside the robot's odometry
(operational) and Gazebo ground-truth pose (EVALUATION-ONLY, for scoring only —
never a model input). One CSV row per camera_observation message received.

This is NOT the frozen commissioning recorder (record_operational_logs.py, which
requires the full paper-campaign provenance set). It is a diagnostic logger to get
M1 reliability maps + M2 fusion-vs-baseline data with the diagnostic detector.
Writes on SIGINT or after --duration-s.
"""
from __future__ import annotations
import argparse, csv, json, math, signal, time
from pathlib import Path
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage

CAMS = ["camera_A", "camera_B", "camera_C", "camera_D"]


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _num(v, d=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


class Recorder(Node):
    def __init__(self, odom_topic, gt_topic, child_frame):
        super().__init__("trajectory_recorder")
        self.rows = []
        self.odom = (float("nan"), float("nan"), float("nan"))
        self.gt = (float("nan"), float("nan"), float("nan"))
        self.child = child_frame
        self.n = {c: 0 for c in CAMS}
        for c in CAMS:
            self.create_subscription(String, f"/perception/camera_observation/{c}", self._cam(c), 30)
        self.create_subscription(Odometry, odom_topic, self._odom, 30)
        self.create_subscription(TFMessage, gt_topic, self._gt, 30)

    def _odom(self, m):
        p = m.pose.pose.position
        self.odom = (p.x, p.y, _yaw(m.pose.pose.orientation))

    def _gt(self, m):
        for t in m.transforms:
            if self.child in t.child_frame_id:
                tr = t.transform.translation
                self.gt = (tr.x, tr.y, _yaw(t.transform.rotation))
                return

    def _cam(self, c):
        def f(m):
            try:
                d = json.loads(m.data)
            except (ValueError, TypeError):
                return
            self.n[c] += 1
            pix = d.get("pixel_uv") or [float("nan"), float("nan")]
            bb = d.get("bbox_bottom_uv") or [float("nan"), float("nan")]
            cov = d.get("conditional_cov_uv") or [[float("nan"), float("nan")], [float("nan"), float("nan")]]
            self.rows.append(dict(
                t=_num(d.get("timestamp_s"), time.time()), camera=c,
                detected=int(bool(d.get("detection_valid"))),
                score=_num(d.get("detector_score_raw", d.get("detector_score", 0.0)), 0.0),
                pix_u=_num(pix[0]), pix_v=_num(pix[1]),
                bbot_u=_num(bb[0]), bbot_v=_num(bb[1]),
                cov_uu=_num(cov[0][0]), cov_uv=_num(cov[0][1]), cov_vv=_num(cov[1][1]),
                meas_age=_num(d.get("measurement_age_s")),
                odom_x=self.odom[0], odom_y=self.odom[1], odom_yaw=self.odom[2],
                gt_x=self.gt[0], gt_y=self.gt[1], gt_yaw=self.gt[2],   # EVALUATION-ONLY
            ))
        return f

    def write(self, out: Path):
        out.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            print("recorder: NO rows captured"); return
        cols = list(self.rows[0].keys())
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(self.rows)
        det = sum(r["detected"] for r in self.rows)
        print(f"recorder wrote {len(self.rows)} rows ({det} detections) to {out}")
        for c in CAMS:
            cr = [r for r in self.rows if r["camera"] == c]
            print(f"  {c}: {len(cr)} rows, {sum(r['detected'] for r in cr)} detections")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--gt-topic", default="/ground_truth_tf")
    ap.add_argument("--child-frame", default="turtlebot3")
    ap.add_argument("--duration-s", type=float, default=0.0)
    args = ap.parse_args()
    rclpy.init()
    node = Recorder(args.odom_topic, args.gt_topic, args.child_frame)
    out = Path(args.out)
    stop = {"v": False}
    signal.signal(signal.SIGINT, lambda *a: stop.__setitem__("v", True))
    signal.signal(signal.SIGTERM, lambda *a: stop.__setitem__("v", True))
    t0 = time.time()
    try:
        while not stop["v"]:
            rclpy.spin_once(node, timeout_sec=0.2)
            if args.duration_s > 0 and time.time() - t0 > args.duration_s:
                break
    finally:
        node.write(out)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
