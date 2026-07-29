#!/usr/bin/env python3
"""Reliability-aware scheduled multi-camera detector.

Instead of running four detectors every cycle (~1 Hz, batched, belief drifts
between corrections), this runs ONE inference per cycle on the camera the
coverage map says best sees the robot's current belief position, and hands over
to the next camera as the robot drives. Single-image inference restores
~3-4 Hz corrections (single-camera rate) while keeping full-warehouse coverage
through hand-over. It projects the detection with that camera's calibration and
publishes the world-frame correction directly to /state/bev.
"""
from __future__ import annotations
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge
from ultralytics import YOLO

from reliability.projection import camera_model_from_world, load_projection_calibration

CAMS = [("camera_A", "external_camera",   "/external_camera/image_raw"),
        ("camera_B", "external_camera_b", "/external_camera_b/image_raw"),
        ("camera_C", "external_camera_c", "/external_camera_c/image_raw"),
        ("camera_D", "external_camera_d", "/external_camera_d/image_raw")]


def _project(camera, u, v, contact_z, offset, slope):
    point = camera.pixel_to_world_at_z(u, v, contact_z) if contact_z > 0 else camera.pixel_to_world(u, v)
    if point is None:
        return None
    px, py = float(point[0]), float(point[1])
    if not offset and not slope:
        return (px, py)
    bx, by = px - float(camera.cam_pos[0]), py - float(camera.cam_pos[1])
    norm = math.hypot(bx, by)
    if norm <= 1e-9:
        return (px, py)
    s = (offset + slope * norm) / norm
    return (px + bx * s, py + by * s)


class ScheduledCameraDetector(Node):
    def __init__(self) -> None:
        super().__init__("scheduled_camera_detector")
        gp = self.declare_parameter
        gp("model_path", ""); gp("world_sdf", ""); gp("coverage_artifact", "")
        gp("projection_calibration", ""); gp("device", "0"); gp("imgsz", 640)
        gp("conf", 0.05); gp("iou", 0.45); gp("contact_z_m", 0.05)
        gp("report_std_m", 0.15); gp("rate_hz", 5.0); gp("frame_id", "map_bev")
        gp("spawn_x", 0.0); gp("spawn_y", 0.0); gp("min_coverage", 0.02)
        g = lambda n: self.get_parameter(n).value
        world = str(g("world_sdf")); self.device = str(g("device"))
        self.imgsz = int(g("imgsz")); self.conf = float(g("conf")); self.iou = float(g("iou"))
        self.contact_z = float(g("contact_z_m")); self.rvar = float(g("report_std_m")) ** 2
        self.frame_id = str(g("frame_id")); self.min_cov = float(g("min_coverage"))
        self.belief = (float(g("spawn_x")), float(g("spawn_y")))

        self.model = YOLO(str(g("model_path")))
        self.cam_models = {cid: camera_model_from_world(world, include_name=mname) for cid, mname, _ in CAMS}
        try:
            self.calib = load_projection_calibration(str(g("projection_calibration")))
        except Exception:
            self.calib = {}
        d = np.load(str(g("coverage_artifact")))
        self.xs = np.asarray(d["xs"], float); self.ys = np.asarray(d["ys"], float)
        self.cov = {cid: np.asarray(d[f"P_camera_{cid.split('_')[1]}_map"], float) for cid, _, _ in CAMS}

        self.bridge = CvBridge()
        self.latest = {cid: None for cid, _, _ in CAMS}
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=2)
        for cid, _, topic in CAMS:
            self.create_subscription(Image, topic, self._img_cb(cid), qos)
        self.create_subscription(PoseWithCovarianceStamped, "/planner_belief", self._belief_cb, 10)
        self.pub = self.create_publisher(PoseWithCovarianceStamped, "/state/bev", 10)
        # warm up the model off the hot path
        try:
            self.model.predict(source=np.zeros((720, 1280, 3), np.uint8), imgsz=self.imgsz,
                               conf=0.0, iou=self.iou, verbose=False,
                               device=(self.device or None))
        except Exception as exc:
            self.get_logger().warn(f"warmup failed: {exc}")
        self._counts = {cid: 0 for cid, _, _ in CAMS}
        self.create_timer(1.0 / max(float(g("rate_hz")), 0.5), self._tick)
        self.get_logger().info("scheduled_camera_detector: one inference/cycle on the coverage-best camera -> /state/bev")

    def _img_cb(self, cid):
        def cb(msg: Image):
            self.latest[cid] = msg  # store raw; convert only the image we infer
        return cb

    def _belief_cb(self, msg: PoseWithCovarianceStamped):
        self.belief = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y))

    def _cov_at(self, cid, x, y):
        xi = int(np.clip(np.searchsorted(self.xs, x) - 1, 0, len(self.xs) - 1))
        yi = int(np.clip(np.searchsorted(self.ys, y) - 1, 0, len(self.ys) - 1))
        return float(self.cov[cid][yi, xi])

    def _tick(self):
        bx, by = self.belief
        ranked = sorted(((self._cov_at(cid, bx, by), cid) for cid, _, _ in CAMS), reverse=True)
        # Try cameras in coverage order until one detects (needed for robustness:
        # the best-coverage camera misses ~30% of the time, and a bootstrap with
        # zero corrections deadlocks the planner). Convert only the image we infer.
        for score, cid in ranked:
            if score < self.min_cov or self.latest[cid] is None:
                continue
            try:
                img = self.bridge.imgmsg_to_cv2(self.latest[cid], "bgr8")
                res = self.model.predict(source=img, imgsz=self.imgsz, conf=self.conf,
                                         iou=self.iou, verbose=False, device=(self.device or None))
            except Exception as exc:
                self.get_logger().warn(f"predict failed on {cid}: {exc}"); return
            boxes = res[0].boxes
            if boxes is None or len(boxes) == 0:
                continue  # this camera missed; try next-best
            xyxy = boxes.xyxy.cpu().numpy(); confs = boxes.conf.cpu().numpy()
            k = int(np.argmax(confs))
            x1, y1, x2, y2 = xyxy[k]
            u = float((x1 + x2) / 2.0); v = float(y2)  # bottom-centre
            cal = self.calib.get(cid, {})
            world = _project(self.cam_models[cid], u, v, self.contact_z,
                             float(cal.get("intercept_m", 0.0)), float(cal.get("slope_per_m", 0.0)))
            if world is None:
                continue
            now = self.get_clock().now()
            m = PoseWithCovarianceStamped()
            m.header.stamp = now.to_msg(); m.header.frame_id = self.frame_id
            m.pose.pose.position.x = float(world[0]); m.pose.pose.position.y = float(world[1])
            m.pose.pose.orientation.w = 1.0
            cov = [0.0] * 36; cov[0] = self.rvar; cov[7] = self.rvar; cov[35] = 1.0e6
            m.pose.covariance = cov
            self.pub.publish(m)
            self._counts[cid] += 1
            if sum(self._counts.values()) % 25 == 0:
                self.get_logger().info(f"handover counts {self._counts} belief=({bx:.1f},{by:.1f}) -> {cid}")
            return  # one correction per cycle


def main(args=None) -> int:
    rclpy.init(args=args)
    node = ScheduledCameraDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    main()
