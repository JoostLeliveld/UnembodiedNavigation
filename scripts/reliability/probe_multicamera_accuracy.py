#!/usr/bin/env python3
"""Live Gazebo probe for extension two-camera detector/calibration accuracy.

This is an evaluation-only utility. It teleports the robot through a small set
of poses in the extension world, compares each detector's selected bbox-bottom
pixel with the projected robot bbox from Gazebo truth, and writes a compact JSON
report. The script never feeds truth back into runtime state, planning, or model
features.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
for rel in (
    "src/experiments",
    "src/perception",
    "src/unav_common",
):
    path = str((REPO_ROOT / rel).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage

from experiments.core.world_profiles import (
    compute_look_at_from_pose,
    parse_camera_pose_from_world,
)
from perception.core.detection_diagnostics import diagnostics_from_message
from unav_common.camera_model import ObliqueCameraModel


DEFAULT_POSES = (
    (-3.60, -3.60, 0.00, "camera_A_only_marker"),
    (0.00, -0.30, 0.00, "overlap_marker"),
    (2.70, -0.60, 0.00, "handover_open"),
    (4.20, 1.60, 1.57, "camera_B_only_marker"),
)


@dataclass(frozen=True)
class CameraProbeResult:
    camera_id: str
    pose_label: str
    commanded_x_m: float
    commanded_y_m: float
    commanded_yaw_rad: float
    truth_x_m: float | None
    truth_y_m: float | None
    truth_yaw_rad: float | None
    truth_dt_s: float | None
    detected: bool
    score: float | None
    selected_u_px: float | None
    selected_v_px: float | None
    expected_u_px: float | None
    expected_v_px: float | None
    selected_pixel_error_px: float | None
    bbox_iou: float | None
    bbox_area_px: float | None
    expected_bbox_area_px: float | None
    homography_xy_error_m: float | None
    measurement_age_s: float | None
    frame_age_at_publish_s: float | None
    diagnostic_stamp_s: float | None
    status: str


class MultiCameraAccuracyProbe(Node):
    def __init__(
        self,
        *,
        world_name: str,
        cameras: dict[str, ObliqueCameraModel],
        settle_s: float,
        sample_timeout_s: float,
        robot_z_m: float,
        robot_box_length_m: float,
        robot_box_width_m: float,
        robot_box_height_m: float,
    ) -> None:
        super().__init__("multicamera_accuracy_probe")
        self.service_name = f"/world/{world_name}/set_pose"
        self.client = self.create_client(SetEntityPose, self.service_name)
        self.zero_cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.cameras = dict(cameras)
        self.settle_s = float(settle_s)
        self.sample_timeout_s = float(sample_timeout_s)
        self.robot_z_m = float(robot_z_m)
        self.robot_box_length_m = float(robot_box_length_m)
        self.robot_box_width_m = float(robot_box_width_m)
        self.robot_box_height_m = float(robot_box_height_m)
        self.clock_s = math.nan
        self.truth_history: list[tuple[float, float, float, float]] = []
        self.last_diag: dict[str, tuple[float, dict[str, float]]] = {}
        clock_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Clock, "/clock", self._clock_cb, clock_qos)
        self.create_subscription(TFMessage, "/ground_truth_tf", self._truth_cb, 10)
        for camera_id in sorted(self.cameras):
            topic = f"/perception/{camera_id}/detection_diagnostics"
            self.create_subscription(
                Float64MultiArray,
                topic,
                lambda msg, cid=camera_id: self._diag_cb(cid, msg),
                10,
            )

    def wait_for_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + max(float(timeout_s), 0.0)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if (
                self.client.wait_for_service(timeout_sec=0.05)
                and self.truth_history
                and math.isfinite(self.clock_s)
                and self.clock_s > 0.0
            ):
                return
        raise RuntimeError(
            f"Timed out waiting for {self.service_name}, /clock, and /ground_truth_tf. "
            "Launch warehouse_multicamera_extension.launch.py first."
        )

    def probe_pose(
        self,
        *,
        x_m: float,
        y_m: float,
        yaw_rad: float,
        label: str,
    ) -> list[CameraProbeResult]:
        before = {camera_id: value[0] for camera_id, value in self.last_diag.items()}
        self._set_robot_pose(x_m=x_m, y_m=y_m, yaw_rad=yaw_rad)
        settle_end = time.monotonic() + max(self.settle_s, 0.0)
        while rclpy.ok() and time.monotonic() < settle_end:
            rclpy.spin_once(self, timeout_sec=0.05)
        min_diag_stamp_s = float(self.clock_s) if math.isfinite(self.clock_s) else -math.inf

        deadline = time.monotonic() + max(self.sample_timeout_s, 0.1)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if all(
                camera_id in self.last_diag
                and self.last_diag[camera_id][0] > before.get(camera_id, -math.inf)
                and self.last_diag[camera_id][0] >= min_diag_stamp_s
                for camera_id in self.cameras
            ):
                break

        results = []
        for camera_id, camera in sorted(self.cameras.items()):
            stamp, diag = self.last_diag.get(camera_id, (math.nan, {}))
            results.append(
                self._score_camera(
                    camera_id=camera_id,
                    camera=camera,
                    pose_label=label,
                    commanded_x_m=x_m,
                    commanded_y_m=y_m,
                    commanded_yaw_rad=yaw_rad,
                    diag_stamp=stamp,
                    diag=diag,
                    previous_stamp=before.get(camera_id, -math.inf),
                    min_diag_stamp_s=min_diag_stamp_s,
                )
            )
        return results

    def _publish_zero_cmd(self, repetitions: int = 3) -> None:
        msg = Twist()
        for _ in range(max(int(repetitions), 1)):
            self.zero_cmd_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.01)

    def _set_robot_pose(self, *, x_m: float, y_m: float, yaw_rad: float) -> None:
        self._publish_zero_cmd()
        req = SetEntityPose.Request()
        req.entity = Entity(name="turtlebot3")
        req.entity.type = Entity.MODEL
        req.pose.position.x = float(x_m)
        req.pose.position.y = float(y_m)
        req.pose.position.z = float(self.robot_z_m)
        half = 0.5 * float(yaw_rad)
        req.pose.orientation.z = math.sin(half)
        req.pose.orientation.w = math.cos(half)
        future = self.client.call_async(req)
        deadline = time.monotonic() + 10.0
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.monotonic() > deadline:
                raise RuntimeError(f"Timed out waiting for {self.service_name}")
        result = future.result()
        if result is None or not bool(getattr(result, "success", False)):
            raise RuntimeError(f"Set pose failed at x={x_m:.3f}, y={y_m:.3f}, yaw={yaw_rad:.3f}")
        self._publish_zero_cmd()

    def _clock_cb(self, msg: Clock) -> None:
        self.clock_s = float(msg.clock.sec) + float(msg.clock.nanosec) * 1.0e-9

    def _truth_cb(self, msg: TFMessage) -> None:
        for tr in msg.transforms:
            if tr.child_frame_id != "turtlebot3":
                continue
            stamp = float(tr.header.stamp.sec) + float(tr.header.stamp.nanosec) * 1.0e-9
            if not math.isfinite(stamp) or stamp <= 0.0:
                stamp = float(self.get_clock().now().nanoseconds) * 1.0e-9
            q = tr.transform.rotation
            yaw = _yaw_from_quaternion(q.x, q.y, q.z, q.w)
            self.truth_history.append(
                (
                    stamp,
                    float(tr.transform.translation.x),
                    float(tr.transform.translation.y),
                    yaw,
                )
            )
        if len(self.truth_history) > 400:
            del self.truth_history[: len(self.truth_history) - 400]

    def _diag_cb(self, camera_id: str, msg: Float64MultiArray) -> None:
        diag = diagnostics_from_message(msg)
        stamp = float(diag.get("stamp", math.nan))
        self.last_diag[camera_id] = (stamp, diag)

    def _latest_truth(self) -> tuple[float, float, float, float] | None:
        if not self.truth_history:
            return None
        return self.truth_history[-1]

    def _score_camera(
        self,
        *,
        camera_id: str,
        camera: ObliqueCameraModel,
        pose_label: str,
        commanded_x_m: float,
        commanded_y_m: float,
        commanded_yaw_rad: float,
        diag_stamp: float,
        diag: dict[str, float],
        previous_stamp: float,
        min_diag_stamp_s: float,
    ) -> CameraProbeResult:
        truth = self._latest_truth()
        truth_stamp = truth_x = truth_y = truth_yaw = None
        truth_dt = None
        if truth is not None:
            truth_stamp, truth_x, truth_y, truth_yaw = truth
            if (
                math.isfinite(float(truth_stamp))
                and math.isfinite(float(diag_stamp))
                and abs(float(truth_stamp) - float(diag_stamp)) < 60.0
            ):
                truth_dt = abs(float(truth_stamp) - float(diag_stamp))
        detected = bool(diag) and bool(diag.get("yolo_detected_after_threshold", 0.0) >= 0.5)
        new_sample = bool(
            math.isfinite(diag_stamp)
            and diag_stamp > previous_stamp
            and diag_stamp >= min_diag_stamp_s
        )
        status = "ok" if detected and new_sample and truth is not None else "missing_detection_or_truth"

        expected_bbox = None
        selected_error = None
        expected_uv = (None, None)
        expected_area = None
        iou = None
        xy_error = None
        if truth is not None:
            expected_bbox = _project_robot_bbox(
                camera,
                x=float(truth_x),
                y=float(truth_y),
                yaw=float(truth_yaw),
                z=float(self.robot_z_m),
                box_length=float(self.robot_box_length_m),
                box_width=float(self.robot_box_width_m),
                box_height=float(self.robot_box_height_m),
            )
        if expected_bbox is not None:
            expected_uv = (
                0.5 * (expected_bbox[0] + expected_bbox[2]),
                expected_bbox[3],
            )
            expected_area = _bbox_area(expected_bbox)
        selected_u = _finite_or_none(diag.get("u_mid")) if diag else None
        selected_v = _finite_or_none(diag.get("v_mid")) if diag else None
        if selected_u is not None and selected_v is not None and expected_uv[0] is not None:
            selected_error = math.hypot(selected_u - expected_uv[0], selected_v - expected_uv[1])
            xy = camera.pixel_to_world(selected_u, selected_v)
            if xy is not None and truth_x is not None and truth_y is not None:
                xy_error = math.hypot(xy[0] - float(truth_x), xy[1] - float(truth_y))

        detected_bbox = _detected_bbox(diag) if diag else None
        if detected_bbox is not None and expected_bbox is not None:
            iou = _bbox_iou(detected_bbox, expected_bbox)

        return CameraProbeResult(
            camera_id=camera_id,
            pose_label=pose_label,
            commanded_x_m=float(commanded_x_m),
            commanded_y_m=float(commanded_y_m),
            commanded_yaw_rad=float(commanded_yaw_rad),
            truth_x_m=truth_x,
            truth_y_m=truth_y,
            truth_yaw_rad=truth_yaw,
            truth_dt_s=truth_dt,
            detected=bool(detected and new_sample),
            score=_finite_or_none(diag.get("yolo_score_selected")) if diag else None,
            selected_u_px=selected_u,
            selected_v_px=selected_v,
            expected_u_px=expected_uv[0],
            expected_v_px=expected_uv[1],
            selected_pixel_error_px=selected_error,
            bbox_iou=iou,
            bbox_area_px=_bbox_area(detected_bbox) if detected_bbox is not None else None,
            expected_bbox_area_px=expected_area,
            homography_xy_error_m=xy_error,
            measurement_age_s=_finite_or_none(diag.get("yolo_latency_s")) if diag else None,
            frame_age_at_publish_s=_finite_or_none(diag.get("frame_age_at_publish_s")) if diag else None,
            diagnostic_stamp_s=diag_stamp if math.isfinite(diag_stamp) else None,
            status=status,
        )


def _camera_from_world(world_sdf: Path, model_name: str) -> ObliqueCameraModel:
    pose = parse_camera_pose_from_world(str(world_sdf), camera_model=model_name)
    cam_pos = pose[:3]
    look_at = compute_look_at_from_pose(cam_pos, pose[3], pose[4], pose[5])
    return ObliqueCameraModel(
        cam_pos=cam_pos,
        look_at=look_at,
        img_width=1280,
        img_height=720,
        fov_h_rad=1.5708,
    )


def _project_world_point(camera: ObliqueCameraModel, xyz: np.ndarray) -> np.ndarray | None:
    cam_pt = camera.R @ (np.asarray(xyz, dtype=float) - camera.cam_pos)
    if cam_pt[2] <= 1e-6:
        return None
    pixel_h = camera.K @ cam_pt
    return np.asarray([pixel_h[0] / pixel_h[2], pixel_h[1] / pixel_h[2]], dtype=float)


def _project_robot_bbox(
    camera: ObliqueCameraModel,
    *,
    x: float,
    y: float,
    yaw: float,
    z: float,
    box_length: float,
    box_width: float,
    box_height: float,
) -> tuple[float, float, float, float] | None:
    hx = 0.5 * float(box_length)
    hy = 0.5 * float(box_width)
    local = np.asarray(
        [
            [-hx, -hy, 0.0],
            [-hx, hy, 0.0],
            [hx, -hy, 0.0],
            [hx, hy, 0.0],
            [-hx, -hy, box_height],
            [-hx, hy, box_height],
            [hx, -hy, box_height],
            [hx, hy, box_height],
        ],
        dtype=float,
    )
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    rot = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    world = (rot @ local.T).T
    world[:, 0] += float(x)
    world[:, 1] += float(y)
    world[:, 2] += float(z)
    pts = []
    for corner in world:
        uv = _project_world_point(camera, corner)
        if uv is None:
            return None
        pts.append(uv)
    arr = np.asarray(pts, dtype=float)
    return (
        float(np.min(arr[:, 0])),
        float(np.min(arr[:, 1])),
        float(np.max(arr[:, 0])),
        float(np.max(arr[:, 1])),
    )


def _detected_bbox(diag: dict[str, float]) -> tuple[float, float, float, float] | None:
    values = (
        _finite_or_none(diag.get("bbox_xmin")),
        _finite_or_none(diag.get("bbox_ymin")),
        _finite_or_none(diag.get("bbox_xmax")),
        _finite_or_none(diag.get("bbox_ymax")),
    )
    if any(value is None for value in values):
        return None
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _bbox_area(bbox: tuple[float, float, float, float] | None) -> float | None:
    if bbox is None:
        return None
    return max(float(bbox[2] - bbox[0]), 0.0) * max(float(bbox[3] - bbox[1]), 0.0)


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    inter = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
    union = (_bbox_area(a) or 0.0) + (_bbox_area(b) or 0.0) - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _finite_or_none(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _parse_pose(text: str) -> tuple[float, float, float, str]:
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError("pose must be x,y,yaw[,label]")
    try:
        x = float(parts[0])
        y = float(parts[1])
        yaw = float(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pose must start with numeric x,y,yaw") from exc
    label = parts[3] if len(parts) == 4 and parts[3] else f"pose_{x:.2f}_{y:.2f}"
    return x, y, yaw, label


def _stats(values: Iterable[float | None]) -> dict[str, float | int | None]:
    vals = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _summarize(results: list[CameraProbeResult]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for camera_id in sorted({result.camera_id for result in results}):
        rows = [result for result in results if result.camera_id == camera_id]
        out[camera_id] = {
            "samples": len(rows),
            "detections": sum(1 for row in rows if row.detected),
            "detection_rate": sum(1 for row in rows if row.detected) / max(len(rows), 1),
            "selected_pixel_error_px": _stats(row.selected_pixel_error_px for row in rows if row.detected),
            "bbox_iou": _stats(row.bbox_iou for row in rows if row.detected),
            "homography_xy_error_m": _stats(row.homography_xy_error_m for row in rows if row.detected),
            "frame_age_at_publish_s": _stats(row.frame_age_at_publish_s for row in rows if row.detected),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--world-sdf",
        default=str((REPO_ROOT / "src" / "sim" / "gazebo_worlds" / "worlds" / "warehouse_multicamera_extension.world.sdf").resolve()),
    )
    parser.add_argument("--world-name", default="warehouse_multicamera_extension")
    parser.add_argument("--camera", action="append", default=[], help="camera_id=model_name, repeatable")
    parser.add_argument("--pose", action="append", type=_parse_pose, default=[], help="x,y,yaw[,label], repeatable")
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--sample-timeout-s", type=float, default=10.0)
    parser.add_argument("--ready-timeout-s", type=float, default=20.0)
    parser.add_argument("--robot-z", type=float, default=0.05)
    parser.add_argument("--robot-box-length", type=float, default=0.22)
    parser.add_argument("--robot-box-width", type=float, default=0.22)
    parser.add_argument("--robot-box-height", type=float, default=0.20)
    parser.add_argument("--out", default="", help="Optional JSON output path")
    args = parser.parse_args()

    camera_specs = args.camera or ["camera_A=external_camera", "camera_B=external_camera_b"]
    cameras: dict[str, ObliqueCameraModel] = {}
    world_sdf = Path(args.world_sdf).expanduser().resolve()
    for spec in camera_specs:
        if "=" not in spec:
            parser.error(f"--camera must be camera_id=model_name, got {spec!r}")
        camera_id, model_name = [part.strip() for part in spec.split("=", 1)]
        cameras[camera_id] = _camera_from_world(world_sdf, model_name)

    poses = tuple(args.pose) if args.pose else DEFAULT_POSES
    rclpy.init()
    node = MultiCameraAccuracyProbe(
        world_name=str(args.world_name),
        cameras=cameras,
        settle_s=float(args.settle_s),
        sample_timeout_s=float(args.sample_timeout_s),
        robot_z_m=float(args.robot_z),
        robot_box_length_m=float(args.robot_box_length),
        robot_box_width_m=float(args.robot_box_width),
        robot_box_height_m=float(args.robot_box_height),
    )
    results: list[CameraProbeResult] = []
    try:
        node.wait_for_ready(timeout_s=float(args.ready_timeout_s))
        for x_m, y_m, yaw_rad, label in poses:
            results.extend(
                node.probe_pose(
                    x_m=x_m,
                    y_m=y_m,
                    yaw_rad=yaw_rad,
                    label=label,
                )
            )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    report = {
        "world_sdf": str(world_sdf),
        "world_name": str(args.world_name),
        "poses": [
            {"x_m": x_m, "y_m": y_m, "yaw_rad": yaw_rad, "label": label}
            for x_m, y_m, yaw_rad, label in poses
        ],
        "summary": _summarize(results),
        "results": [asdict(result) for result in results],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
