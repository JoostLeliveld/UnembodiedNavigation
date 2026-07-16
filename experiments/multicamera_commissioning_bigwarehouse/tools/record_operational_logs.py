#!/usr/bin/env python3
"""Record operational multi-camera commissioning logs in ``warehouse_full_4cam``.

The recorder subscribes only to the detector's JSON ``CameraObservation``
contracts and noisy odometry.  It projects valid bottom-centre pixels through
each known camera calibration into the shared warehouse frame and writes the
CSV layout consumed by ``reliability_tools export-multicamera``.  It never
subscribes to Gazebo ground truth or writes evaluation labels.

Example (run after the study launch):

    source install/setup.bash
    python3 experiments/multicamera_commissioning_bigwarehouse/tools/record_operational_logs.py \
      --out-dir logs/multicamera_commissioning_bigwarehouse/run_001/raw \
      --duration-s 120 \
      --odom-origin-x-m=-1.8 --odom-origin-y-m=-5.4 --odom-origin-yaw-rad=1.5708

"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any


REPO = Path(__file__).resolve().parents[3]
for relative in ("src/reliability", "src/unav_common"):
    source = str(REPO / relative)
    if source not in sys.path:
        sys.path.insert(0, source)

from reliability import CameraObservation  # noqa: E402
from reliability.projection import (  # noqa: E402
    camera_model_from_world as _library_camera_model_from_world,
    load_projection_calibration,
    project_observation_to_world as _library_project_observation_to_world,
)
from unav_common.camera_model import ObliqueCameraModel  # noqa: E402

try:  # Keep calibration helpers importable for non-ROS asset tests.
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from std_msgs.msg import String
except ImportError:  # pragma: no cover - only used in non-ROS tooling contexts
    rclpy = None
    Odometry = Any
    String = Any
    Node = object
    Parameter = Any


DEFAULT_WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
DEFAULT_CAMERA_TOPICS = {
    "camera_A": "/perception/camera_observation/camera_A",
    "camera_B": "/perception/camera_observation/camera_B",
    "camera_C": "/perception/camera_observation/camera_C",
    "camera_D": "/perception/camera_observation/camera_D",
}
DEFAULT_CAMERA_MODELS = {
    "camera_A": "external_camera",
    "camera_B": "external_camera_b",
    "camera_C": "external_camera_c",
    "camera_D": "external_camera_d",
}
PERCEPTION_FIELDS = (
    "diag_stamp",
    "detected",
    "yolo_detected_after_threshold",
    "yolo_score_raw",
    "yolo_score_selected",
    "obs_u",
    "obs_v",
    "pixel_pose_available",
    "pixel_pose_fresh",
    "pixel_pose_age_s",
    "pred_world_x",
    "pred_world_y",
    "camera_id",
    "calibration_id",
    "image_frame_id",
)
ODOMETRY_FIELDS = (
    "stamp",
    "odom_noisy_x",
    "odom_noisy_y",
    "odom_noisy_cov_xx",
    "odom_noisy_cov_xy",
    "odom_noisy_cov_yy",
)


# Projection now lives in reliability.projection (single implementation shared
# with the live camera manager node); the recorder keeps these names as thin
# aliases for existing importers/tests.
camera_model_from_world = _library_camera_model_from_world
project_observation_to_world = _library_project_observation_to_world


class OperationalLogRecorder(Node):
    """ROS collector for per-camera operational records and noisy odometry."""

    def __init__(
        self,
        *,
        out_dir: Path,
        camera_topics: dict[str, str],
        camera_models: dict[str, ObliqueCameraModel],
        odom_topic: str,
        contact_z_m: float,
        fresh_age_s: float,
        odom_origin_x_m: float,
        odom_origin_y_m: float,
        odom_origin_yaw_rad: float,
        use_sim_time: bool,
        projection_calibrations: dict[str, dict[str, float]] | None = None,
    ) -> None:
        super().__init__(
            "bigwarehouse_multicamera_operational_recorder",
            parameter_overrides=[Parameter("use_sim_time", value=bool(use_sim_time))],
        )
        self.out_dir = out_dir
        self.camera_models = dict(camera_models)
        self.projection_calibrations = dict(projection_calibrations or {})
        self.contact_z_m = float(contact_z_m)
        self.fresh_age_s = float(fresh_age_s)
        self.odom_origin_x_m = float(odom_origin_x_m)
        self.odom_origin_y_m = float(odom_origin_y_m)
        self.odom_origin_yaw_rad = float(odom_origin_yaw_rad)
        self._origin_cos = math.cos(self.odom_origin_yaw_rad)
        self._origin_sin = math.sin(self.odom_origin_yaw_rad)
        self.latest_odom_xy = (0.0, 0.0)
        self.latest_odom_stamp = math.nan
        self.handles: dict[str, Any] = {}
        self.writers: dict[str, csv.DictWriter] = {}
        self.counts = {camera_id: 0 for camera_id in camera_topics}

        out_dir.mkdir(parents=True, exist_ok=True)
        for camera_id, topic in sorted(camera_topics.items()):
            if camera_id not in camera_models:
                raise RuntimeError(f"No calibration model supplied for {camera_id}")
            handle = (out_dir / f"{camera_id}_perception.csv").open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(handle, fieldnames=PERCEPTION_FIELDS)
            writer.writeheader()
            self.handles[camera_id] = handle
            self.writers[camera_id] = writer
            self.create_subscription(String, topic, self._camera_callback(camera_id), 20)
            self.get_logger().info(f"recording {camera_id}: {topic}")

        self.odom_handle = (out_dir / "experiment.csv").open("w", newline="", encoding="utf-8")
        self.odom_writer = csv.DictWriter(self.odom_handle, fieldnames=ODOMETRY_FIELDS)
        self.odom_writer.writeheader()
        self.create_subscription(Odometry, odom_topic, self._odom_callback, 50)
        self.get_logger().info(
            f"recording odometry: {odom_topic} local->warehouse origin="
            f"({self.odom_origin_x_m:.3f}, {self.odom_origin_y_m:.3f}, "
            f"yaw={self.odom_origin_yaw_rad:.3f})"
        )

    def _odom_to_warehouse(self, local_x: float, local_y: float) -> tuple[float, float]:
        """Map Gazebo's spawn-local odometry into the fixed warehouse frame.

        The origin is a documented launch condition, not a simulator pose or
        evaluation label.  It is necessary because the DiffDrive plugin resets
        its odometry to (0, 0, 0) at every arbitrary robot spawn.
        """

        return (
            self.odom_origin_x_m + self._origin_cos * local_x - self._origin_sin * local_y,
            self.odom_origin_y_m + self._origin_sin * local_x + self._origin_cos * local_y,
        )

    def _odom_covariance_to_warehouse(
        self,
        local_xx: float,
        local_xy: float,
        local_yy: float,
    ) -> tuple[float, float, float]:
        """Rotate a planar local-odom covariance into the warehouse frame."""

        if not all(math.isfinite(value) for value in (local_xx, local_xy, local_yy)):
            return math.nan, math.nan, math.nan
        c = self._origin_cos
        s = self._origin_sin
        world_xx = c * c * local_xx - 2.0 * s * c * local_xy + s * s * local_yy
        world_xy = s * c * local_xx + (c * c - s * s) * local_xy - s * c * local_yy
        world_yy = s * s * local_xx + 2.0 * s * c * local_xy + c * c * local_yy
        return world_xx, world_xy, world_yy

    def _odom_callback(self, message: Odometry) -> None:
        stamp = _stamp_s(message.header.stamp)
        local_x = float(message.pose.pose.position.x)
        local_y = float(message.pose.pose.position.y)
        self.latest_odom_xy = self._odom_to_warehouse(local_x, local_y)
        self.latest_odom_stamp = stamp
        local_covariance = message.pose.covariance
        local_xx = float(local_covariance[0]) if len(local_covariance) >= 8 else math.nan
        local_xy = float(local_covariance[1]) if len(local_covariance) >= 8 else math.nan
        local_yy = float(local_covariance[7]) if len(local_covariance) >= 8 else math.nan
        world_xx, world_xy, world_yy = self._odom_covariance_to_warehouse(
            local_xx,
            local_xy,
            local_yy,
        )
        self.odom_writer.writerow(
            {
                "stamp": _csv_float(stamp),
                "odom_noisy_x": _csv_float(self.latest_odom_xy[0]),
                "odom_noisy_y": _csv_float(self.latest_odom_xy[1]),
                "odom_noisy_cov_xx": _csv_float(world_xx),
                "odom_noisy_cov_xy": _csv_float(world_xy),
                "odom_noisy_cov_yy": _csv_float(world_yy),
            }
        )
        self.odom_handle.flush()

    def _camera_callback(self, expected_camera_id: str):
        def callback(message: String) -> None:
            try:
                observation = CameraObservation.from_json(message.data)
            except Exception as exc:  # noqa: BLE001 - malformed input must not end a collection run
                self.get_logger().warn(f"{expected_camera_id}: rejected camera observation: {exc}")
                return
            if observation.camera_id != expected_camera_id:
                self.get_logger().warn(
                    f"{expected_camera_id}: ignoring observation labelled {observation.camera_id!r}"
                )
                return
            world_xy = project_observation_to_world(
                observation,
                self.camera_models[expected_camera_id],
                contact_z_m=self.contact_z_m,
                along_bearing_offset_m=self.projection_calibrations.get(
                    expected_camera_id, {}
                ).get("intercept_m", 0.0),
                along_bearing_slope_per_m=self.projection_calibrations.get(
                    expected_camera_id, {}
                ).get("slope_per_m", 0.0),
            )
            fresh = observation.measurement_age_s <= self.fresh_age_s
            u, v = observation.pixel_uv if observation.pixel_uv is not None else (math.nan, math.nan)
            self.writers[expected_camera_id].writerow(
                {
                    "diag_stamp": _csv_float(observation.timestamp_s),
                    "detected": int(observation.detection_valid),
                    "yolo_detected_after_threshold": int(observation.detection_valid),
                    "yolo_score_raw": _csv_float(observation.detector_score),
                    "yolo_score_selected": _csv_float(observation.detector_score),
                    "obs_u": _csv_float(u),
                    "obs_v": _csv_float(v),
                    "pixel_pose_available": int(world_xy is not None),
                    "pixel_pose_fresh": int(fresh),
                    "pixel_pose_age_s": _csv_float(observation.measurement_age_s),
                    "pred_world_x": _csv_float(world_xy[0]) if world_xy is not None else "",
                    "pred_world_y": _csv_float(world_xy[1]) if world_xy is not None else "",
                    "camera_id": observation.camera_id,
                    "calibration_id": observation.calibration_id,
                    "image_frame_id": observation.image_frame_id,
                }
            )
            self.handles[expected_camera_id].flush()
            self.counts[expected_camera_id] += 1

        return callback

    def close(self) -> None:
        for camera_id, handle in self.handles.items():
            handle.close()
            self.get_logger().info(f"{camera_id}: wrote {self.counts[camera_id]} operational observations")
        self.odom_handle.close()


def _stamp_s(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def _csv_float(value: float) -> str:
    number = float(value)
    return f"{number:.9f}" if math.isfinite(number) else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--world-sdf", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--odom-topic", default="/odom_noisy")
    parser.add_argument("--contact-z-m", type=float, default=0.05)
    parser.add_argument("--fresh-age-s", type=float, default=0.15)
    parser.add_argument("--odom-origin-x-m", type=float, required=True)
    parser.add_argument("--odom-origin-y-m", type=float, required=True)
    parser.add_argument("--odom-origin-yaw-rad", type=float, required=True)
    parser.add_argument("--use-sim-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--projection-calibration",
        type=Path,
        default=None,
        help=(
            "Optional projection_calibration.json with per-camera along-bearing "
            "offsets (commissioning constants; corrects the near-edge box-bottom "
            "pull toward the camera)"
        ),
    )
    args = parser.parse_args()
    if rclpy is None:
        raise RuntimeError("rclpy is required; source the ROS workspace before running this recorder")
    if args.contact_z_m < 0.0 or args.fresh_age_s < 0.0:
        parser.error("--contact-z-m and --fresh-age-s must be non-negative")
    if not all(
        math.isfinite(value)
        for value in (args.odom_origin_x_m, args.odom_origin_y_m, args.odom_origin_yaw_rad)
    ):
        parser.error("the documented odometry-origin pose must be finite")

    world_sdf = args.world_sdf.expanduser().resolve()
    camera_models = {
        camera_id: camera_model_from_world(world_sdf, include_name=model_name)
        for camera_id, model_name in DEFAULT_CAMERA_MODELS.items()
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "world_sdf": str(world_sdf),
        "camera_topics": DEFAULT_CAMERA_TOPICS,
        "camera_models": DEFAULT_CAMERA_MODELS,
        "odom_topic": str(args.odom_topic),
        "odom_origin_warehouse": {
            "x_m": float(args.odom_origin_x_m),
            "y_m": float(args.odom_origin_y_m),
            "yaw_rad": float(args.odom_origin_yaw_rad),
            "source": "documented_launch_spawn_not_ground_truth",
        },
        "contact_z_m": float(args.contact_z_m),
        "fresh_age_s": float(args.fresh_age_s),
        "duration_clock": "simulation" if args.use_sim_time else "wall",
        "odom_covariance": {
            "columns": ["odom_noisy_cov_xx", "odom_noisy_cov_xy", "odom_noisy_cov_yy"],
            "frame": "warehouse",
            "source": "published_noisy_odometry_pose_covariance",
            "legacy_fallback": "empty covariance columns are handled explicitly by the GP-input adapter",
        },
        "contains_ground_truth": False,
    }
    projection_calibrations: dict[str, dict[str, float]] = {}
    if args.projection_calibration is not None:
        projection_calibrations = load_projection_calibration(args.projection_calibration)
    manifest["projection_calibration"] = {
        "path": str(args.projection_calibration) if args.projection_calibration else None,
        "along_bearing_calibrations": projection_calibrations,
    }
    (args.out_dir / "operational_recording_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rclpy.init()
    node = OperationalLogRecorder(
        out_dir=args.out_dir,
        camera_topics=DEFAULT_CAMERA_TOPICS,
        camera_models=camera_models,
        odom_topic=str(args.odom_topic),
        contact_z_m=float(args.contact_z_m),
        fresh_age_s=float(args.fresh_age_s),
        odom_origin_x_m=float(args.odom_origin_x_m),
        odom_origin_y_m=float(args.odom_origin_y_m),
        odom_origin_yaw_rad=float(args.odom_origin_yaw_rad),
        use_sim_time=bool(args.use_sim_time),
        projection_calibrations=projection_calibrations,
    )
    try:
        if args.duration_s > 0.0:
            started_at_s: float | None = None
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.10)
                now_s = (
                    node.get_clock().now().nanoseconds * 1e-9
                    if args.use_sim_time
                    else time.monotonic()
                )
                # With simulation time, the node starts before the first
                # /clock message. Anchor the requested duration only once the
                # clock has begun rather than treating its later first value as
                # elapsed collection time.
                if started_at_s is None:
                    if not args.use_sim_time or now_s > 0.0:
                        started_at_s = now_s
                elif now_s - started_at_s >= args.duration_s:
                    break
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
