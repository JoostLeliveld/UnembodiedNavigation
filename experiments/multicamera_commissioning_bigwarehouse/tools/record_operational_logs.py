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
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[3]
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
for relative in ("src/reliability", "src/unav_common"):
    source = str(REPO / relative)
    if source not in sys.path:
        sys.path.insert(0, source)
perception_source = str(REPO / "src/perception")
if perception_source not in sys.path:
    sys.path.insert(0, perception_source)

from reliability import CameraObservation  # noqa: E402
from reliability.projection import (  # noqa: E402
    camera_model_from_world as _library_camera_model_from_world,
    project_observation_to_world as _library_project_observation_to_world,
)
from unav_common.camera_model import ObliqueCameraModel  # noqa: E402
from perception.core.four_camera_runtime_contract import (  # noqa: E402
    BATCHED_CAMERA_ORDER,
    BATCHED_RUNTIME_MODE,
    BATCHED_EXECUTABLE,
    EXECUTABLE_SOURCE_KEY,
    FOUR_CAMERA_BATCH_SOURCE_KEY,
    REQUIRED_SOURCE_KEYS,
    RUNTIME_CONTRACT_SOURCE_KEY,
    RUNTIME_CONTRACT_TOPIC,
    RuntimeContractError,
    YOLO_SELECTION_SOURCE_KEY,
    build_batched_runtime_contract,
    compare_batched_runtime_contract,
    runtime_contract_sha256,
)
import campaign_ledger  # noqa: E402

try:  # Keep calibration helpers importable for non-ROS asset tests.
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String
except ImportError:  # pragma: no cover - only used in non-ROS tooling contexts
    rclpy = None
    Odometry = Any
    String = Any
    Node = object
    Parameter = Any
    DurabilityPolicy = Any
    HistoryPolicy = Any
    QoSProfile = Any
    ReliabilityPolicy = Any


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
EVIDENCE_ROLES = ("fit", "qualification", "diagnostic")
DETECTOR_CAMERA_ORDER = tuple(DEFAULT_CAMERA_TOPICS)
BATCHED_FRAME_POLICY = "strict_new_unique_stamp_latest_only_no_reuse"
DEFAULT_DETECTOR_CONTRACT_TIMEOUT_S = 60.0
DETECTOR_RUNTIME_SOURCE_PATHS = {
    EXECUTABLE_SOURCE_KEY: REPO
    / "src/perception/perception/nodes/batched_four_camera_yolo_node.py",
    FOUR_CAMERA_BATCH_SOURCE_KEY: REPO / "src/perception/perception/core/four_camera_batch.py",
    RUNTIME_CONTRACT_SOURCE_KEY: REPO
    / "src/perception/perception/core/four_camera_runtime_contract.py",
    YOLO_SELECTION_SOURCE_KEY: REPO / "src/perception/perception/core/yolo_selection.py",
}


def _method_freeze(analysis_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(analysis_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"analysis plan must be a YAML mapping: {analysis_path}")
    freeze = payload.get("method_freeze")
    if not isinstance(freeze, dict) or not isinstance(freeze.get("source_files"), list):
        raise RuntimeError(f"analysis plan lacks method_freeze.source_files: {analysis_path}")
    source_sha256: dict[str, str] = {}
    study_dir = analysis_path.parent.parent
    for relative in freeze["source_files"]:
        source = (study_dir / str(relative)).resolve()
        if not source.is_file():
            raise RuntimeError(f"frozen method source is missing: {source}")
        source_sha256[str(relative)] = _sha256_file(source)
    return {
        "analysis_plan_id": str(payload.get("analysis_plan_id", "")),
        "analysis_plan_sha256": _sha256_file(analysis_path),
        "source_sha256": source_sha256,
    }
PERCEPTION_FIELDS = (
    "recorder_wall_elapsed_s",
    "diag_stamp",
    "detected",
    "yolo_detected_after_threshold",
    "yolo_score_raw",
    "yolo_score_selected",
    "obs_u",
    "obs_v",
    "selected_pixel_source",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "bbox_bottom_u",
    "bbox_bottom_v",
    "mask_available",
    "mask_area_px",
    "mask_polygon_points",
    "mask_bottom_u",
    "mask_bottom_v",
    "yolo_inference_wall_ms",
    "detector_callback_wall_ms",
    "image_receive_stamp_s",
    "inference_start_stamp_s",
    "inference_finish_stamp_s",
    "publish_stamp_s",
    "frame_age_at_publish_s",
    "availability_probability",
    "association_probability",
    "conditional_cov_uu",
    "conditional_cov_uv",
    "conditional_cov_vv",
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
    "recorder_wall_elapsed_s",
    "stamp",
    "odom_noisy_x",
    "odom_noisy_y",
    "odom_noisy_cov_xx",
    "odom_noisy_cov_xy",
    "odom_noisy_cov_yy",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_contract_qos() -> Any:
    """Use the detector's retained identity, never a best-effort sample."""

    if QoSProfile is Any:
        raise RuntimeError("rclpy QoS support is required for runtime-contract recording")
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def _frozen_detector_source_hashes(
    *, analysis_path: Path, method_freeze: dict[str, Any]
) -> dict[str, str]:
    """Map analysis-freeze entries to the source modules attested by the node."""

    source_sha256 = method_freeze.get("source_sha256")
    if not isinstance(source_sha256, dict):
        raise RuntimeError("method freeze lacks source SHA-256 values")
    study_dir = analysis_path.parent.parent
    resolved_freeze = {
        (study_dir / str(relative)).resolve(): str(digest)
        for relative, digest in source_sha256.items()
    }
    hashes: dict[str, str] = {}
    missing: list[str] = []
    for key in REQUIRED_SOURCE_KEYS:
        source_path = DETECTOR_RUNTIME_SOURCE_PATHS[key].resolve()
        digest = resolved_freeze.get(source_path)
        if digest is None:
            missing.append(str(source_path))
            continue
        if _sha256_file(source_path) != digest:
            raise RuntimeError(
                "detector source changed after the method freeze was created: "
                f"{source_path}"
            )
        hashes[key] = digest
    if missing:
        raise RuntimeError(
            "analysis method freeze omits runtime-contract source files: "
            + ", ".join(missing)
        )
    return hashes


def _config_model_path(config_path: Path, raw_model: Any) -> Path:
    if not isinstance(raw_model, str) or not raw_model.strip():
        raise RuntimeError("detector runtime config lacks runtime_pilot.model")
    candidate = Path(raw_model).expanduser()
    # Study configs conventionally use repository-relative asset paths.  Keep
    # absolute paths available for a future immutable successor config.
    if not candidate.is_absolute():
        candidate = REPO / candidate
    return candidate.resolve()


def _validate_batched_cli_against_runtime_config(
    *,
    config_path: Path,
    expected_contract: dict[str, Any],
    evidence_role: str = "diagnostic",
) -> dict[str, Any]:
    """Ensure frozen config and recorder CLI describe the same deployment.

    The CLI is useful for auditable shell invocations, but it is not allowed to
    silently override the versioned detector configuration.
    """

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("runtime_pilot"), dict):
        raise RuntimeError(f"detector runtime config lacks runtime_pilot mapping: {config_path}")
    runtime = payload["runtime_pilot"]
    expected = {
        "detector_mode": BATCHED_RUNTIME_MODE,
        "runtime_executable": BATCHED_EXECUTABLE,
        "launch_switch": "yolo_batched_four_camera",
        "model_format": "native_ultralytics",
        "model_instances": 1,
        "batch_size": 4,
        "camera_order": list(BATCHED_CAMERA_ORDER),
        "device": expected_contract["shared_device"],
        "cpu_threads": expected_contract["cpu_threads"],
        "cpu_interop_threads": expected_contract["interop_threads"],
        "opencv_threads": expected_contract["opencv_threads"],
        "max_batch_stamp_skew_s": expected_contract["max_batch_stamp_skew_s"],
        "max_pending_wall_s": expected_contract["max_pending_wall_s"],
        "frame_policy": expected_contract["frame_policy"],
        "inference_timing_semantics": expected_contract[
            "inference_timing_semantics"
        ],
        "fault_policy": expected_contract["fault_policy"],
        "synchronization_miss_policy": expected_contract[
            "synchronization_miss_policy"
        ],
        "masks": expected_contract["use_masks"],
        "confidence_threshold": expected_contract["confidence_threshold"],
        "iou_threshold": expected_contract["iou_threshold"],
    }
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual_value = runtime.get(key)
        if isinstance(expected_value, float):
            try:
                matches = math.isclose(
                    float(actual_value), expected_value, rel_tol=0.0, abs_tol=1.0e-12
                )
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual_value == expected_value
        if not matches:
            mismatches.append(key)
    configured_model = _config_model_path(config_path, runtime.get("model"))
    if configured_model != Path(expected_contract["model_path"]).resolve():
        mismatches.append("model")
    image_sizes = runtime.get("image_sizes")
    if not isinstance(image_sizes, list) or expected_contract["imgsz"] not in image_sizes:
        mismatches.append("image_sizes")
    if str(evidence_role) != "diagnostic":
        selection = runtime.get("evidence_selection")
        selected_imgsz = (
            selection.get("selected_image_size")
            if isinstance(selection, dict)
            else None
        )
        if (
            payload.get("lock_status") != "locked_before_evidence"
            or not isinstance(selection, dict)
            or selection.get("permitted") is not True
            or isinstance(selected_imgsz, bool)
            or not isinstance(selected_imgsz, int)
            or selected_imgsz != expected_contract["imgsz"]
            or image_sizes != [selected_imgsz]
        ):
            mismatches.append("evidence_selection")
    if mismatches:
        raise RuntimeError(
            "detector runtime CLI differs from frozen runtime config "
            f"{config_path}: " + ", ".join(sorted(set(mismatches)))
        )
    return runtime


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=REPO, check=False, capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


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
        fresh_age_s: float,
        odom_origin_x_m: float,
        odom_origin_y_m: float,
        odom_origin_yaw_rad: float,
        use_sim_time: bool,
        expected_detector_contract: dict[str, Any],
        max_stream_wall_age_s: float,
        detector_contract_topic: str = RUNTIME_CONTRACT_TOPIC,
    ) -> None:
        super().__init__(
            "bigwarehouse_multicamera_operational_recorder",
            parameter_overrides=[Parameter("use_sim_time", value=bool(use_sim_time))],
        )
        self.out_dir = out_dir
        self.camera_models = dict(camera_models)
        self.fresh_age_s = float(fresh_age_s)
        self.max_stream_wall_age_s = float(max_stream_wall_age_s)
        if (
            not math.isfinite(self.max_stream_wall_age_s)
            or self.max_stream_wall_age_s <= 0.0
        ):
            raise RuntimeError("max_stream_wall_age_s must be finite and positive")
        self.odom_origin_x_m = float(odom_origin_x_m)
        self.odom_origin_y_m = float(odom_origin_y_m)
        self.odom_origin_yaw_rad = float(odom_origin_yaw_rad)
        self._origin_cos = math.cos(self.odom_origin_yaw_rad)
        self._origin_sin = math.sin(self.odom_origin_yaw_rad)
        try:
            # Validate the frozen side before subscribing.  A malformed local
            # expectation is as unsafe as a malformed message from the live
            # detector and must not leave a recorder waiting indefinitely.
            self.expected_detector_contract = build_batched_runtime_contract(
                **{
                    "model_path": expected_detector_contract["model_path"],
                    "model_sha256": expected_detector_contract["model_sha256"],
                    "imgsz": expected_detector_contract["imgsz"],
                    "confidence_threshold": expected_detector_contract[
                        "confidence_threshold"
                    ],
                    "iou_threshold": expected_detector_contract["iou_threshold"],
                    "use_masks": expected_detector_contract["use_masks"],
                    "shared_device": expected_detector_contract["shared_device"],
                    "cpu_threads": expected_detector_contract["cpu_threads"],
                    "interop_threads": expected_detector_contract["interop_threads"],
                    "opencv_threads": expected_detector_contract["opencv_threads"],
                    "max_batch_stamp_skew_s": expected_detector_contract[
                        "max_batch_stamp_skew_s"
                    ],
                    "max_pending_wall_s": expected_detector_contract[
                        "max_pending_wall_s"
                    ],
                    "source_hashes": expected_detector_contract["source_hashes"],
                }
            )
        except (KeyError, RuntimeContractError, TypeError, ValueError) as exc:
            raise RuntimeError("invalid expected detector runtime contract") from exc
        self.expected_detector_contract_sha256 = runtime_contract_sha256(
            self.expected_detector_contract
        )
        self.detector_contract_topic = str(detector_contract_topic)
        if self.detector_contract_topic != RUNTIME_CONTRACT_TOPIC:
            raise RuntimeError(
                "operational recorder only accepts the fixed four-camera runtime-contract topic"
            )
        self.detector_contract_armed = False
        self.detector_contract_error: str | None = None
        self.observed_detector_contract: dict[str, Any] | None = None
        self.observed_detector_contract_sha256: str | None = None
        self.detector_contract_received_wall_s: float | None = None
        self.prearm_camera_messages = {camera_id: 0 for camera_id in camera_topics}
        self.prearm_odom_messages = 0
        self.last_camera_observation_wall_s: dict[str, float | None] = {
            camera_id: None for camera_id in camera_topics
        }
        self.last_odom_observation_wall_s: float | None = None
        self.stream_watchdog_failure: str | None = None
        self.latest_odom_xy = (0.0, 0.0)
        self.latest_odom_stamp = math.nan
        self.handles: dict[str, Any] = {}
        self.writers: dict[str, csv.DictWriter] = {}
        self.counts = {camera_id: 0 for camera_id in camera_topics}
        self.first_camera_stamp_s: dict[str, float | None] = {
            camera_id: None for camera_id in camera_topics
        }
        self.last_camera_stamp_s: dict[str, float | None] = {
            camera_id: None for camera_id in camera_topics
        }
        self.odom_count = 0
        self.first_odom_stamp_s: float | None = None
        self.last_odom_stamp_s: float | None = None
        self.first_sim_stamp_s: float | None = None
        self.last_sim_stamp_s: float | None = None
        self.started_wall_monotonic_s = time.monotonic()
        self.stream_integrity_errors: list[str] = []
        self.calibration_ids: dict[str, set[str]] = {camera_id: set() for camera_id in camera_topics}
        self.image_frame_ids: dict[str, set[str]] = {camera_id: set() for camera_id in camera_topics}
        self._part_to_final: list[tuple[Path, Path]] = []

        out_dir.mkdir(parents=True, exist_ok=True)
        # This subscription is deliberately established before observations
        # and odometry.  Transient-local QoS also lets a recorder started after
        # detector warm-up receive the retained identity before any route arms.
        self.create_subscription(
            String,
            self.detector_contract_topic,
            self._detector_contract_callback,
            _runtime_contract_qos(),
        )
        self.get_logger().info(
            "waiting for validated retained detector runtime contract: "
            f"{self.detector_contract_topic}"
        )
        for camera_id, topic in sorted(camera_topics.items()):
            if camera_id not in camera_models:
                raise RuntimeError(f"No calibration model supplied for {camera_id}")
            final_path = out_dir / f"{camera_id}_perception.csv"
            part_path = final_path.with_suffix(final_path.suffix + ".part")
            if final_path.exists() or part_path.exists():
                raise RuntimeError(f"Refusing to overwrite recorder output: {final_path}")
            handle = part_path.open("x", newline="", encoding="utf-8")
            writer = csv.DictWriter(handle, fieldnames=PERCEPTION_FIELDS)
            writer.writeheader()
            self.handles[camera_id] = handle
            self.writers[camera_id] = writer
            self._part_to_final.append((part_path, final_path))
            self.create_subscription(String, topic, self._camera_callback(camera_id), 20)
            self.get_logger().info(f"recording {camera_id}: {topic}")

        odom_final_path = out_dir / "experiment.csv"
        odom_part_path = odom_final_path.with_suffix(odom_final_path.suffix + ".part")
        if odom_final_path.exists() or odom_part_path.exists():
            raise RuntimeError(f"Refusing to overwrite recorder output: {odom_final_path}")
        self.odom_handle = odom_part_path.open("x", newline="", encoding="utf-8")
        self.odom_writer = csv.DictWriter(self.odom_handle, fieldnames=ODOMETRY_FIELDS)
        self.odom_writer.writeheader()
        self._part_to_final.append((odom_part_path, odom_final_path))
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

    def _detector_contract_callback(self, message: String) -> None:
        """Arm exactly once when the live model proves the frozen identity."""

        try:
            raw = getattr(message, "data", None)
            if not isinstance(raw, str):
                raise RuntimeContractError("runtime-contract message data is not a string")
            parsed = json.loads(raw)
            observed = compare_batched_runtime_contract(
                parsed, self.expected_detector_contract
            )
        except Exception as exc:  # A bad retained contract must stop the run, not be ignored.
            self.detector_contract_error = (
                "detector runtime contract rejected: " f"{type(exc).__name__}: {exc}"
            )
            self.get_logger().error(self.detector_contract_error)
            return

        observed_sha256 = runtime_contract_sha256(observed)
        if self.observed_detector_contract_sha256 is not None:
            if observed_sha256 != self.observed_detector_contract_sha256:
                self.detector_contract_error = (
                    "detector runtime contract changed after recorder arming"
                )
                self.get_logger().error(self.detector_contract_error)
            return
        self.observed_detector_contract = observed
        self.observed_detector_contract_sha256 = observed_sha256
        self.detector_contract_received_wall_s = time.monotonic()
        self.detector_contract_armed = True
        self.get_logger().info(
            "armed after validated detector runtime contract "
            f"sha256={observed_sha256}"
        )

    def stream_watchdog_failures(self, *, now_wall_s: float) -> list[str]:
        """Return post-arm stream gaps that make live evidence unsafe.

        The timer begins only after the retained detector contract is accepted.
        Pre-arm callbacks are deliberately ignored; a stream must then produce
        valid, recordable observations rather than merely keeping its ROS topic
        alive with malformed traffic.
        """

        if not self.detector_contract_armed:
            return []
        now = float(now_wall_s)
        if not math.isfinite(now):
            return ["stream watchdog received a non-finite wall clock"]
        armed_at = self.detector_contract_received_wall_s
        if armed_at is None or not math.isfinite(armed_at):
            return ["stream watchdog is armed without a finite contract receipt time"]
        failures: list[str] = []
        for camera_id in sorted(self.last_camera_observation_wall_s):
            last = self.last_camera_observation_wall_s[camera_id]
            reference = armed_at if last is None else last
            age = max(now - reference, 0.0)
            if age > self.max_stream_wall_age_s:
                phase = "since contract arming" if last is None else "since last valid observation"
                failures.append(
                    f"{camera_id} stream watchdog exceeded "
                    f"{self.max_stream_wall_age_s:.3f}s {phase} (age={age:.3f}s)"
                )
        odom_reference = (
            armed_at
            if self.last_odom_observation_wall_s is None
            else self.last_odom_observation_wall_s
        )
        odom_age = max(now - odom_reference, 0.0)
        if odom_age > self.max_stream_wall_age_s:
            phase = (
                "since contract arming"
                if self.last_odom_observation_wall_s is None
                else "since last valid observation"
            )
            failures.append(
                f"odometry stream watchdog exceeded {self.max_stream_wall_age_s:.3f}s "
                f"{phase} (age={odom_age:.3f}s)"
            )
        return failures

    def _odom_callback(self, message: Odometry) -> None:
        if not self.detector_contract_armed:
            self.prearm_odom_messages += 1
            return
        stamp = _stamp_s(message.header.stamp)
        if (
            not math.isfinite(stamp)
            or (
                self.last_odom_stamp_s is not None
                and stamp <= self.last_odom_stamp_s
            )
        ):
            problem = f"odometry timestamp is non-finite, duplicate, or out of order: {stamp!r}"
            self.stream_integrity_errors.append(problem)
            self.get_logger().error(problem)
            return
        self._record_sim_stamp(stamp)
        if math.isfinite(stamp):
            if self.first_odom_stamp_s is None:
                self.first_odom_stamp_s = float(stamp)
            self.last_odom_stamp_s = float(stamp)
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
                "recorder_wall_elapsed_s": _csv_float(time.monotonic() - self.started_wall_monotonic_s),
                "stamp": _csv_float(stamp),
                "odom_noisy_x": _csv_float(self.latest_odom_xy[0]),
                "odom_noisy_y": _csv_float(self.latest_odom_xy[1]),
                "odom_noisy_cov_xx": _csv_float(world_xx),
                "odom_noisy_cov_xy": _csv_float(world_xy),
                "odom_noisy_cov_yy": _csv_float(world_yy),
            }
        )
        self.odom_handle.flush()
        self.odom_count += 1
        self.last_odom_observation_wall_s = time.monotonic()

    def _record_sim_stamp(self, stamp: float) -> None:
        if not math.isfinite(stamp):
            return
        if self.first_sim_stamp_s is None:
            self.first_sim_stamp_s = float(stamp)
        self.last_sim_stamp_s = float(stamp)

    def _camera_callback(self, expected_camera_id: str):
        def callback(message: String) -> None:
            if not self.detector_contract_armed:
                self.prearm_camera_messages[expected_camera_id] += 1
                return
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
            previous_stamp = self.last_camera_stamp_s[expected_camera_id]
            if (
                not math.isfinite(observation.timestamp_s)
                or (
                    previous_stamp is not None
                    and observation.timestamp_s <= previous_stamp
                )
            ):
                problem = (
                    f"{expected_camera_id}: observation timestamp is non-finite, "
                    f"duplicate, or out of order: {observation.timestamp_s!r}"
                )
                self.stream_integrity_errors.append(problem)
                self.get_logger().error(problem)
                return
            self._record_sim_stamp(observation.timestamp_s)
            if math.isfinite(observation.timestamp_s):
                if self.first_camera_stamp_s[expected_camera_id] is None:
                    self.first_camera_stamp_s[expected_camera_id] = float(
                        observation.timestamp_s
                    )
                self.last_camera_stamp_s[expected_camera_id] = float(
                    observation.timestamp_s
                )
            # Inverse perspective mapping, no parameters (e7, 2026-08-07).
            world_xy = project_observation_to_world(
                observation, self.camera_models[expected_camera_id]
            )
            fresh = observation.measurement_age_s <= self.fresh_age_s
            u, v = observation.pixel_uv if observation.pixel_uv is not None else (math.nan, math.nan)
            bbox = observation.bbox_xyxy or (math.nan, math.nan, math.nan, math.nan)
            bbox_bottom = observation.bbox_bottom_uv or (math.nan, math.nan)
            mask_bottom = observation.mask_bottom_uv or (math.nan, math.nan)
            conditional_covariance = observation.conditional_cov_uv
            self.writers[expected_camera_id].writerow(
                {
                    "recorder_wall_elapsed_s": _csv_float(time.monotonic() - self.started_wall_monotonic_s),
                    "diag_stamp": _csv_float(observation.timestamp_s),
                    "detected": int(observation.detection_valid),
                    "yolo_detected_after_threshold": int(observation.detection_valid),
                    "yolo_score_raw": _csv_float(observation.detector_score_raw),
                    "yolo_score_selected": _csv_float(observation.detector_score),
                    "obs_u": _csv_float(u),
                    "obs_v": _csv_float(v),
                    "selected_pixel_source": observation.selected_pixel_source,
                    "bbox_xmin": _csv_float(bbox[0]),
                    "bbox_ymin": _csv_float(bbox[1]),
                    "bbox_xmax": _csv_float(bbox[2]),
                    "bbox_ymax": _csv_float(bbox[3]),
                    "bbox_bottom_u": _csv_float(bbox_bottom[0]),
                    "bbox_bottom_v": _csv_float(bbox_bottom[1]),
                    "mask_available": int(observation.mask_available),
                    "mask_area_px": _csv_float(observation.mask_area_px),
                    "mask_polygon_points": _csv_float(observation.mask_polygon_points),
                    "mask_bottom_u": _csv_float(mask_bottom[0]),
                    "mask_bottom_v": _csv_float(mask_bottom[1]),
                    "yolo_inference_wall_ms": _csv_float(observation.yolo_inference_wall_ms),
                    "detector_callback_wall_ms": _csv_float(observation.detector_callback_wall_ms),
                    "image_receive_stamp_s": _csv_float(observation.image_receive_stamp_s),
                    "inference_start_stamp_s": _csv_float(observation.inference_start_stamp_s),
                    "inference_finish_stamp_s": _csv_float(observation.inference_finish_stamp_s),
                    "publish_stamp_s": _csv_float(observation.publish_stamp_s),
                    "frame_age_at_publish_s": _csv_float(observation.frame_age_at_publish_s),
                    "availability_probability": _csv_float(observation.availability_probability),
                    "association_probability": _csv_float(observation.association_probability),
                    "conditional_cov_uu": _csv_float(conditional_covariance[0][0]),
                    "conditional_cov_uv": _csv_float(conditional_covariance[0][1]),
                    "conditional_cov_vv": _csv_float(conditional_covariance[1][1]),
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
            self.calibration_ids[expected_camera_id].add(str(observation.calibration_id))
            self.image_frame_ids[expected_camera_id].add(str(observation.image_frame_id))
            self.last_camera_observation_wall_s[expected_camera_id] = time.monotonic()

        return callback

    def close(self, *, finalize: bool) -> None:
        for camera_id, handle in self.handles.items():
            handle.close()
            self.get_logger().info(f"{camera_id}: wrote {self.counts[camera_id]} operational observations")
        self.odom_handle.close()
        if finalize:
            for part_path, final_path in self._part_to_final:
                os.replace(part_path, final_path)


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
    parser.add_argument(
        "--completion-manifest",
        type=Path,
        default=None,
        help="Stop successfully only after this JSON exists with status=completed.",
    )
    parser.add_argument(
        "--wall-timeout-s",
        type=float,
        default=0.0,
        help="Wall-clock deadman; 0 derives 10x simulation duration + 60 s when duration is set.",
    )
    parser.add_argument("--min-camera-rows", type=int, default=1)
    parser.add_argument("--min-odom-rows", type=int, default=1)
    parser.add_argument("--odom-topic", default="/odom_noisy")
    parser.add_argument("--fresh-age-s", type=float, default=0.15)
    parser.add_argument(
        "--max-stream-wall-age-s",
        type=float,
        default=3.0,
        help=(
            "Post-contract live watchdog limit. It must equal the enforce "
            "runtime-readiness max_stream_wall_age_s threshold for evidence runs."
        ),
    )
    parser.add_argument("--odom-origin-x-m", type=float, required=True)
    parser.add_argument("--odom-origin-y-m", type=float, required=True)
    parser.add_argument("--odom-origin-yaw-rad", type=float, required=True)
    parser.add_argument("--use-sim-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan-row-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--analysis-split", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--campaign-ledger",
        type=Path,
        required=True,
        help="Immutable plan contract that owns --plan-row-id and --out-dir.",
    )
    parser.add_argument(
        "--evidence-role",
        choices=EVIDENCE_ROLES,
        default="diagnostic",
        help=(
            "Pre-declared before recording. D0 readiness accepts qualification "
            "runs only; diagnostic is the fail-closed default."
        ),
    )
    parser.add_argument("--detector-model", type=Path, required=True)
    parser.add_argument(
        "--detector-runtime-config",
        type=Path,
        default=REPO
        / "experiments/multicamera_commissioning_bigwarehouse/config/detector_4cam_v2.yaml",
        help=(
            "Versioned detector runtime config; it must also be listed in "
            "--frozen-config so CLI settings cannot override the freeze."
        ),
    )
    parser.add_argument("--detector-imgsz", type=int, required=True)
    parser.add_argument("--detector-conf", type=float, required=True)
    parser.add_argument("--detector-iou", type=float, required=True)
    parser.add_argument(
        "--detector-runtime-mode",
        choices=("batched_four_camera", "separate_processes"),
        required=True,
    )
    parser.add_argument("--detector-executable", required=True)
    parser.add_argument(
        "--detector-model-format", choices=("native_ultralytics",), required=True
    )
    parser.add_argument("--detector-model-instances", type=int, required=True)
    parser.add_argument("--detector-batch-size", type=int, required=True)
    parser.add_argument(
        "--detector-camera-order",
        nargs=4,
        metavar=("A", "B", "C", "D"),
        required=True,
    )
    parser.add_argument("--detector-shared-device", default=None)
    parser.add_argument("--detector-cpu-threads", type=int, default=None)
    parser.add_argument("--detector-interop-threads", type=int, default=None)
    parser.add_argument("--detector-opencv-threads", type=int, default=None)
    parser.add_argument("--detector-max-batch-stamp-skew-s", type=float, default=None)
    parser.add_argument("--detector-max-pending-wall-s", type=float, default=None)
    parser.add_argument("--detector-frame-policy", default=None)
    parser.add_argument(
        "--detector-yolo-batched-four-camera",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Explicit value of the warehouse launch topology switch.",
    )
    parser.add_argument(
        "--detector-contract-timeout-s",
        type=float,
        default=DEFAULT_DETECTOR_CONTRACT_TIMEOUT_S,
        help=(
            "Wall-clock bound for receiving the retained observed detector "
            "contract before this recorder fails closed."
        ),
    )
    parser.add_argument(
        "--detector-use-masks",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Must be explicitly supplied as --detector-use-masks or --no-detector-use-masks.",
    )
    for camera_suffix in ("a", "b", "c", "d"):
        parser.add_argument(f"--device-camera-{camera_suffix}", default=None)
    parser.add_argument(
        "--study-config",
        type=Path,
        default=REPO / "experiments/multicamera_commissioning_bigwarehouse/config/study.yaml",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO / "experiments/multicamera_commissioning_bigwarehouse/config/paper_protocol.yaml",
    )
    parser.add_argument(
        "--analysis-plan",
        type=Path,
        default=REPO / "experiments/multicamera_commissioning_bigwarehouse/config/paper_analysis_plan.yaml",
    )
    parser.add_argument(
        "--frozen-config",
        type=Path,
        action="append",
        required=True,
        help=(
            "Frozen campaign config recorded in every run; repeat exactly the "
            "--config set used to create campaign_ledger.json."
        ),
    )
    parser.add_argument(
        "--projection-calibration",
        type=Path,
        required=True,
        help=(
            "Optional projection_calibration.json with per-camera along-bearing "
            "offsets (commissioning constants; corrects the near-edge box-bottom "
            "pull toward the camera)"
        ),
    )
    args = parser.parse_args()
    if rclpy is None:
        raise RuntimeError("rclpy is required; source the ROS workspace before running this recorder")
    if args.fresh_age_s < 0.0 or args.duration_s < 0.0:
        parser.error("--fresh-age-s and --duration-s must be non-negative")
    if (
        not math.isfinite(args.max_stream_wall_age_s)
        or args.max_stream_wall_age_s <= 0.0
    ):
        parser.error("--max-stream-wall-age-s must be finite and positive")
    if args.detector_use_masks is None:
        parser.error("explicitly pass --detector-use-masks or --no-detector-use-masks")
    if args.detector_imgsz <= 0 or args.min_camera_rows < 1 or args.min_odom_rows < 1:
        parser.error("detector image size and minimum row gates must be valid")
    if not 0.0 <= args.detector_conf <= 1.0 or not 0.0 <= args.detector_iou <= 1.0:
        parser.error("detector confidence and IoU must lie in [0, 1]")
    if args.duration_s <= 0.0 and args.completion_manifest is None:
        parser.error("set --duration-s or --completion-manifest; unbounded recording is forbidden")
    if (
        not math.isfinite(args.detector_contract_timeout_s)
        or args.detector_contract_timeout_s <= 0.0
    ):
        parser.error("--detector-contract-timeout-s must be finite and positive")
    if not all(
        math.isfinite(value)
        for value in (args.odom_origin_x_m, args.odom_origin_y_m, args.odom_origin_yaw_rad)
    ):
        parser.error("the documented odometry-origin pose must be finite")
    if tuple(args.detector_camera_order) != DETECTOR_CAMERA_ORDER:
        parser.error(
            "--detector-camera-order must be exactly camera_A camera_B camera_C camera_D"
        )
    device_values = (
        args.device_camera_a,
        args.device_camera_b,
        args.device_camera_c,
        args.device_camera_d,
    )
    if args.detector_runtime_mode == "batched_four_camera":
        expected = {
            "executable": "batched_four_camera_yolo_node",
            "model_instances": 1,
            "batch_size": 4,
            "yolo_batched_four_camera": True,
            "cpu_threads": 2,
            "interop_threads": 1,
            "opencv_threads": 1,
            "frame_policy": BATCHED_FRAME_POLICY,
        }
        actual = {
            "executable": args.detector_executable,
            "model_instances": args.detector_model_instances,
            "batch_size": args.detector_batch_size,
            "yolo_batched_four_camera": args.detector_yolo_batched_four_camera,
            "cpu_threads": args.detector_cpu_threads,
            "interop_threads": args.detector_interop_threads,
            "opencv_threads": args.detector_opencv_threads,
            "frame_policy": args.detector_frame_policy,
        }
        wrong = [key for key, value in expected.items() if actual.get(key) != value]
        if wrong:
            parser.error("batched detector contract mismatch: " + ", ".join(wrong))
        if args.detector_shared_device is None:
            parser.error("batched detector requires --detector-shared-device")
        if str(args.detector_shared_device) != "0":
            parser.error("batched detector shared device must match the frozen value '0'")
        if any(value is not None for value in device_values):
            parser.error("batched detector uses one shared device, not per-camera devices")
        if (
            args.detector_max_batch_stamp_skew_s is None
            or args.detector_max_pending_wall_s is None
            or args.detector_max_batch_stamp_skew_s < 0.0
            or args.detector_max_pending_wall_s <= 0.0
        ):
            parser.error("batched detector timing bounds must be explicitly positive")
        if not math.isclose(args.detector_max_batch_stamp_skew_s, 0.10):
            parser.error("batched detector max stamp skew must match frozen value 0.10 s")
        if not math.isclose(args.detector_max_pending_wall_s, 0.50):
            parser.error("batched detector max pending wall time must match frozen value 0.50 s")
    else:
        if args.detector_executable != "yolo_robot_detector_node":
            parser.error("separate-process detector executable must be yolo_robot_detector_node")
        if args.detector_model_instances != 4 or args.detector_batch_size != 1:
            parser.error("separate-process detector requires four model instances and batch size one")
        if args.detector_yolo_batched_four_camera is not False:
            parser.error("separate-process detector requires --no-detector-yolo-batched-four-camera")
        if any(value is None for value in device_values):
            parser.error("separate-process detector requires all four per-camera devices")
        batch_only_values = (
            args.detector_shared_device,
            args.detector_cpu_threads,
            args.detector_interop_threads,
            args.detector_opencv_threads,
            args.detector_max_batch_stamp_skew_s,
            args.detector_max_pending_wall_s,
            args.detector_frame_policy,
        )
        if any(value is not None for value in batch_only_values):
            parser.error("separate-process detector cannot declare batched-only settings")
        parser.error(
            "separate-process fallback remains launch-compatible for timing diagnostics, "
            "but cannot be collected as operational evidence because it has no single "
            "observed four-camera runtime contract"
        )

    world_sdf = args.world_sdf.expanduser().resolve()
    detector_model = args.detector_model.expanduser().resolve()
    detector_runtime_config = args.detector_runtime_config.expanduser().resolve()
    projection_path = args.projection_calibration.expanduser().resolve()
    frozen_config_paths = [path.expanduser().resolve() for path in args.frozen_config]
    provenance_paths = {
        "world_sdf": world_sdf,
        "detector_model": detector_model,
        "detector_runtime_config": detector_runtime_config,
        "projection_calibration": projection_path,
        "study_config": args.study_config.expanduser().resolve(),
        "protocol": args.protocol.expanduser().resolve(),
        "analysis_plan": args.analysis_plan.expanduser().resolve(),
    }
    missing_paths = [str(path) for path in provenance_paths.values() if not path.is_file()]
    missing_paths.extend(str(path) for path in frozen_config_paths if not path.is_file())
    if missing_paths:
        parser.error("missing provenance inputs: " + ", ".join(missing_paths))
    if detector_runtime_config not in frozen_config_paths:
        parser.error(
            "--detector-runtime-config must be repeated exactly as a --frozen-config input"
        )
    method_freeze = _method_freeze(provenance_paths["analysis_plan"])
    try:
        expected_detector_contract = build_batched_runtime_contract(
            model_path=detector_model,
            model_sha256=_sha256_file(detector_model),
            imgsz=int(args.detector_imgsz),
            confidence_threshold=float(args.detector_conf),
            iou_threshold=float(args.detector_iou),
            use_masks=bool(args.detector_use_masks),
            shared_device=str(args.detector_shared_device),
            cpu_threads=int(args.detector_cpu_threads),
            interop_threads=int(args.detector_interop_threads),
            opencv_threads=int(args.detector_opencv_threads),
            max_batch_stamp_skew_s=float(args.detector_max_batch_stamp_skew_s),
            max_pending_wall_s=float(args.detector_max_pending_wall_s),
            source_hashes=_frozen_detector_source_hashes(
                analysis_path=provenance_paths["analysis_plan"],
                method_freeze=method_freeze,
            ),
        )
        _validate_batched_cli_against_runtime_config(
            config_path=detector_runtime_config,
            expected_contract=expected_detector_contract,
            evidence_role=args.evidence_role,
        )
    except (RuntimeContractError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(f"detector runtime freeze validation failed: {exc}")
    try:
        campaign_contract = campaign_ledger.recorder_preflight_contract(
            ledger_path=args.campaign_ledger,
            plan_row_id=str(args.plan_row_id),
            attempt_id=str(args.attempt_id),
            seed=int(args.seed),
            evidence_role=str(args.evidence_role),
            analysis_split=str(args.analysis_split),
            output_dir=args.out_dir,
            artifact_subdir="raw",
            expected_inputs={
                "study": provenance_paths["study_config"],
                "protocol": provenance_paths["protocol"],
                "model": detector_model,
                "calibration": projection_path,
            },
            frozen_config_paths=frozen_config_paths,
        )
    except campaign_ledger.LedgerError as exc:
        parser.error(f"campaign preflight failed: {exc}")
    if campaign_contract.get("method_freeze") != method_freeze:
        parser.error(
            "campaign method-freeze snapshot differs from the locally resolved source bytes"
        )
    campaign_method_freeze_sha256 = campaign_contract.get("method_freeze_sha256")
    if not isinstance(campaign_method_freeze_sha256, str) or len(
        campaign_method_freeze_sha256
    ) != 64:
        parser.error("campaign method-freeze SHA-256 is missing or malformed")
    camera_models = {
        camera_id: camera_model_from_world(world_sdf, include_name=model_name)
        for camera_id, model_name in DEFAULT_CAMERA_MODELS.items()
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    final_manifest_path = args.out_dir / "operational_recording_manifest.json"
    progress_manifest_path = args.out_dir / "operational_recording_manifest.in_progress.json"
    failed_manifest_path = args.out_dir / "operational_recording_manifest.failed.json"
    if any(path.exists() for path in (final_manifest_path, progress_manifest_path, failed_manifest_path)):
        raise RuntimeError(f"Refusing to reuse recorder output directory: {args.out_dir}")
    started_utc = _utc_now()
    devices = (
        {}
        if args.detector_runtime_mode == "batched_four_camera"
        else {
            "camera_A": str(args.device_camera_a),
            "camera_B": str(args.device_camera_b),
            "camera_C": str(args.device_camera_c),
            "camera_D": str(args.device_camera_d),
        }
    )
    detector_topology = {
        "runtime_mode": str(args.detector_runtime_mode),
        "executable": str(args.detector_executable),
        "launch_switch_yolo_batched_four_camera": bool(
            args.detector_yolo_batched_four_camera
        ),
        "model_format": str(args.detector_model_format),
        "model_instances": int(args.detector_model_instances),
        "batch_size": int(args.detector_batch_size),
        "camera_order": list(args.detector_camera_order),
    }
    if args.detector_runtime_mode == "batched_four_camera":
        detector_topology.update(
            {
                "shared_device": str(args.detector_shared_device),
                "cpu_threads": int(args.detector_cpu_threads),
                "interop_threads": int(args.detector_interop_threads),
                "opencv_threads": int(args.detector_opencv_threads),
                "max_batch_stamp_skew_s": float(
                    args.detector_max_batch_stamp_skew_s
                ),
                "max_pending_wall_s": float(args.detector_max_pending_wall_s),
                "frame_policy": str(args.detector_frame_policy),
                "inference_timing_semantics": (
                    "full_batch_wall_ms_repeated_per_camera_result"
                ),
                "fault_policy": (
                    "fatal_process_exit_and_launch_shutdown_no_synthetic_miss"
                ),
                "synchronization_miss_policy": (
                    "no_output_detected_by_liveness_gate"
                ),
            }
        )
    else:
        detector_topology.update({"instances": 4, "devices": devices})
    manifest = {
        "status": "in_progress",
        "started_utc": started_utc,
        "run_id": str(args.run_id),
        "plan_row_id": str(args.plan_row_id),
        "attempt_id": str(args.attempt_id),
        "analysis_split": str(args.analysis_split),
        "seed": int(args.seed),
        "evidence_role": str(args.evidence_role),
        "world_sdf": str(world_sdf),
        "provenance": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in provenance_paths.items()
        },
        "frozen_configs": [
            {"path": str(path), "sha256": _sha256_file(path)}
            for path in frozen_config_paths
        ],
        "method_freeze": method_freeze,
        # This is the plan-time provenance digest, rather than a fresh hash
        # calculated by the recorder.  Keeping it beside the snapshot makes
        # the ledger binding visible even when this manifest is inspected on
        # its own.
        "method_freeze_sha256": campaign_method_freeze_sha256,
        "campaign_contract": campaign_contract,
        "transport_environment": campaign_contract["transport_environment"],
        "git": _git_state(),
        "camera_topics": DEFAULT_CAMERA_TOPICS,
        "camera_models": DEFAULT_CAMERA_MODELS,
        "odom_topic": str(args.odom_topic),
        "odom_origin_warehouse": {
            "x_m": float(args.odom_origin_x_m),
            "y_m": float(args.odom_origin_y_m),
            "yaw_rad": float(args.odom_origin_yaw_rad),
            "source": "documented_launch_spawn_not_ground_truth",
        },
        "fresh_age_s": float(args.fresh_age_s),
        "stream_liveness_watchdog": {
            "clock": "steady_monotonic",
            "starts_after": "validated_detector_runtime_contract",
            "max_stream_wall_age_s": float(args.max_stream_wall_age_s),
            "streams": ["camera_A", "camera_B", "camera_C", "camera_D", "odometry"],
            "prearm_messages_do_not_count": True,
            "policy": "fatal_recorder_exit_and_part_artifact_quarantine",
        },
        "detector_runtime": {
            "model_path": str(detector_model),
            "model_sha256": expected_detector_contract["model_sha256"],
            "imgsz": int(args.detector_imgsz),
            "confidence_threshold": float(args.detector_conf),
            "iou_threshold": float(args.detector_iou),
            "use_masks": bool(args.detector_use_masks),
            "devices": devices,
            "topology": detector_topology,
            "observed_contract": None,
            "observed_contract_sha256": None,
            "observed_contract_requirement": {
                "topic": RUNTIME_CONTRACT_TOPIC,
                "qos": "reliable_transient_local_keep_last_1",
                "timeout_wall_s": float(args.detector_contract_timeout_s),
                "expected_contract": expected_detector_contract,
                "expected_contract_sha256": runtime_contract_sha256(
                    expected_detector_contract
                ),
                "state": "waiting_before_arming",
            },
        },
        "duration_clock": "simulation" if args.use_sim_time else "wall",
        "odom_covariance": {
            "columns": ["odom_noisy_cov_xx", "odom_noisy_cov_xy", "odom_noisy_cov_yy"],
            "frame": "warehouse",
            "source": "published_noisy_odometry_pose_covariance",
            "legacy_fallback": "empty covariance columns are handled explicitly by the GP-input adapter",
        },
        "contains_ground_truth": False,
        "minimum_stream_rows": {
            "per_camera": int(args.min_camera_rows),
            "odometry": int(args.min_odom_rows),
        },
    }
    projection_payload = json.loads(projection_path.read_text(encoding="utf-8"))
    manifest["projection_calibration"] = {
        "path": str(projection_path),
        "sha256": _sha256_file(projection_path),
        "kind": projection_payload.get("kind"),
        "calibration_id": projection_payload.get("calibration_id"),
    }
    _write_json_atomic(progress_manifest_path, manifest)

    rclpy.init()
    node = OperationalLogRecorder(
        out_dir=args.out_dir,
        camera_topics=DEFAULT_CAMERA_TOPICS,
        camera_models=camera_models,
        odom_topic=str(args.odom_topic),
        fresh_age_s=float(args.fresh_age_s),
        max_stream_wall_age_s=float(args.max_stream_wall_age_s),
        odom_origin_x_m=float(args.odom_origin_x_m),
        odom_origin_y_m=float(args.odom_origin_y_m),
        odom_origin_yaw_rad=float(args.odom_origin_yaw_rad),
        use_sim_time=bool(args.use_sim_time),
        expected_detector_contract=expected_detector_contract,
        detector_contract_topic=RUNTIME_CONTRACT_TOPIC,
    )
    status = "failed"
    stop_reason = "unknown"
    failure_message = ""
    wall_start = time.monotonic()
    derived_wall_timeout_s = (
        float(args.wall_timeout_s)
        if args.wall_timeout_s > 0.0
        else (10.0 * float(args.duration_s) + 60.0 if args.duration_s > 0.0 else 0.0)
    )
    started_at_s: float | None = None
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.10)
            if node.detector_contract_error is not None:
                raise RuntimeError(node.detector_contract_error)
            if not node.detector_contract_armed:
                if (
                    time.monotonic() - wall_start
                    > float(args.detector_contract_timeout_s)
                ):
                    raise RuntimeError(
                        "timed out waiting for a matching retained detector runtime contract "
                        f"on {RUNTIME_CONTRACT_TOPIC} after "
                        f"{args.detector_contract_timeout_s:.1f}s"
                    )
                # Neither detector nor odometry rows are accepted before this
                # point; their callbacks only count pre-arm messages.
                continue
            watchdog_failures = node.stream_watchdog_failures(now_wall_s=time.monotonic())
            if watchdog_failures:
                node.stream_watchdog_failure = "; ".join(watchdog_failures)
                raise RuntimeError(node.stream_watchdog_failure)
            now_s = (
                node.get_clock().now().nanoseconds * 1e-9
                if args.use_sim_time
                else time.monotonic()
            )
            if started_at_s is None:
                if not args.use_sim_time or now_s > 0.0:
                    started_at_s = now_s
            elif args.duration_s > 0.0 and now_s - started_at_s >= args.duration_s:
                status = "completed"
                stop_reason = "requested_duration_reached"
                break
            if args.completion_manifest is not None and args.completion_manifest.is_file():
                completion = json.loads(args.completion_manifest.read_text(encoding="utf-8"))
                if completion.get("status") == "completed":
                    status = "completed"
                    stop_reason = "route_completion_manifest"
                    break
            if derived_wall_timeout_s > 0.0 and time.monotonic() - wall_start > derived_wall_timeout_s:
                raise RuntimeError(
                    f"wall-clock deadman reached after {derived_wall_timeout_s:.1f}s before completion"
                )
    except KeyboardInterrupt:
        status = "interrupted"
        stop_reason = "keyboard_interrupt"
    except BaseException as exc:
        status = "failed"
        stop_reason = "exception"
        failure_message = f"{type(exc).__name__}: {exc}"
    finally:
        if status == "completed":
            if not node.detector_contract_armed:
                status = "failed"
                stop_reason = "detector_runtime_contract_not_armed"
                failure_message = "no matching observed detector runtime contract"
            if status == "completed":
                low_cameras = {
                    camera: count for camera, count in node.counts.items()
                    if count < int(args.min_camera_rows)
                }
                if low_cameras or node.odom_count < int(args.min_odom_rows):
                    status = "failed"
                    stop_reason = "minimum_stream_rows_not_met"
                    failure_message = (
                        f"camera rows below {args.min_camera_rows}: {low_cameras}; "
                        f"odom rows {node.odom_count} < {args.min_odom_rows}"
                    )
                if node.stream_integrity_errors:
                    status = "failed"
                    stop_reason = "stream_integrity_error"
                    failure_message = "; ".join(node.stream_integrity_errors[:10])
        node.close(finalize=(status == "completed"))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    runtime_contract_requirement = manifest["detector_runtime"][
        "observed_contract_requirement"
    ]
    runtime_contract_requirement.update(
        {
            "state": (
                "armed"
                if node.detector_contract_armed and node.detector_contract_error is None
                else "rejected_or_unavailable"
            ),
            "received_wall_elapsed_s": (
                None
                if node.detector_contract_received_wall_s is None
                else max(
                    node.detector_contract_received_wall_s
                    - node.started_wall_monotonic_s,
                    0.0,
                )
            ),
            "error": node.detector_contract_error,
            "prearm_camera_messages": dict(node.prearm_camera_messages),
            "prearm_odom_messages": int(node.prearm_odom_messages),
        }
    )
    manifest["detector_runtime"]["observed_contract"] = node.observed_detector_contract
    manifest["detector_runtime"]["observed_contract_sha256"] = (
        node.observed_detector_contract_sha256
    )
    watchdog_manifest = manifest["stream_liveness_watchdog"]
    watchdog_manifest.update(
        {
            "terminal_failure": node.stream_watchdog_failure,
            "last_valid_camera_wall_elapsed_s": {
                camera: (
                    None
                    if stamp is None
                    else max(stamp - node.started_wall_monotonic_s, 0.0)
                )
                for camera, stamp in node.last_camera_observation_wall_s.items()
            },
            "last_valid_odom_wall_elapsed_s": (
                None
                if node.last_odom_observation_wall_s is None
                else max(
                    node.last_odom_observation_wall_s - node.started_wall_monotonic_s,
                    0.0,
                )
            ),
        }
    )
    manifest.update({
        "status": status,
        "finished_utc": _utc_now(),
        "stop_reason": stop_reason,
        "failure_message": failure_message or None,
        "wall_elapsed_s": float(time.monotonic() - wall_start),
        "first_sim_stamp_s": node.first_sim_stamp_s,
        "last_sim_stamp_s": node.last_sim_stamp_s,
        "camera_rows": dict(node.counts),
        "odometry_rows": int(node.odom_count),
        "camera_stamp_ranges_s": {
            camera: {
                "first": node.first_camera_stamp_s[camera],
                "last": node.last_camera_stamp_s[camera],
            }
            for camera in sorted(node.counts)
        },
        "odometry_stamp_range_s": {
            "first": node.first_odom_stamp_s,
            "last": node.last_odom_stamp_s,
        },
        "stream_integrity_errors": list(node.stream_integrity_errors),
        "observed_calibration_ids": {
            camera: sorted(values) for camera, values in node.calibration_ids.items()
        },
        "observed_image_frame_ids": {
            camera: sorted(values) for camera, values in node.image_frame_ids.items()
        },
    })
    if args.completion_manifest is not None and args.completion_manifest.is_file():
        manifest["route_completion_manifest"] = {
            "path": str(args.completion_manifest.resolve()),
            "sha256": _sha256_file(args.completion_manifest.resolve()),
        }
    progress_manifest_path.unlink(missing_ok=True)
    if status == "completed":
        _write_json_atomic(final_manifest_path, manifest)
        return 0
    _write_json_atomic(failed_manifest_path, manifest)
    raise RuntimeError(f"operational recording did not complete: {stop_reason}: {failure_message}")


if __name__ == "__main__":
    raise SystemExit(main())
