"""Single-model, deterministic four-camera YOLO inference runtime.

This node is intentionally specific to the commissioned A--D camera contract.
It holds at most one not-yet-used frame per camera and invokes one native
Ultralytics model with a four-image source list in A--D order.  It never fills a
batch by repeating a previous image.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, NoReturn

import cv2
import numpy as np
import rclpy
import torch
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, String
from ultralytics import YOLO

import perception.core.four_camera_batch as four_camera_batch_module
import perception.core.four_camera_runtime_contract as runtime_contract_module
import perception.core.yolo_selection as yolo_selection_module
from perception.core.detection_diagnostics import (
    diagnostics_from_message,
    diagnostics_message,
)
from perception.core.four_camera_batch import (
    CAMERA_ORDER,
    BatchContractError,
    FourCameraBatcher,
    PendingFrame,
    frame_age_at_publish_s,
    stamp_parts_to_ns,
    validate_batch_results,
)
from perception.core.four_camera_runtime_contract import (
    EXECUTABLE_SOURCE_KEY,
    FOUR_CAMERA_BATCH_SOURCE_KEY,
    RUNTIME_CONTRACT_SOURCE_KEY,
    RUNTIME_CONTRACT_TOPIC,
    YOLO_SELECTION_SOURCE_KEY,
    build_batched_runtime_contract,
    runtime_contract_json,
    sha256_file,
    source_hashes_from_paths,
)
from perception.core.ros_image import image_msg_to_bgr8
from perception.core.yolo_selection import select_best_detection, target_class_ids
from perception.nodes.yolo_robot_detector_node import (
    _as_bool,
    _confidence_logit,
    _selected_pixel_source_code,
)


CAMERA_TOPICS = {
    "camera_A": "/external_camera/image_raw",
    "camera_B": "/external_camera_b/image_raw",
    "camera_C": "/external_camera_c/image_raw",
    "camera_D": "/external_camera_d/image_raw",
}


def _runtime_contract_qos() -> QoSProfile:
    """Retain exactly one detector identity for recorders that start later."""

    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


@dataclass(frozen=True)
class _CameraOutput:
    pixel_publisher: Any
    diagnostics_publisher: Any
    observation_publisher: Any


@dataclass(frozen=True)
class _BatchTiming:
    predict_start_stamp_s: float = math.nan
    predict_finish_stamp_s: float = math.nan
    inference_wall_ms: float = math.nan


class BatchedFourCameraYoloNode(Node):
    """Run one native YOLO model over strict, fresh A--D frame batches."""

    def __init__(self) -> None:
        super().__init__("batched_four_camera_yolo_node")

        self.declare_parameter("model_path", "")
        self.declare_parameter("device", "")
        self.declare_parameter("cpu_num_threads", 2)
        self.declare_parameter("cpu_num_interop_threads", 1)
        self.declare_parameter("opencv_num_threads", 1)
        self.declare_parameter("image_size", 640)
        self.declare_parameter("confidence_threshold", 0.25)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("class_name", "robot")
        self.declare_parameter("class_id", -1)
        self.declare_parameter("use_masks", True)
        self.declare_parameter("mask_min_area", 12.0)
        self.declare_parameter("mask_bottom_band_px", 3.0)
        self.declare_parameter("min_bbox_area_px", 0.0)
        self.declare_parameter("debug_frame_dir", "")
        self.declare_parameter("pixel_noise_sigma", 0.0)
        self.declare_parameter("seed", 0)
        self.declare_parameter("warmup_iters", 3)
        self.declare_parameter("max_batch_stamp_skew_s", 0.10)
        self.declare_parameter("max_pending_wall_s", 0.50)
        self.declare_parameter("camera_observation_r_visible_uv", 2.5)
        self.declare_parameter("camera_observation_r_miss_uv", 40.0)

        raw_model_path = str(self.get_parameter("model_path").value).strip()
        if not raw_model_path:
            raise RuntimeError("model_path must be set for batched_four_camera_yolo_node")
        model_path = Path(raw_model_path).expanduser()
        if not model_path.is_file():
            raise RuntimeError(f"model_path does not exist: {model_path}")
        if model_path.suffix.lower() != ".pt":
            raise RuntimeError(
                "batched commissioning requires a native Ultralytics .pt checkpoint; "
                f"got {model_path}"
            )
        self.model_path = model_path.resolve()
        # Keep this a string end to end.  In particular, ``0`` means CUDA GPU
        # zero to Ultralytics but would violate this declared ROS parameter if
        # launch serialized it as an integer.
        self.device = str(self.get_parameter("device").value).strip()
        self.cpu_num_threads = max(1, int(self.get_parameter("cpu_num_threads").value))
        self.cpu_num_interop_threads = max(
            1, int(self.get_parameter("cpu_num_interop_threads").value)
        )
        self.opencv_num_threads = max(1, int(self.get_parameter("opencv_num_threads").value))
        torch.set_num_threads(self.cpu_num_threads)
        try:
            torch.set_num_interop_threads(self.cpu_num_interop_threads)
        except RuntimeError as exc:
            self.get_logger().warn(f"could not set PyTorch interop threads: {exc}")
        cv2.setNumThreads(self.opencv_num_threads)
        # Attest the effective process settings, not merely requested ROS
        # parameters. If a runtime cannot apply a setting, the observed
        # contract must differ and the recorder will fail closed.
        self.actual_cpu_num_threads = int(torch.get_num_threads())
        self.actual_cpu_num_interop_threads = int(torch.get_num_interop_threads())
        self.actual_opencv_num_threads = int(cv2.getNumThreads())

        self.image_size = int(self.get_parameter("image_size").value)
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.iou_threshold = float(self.get_parameter("iou_threshold").value)
        self.class_name = str(self.get_parameter("class_name").value)
        self.class_id = int(self.get_parameter("class_id").value)
        use_masks_value = self.get_parameter("use_masks").value
        self.use_masks = use_masks_value if isinstance(use_masks_value, bool) else _as_bool(use_masks_value)
        self.mask_min_area = float(self.get_parameter("mask_min_area").value)
        self.mask_bottom_band_px = float(self.get_parameter("mask_bottom_band_px").value)
        self.min_bbox_area_px = float(self.get_parameter("min_bbox_area_px").value)
        self.pixel_noise_sigma = float(self.get_parameter("pixel_noise_sigma").value)
        self.rng = np.random.default_rng(int(self.get_parameter("seed").value))
        self.warmup_iters = max(0, int(self.get_parameter("warmup_iters").value))
        self.r_visible_uv = float(self.get_parameter("camera_observation_r_visible_uv").value)
        self.r_miss_uv = float(self.get_parameter("camera_observation_r_miss_uv").value)
        self.debug_frame_dir = str(self.get_parameter("debug_frame_dir").value).strip()
        if self.debug_frame_dir:
            Path(self.debug_frame_dir).expanduser().mkdir(parents=True, exist_ok=True)

        if self.image_size <= 0:
            raise RuntimeError("image_size must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise RuntimeError("confidence_threshold must be in [0, 1]")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise RuntimeError("iou_threshold must be in [0, 1]")
        if self.pixel_noise_sigma < 0.0 or self.min_bbox_area_px < 0.0:
            raise RuntimeError("pixel_noise_sigma and min_bbox_area_px must be non-negative")

        self.max_batch_stamp_skew_s = float(
            self.get_parameter("max_batch_stamp_skew_s").value
        )
        self.max_pending_wall_s = float(self.get_parameter("max_pending_wall_s").value)
        self.batcher = FourCameraBatcher(
            camera_order=CAMERA_ORDER,
            max_stamp_skew_s=self.max_batch_stamp_skew_s,
            max_pending_wall_s=self.max_pending_wall_s,
        )
        self._warning_counts: dict[str, int] = {}

        # Native checkpoint only: one wrapper, one model allocation, one
        # explicitly typed device.  TorchScript is intentionally not selected
        # because the existing export is fixed-shape and bypasses the matched
        # native preprocessing path.
        model_sha256_before_load = sha256_file(self.model_path)
        self.model = YOLO(str(self.model_path))
        self.target_ids = target_class_ids(
            getattr(self.model, "names", {}), self.class_name, self.class_id
        )
        if self.target_ids == set():
            raise RuntimeError(
                "class filter does not match any YOLO class names: "
                f"class_name={self.class_name!r}, class_id={self.class_id}, "
                f"names={getattr(self.model, 'names', {})!r}"
            )

        try:
            from reliability.single_camera_adapter import (
                SingleCameraAdapterConfig,
                camera_observation_from_diagnostics,
            )
        except Exception as exc:
            raise RuntimeError(
                "the CameraObservation adapter is required by the batched runtime"
            ) from exc
        self._camera_observation_from_diagnostics = camera_observation_from_diagnostics
        self._observation_configs = {
            camera_id: SingleCameraAdapterConfig(
                camera_id=camera_id,
                calibration_id=f"warehouse_full_4cam_{camera_id}",
                image_frame_id=camera_id,
                r_visible_uv=self.r_visible_uv,
                r_miss_uv=self.r_miss_uv,
            )
            for camera_id in CAMERA_ORDER
        }

        if self.warmup_iters:
            dummy_batch = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in CAMERA_ORDER]
            warmup_start = time.perf_counter()
            for _ in range(self.warmup_iters):
                try:
                    validate_batch_results(self._predict_batch(dummy_batch), len(CAMERA_ORDER))
                except Exception as exc:
                    raise RuntimeError(f"four-image detector warmup failed: {exc}") from exc
            self.get_logger().info(
                f"four-image warmup: {self.warmup_iters} iterations in "
                f"{(time.perf_counter() - warmup_start) * 1.0e3:.0f} ms"
            )

        # Hash after the model is loaded and warmed, then reject a checkpoint
        # replacement race rather than attesting bytes that were not the model
        # used for the first live batch.
        self.model_sha256 = sha256_file(self.model_path)
        if self.model_sha256 != model_sha256_before_load:
            raise RuntimeError(
                "model checkpoint bytes changed while the batched detector was loading"
            )
        self._publish_runtime_contract()

        self.outputs: dict[str, _CameraOutput] = {}
        for camera_id in CAMERA_ORDER:
            self.outputs[camera_id] = _CameraOutput(
                pixel_publisher=self.create_publisher(
                    PoseStamped, f"/perception/{camera_id}/pixel_pose", 10
                ),
                diagnostics_publisher=self.create_publisher(
                    Float64MultiArray,
                    f"/perception/{camera_id}/detection_diagnostics",
                    10,
                ),
                observation_publisher=self.create_publisher(
                    String, f"/perception/camera_observation/{camera_id}", 10
                ),
            )

        self.subscriptions = []
        for camera_id in CAMERA_ORDER:
            self.subscriptions.append(
                self.create_subscription(
                    Image,
                    CAMERA_TOPICS[camera_id],
                    lambda msg, selected_camera=camera_id: self._image_callback(
                        selected_camera, msg
                    ),
                    1,
                )
            )

        self.get_logger().info(
            "Batched four-camera YOLO started "
            f"(model={self.model_path}, device={self.device or 'auto'}, "
            f"batch_order={','.join(CAMERA_ORDER)}, batch_size={len(CAMERA_ORDER)}, "
            f"imgsz={self.image_size}, masks={self.use_masks}, "
            f"torch_threads={self.actual_cpu_num_threads}, "
            f"torch_interop={self.actual_cpu_num_interop_threads}, "
            f"opencv_threads={self.actual_opencv_num_threads}, "
            "input_policy=strict-new-latest-only)"
        )

    def _publish_runtime_contract(self) -> None:
        """Publish the retained identity only after a usable model is ready.

        The recorder subscribes to this topic before accepting detector or
        odometry rows.  It compares the JSON byte-for-byte (via the embedded
        digest) with its frozen CLI/configuration expectation, so this is not a
        best-effort diagnostic announcement.
        """

        try:
            source_hashes = source_hashes_from_paths(
                {
                    EXECUTABLE_SOURCE_KEY: Path(__file__),
                    FOUR_CAMERA_BATCH_SOURCE_KEY: Path(four_camera_batch_module.__file__),
                    RUNTIME_CONTRACT_SOURCE_KEY: Path(runtime_contract_module.__file__),
                    YOLO_SELECTION_SOURCE_KEY: Path(yolo_selection_module.__file__),
                }
            )
            self.runtime_contract = build_batched_runtime_contract(
                model_path=self.model_path,
                model_sha256=self.model_sha256,
                imgsz=self.image_size,
                confidence_threshold=self.confidence_threshold,
                iou_threshold=self.iou_threshold,
                use_masks=self.use_masks,
                shared_device=self.device,
                cpu_threads=self.actual_cpu_num_threads,
                interop_threads=self.actual_cpu_num_interop_threads,
                opencv_threads=self.actual_opencv_num_threads,
                max_batch_stamp_skew_s=self.max_batch_stamp_skew_s,
                max_pending_wall_s=self.max_pending_wall_s,
                source_hashes=source_hashes,
            )
            self.runtime_contract_publisher = self.create_publisher(
                String,
                RUNTIME_CONTRACT_TOPIC,
                _runtime_contract_qos(),
            )
            message = String()
            message.data = runtime_contract_json(self.runtime_contract)
            self.runtime_contract_publisher.publish(message)
        except Exception as exc:
            raise RuntimeError(
                "could not publish the required four-camera detector runtime contract"
            ) from exc
        self.get_logger().info(
            "published retained four-camera detector runtime contract "
            f"sha256={self.runtime_contract['contract_sha256']} "
            f"topic={RUNTIME_CONTRACT_TOPIC}"
        )

    def _clock_s(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1.0e-9

    def _predict_batch(self, images_bgr: list[np.ndarray]):
        if len(images_bgr) != len(CAMERA_ORDER):
            raise BatchContractError(
                f"predict requires exactly {len(CAMERA_ORDER)} images"
            )
        kwargs = {
            "source": list(images_bgr),
            "imgsz": self.image_size,
            "conf": 0.0,
            "iou": self.iou_threshold,
            "batch": len(CAMERA_ORDER),
            "stream": False,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device
        return self.model.predict(**kwargs)

    def _warn_bounded(self, key: str, message: str) -> None:
        count = self._warning_counts.get(key, 0) + 1
        self._warning_counts[key] = count
        if count <= 3 or count & (count - 1) == 0:
            self.get_logger().warn(f"{message} (count={count})")

    def _fatal(self, message: str, cause: Exception | None = None) -> NoReturn:
        """Log once and terminate rather than forge an availability miss."""

        self.get_logger().fatal(message)
        if cause is None:
            raise RuntimeError(message)
        raise RuntimeError(message) from cause

    def _image_callback(self, camera_id: str, msg: Image) -> None:
        receive_wall_s = time.perf_counter()
        receive_stamp_s = self._clock_s()
        try:
            stamp_ns = stamp_parts_to_ns(msg.header.stamp.sec, msg.header.stamp.nanosec)
            decision = self.batcher.offer(
                PendingFrame(
                    camera_id=camera_id,
                    stamp_ns=stamp_ns,
                    receive_stamp_s=receive_stamp_s,
                    receive_wall_s=receive_wall_s,
                    payload=msg,
                )
            )
        except Exception as exc:
            self._fatal(f"malformed input contract for {camera_id}: {exc}", exc)

        if decision.status in {"duplicate", "out_of_order", "stamp_skew"}:
            self._warn_bounded(
                camera_id + ":" + decision.status,
                f"{decision.status} frame rejected for {camera_id}; "
                f"dropped={decision.dropped_camera_ids}",
            )
        elif decision.dropped_camera_ids:
            self._warn_bounded(
                "pending_expired",
                f"expired pending frames: {decision.dropped_camera_ids}",
            )
        if decision.batch is not None:
            self._process_batch(decision.batch)

    def _process_batch(self, batch: tuple[PendingFrame, ...]) -> None:
        if tuple(item.camera_id for item in batch) != CAMERA_ORDER:
            self._fatal("internal four-camera batch order violation")

        images: list[np.ndarray] = []
        try:
            for item in batch:
                image = image_msg_to_bgr8(item.payload)
                if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
                    raise BatchContractError(
                        f"{item.camera_id} image conversion returned shape {image.shape!r}"
                    )
                images.append(image)
        except Exception as exc:
            self._fatal(f"four-camera image conversion failed: {exc}", exc)

        predict_start_stamp_s = self._clock_s()
        predict_start_wall_s = time.perf_counter()
        try:
            raw_results = self._predict_batch(images)
        except Exception as exc:
            elapsed_ms = max((time.perf_counter() - predict_start_wall_s) * 1.0e3, 0.0)
            self._fatal(
                f"four-camera batch inference failed after {elapsed_ms:.1f} ms: {exc}",
                exc,
            )

        predict_finish_stamp_s = self._clock_s()
        timing = _BatchTiming(
            predict_start_stamp_s=predict_start_stamp_s,
            predict_finish_stamp_s=predict_finish_stamp_s,
            inference_wall_ms=max(
                (time.perf_counter() - predict_start_wall_s) * 1.0e3, 0.0
            ),
        )
        try:
            results = validate_batch_results(raw_results, len(CAMERA_ORDER))
        except BatchContractError as exc:
            self._fatal(f"malformed four-camera results: {exc}", exc)

        prepared: list[dict[str, Any]] = []
        for item, image, result in zip(batch, images, results, strict=True):
            try:
                prepared.append(self._prepare_result(result))
            except Exception as exc:
                self._fatal(
                    f"malformed result contract for {item.camera_id}: {exc}",
                    exc,
                )
        # Validate and prepare all selections before publishing any camera from
        # this batch.  A malformed B/C/D result therefore cannot leave a
        # plausible partial A-only evidence row behind.
        for item, image, selection in zip(batch, images, prepared, strict=True):
            self._publish_result(item, image, selection, timing)

    def _prepare_result(self, result: Any) -> dict[str, Any]:
        selection = select_best_detection(
            result,
            target_ids=self.target_ids,
            confidence_threshold=self.confidence_threshold,
            use_masks=self.use_masks,
            mask_min_area=self.mask_min_area,
            mask_bottom_band_px=self.mask_bottom_band_px,
        )
        bbox = selection.get("bbox_xyxy")
        if bbox is not None and self.min_bbox_area_px > 0.0:
            x0, y0, x1, y1 = np.asarray(bbox, dtype=float).reshape(4)
            bbox_area = float(max(x1 - x0, 0.0) * max(y1 - y0, 0.0))
            if bbox_area < self.min_bbox_area_px:
                selection["detected_after_threshold"] = False

        if bbox is not None:
            selected_u = float(selection.get("selected_u", math.nan))
            selected_v = float(selection.get("selected_v", math.nan))
            if self.pixel_noise_sigma > 0.0:
                selected_u += float(self.rng.normal(0.0, self.pixel_noise_sigma))
                selected_v += float(self.rng.normal(0.0, self.pixel_noise_sigma))
            selection["selected_u"] = selected_u
            selection["selected_v"] = selected_v
            if not math.isfinite(selected_u) or not math.isfinite(selected_v):
                raise BatchContractError("selected pixel is non-finite")

        return selection

    def _publish_result(
        self,
        item: PendingFrame,
        image_bgr: np.ndarray,
        selection: dict[str, Any],
        timing: _BatchTiming,
    ) -> None:
        if bool(selection.get("detected_after_threshold", False)):
            selected_u = float(selection.get("selected_u", math.nan))
            selected_v = float(selection.get("selected_v", math.nan))
            if not math.isfinite(selected_u) or not math.isfinite(selected_v):
                self._fatal(f"non-finite selected pixel for {item.camera_id}")
        self._write_debug_frame(item, image_bgr, selection)
        self._publish_camera(item, image_bgr.shape[:2], selection, timing)
        if not bool(selection.get("detected_after_threshold", False)):
            return

        selected_u = float(selection.get("selected_u", math.nan))
        selected_v = float(selection.get("selected_v", math.nan))
        pose = PoseStamped()
        pose.header = item.payload.header
        pose.header.frame_id = "image"
        pose.pose.position.x = selected_u
        pose.pose.position.y = selected_v
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        try:
            self.outputs[item.camera_id].pixel_publisher.publish(pose)
        except Exception as exc:
            self._fatal(f"pixel-pose publish failed for {item.camera_id}: {exc}", exc)

    def _publish_camera(
        self,
        item: PendingFrame,
        image_shape: tuple[int, int],
        selection: dict[str, Any] | None,
        timing: _BatchTiming,
    ) -> None:
        selection = dict(selection or {})
        stamp_s = item.stamp_ns * 1.0e-9
        bbox = selection.get("bbox_xyxy")
        if bbox is None:
            x0 = y0 = x1 = y1 = math.nan
        else:
            x0, y0, x1, y1 = [
                float(value) for value in np.asarray(bbox, dtype=float).reshape(4)
            ]
        selected_u = float(selection.get("selected_u", math.nan))
        selected_v = float(selection.get("selected_v", math.nan))
        raw_score = float(selection.get("raw_best_score", 0.0) or 0.0)
        selected_score = float(selection.get("selected_score", raw_score) or raw_score)
        mask_available = int(selection.get("mask_available", 0) or 0)
        mask_area = float(selection.get("mask_area", math.nan))
        mask_bottom_u = (
            float(selection.get("mask_bottom_u", math.nan)) if mask_available else math.nan
        )
        mask_bottom_v = (
            float(selection.get("mask_bottom_v", math.nan)) if mask_available else math.nan
        )
        polygon = (selection.get("detection") or {}).get("polygon")
        mask_points = float(polygon.shape[0]) if mask_available and polygon is not None else math.nan
        height, width = image_shape
        if math.isfinite(selected_u) and math.isfinite(selected_v) and height > 0 and width > 0:
            border_margin = float(
                min(
                    selected_u,
                    selected_v,
                    float(width) - 1.0 - selected_u,
                    float(height) - 1.0 - selected_v,
                )
            )
        else:
            border_margin = math.nan
        bbox_area = (
            float(max(x1 - x0, 0.0) * max(y1 - y0, 0.0))
            if math.isfinite(x0)
            else math.nan
        )
        publish_stamp_s = self._clock_s()
        callback_ms = max((time.perf_counter() - item.receive_wall_s) * 1.0e3, 0.0)
        if math.isfinite(timing.predict_start_stamp_s) and math.isfinite(
            timing.predict_finish_stamp_s
        ):
            latency_s = max(
                timing.predict_finish_stamp_s - timing.predict_start_stamp_s, 0.0
            )
        else:
            latency_s = math.nan
        try:
            frame_age_s = frame_age_at_publish_s(
                publish_stamp_s=publish_stamp_s,
                image_stamp_s=stamp_s,
            )
        except BatchContractError as exc:
            self._fatal(
                f"future image-stamp integrity fault for {item.camera_id}: {exc}",
                exc,
            )
        detected = bool(selection.get("detected_after_threshold", False))
        message = diagnostics_message(
            stamp=stamp_s,
            detected=detected,
            u_mid=selected_u,
            v_mid=selected_v,
            yaw_est=math.nan,
            u_red=math.nan,
            v_red=math.nan,
            red_area_px=math.nan,
            u_blue=math.nan,
            v_blue=math.nan,
            blue_area_px=math.nan,
            separation_px=math.nan,
            border_margin_px=border_margin,
            yolo_score_raw=raw_score,
            yolo_score_selected=selected_score,
            yolo_detected_after_threshold=1.0 if detected else 0.0,
            yolo_best_class_id=float(selection.get("best_class_id", math.nan)),
            yolo_target_candidate_count=float(selection.get("n_candidates", 0)),
            bbox_area_px=bbox_area,
            bbox_xmin=x0,
            bbox_ymin=y0,
            bbox_xmax=x1,
            bbox_ymax=y1,
            logit_margin=math.nan,
            class_entropy=math.nan,
            mask_area_px=mask_area,
            mask_bottom_u=mask_bottom_u,
            mask_bottom_v=mask_bottom_v,
            mask_used=1.0 if mask_available else 0.0,
            mask_polygon_points=mask_points,
            confidence_logit=_confidence_logit(raw_score) if raw_score > 0.0 else math.nan,
            mask_compactness=math.nan,
            mask_border_frac=math.nan,
            mask_score=selected_score if mask_available else math.nan,
            selected_pixel_source_code=_selected_pixel_source_code(
                selection.get("selected_pixel_source", "none")
            ),
            yolo_inference_ms=timing.inference_wall_ms,
            detector_callback_ms=callback_ms,
            yolo_receive_stamp=item.receive_stamp_s,
            yolo_start_stamp=timing.predict_start_stamp_s,
            yolo_finish_stamp=timing.predict_finish_stamp_s,
            yolo_publish_stamp=publish_stamp_s,
            yolo_latency_s=latency_s,
            frame_age_at_publish_s=frame_age_s,
        )
        output = self.outputs[item.camera_id]
        observation_message = self._observation_message(item.camera_id, message)
        try:
            output.diagnostics_publisher.publish(message)
            output.observation_publisher.publish(observation_message)
        except Exception as exc:
            self._fatal(f"operational publish failed for {item.camera_id}: {exc}", exc)

    def _observation_message(
        self, camera_id: str, diagnostics: Float64MultiArray
    ) -> String:
        try:
            observation = self._camera_observation_from_diagnostics(
                diagnostics_from_message(diagnostics),
                config=self._observation_configs[camera_id],
            )
            message = String()
            message.data = observation.to_json()
            return message
        except Exception as exc:
            self._fatal(
                f"CameraObservation contract failed for {camera_id}: {exc}", exc
            )

    def _write_debug_frame(
        self, item: PendingFrame, image_bgr: np.ndarray, selection: dict[str, Any]
    ) -> None:
        if not self.debug_frame_dir:
            return
        try:
            image = image_bgr.copy()
            bbox = selection.get("bbox_xyxy")
            if bbox is not None:
                x0, y0, x1, y1 = np.asarray(bbox, dtype=float).reshape(4)
                cv2.rectangle(
                    image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 1
                )
            u = float(selection.get("selected_u", math.nan))
            v = float(selection.get("selected_v", math.nan))
            if math.isfinite(u) and math.isfinite(v):
                cv2.circle(image, (int(u), int(v)), 2, (0, 255, 255), -1)
            filename = f"{item.camera_id}_{item.stamp_ns:019d}.png"
            cv2.imwrite(str(Path(self.debug_frame_dir).expanduser() / filename), image)
        except Exception as exc:
            self._warn_bounded(item.camera_id + ":debug", f"debug frame write failed: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BatchedFourCameraYoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
