#!/usr/bin/env python3
"""Steady-clock ROS readiness barrier for the four-camera study.

The barrier observes operational stream timing and the detector's retained
runtime identity. It subscribes to the ground-truth TF topic solely as a
heartbeat: the callback counts messages and never reads transforms, frame IDs,
poses, or any other truth value.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import yaml

try:
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import String
    from tf2_msgs.msg import TFMessage
except ImportError:  # pragma: no cover - pure state/summary tests do not need ROS.
    rclpy = None
    Odometry = Any
    Node = object
    Clock = Any
    String = Any
    TFMessage = Any


REPO = Path(__file__).resolve().parents[3]
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
reliability_source = str(REPO / "src/reliability")
if reliability_source not in sys.path:
    sys.path.insert(0, reliability_source)
perception_source = str(REPO / "src/perception")
if perception_source not in sys.path:
    sys.path.insert(0, perception_source)

from reliability import CameraObservation  # noqa: E402
from perception.core.four_camera_runtime_contract import (  # noqa: E402
    BATCHED_CAMERA_ORDER,
    BATCHED_EXECUTABLE,
    BATCHED_RUNTIME_MODE,
    EXECUTABLE_SOURCE_KEY,
    FOUR_CAMERA_BATCH_SOURCE_KEY,
    REQUIRED_SOURCE_KEYS,
    RUNTIME_CONTRACT_TOPIC,
    RUNTIME_CONTRACT_SOURCE_KEY,
    RuntimeContractError,
    YOLO_SELECTION_SOURCE_KEY,
    build_batched_runtime_contract,
    compare_batched_runtime_contract,
    runtime_contract_sha256,
    validate_batched_runtime_contract,
)
import campaign_ledger  # noqa: E402


CAMERA_TOPICS: dict[str, str] = {
    "camera_A": "/perception/camera_observation/camera_A",
    "camera_B": "/perception/camera_observation/camera_B",
    "camera_C": "/perception/camera_observation/camera_C",
    "camera_D": "/perception/camera_observation/camera_D",
}
CORE_TOPICS: dict[str, str] = {
    "clock": "/clock",
    "odom": "/odom",
    "odom_noisy": "/odom_noisy",
    "ground_truth_heartbeat": "/ground_truth_tf",
}
DETECTOR_RUNTIME_CONTRACT_TOPIC = RUNTIME_CONTRACT_TOPIC
DEFAULT_STUDY_CONFIG = (
    REPO / "experiments/multicamera_commissioning_bigwarehouse/config/study.yaml"
)
DEFAULT_PROTOCOL = (
    REPO / "experiments/multicamera_commissioning_bigwarehouse/config/paper_protocol.yaml"
)
DEFAULT_ANALYSIS_PLAN = (
    REPO / "experiments/multicamera_commissioning_bigwarehouse/config/paper_analysis_plan.yaml"
)
DEFAULT_DETECTOR_RUNTIME_CONFIG = (
    REPO / "experiments/multicamera_commissioning_bigwarehouse/config/detector_4cam_v2.yaml"
)
DETECTOR_RUNTIME_SOURCE_PATHS = {
    EXECUTABLE_SOURCE_KEY: REPO
    / "src/perception/perception/nodes/batched_four_camera_yolo_node.py",
    FOUR_CAMERA_BATCH_SOURCE_KEY: REPO / "src/perception/perception/core/four_camera_batch.py",
    RUNTIME_CONTRACT_SOURCE_KEY: REPO
    / "src/perception/perception/core/four_camera_runtime_contract.py",
    YOLO_SELECTION_SOURCE_KEY: REPO / "src/perception/perception/core/yolo_selection.py",
}


@dataclass(frozen=True)
class ReadinessConfig:
    mode: str = "pilot"
    timeout_s: float = 60.0
    min_clock_messages: int = 20
    min_odom_messages: int = 20
    min_odom_noisy_messages: int = 20
    min_ground_truth_messages: int = 5
    min_camera_messages: int = 10
    min_unique_camera_stamps: int = 10
    min_clock_sim_span_s: float = 1.0
    max_stream_wall_age_s: float = 3.0
    max_frame_age_s: float = 0.15
    min_fresh_fraction: float = 1.0
    sample_window: int = 20

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in {"pilot", "enforce"}:
            raise ValueError("mode must be 'pilot' or 'enforce'")
        object.__setattr__(self, "mode", mode)
        for name in (
            "timeout_s",
            "min_clock_sim_span_s",
            "max_stream_wall_age_s",
            "max_frame_age_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        fraction = float(self.min_fresh_fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("min_fresh_fraction must be in [0, 1]")
        object.__setattr__(self, "min_fresh_fraction", fraction)
        for name in (
            "min_clock_messages",
            "min_odom_messages",
            "min_odom_noisy_messages",
            "min_ground_truth_messages",
            "min_camera_messages",
            "min_unique_camera_stamps",
            "sample_window",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))


@dataclass
class StreamSeries:
    message_count: int = 0
    first_wall_s: float | None = None
    last_wall_s: float | None = None
    last_stamp_progress_wall_s: float | None = None
    last_sim_stamp_s: float | None = None
    out_of_order_stamp_count: int = 0
    sim_stamps: list[float] = field(default_factory=list)

    def observe(self, *, wall_s: float, sim_stamp_s: float | None = None) -> None:
        wall = _finite(wall_s, "wall_s")
        self.message_count += 1
        if self.first_wall_s is None:
            self.first_wall_s = wall
        self.last_wall_s = wall
        if sim_stamp_s is not None:
            stamp = float(sim_stamp_s)
            if math.isfinite(stamp):
                if self.last_sim_stamp_s is None or stamp > self.last_sim_stamp_s:
                    self.last_stamp_progress_wall_s = wall
                    self.last_sim_stamp_s = stamp
                elif stamp < self.last_sim_stamp_s:
                    self.out_of_order_stamp_count += 1
                self.sim_stamps.append(stamp)

    def summary(self, *, now_wall_s: float) -> dict[str, Any]:
        finite_stamps = self.sim_stamps
        return {
            "message_count": self.message_count,
            "last_wall_age_s": (
                None if self.last_wall_s is None else max(0.0, now_wall_s - self.last_wall_s)
            ),
            "last_stamp_progress_wall_age_s": (
                None
                if self.last_stamp_progress_wall_s is None
                else max(0.0, now_wall_s - self.last_stamp_progress_wall_s)
            ),
            "wall_span_s": (
                None
                if self.first_wall_s is None or self.last_wall_s is None
                else max(0.0, self.last_wall_s - self.first_wall_s)
            ),
            "sim_stamp_count": len(finite_stamps),
            "unique_sim_stamp_count": len(set(finite_stamps)),
            "out_of_order_stamp_count": self.out_of_order_stamp_count,
            "sim_span_s": (
                None if len(finite_stamps) < 2 else max(finite_stamps) - min(finite_stamps)
            ),
        }


@dataclass(frozen=True)
class CameraTimingSample:
    timestamp_s: float
    wall_received_s: float
    frame_age_s: float | None
    inference_wall_ms: float | None
    callback_wall_ms: float | None


@dataclass
class CameraSeries:
    raw_message_count: int = 0
    valid_message_count: int = 0
    malformed_message_count: int = 0
    wrong_camera_id_count: int = 0
    duplicate_stamp_count: int = 0
    out_of_order_stamp_count: int = 0
    last_raw_wall_s: float | None = None
    last_unique_wall_s: float | None = None
    samples: list[CameraTimingSample] = field(default_factory=list)
    _seen_stamps: set[float] = field(default_factory=set)

    def observe_raw(self, *, wall_s: float) -> None:
        self.raw_message_count += 1
        self.last_raw_wall_s = _finite(wall_s, "wall_s")

    def observe_valid(
        self,
        *,
        timestamp_s: float,
        wall_s: float,
        frame_age_s: float | None,
        inference_wall_ms: float | None,
        callback_wall_ms: float | None,
    ) -> bool:
        stamp = _finite(timestamp_s, "timestamp_s")
        wall = _finite(wall_s, "wall_s")
        self.valid_message_count += 1
        if stamp in self._seen_stamps:
            self.duplicate_stamp_count += 1
            return False
        if self.samples and stamp < self.samples[-1].timestamp_s:
            self.out_of_order_stamp_count += 1
        self._seen_stamps.add(stamp)
        self.last_unique_wall_s = wall
        self.samples.append(
            CameraTimingSample(
                timestamp_s=stamp,
                wall_received_s=wall,
                frame_age_s=_finite_or_none(frame_age_s),
                inference_wall_ms=_finite_or_none(inference_wall_ms),
                callback_wall_ms=_finite_or_none(callback_wall_ms),
            )
        )
        return True

    @property
    def unique_stamp_count(self) -> int:
        return len(self.samples)

    def summary(
        self,
        *,
        now_wall_s: float,
        max_frame_age_s: float,
        sample_window: int,
    ) -> dict[str, Any]:
        recent = self.samples[-sample_window:]
        wall_times = [sample.wall_received_s for sample in recent]
        sim_stamps = [sample.timestamp_s for sample in recent]
        ages = [sample.frame_age_s for sample in recent if sample.frame_age_s is not None]
        inference = [
            sample.inference_wall_ms
            for sample in recent
            if sample.inference_wall_ms is not None
        ]
        callbacks = [
            sample.callback_wall_ms
            for sample in recent
            if sample.callback_wall_ms is not None
        ]
        fresh_count = sum(
            sample.frame_age_s is not None and sample.frame_age_s <= max_frame_age_s
            for sample in recent
        )
        return {
            "raw_message_count": self.raw_message_count,
            "valid_message_count": self.valid_message_count,
            "unique_observation_stamp_count": self.unique_stamp_count,
            "duplicate_stamp_count": self.duplicate_stamp_count,
            "out_of_order_stamp_count": self.out_of_order_stamp_count,
            "malformed_message_count": self.malformed_message_count,
            "wrong_camera_id_count": self.wrong_camera_id_count,
            "last_raw_wall_age_s": (
                None
                if self.last_raw_wall_s is None
                else max(0.0, now_wall_s - self.last_raw_wall_s)
            ),
            "last_wall_age_s": (
                None
                if self.last_unique_wall_s is None
                else max(0.0, now_wall_s - self.last_unique_wall_s)
            ),
            "summary_window_count": len(recent),
            "wall_hz": _rate_hz(wall_times),
            "sim_hz": _rate_hz(sim_stamps),
            "frame_age_s": {
                "finite_count": len(ages),
                "missing_count": len(recent) - len(ages),
                "p50": _percentile(ages, 0.50),
                "p90": _percentile(ages, 0.90),
                "max": max(ages) if ages else None,
                "threshold_s": max_frame_age_s,
                "fresh_count": fresh_count,
                "fresh_fraction": fresh_count / len(recent) if recent else 0.0,
            },
            "inference_wall_ms": {
                "finite_count": len(inference),
                "p50": _percentile(inference, 0.50),
                "p90": _percentile(inference, 0.90),
            },
            "callback_wall_ms": {
                "finite_count": len(callbacks),
                "p50": _percentile(callbacks, 0.50),
                "p90": _percentile(callbacks, 0.90),
            },
        }


class RuntimeReadinessState:
    """ROS-independent stream accumulator and fail-closed evaluator."""

    def __init__(
        self,
        *,
        camera_ids: tuple[str, ...],
        started_wall_s: float,
        expected_detector_contract: Mapping[str, Any] | None = None,
    ) -> None:
        self.started_wall_s = _finite(started_wall_s, "started_wall_s")
        self._camera_ids = tuple(camera_ids)
        self.core = {name: StreamSeries() for name in CORE_TOPICS}
        self.cameras = {camera_id: CameraSeries() for camera_id in self._camera_ids}
        try:
            self.expected_detector_contract = (
                None
                if expected_detector_contract is None
                else validate_batched_runtime_contract(expected_detector_contract)
            )
        except (RuntimeContractError, TypeError, ValueError) as exc:
            raise RuntimeError("invalid expected detector runtime contract") from exc
        self.expected_detector_contract_sha256 = (
            None
            if self.expected_detector_contract is None
            else runtime_contract_sha256(self.expected_detector_contract)
        )
        self.detector_runtime_contract_message_count = 0
        self.detector_runtime_contract: dict[str, Any] | None = None
        self.detector_runtime_contract_sha256: str | None = None
        self.detector_runtime_contract_error: str | None = None
        self.detector_runtime_contract_received_wall_s: float | None = None
        self.post_contract_started_wall_s: float | None = None
        self.precontract_core_messages = {name: 0 for name in CORE_TOPICS}
        self.precontract_camera_messages = {camera_id: 0 for camera_id in self._camera_ids}

    @property
    def detector_runtime_contract_armed(self) -> bool:
        return (
            self.detector_runtime_contract is not None
            and self.detector_runtime_contract_error is None
        )

    def _reset_post_contract_streams(self, *, wall_s: float) -> None:
        """Discard every timing sample that arrived before identity validation."""

        self.core = {name: StreamSeries() for name in CORE_TOPICS}
        self.cameras = {camera_id: CameraSeries() for camera_id in self._camera_ids}
        self.post_contract_started_wall_s = _finite(wall_s, "wall_s")

    def observe_clock(self, *, wall_s: float, sim_stamp_s: float) -> None:
        if not self.detector_runtime_contract_armed:
            self.precontract_core_messages["clock"] += 1
            return
        self.core["clock"].observe(wall_s=wall_s, sim_stamp_s=sim_stamp_s)

    def observe_odom(self, stream: str, *, wall_s: float, sim_stamp_s: float) -> None:
        if stream not in {"odom", "odom_noisy"}:
            raise ValueError(f"unknown odometry stream {stream!r}")
        if not self.detector_runtime_contract_armed:
            self.precontract_core_messages[stream] += 1
            return
        self.core[stream].observe(wall_s=wall_s, sim_stamp_s=sim_stamp_s)

    def observe_ground_truth_heartbeat(self, *, wall_s: float) -> None:
        # Intentionally no message argument: truth content cannot cross this API.
        if not self.detector_runtime_contract_armed:
            self.precontract_core_messages["ground_truth_heartbeat"] += 1
            return
        self.core["ground_truth_heartbeat"].observe(wall_s=wall_s)

    def observe_detector_runtime_contract(self, raw_json: str, *, wall_s: float) -> None:
        """Record the retained live identity before timing can clear readiness.

        The normal campaign path compares the publication against a contract
        derived from the frozen locked detector config, model bytes, and
        method-freeze source hashes.  The optional no-expectation form remains
        useful for small pure timing-state tests, but never powers ``main``.
        """

        self.detector_runtime_contract_message_count += 1
        try:
            if not isinstance(raw_json, str):
                raise RuntimeContractError("runtime-contract message data is not a string")
            parsed = json.loads(raw_json)
            contract = (
                validate_batched_runtime_contract(parsed)
                if self.expected_detector_contract is None
                else compare_batched_runtime_contract(
                    parsed, self.expected_detector_contract
                )
            )
        except Exception as exc:
            self.detector_runtime_contract_error = (
                "detector runtime contract rejected: " f"{type(exc).__name__}: {exc}"
            )
            return
        digest = runtime_contract_sha256(contract)
        if self.detector_runtime_contract_sha256 is not None:
            if digest != self.detector_runtime_contract_sha256:
                self.detector_runtime_contract_error = (
                    "detector runtime contract changed during readiness"
                )
            return
        self.detector_runtime_contract = contract
        self.detector_runtime_contract_sha256 = digest
        received_wall_s = _finite(wall_s, "wall_s")
        self.detector_runtime_contract_received_wall_s = received_wall_s
        self._reset_post_contract_streams(wall_s=received_wall_s)

    def _detector_runtime_contract_summary(self, *, now_wall_s: float) -> dict[str, Any]:
        return {
            "topic": DETECTOR_RUNTIME_CONTRACT_TOPIC,
            "qos": "reliable_transient_local_keep_last_1",
            "message_count": self.detector_runtime_contract_message_count,
            "validated": self.detector_runtime_contract is not None
            and self.detector_runtime_contract_error is None,
            "contract_sha256": self.detector_runtime_contract_sha256,
            "contract": self.detector_runtime_contract,
            "expected_contract_sha256": self.expected_detector_contract_sha256,
            "expected_contract": self.expected_detector_contract,
            "error": self.detector_runtime_contract_error,
            "received_wall_elapsed_s": (
                None
                if self.detector_runtime_contract_received_wall_s is None
                else max(
                    0.0,
                    self.detector_runtime_contract_received_wall_s - self.started_wall_s,
                )
            ),
            "post_contract_window_started_wall_elapsed_s": (
                None
                if self.post_contract_started_wall_s is None
                else max(0.0, self.post_contract_started_wall_s - self.started_wall_s)
            ),
            "precontract_messages_discarded": {
                "core": dict(self.precontract_core_messages),
                "cameras": dict(self.precontract_camera_messages),
            },
            "last_checked_wall_elapsed_s": max(0.0, now_wall_s - self.started_wall_s),
            "validation_scope": (
                "schema_and_self_hash_only_test_mode"
                if self.expected_detector_contract is None
                else "schema_self_hash_and_frozen_locked_config_model_bytes_source_hashes"
            ),
        }

    def observe_camera_raw(self, camera_id: str, *, wall_s: float) -> None:
        if not self.detector_runtime_contract_armed:
            self.precontract_camera_messages[camera_id] += 1
            return
        self.cameras[camera_id].observe_raw(wall_s=wall_s)

    def observe_camera(self, camera_id: str, sample: CameraTimingSample) -> bool:
        if not self.detector_runtime_contract_armed:
            self.precontract_camera_messages[camera_id] += 1
            return False
        return self.cameras[camera_id].observe_valid(
            timestamp_s=sample.timestamp_s,
            wall_s=sample.wall_received_s,
            frame_age_s=sample.frame_age_s,
            inference_wall_ms=sample.inference_wall_ms,
            callback_wall_ms=sample.callback_wall_ms,
        )

    def evaluate(self, config: ReadinessConfig, *, now_wall_s: float) -> dict[str, Any]:
        now = _finite(now_wall_s, "now_wall_s")
        failures: list[str] = []
        warnings: list[str] = []
        detector_contract_summary = self._detector_runtime_contract_summary(
            now_wall_s=now
        )
        if self.detector_runtime_contract_error is not None:
            failures.append(self.detector_runtime_contract_error)
        elif self.detector_runtime_contract is None:
            failures.append(
                f"{DETECTOR_RUNTIME_CONTRACT_TOPIC}: missing a validated retained "
                "detector runtime contract"
            )
        core_summary = {
            name: series.summary(now_wall_s=now) for name, series in self.core.items()
        }
        minimums = {
            "clock": config.min_clock_messages,
            "odom": config.min_odom_messages,
            "odom_noisy": config.min_odom_noisy_messages,
            "ground_truth_heartbeat": config.min_ground_truth_messages,
        }
        for name, minimum in minimums.items():
            summary = core_summary[name]
            if summary["message_count"] < minimum:
                failures.append(
                    f"{CORE_TOPICS[name]}: {summary['message_count']} messages < required {minimum}"
                )
            age_key = (
                "last_wall_age_s"
                if name == "ground_truth_heartbeat"
                else "last_stamp_progress_wall_age_s"
            )
            age = summary[age_key]
            if age is None or age > config.max_stream_wall_age_s:
                failures.append(
                    f"{CORE_TOPICS[name]}: missing or stale for more than "
                    f"{config.max_stream_wall_age_s:.3f} wall-s"
                )
        clock_span = core_summary["clock"]["sim_span_s"]
        if clock_span is None or clock_span < config.min_clock_sim_span_s:
            failures.append(
                f"/clock advanced {clock_span or 0.0:.3f} sim-s < required "
                f"{config.min_clock_sim_span_s:.3f} sim-s"
            )
        for name in ("odom", "odom_noisy"):
            if (
                core_summary[name]["message_count"] >= minimums[name]
                and core_summary[name]["unique_sim_stamp_count"] < 2
            ):
                failures.append(f"{CORE_TOPICS[name]}: message stamps are not advancing")
        for name in ("clock", "odom", "odom_noisy"):
            if core_summary[name]["out_of_order_stamp_count"]:
                failures.append(
                    f"{CORE_TOPICS[name]}: {core_summary[name]['out_of_order_stamp_count']} "
                    "out-of-order stamps"
                )

        camera_summary: dict[str, dict[str, Any]] = {}
        age_gate_by_camera: dict[str, bool] = {}
        for camera_id, series in self.cameras.items():
            summary = series.summary(
                now_wall_s=now,
                max_frame_age_s=config.max_frame_age_s,
                sample_window=config.sample_window,
            )
            camera_summary[camera_id] = summary
            topic = CAMERA_TOPICS.get(camera_id, camera_id)
            if summary["valid_message_count"] < config.min_camera_messages:
                failures.append(
                    f"{topic}: {summary['valid_message_count']} valid messages < required "
                    f"{config.min_camera_messages}"
                )
            if summary["unique_observation_stamp_count"] < config.min_unique_camera_stamps:
                failures.append(
                    f"{topic}: {summary['unique_observation_stamp_count']} unique stamps < required "
                    f"{config.min_unique_camera_stamps}"
                )
            age = summary["last_wall_age_s"]
            if age is None or age > config.max_stream_wall_age_s:
                failures.append(
                    f"{topic}: missing or stale for more than "
                    f"{config.max_stream_wall_age_s:.3f} wall-s"
                )
            if summary["out_of_order_stamp_count"]:
                failures.append(
                    f"{topic}: {summary['out_of_order_stamp_count']} out-of-order observation stamps"
                )
            age_stats = summary["frame_age_s"]
            age_pass = (
                summary["summary_window_count"] >= config.min_unique_camera_stamps
                and age_stats["finite_count"] == summary["summary_window_count"]
                and age_stats["fresh_fraction"] >= config.min_fresh_fraction
            )
            age_gate_by_camera[camera_id] = age_pass
            if not age_pass:
                message = (
                    f"{topic}: recent fresh fraction {age_stats['fresh_fraction']:.3f} < "
                    f"{config.min_fresh_fraction:.3f} at max age {config.max_frame_age_s:.3f}s"
                )
                if config.mode == "enforce":
                    failures.append(message)
                else:
                    warnings.append("pilot age diagnostic: " + message)
            for metric in ("inference_wall_ms", "callback_wall_ms"):
                if summary[metric]["finite_count"] < summary["summary_window_count"]:
                    warnings.append(f"{topic}: incomplete {metric} telemetry")

        sim_rtf = _sim_rtf(self.core["clock"])
        if sim_rtf is None or sim_rtf <= 0.0:
            failures.append("/clock: simulation real-time factor is unavailable or non-positive")
        return {
            "schema_version": 1,
            "mode": config.mode,
            "pass": not failures,
            "evidence_eligible": config.mode == "enforce" and not failures,
            "would_pass_age_enforcement": all(age_gate_by_camera.values()),
            "age_enforced": config.mode == "enforce",
            "failures": sorted(set(failures)),
            "warnings": sorted(set(warnings)),
            "elapsed_wall_s": max(0.0, now - self.started_wall_s),
            "thresholds": {
                "timeout_s": config.timeout_s,
                "max_stream_wall_age_s": config.max_stream_wall_age_s,
                "max_frame_age_s": config.max_frame_age_s,
                "min_fresh_fraction": config.min_fresh_fraction,
                "sample_window": config.sample_window,
                "min_unique_camera_stamps": config.min_unique_camera_stamps,
                "minimum_message_counts": minimums
                | {"each_camera": config.min_camera_messages},
            },
            "simulation": {
                "real_time_factor": sim_rtf,
                "clock_sim_span_s": clock_span,
                "clock_wall_span_s": core_summary["clock"]["wall_span_s"],
            },
            "streams": core_summary,
            "cameras": camera_summary,
            "age_gate_by_camera": age_gate_by_camera,
            "detector_runtime_contract": detector_contract_summary,
            "ground_truth_firewall": {
                "topic": CORE_TOPICS["ground_truth_heartbeat"],
                "message_count": core_summary["ground_truth_heartbeat"]["message_count"],
                "values_read": False,
                "purpose": "stream existence/count heartbeat only",
            },
        }


class RuntimeReadinessNode(Node):
    """ROS adapter that feeds only timing/count data into readiness state."""

    def __init__(
        self,
        *,
        camera_topics: Mapping[str, str],
        expected_detector_contract: Mapping[str, Any],
    ) -> None:
        super().__init__("bigwarehouse_runtime_readiness")
        self.camera_topics = dict(camera_topics)
        self.state = RuntimeReadinessState(
            camera_ids=tuple(self.camera_topics),
            started_wall_s=time.monotonic(),
            expected_detector_contract=expected_detector_contract,
        )
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        contract_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        # Subscribe first: an enforce pass is a pre-route barrier only if the
        # retained live detector identity has already been observed.
        self.create_subscription(
            String,
            DETECTOR_RUNTIME_CONTRACT_TOPIC,
            self._detector_runtime_contract_callback,
            contract_qos,
        )
        self.create_subscription(Clock, CORE_TOPICS["clock"], self._clock_callback, qos)
        self.create_subscription(
            Odometry,
            CORE_TOPICS["odom"],
            lambda message: self._odom_callback("odom", message),
            qos,
        )
        self.create_subscription(
            Odometry,
            CORE_TOPICS["odom_noisy"],
            lambda message: self._odom_callback("odom_noisy", message),
            qos,
        )
        self.create_subscription(
            TFMessage,
            CORE_TOPICS["ground_truth_heartbeat"],
            self._ground_truth_heartbeat_callback,
            qos,
        )
        for camera_id, topic in self.camera_topics.items():
            self.create_subscription(
                String,
                topic,
                self._camera_callback(camera_id),
                qos,
            )

    def _detector_runtime_contract_callback(self, message: String) -> None:
        self.state.observe_detector_runtime_contract(
            message.data,
            wall_s=time.monotonic(),
        )

    def _clock_callback(self, message: Clock) -> None:
        wall_s = time.monotonic()
        if not self.state.detector_runtime_contract_armed:
            # Count the callback but do not even deserialize its timing value
            # before the retained detector identity is validated.
            self.state.observe_clock(wall_s=wall_s, sim_stamp_s=0.0)
            return
        self.state.observe_clock(
            wall_s=wall_s,
            sim_stamp_s=_stamp_s(message.clock),
        )

    def _odom_callback(self, stream: str, message: Odometry) -> None:
        wall_s = time.monotonic()
        if not self.state.detector_runtime_contract_armed:
            # Pre-contract odometry samples may have belonged to a different
            # detector deployment. Count them only; never deserialize stamps.
            self.state.observe_odom(stream, wall_s=wall_s, sim_stamp_s=0.0)
            return
        self.state.observe_odom(
            stream,
            wall_s=wall_s,
            sim_stamp_s=_stamp_s(message.header.stamp),
        )

    def _ground_truth_heartbeat_callback(self, _message: TFMessage) -> None:
        # Deliberately do not inspect `_message`; this is only a liveness count.
        self.state.observe_ground_truth_heartbeat(wall_s=time.monotonic())

    def _camera_callback(self, expected_camera_id: str):
        def callback(message: String) -> None:
            wall_s = time.monotonic()
            self.state.observe_camera_raw(expected_camera_id, wall_s=wall_s)
            if not self.state.detector_runtime_contract_armed:
                # The payload may contain timing/identity from a process that
                # ran before the retained contract was received. Do not parse
                # it or count it twice as a valid timing sample.
                return
            series = self.state.cameras[expected_camera_id]
            try:
                observation = CameraObservation.from_json(message.data)
            except Exception:
                series.malformed_message_count += 1
                return
            if observation.camera_id != expected_camera_id:
                series.wrong_camera_id_count += 1
                return
            self.state.observe_camera(
                expected_camera_id,
                CameraTimingSample(
                    timestamp_s=observation.timestamp_s,
                    wall_received_s=wall_s,
                    frame_age_s=observation.frame_age_at_publish_s,
                    inference_wall_ms=observation.yolo_inference_wall_ms,
                    callback_wall_ms=observation.detector_callback_wall_ms,
                ),
            )

        return callback


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _method_freeze(analysis_path: Path) -> dict[str, Any]:
    """Recreate the recorder's exact method-freeze snapshot."""

    payload = yaml.safe_load(analysis_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"analysis plan must be a YAML mapping: {analysis_path}")
    freeze = payload.get("method_freeze")
    if not isinstance(freeze, dict) or not isinstance(freeze.get("source_files"), list):
        raise RuntimeError(f"analysis plan lacks method_freeze.source_files: {analysis_path}")
    study_dir = analysis_path.parent.parent
    source_sha256: dict[str, str] = {}
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


def _frozen_detector_source_hashes(
    *, analysis_path: Path, method_freeze: Mapping[str, Any]
) -> dict[str, str]:
    """Map plan-frozen source entries to the modules attested live by YOLO."""

    source_sha256 = method_freeze.get("source_sha256")
    if not isinstance(source_sha256, Mapping):
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
    if not candidate.is_absolute():
        candidate = REPO / candidate
    return candidate.resolve()


def _config_matches_contract_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12
            )
        except (TypeError, ValueError):
            return False
    return actual == expected


def _expected_contract_from_runtime_config(
    *,
    config_path: Path,
    detector_model: Path,
    detector_imgsz: int,
    method_freeze: Mapping[str, Any],
    analysis_path: Path,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive an exact expected live contract from a frozen deployment config.

    A readiness run may not let a shell argument choose a detector setting. It
    must prove the exact, versioned deployment selected before evidence.  Pilot
    mode can inspect one declared candidate image size, while enforce mode
    accepts only a successor that locks that one image size before collection.
    """

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"detector runtime config must be a YAML mapping: {config_path}")
    runtime = payload.get("runtime_pilot")
    if not isinstance(runtime, Mapping):
        raise RuntimeError(f"detector runtime config lacks runtime_pilot mapping: {config_path}")
    if isinstance(detector_imgsz, bool) or int(detector_imgsz) != detector_imgsz:
        raise RuntimeError("--detector-imgsz must be a positive integer")
    selected_imgsz = int(detector_imgsz)
    if selected_imgsz <= 0:
        raise RuntimeError("--detector-imgsz must be a positive integer")

    image_sizes = runtime.get("image_sizes")
    if (
        not isinstance(image_sizes, list)
        or not image_sizes
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in image_sizes)
        or len(set(image_sizes)) != len(image_sizes)
        or selected_imgsz not in image_sizes
    ):
        raise RuntimeError(
            "--detector-imgsz must be one of the unique positive runtime_pilot.image_sizes"
        )
    selection = runtime.get("evidence_selection")
    if not isinstance(selection, Mapping):
        raise RuntimeError("detector runtime config lacks runtime_pilot.evidence_selection")
    if mode == "enforce":
        if selection.get("permitted") is not True:
            raise RuntimeError(
                "enforce readiness rejects a detector config whose "
                "evidence_selection.permitted is not true"
            )
        if payload.get("lock_status") != "locked_before_evidence":
            raise RuntimeError(
                "enforce readiness requires a detector runtime config with "
                "lock_status=locked_before_evidence"
            )
        locked_imgsz = selection.get("selected_image_size")
        if (
            isinstance(locked_imgsz, bool)
            or not isinstance(locked_imgsz, int)
            or locked_imgsz != selected_imgsz
            or image_sizes != [locked_imgsz]
        ):
            raise RuntimeError(
                "enforce readiness requires one locked selected_image_size that exactly "
                "matches --detector-imgsz and runtime_pilot.image_sizes"
            )

    configured_model = _config_model_path(config_path, runtime.get("model"))
    if configured_model != detector_model.resolve():
        raise RuntimeError(
            "detector runtime config model differs from --detector-model: "
            f"{configured_model} != {detector_model.resolve()}"
        )
    try:
        expected = build_batched_runtime_contract(
            model_path=detector_model,
            model_sha256=_sha256_file(detector_model),
            imgsz=selected_imgsz,
            confidence_threshold=runtime["confidence_threshold"],
            iou_threshold=runtime["iou_threshold"],
            use_masks=runtime["masks"],
            shared_device=runtime["device"],
            cpu_threads=runtime["cpu_threads"],
            interop_threads=runtime["cpu_interop_threads"],
            opencv_threads=runtime["opencv_threads"],
            max_batch_stamp_skew_s=runtime["max_batch_stamp_skew_s"],
            max_pending_wall_s=runtime["max_pending_wall_s"],
            source_hashes=_frozen_detector_source_hashes(
                analysis_path=analysis_path,
                method_freeze=method_freeze,
            ),
        )
    except (KeyError, RuntimeContractError, TypeError, ValueError) as exc:
        raise RuntimeError("detector runtime config is not a valid batched deployment") from exc

    required_runtime_fields = {
        "detector_mode": BATCHED_RUNTIME_MODE,
        "runtime_executable": BATCHED_EXECUTABLE,
        "launch_switch": "yolo_batched_four_camera",
        "model_format": "native_ultralytics",
        "model_instances": 1,
        "batch_size": 4,
        "camera_order": list(BATCHED_CAMERA_ORDER),
        "device": expected["shared_device"],
        "cpu_threads": expected["cpu_threads"],
        "cpu_interop_threads": expected["interop_threads"],
        "opencv_threads": expected["opencv_threads"],
        "max_batch_stamp_skew_s": expected["max_batch_stamp_skew_s"],
        "max_pending_wall_s": expected["max_pending_wall_s"],
        "frame_policy": expected["frame_policy"],
        "inference_timing_semantics": expected["inference_timing_semantics"],
        "fault_policy": expected["fault_policy"],
        "synchronization_miss_policy": expected["synchronization_miss_policy"],
        "masks": expected["use_masks"],
        "confidence_threshold": expected["confidence_threshold"],
        "iou_threshold": expected["iou_threshold"],
    }
    mismatches = [
        key
        for key, expected_value in required_runtime_fields.items()
        if not _config_matches_contract_value(runtime.get(key), expected_value)
    ]
    if mismatches:
        raise RuntimeError(
            "detector runtime config differs from the required batched contract: "
            + ", ".join(sorted(mismatches))
        )
    metadata = {
        "path": str(config_path),
        "sha256": _sha256_file(config_path),
        "artifact_id": payload.get("artifact_id"),
        "lock_status": payload.get("lock_status"),
        "evidence_selection": dict(selection),
        "selected_imgsz": selected_imgsz,
    }
    return expected, metadata


def _campaign_preflight(
    args: argparse.Namespace,
    *,
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path], list[Path]]:
    """Bind this pre-route barrier to one immutable ledger attempt slot."""

    paths = {
        "study": args.study_config.expanduser().resolve(),
        "protocol": args.protocol.expanduser().resolve(),
        "analysis_plan": args.analysis_plan.expanduser().resolve(),
        "detector_runtime_config": args.detector_runtime_config.expanduser().resolve(),
        "model": args.detector_model.expanduser().resolve(),
        "calibration": args.projection_calibration.expanduser().resolve(),
    }
    frozen_configs = [path.expanduser().resolve() for path in args.frozen_config]
    missing = [str(path) for path in paths.values() if not path.is_file()]
    missing.extend(str(path) for path in frozen_configs if not path.is_file())
    if missing:
        raise RuntimeError("missing readiness provenance inputs: " + ", ".join(missing))
    if sum(path == paths["analysis_plan"] for path in frozen_configs) != 1:
        raise RuntimeError("--analysis-plan must be included exactly once in --frozen-config")
    if sum(path == paths["detector_runtime_config"] for path in frozen_configs) != 1:
        raise RuntimeError(
            "--detector-runtime-config must be included exactly once in --frozen-config"
        )
    try:
        campaign_contract = campaign_ledger.recorder_preflight_contract(
            ledger_path=args.campaign_ledger.expanduser().resolve(),
            plan_row_id=str(args.plan_row_id),
            attempt_id=str(args.attempt_id),
            seed=int(args.seed),
            evidence_role=str(args.evidence_role),
            analysis_split=str(args.analysis_split),
            # `runtime_readiness.json` belongs at the attempt root, unlike the
            # operational recorder's `raw/` stream directory.
            output_dir=output.parent,
            artifact_subdir=".",
            expected_inputs={
                "study": paths["study"],
                "protocol": paths["protocol"],
                "model": paths["model"],
                "calibration": paths["calibration"],
            },
            frozen_config_paths=frozen_configs,
        )
    except campaign_ledger.LedgerError as exc:
        raise RuntimeError(f"campaign preflight failed: {exc}") from exc
    method_freeze = _method_freeze(paths["analysis_plan"])
    if campaign_contract.get("method_freeze") != method_freeze:
        raise RuntimeError(
            "campaign method-freeze snapshot differs from the locally resolved source bytes"
        )
    campaign_method_freeze_sha256 = campaign_contract.get("method_freeze_sha256")
    if not isinstance(campaign_method_freeze_sha256, str) or len(
        campaign_method_freeze_sha256
    ) != 64:
        raise RuntimeError("campaign method-freeze SHA-256 is missing or malformed")
    return campaign_contract, method_freeze, paths, frozen_configs


def write_json_atomic_new(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically publish a JSON report and refuse to replace prior evidence."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _stamp_s(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = float(quantile) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _rate_hz(stamps: list[float]) -> float | None:
    if len(stamps) < 2:
        return None
    span = max(stamps) - min(stamps)
    return (len(stamps) - 1) / span if span > 0.0 else None


def _sim_rtf(clock: StreamSeries) -> float | None:
    if (
        clock.first_wall_s is None
        or clock.last_wall_s is None
        or len(clock.sim_stamps) < 2
    ):
        return None
    wall_span = clock.last_wall_s - clock.first_wall_s
    sim_span = max(clock.sim_stamps) - min(clock.sim_stamps)
    if wall_span <= 0.0 or sim_span <= 0.0:
        return None
    return sim_span / wall_span


def _camera_topics(overrides: list[str]) -> dict[str, str]:
    topics = dict(CAMERA_TOPICS)
    for entry in overrides:
        if "=" not in entry:
            raise ValueError(f"camera topic override must be camera_id=/topic, got {entry!r}")
        camera_id, topic = (part.strip() for part in entry.split("=", 1))
        if camera_id not in topics or not topic.startswith("/"):
            raise ValueError(f"invalid camera topic override {entry!r}")
        topics[camera_id] = topic
    return topics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "enforce"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--campaign-ledger",
        type=Path,
        required=True,
        help="Immutable campaign ledger owning this pre-route attempt.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan-row-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--analysis-split", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--evidence-role",
        choices=("fit", "qualification", "diagnostic"),
        required=True,
    )
    parser.add_argument(
        "--frozen-config",
        type=Path,
        action="append",
        required=True,
        help="Repeat the exact campaign --config set, including --analysis-plan.",
    )
    parser.add_argument("--study-config", type=Path, default=DEFAULT_STUDY_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--analysis-plan", type=Path, default=DEFAULT_ANALYSIS_PLAN)
    parser.add_argument("--detector-model", type=Path, required=True)
    parser.add_argument(
        "--detector-runtime-config",
        type=Path,
        default=DEFAULT_DETECTOR_RUNTIME_CONFIG,
        help=(
            "Locked versioned detector deployment config. It must appear exactly "
            "once in --frozen-config."
        ),
    )
    parser.add_argument(
        "--detector-imgsz",
        type=int,
        required=True,
        help=(
            "Exact active YOLO image size. It must be a declared candidate in "
            "the frozen config and, in enforce mode, its sole locked selection."
        ),
    )
    parser.add_argument("--projection-calibration", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--min-clock-messages", type=int, default=20)
    parser.add_argument("--min-odom-messages", type=int, default=20)
    parser.add_argument("--min-odom-noisy-messages", type=int, default=20)
    parser.add_argument("--min-ground-truth-messages", type=int, default=5)
    parser.add_argument("--min-camera-messages", type=int, default=10)
    parser.add_argument("--min-unique-camera-stamps", type=int, default=10)
    parser.add_argument("--min-clock-sim-span-s", type=float, default=1.0)
    parser.add_argument("--max-stream-wall-age-s", type=float, default=3.0)
    parser.add_argument("--max-frame-age-s", type=float, default=0.15)
    parser.add_argument("--min-fresh-fraction", type=float, default=1.0)
    parser.add_argument("--sample-window", type=int, default=20)
    parser.add_argument(
        "--camera-topic",
        action="append",
        default=[],
        metavar="CAMERA_ID=/TOPIC",
        help="Override one of the four default CameraObservation topics",
    )
    args = parser.parse_args()

    if rclpy is None:
        raise RuntimeError("rclpy is required; source the ROS workspace before runtime readiness")
    try:
        config = ReadinessConfig(
            mode=args.mode,
            timeout_s=args.timeout_s,
            min_clock_messages=args.min_clock_messages,
            min_odom_messages=args.min_odom_messages,
            min_odom_noisy_messages=args.min_odom_noisy_messages,
            min_ground_truth_messages=args.min_ground_truth_messages,
            min_camera_messages=args.min_camera_messages,
            min_unique_camera_stamps=args.min_unique_camera_stamps,
            min_clock_sim_span_s=args.min_clock_sim_span_s,
            max_stream_wall_age_s=args.max_stream_wall_age_s,
            max_frame_age_s=args.max_frame_age_s,
            min_fresh_fraction=args.min_fresh_fraction,
            sample_window=args.sample_window,
        )
        camera_topics = _camera_topics(args.camera_topic)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"refusing to overwrite readiness report: {output}")
    try:
        campaign_contract, method_freeze, provenance_paths, frozen_config_paths = (
            _campaign_preflight(args, output=output)
        )
        expected_detector_contract, detector_runtime_config = (
            _expected_contract_from_runtime_config(
                config_path=provenance_paths["detector_runtime_config"],
                detector_model=provenance_paths["model"],
                detector_imgsz=args.detector_imgsz,
                method_freeze=method_freeze,
                analysis_path=provenance_paths["analysis_plan"],
                mode=config.mode,
            )
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    provenance = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in provenance_paths.items()
    }
    frozen_configs = [
        {"path": str(path), "sha256": _sha256_file(path)}
        for path in frozen_config_paths
    ]

    rclpy.init()
    node = RuntimeReadinessNode(
        camera_topics=camera_topics,
        expected_detector_contract=expected_detector_contract,
    )
    deadline = node.state.started_wall_s + config.timeout_s
    timed_out = False
    interrupted = False
    try:
        while rclpy.ok():
            now_wall_s = time.monotonic()
            if now_wall_s >= deadline:
                timed_out = True
                break
            rclpy.spin_once(node, timeout_sec=min(0.1, deadline - now_wall_s))
            if node.state.evaluate(config, now_wall_s=time.monotonic())["pass"]:
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        report = node.state.evaluate(config, now_wall_s=time.monotonic())
        if report["pass"]:
            timed_out = False
        report.update(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "completed" if report["pass"] else "failed",
                "run_id": str(args.run_id),
                "plan_row_id": str(args.plan_row_id),
                "attempt_id": str(args.attempt_id),
                "analysis_split": str(args.analysis_split),
                "seed": int(args.seed),
                "evidence_role": str(args.evidence_role),
                "campaign_contract": campaign_contract,
                "transport_environment": campaign_contract[
                    "transport_environment"
                ],
                "provenance": provenance,
                "frozen_configs": frozen_configs,
                "method_freeze": method_freeze,
                "method_freeze_sha256": campaign_contract["method_freeze_sha256"],
                "detector_runtime_config": detector_runtime_config,
                "contains_ground_truth": False,
                "timed_out": timed_out,
                "interrupted": interrupted,
                "topics": {"core": dict(CORE_TOPICS), "cameras": camera_topics},
            }
        )
        if interrupted:
            report["failures"] = sorted(set(report["failures"] + ["readiness interrupted"]))
            report["pass"] = False
            report["evidence_eligible"] = False
        elif timed_out and not report["pass"]:
            report["failures"] = sorted(
                set(report["failures"] + ["steady-clock readiness timeout expired"])
            )
        report["status"] = "completed" if report["pass"] else "failed"
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    write_json_atomic_new(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
