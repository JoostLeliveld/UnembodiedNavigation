#!/usr/bin/env python3
"""Steady-clock ROS readiness barrier for the four-camera study.

The barrier observes operational stream timing only.  It subscribes to the
ground-truth TF topic solely as a heartbeat: the callback counts messages and
never reads transforms, frame IDs, poses, or any other truth value.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

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
reliability_source = str(REPO / "src/reliability")
if reliability_source not in sys.path:
    sys.path.insert(0, reliability_source)

from reliability import CameraObservation  # noqa: E402


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

    def __init__(self, *, camera_ids: tuple[str, ...], started_wall_s: float) -> None:
        self.started_wall_s = _finite(started_wall_s, "started_wall_s")
        self.core = {name: StreamSeries() for name in CORE_TOPICS}
        self.cameras = {camera_id: CameraSeries() for camera_id in camera_ids}

    def observe_clock(self, *, wall_s: float, sim_stamp_s: float) -> None:
        self.core["clock"].observe(wall_s=wall_s, sim_stamp_s=sim_stamp_s)

    def observe_odom(self, stream: str, *, wall_s: float, sim_stamp_s: float) -> None:
        if stream not in {"odom", "odom_noisy"}:
            raise ValueError(f"unknown odometry stream {stream!r}")
        self.core[stream].observe(wall_s=wall_s, sim_stamp_s=sim_stamp_s)

    def observe_ground_truth_heartbeat(self, *, wall_s: float) -> None:
        # Intentionally no message argument: truth content cannot cross this API.
        self.core["ground_truth_heartbeat"].observe(wall_s=wall_s)

    def observe_camera_raw(self, camera_id: str, *, wall_s: float) -> None:
        self.cameras[camera_id].observe_raw(wall_s=wall_s)

    def observe_camera(self, camera_id: str, sample: CameraTimingSample) -> bool:
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
            "ground_truth_firewall": {
                "topic": CORE_TOPICS["ground_truth_heartbeat"],
                "message_count": core_summary["ground_truth_heartbeat"]["message_count"],
                "values_read": False,
                "purpose": "stream existence/count heartbeat only",
            },
        }


class RuntimeReadinessNode(Node):
    """ROS adapter that feeds only timing/count data into readiness state."""

    def __init__(self, *, camera_topics: Mapping[str, str]) -> None:
        super().__init__("bigwarehouse_runtime_readiness")
        self.camera_topics = dict(camera_topics)
        self.state = RuntimeReadinessState(
            camera_ids=tuple(self.camera_topics),
            started_wall_s=time.monotonic(),
        )
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
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

    def _clock_callback(self, message: Clock) -> None:
        self.state.observe_clock(
            wall_s=time.monotonic(),
            sim_stamp_s=_stamp_s(message.clock),
        )

    def _odom_callback(self, stream: str, message: Odometry) -> None:
        self.state.observe_odom(
            stream,
            wall_s=time.monotonic(),
            sim_stamp_s=_stamp_s(message.header.stamp),
        )

    def _ground_truth_heartbeat_callback(self, _message: TFMessage) -> None:
        # Deliberately do not inspect `_message`; this is only a liveness count.
        self.state.observe_ground_truth_heartbeat(wall_s=time.monotonic())

    def _camera_callback(self, expected_camera_id: str):
        def callback(message: String) -> None:
            wall_s = time.monotonic()
            series = self.state.cameras[expected_camera_id]
            self.state.observe_camera_raw(expected_camera_id, wall_s=wall_s)
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

    rclpy.init()
    node = RuntimeReadinessNode(camera_topics=camera_topics)
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
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    write_json_atomic_new(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
