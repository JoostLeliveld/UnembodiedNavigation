#!/usr/bin/env python3
"""Record EVALUATION-ONLY simulation ground truth during a commissioning run.

This recorder is deliberately a separate process and a separate output file
from ``record_operational_logs.py``.  It subscribes only to ``/ground_truth_tf``
(the unconditional ``ros_gz`` bridge of Gazebo's SceneBroadcaster
``dynamic_pose/info`` stream, see ``src/sim/launch/bringup_sim.launch.py``) and
writes ``ground_truth.csv``.  Nothing here may ever feed the operational
pipeline; the output exists so ``attach_evaluation_truth.py`` can populate the
``evaluation_only/`` half of the export split.

Example (run alongside the operational recorder):

    source install/setup.bash
    python3 experiments/multicamera_commissioning_bigwarehouse/tools/record_evaluation_truth.py \
      --out-dir logs/multicamera_commissioning_bigwarehouse/run_001/evaluation_only \
      --duration-s 120

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
import sys
import time
from typing import Any

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from tf2_msgs.msg import TFMessage
except ImportError:  # pragma: no cover - only used in non-ROS tooling contexts
    rclpy = None
    Node = object
    Parameter = Any
    TFMessage = Any


TRUTH_FIELDS = ("stamp", "gt_x", "gt_y", "gt_yaw")
EVIDENCE_ROLES = ("fit", "qualification", "diagnostic")


REPO = Path(__file__).resolve().parents[3]
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import campaign_ledger  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _stamp_s(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def _csv_float(value: float) -> str:
    number = float(value)
    return f"{number:.9f}" if math.isfinite(number) else ""


def _yaw_from_quaternion(rotation: Any) -> float:
    x = float(rotation.x)
    y = float(rotation.y)
    z = float(rotation.z)
    w = float(rotation.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class EvaluationTruthRecorder(Node):
    """Write the robot's true world pose stream to an evaluation-only CSV."""

    def __init__(
        self,
        *,
        out_dir: Path,
        topic: str,
        child_frame_id: str,
        min_interval_s: float,
        use_sim_time: bool,
    ) -> None:
        super().__init__(
            "evaluation_truth_recorder",
            parameter_overrides=[Parameter("use_sim_time", value=bool(use_sim_time))],
        )
        self.child_frame_id = child_frame_id
        self.min_interval_s = max(0.0, float(min_interval_s))
        self.first_stamp: float | None = None
        self.last_stamp: float | None = None
        self._last_written_stamp: float | None = None
        self.count = 0

        out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = out_dir / "ground_truth.csv"
        self.part_path = self.csv_path.with_suffix(self.csv_path.suffix + ".part")
        if self.csv_path.exists() or self.part_path.exists():
            raise RuntimeError(f"Refusing to overwrite truth output: {self.csv_path}")
        self._csv_handle = self.part_path.open("x", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self._csv_handle, fieldnames=TRUTH_FIELDS)
        self.writer.writeheader()
        self.create_subscription(TFMessage, topic, self._truth_callback, 50)

    def _truth_callback(self, message: TFMessage) -> None:
        for transform in message.transforms:
            if transform.child_frame_id != self.child_frame_id:
                continue
            stamp = _stamp_s(transform.header.stamp)
            if stamp <= 0.0:
                stamp = self.get_clock().now().nanoseconds * 1.0e-9
            if self.first_stamp is None:
                self.first_stamp = stamp
            self.last_stamp = stamp
            if (
                self._last_written_stamp is not None
                and stamp - self._last_written_stamp < self.min_interval_s
            ):
                return
            self._last_written_stamp = stamp
            self.writer.writerow(
                {
                    "stamp": _csv_float(stamp),
                    "gt_x": _csv_float(transform.transform.translation.x),
                    "gt_y": _csv_float(transform.transform.translation.y),
                    "gt_yaw": _csv_float(_yaw_from_quaternion(transform.transform.rotation)),
                }
            )
            self.count += 1
            if self.count % 50 == 0:
                self._csv_handle.flush()
            return

    def elapsed_s(self) -> float:
        if self.first_stamp is None or self.last_stamp is None:
            return 0.0
        return self.last_stamp - self.first_stamp

    def close(self, *, finalize: bool) -> None:
        self._csv_handle.close()
        if finalize:
            os.replace(self.part_path, self.csv_path)
        self.get_logger().info(f"wrote {self.count} ground-truth samples to {self.csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--topic", default="/ground_truth_tf")
    parser.add_argument("--child-frame-id", default="turtlebot3")
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--completion-manifest", type=Path, default=None)
    parser.add_argument("--wall-timeout-s", type=float, default=0.0)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--min-interval-s", type=float, default=0.05)
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
            "Pre-declared before recording and required to match the operational "
            "recorder. D0 readiness accepts qualification runs only."
        ),
    )
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
        default=REPO
        / "experiments/multicamera_commissioning_bigwarehouse/config/paper_analysis_plan.yaml",
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
    args = parser.parse_args()
    if rclpy is None:
        raise SystemExit("rclpy is required to record ground truth")

    if args.duration_s < 0.0 or args.wall_timeout_s < 0.0 or args.min_samples < 1:
        parser.error("durations must be non-negative and --min-samples must be positive")
    if args.duration_s <= 0.0 and args.completion_manifest is None:
        parser.error("set --duration-s or --completion-manifest; unbounded recording is forbidden")
    provenance_paths = {
        "study_config": args.study_config.expanduser().resolve(),
        "protocol": args.protocol.expanduser().resolve(),
        "analysis_plan": args.analysis_plan.expanduser().resolve(),
    }
    frozen_config_paths = [path.expanduser().resolve() for path in args.frozen_config]
    missing = [str(path) for path in provenance_paths.values() if not path.is_file()]
    missing.extend(str(path) for path in frozen_config_paths if not path.is_file())
    if missing:
        parser.error("missing provenance inputs: " + ", ".join(missing))
    try:
        campaign_contract = campaign_ledger.recorder_preflight_contract(
            ledger_path=args.campaign_ledger,
            plan_row_id=str(args.plan_row_id),
            attempt_id=str(args.attempt_id),
            seed=int(args.seed),
            evidence_role=str(args.evidence_role),
            analysis_split=str(args.analysis_split),
            output_dir=args.out_dir,
            artifact_subdir="evaluation_only",
            expected_inputs={
                "study": provenance_paths["study_config"],
                "protocol": provenance_paths["protocol"],
            },
            frozen_config_paths=frozen_config_paths,
        )
    except campaign_ledger.LedgerError as exc:
        parser.error(f"campaign preflight failed: {exc}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    final_manifest_path = args.out_dir / "evaluation_truth_manifest.json"
    progress_manifest_path = args.out_dir / "evaluation_truth_manifest.in_progress.json"
    failed_manifest_path = args.out_dir / "evaluation_truth_manifest.failed.json"
    if any(path.exists() for path in (final_manifest_path, progress_manifest_path, failed_manifest_path)):
        raise RuntimeError(f"Refusing to reuse truth output directory: {args.out_dir}")
    base_manifest = {
        "status": "in_progress",
        "started_utc": _utc_now(),
        "contains_ground_truth": True,
        "evaluation_only": True,
        "run_id": str(args.run_id),
        "plan_row_id": str(args.plan_row_id),
        "attempt_id": str(args.attempt_id),
        "analysis_split": str(args.analysis_split),
        "seed": int(args.seed),
        "evidence_role": str(args.evidence_role),
        "minimum_samples": int(args.min_samples),
        "provenance": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in provenance_paths.items()
        },
        "frozen_configs": [
            {"path": str(path), "sha256": _sha256_file(path)}
            for path in frozen_config_paths
        ],
        "campaign_contract": campaign_contract,
        "transport_environment": campaign_contract["transport_environment"],
    }
    _write_json_atomic(progress_manifest_path, base_manifest)

    rclpy.init()
    recorder = EvaluationTruthRecorder(
        out_dir=args.out_dir,
        topic=args.topic,
        child_frame_id=args.child_frame_id,
        min_interval_s=args.min_interval_s,
        use_sim_time=args.use_sim_time,
    )
    wall_start = time.monotonic()
    status = "failed"
    stop_reason = "unknown"
    failure_message = ""
    wall_timeout_s = (
        float(args.wall_timeout_s)
        if args.wall_timeout_s > 0.0
        else (10.0 * float(args.duration_s) + 60.0 if args.duration_s > 0.0 else 0.0)
    )
    try:
        while rclpy.ok():
            rclpy.spin_once(recorder, timeout_sec=0.1)
            if args.duration_s > 0.0 and recorder.elapsed_s() >= args.duration_s:
                status = "completed"
                stop_reason = "requested_duration_reached"
                break
            if args.completion_manifest is not None and args.completion_manifest.is_file():
                completion = json.loads(args.completion_manifest.read_text(encoding="utf-8"))
                if completion.get("status") == "completed":
                    status = "completed"
                    stop_reason = "route_completion_manifest"
                    break
            if wall_timeout_s > 0.0 and time.monotonic() - wall_start > wall_timeout_s:
                raise RuntimeError("wall-clock deadman reached before truth recording completed")
    except KeyboardInterrupt:
        status = "interrupted"
        stop_reason = "keyboard_interrupt"
    except BaseException as exc:
        status = "failed"
        stop_reason = "exception"
        failure_message = f"{type(exc).__name__}: {exc}"
    finally:
        if status == "completed" and recorder.count < int(args.min_samples):
            status = "failed"
            stop_reason = "minimum_samples_not_met"
            failure_message = f"truth samples {recorder.count} < {args.min_samples}"
        recorder.close(finalize=(status == "completed"))
        recorder.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    manifest = {
        **base_manifest,
        "status": status,
        "finished_utc": _utc_now(),
        "stop_reason": stop_reason,
        "failure_message": failure_message or None,
        "wall_elapsed_s": float(time.monotonic() - wall_start),
        "topic": args.topic,
        "child_frame_id": args.child_frame_id,
        "min_interval_s": float(args.min_interval_s),
        "sample_count": recorder.count,
        "first_stamp_s": recorder.first_stamp,
        "last_stamp_s": recorder.last_stamp,
        "note": (
            "Simulation truth for evaluation_only exports and calibration audits. "
            "Never an operational or model input (leakage firewall)."
        ),
    }
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
    raise RuntimeError(f"evaluation truth recording did not complete: {stop_reason}: {failure_message}")


if __name__ == "__main__":
    raise SystemExit(main())
