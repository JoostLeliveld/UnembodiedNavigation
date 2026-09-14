#!/usr/bin/env python3
"""Run one immutable reference-controlled commissioning drive."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import yaml

from validate_protocol import REPO, sha256_file
from validate_protocol import validate as _validate_v1
from validate_protocol_v2 import validate as _validate_v2


def validate(protocol_path):
    """Dispatch to the validator matching the protocol's declared schema."""
    import yaml as _yaml

    schema = _yaml.safe_load(Path(protocol_path).read_text(encoding="utf-8")).get(
        "schema_version"
    )
    if schema == "reference_controlled_commissioning.v2":
        return _validate_v2(Path(protocol_path))
    return _validate_v1(Path(protocol_path))
from build_tables import build_tables


HERE = Path(__file__).resolve().parent


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _ros_environment(domain_id: int, ros_log_dir: Path) -> dict[str, str]:
    setup = REPO / "install/setup.bash"
    if not setup.is_file():
        raise RuntimeError("install/setup.bash is missing; build the workspace before collection")
    completed = subprocess.run(
        ["bash", "-lc", f"source {setup} && env -0"],
        check=True,
        capture_output=True,
    )
    environment = dict(os.environ)
    for item in completed.stdout.split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            environment[key.decode()] = value.decode()
    environment["ROS_DOMAIN_ID"] = str(domain_id)
    # ROS_DOMAIN_ID does not isolate Gazebo Transport.  Without a unique
    # partition, the next drive can briefly attach to a shutting-down server
    # with the same world name and then observe its simulation clock jump
    # backwards when the new server appears.
    transport_partition = f"reference_controlled_commissioning_{domain_id}"
    environment["IGN_PARTITION"] = transport_partition
    environment["GZ_PARTITION"] = transport_partition
    environment["ROS_LOG_DIR"] = str(ros_log_dir)
    environment["IGN_LOG_PATH"] = str(ros_log_dir.parent / "ignition_logs")
    environment["GZ_LOG_PATH"] = str(ros_log_dir.parent / "ignition_logs")
    return environment


def _stop(process: subprocess.Popen | None, timeout_s: float = 8.0) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3.0)


def _reap_simulator(environment: dict[str, str], timeout_s: float = 20.0) -> dict[str, Any]:
    """Make sure this drive's Gazebo server is gone before the next drive starts.

    `ros2 launch` does not always reap the simulator it spawned, and a surviving server
    keeps its GPU memory.  Two leaked servers exhaust a 4 GB card and the next drive's
    detector dies with CUDA out-of-memory, so a drive that leaks one silently breaks the
    drive after it.  Only processes in this drive's own ROS domain are touched.
    """
    domain = str(environment.get("ROS_DOMAIN_ID", ""))
    matched: list[int] = []
    try:
        listing = subprocess.run(
            ["pgrep", "-f", "ign gazebo"], capture_output=True, text=True, check=False
        )
        for line in listing.stdout.split():
            pid = int(line)
            try:
                environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            except OSError:
                continue
            values = {item.split(b"=", 1)[0].decode(): item.split(b"=", 1)[1].decode()
                      for item in environ if b"=" in item}
            if domain and values.get("ROS_DOMAIN_ID") == domain:
                matched.append(pid)
    except Exception:
        return {"attempted": False, "pids": []}
    for pid in matched:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout_s
    alive = list(matched)
    while alive and time.monotonic() < deadline:
        alive = [pid for pid in alive if Path(f"/proc/{pid}").exists()]
        if alive:
            time.sleep(0.5)
    for pid in alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return {"attempted": True, "pids": matched, "killed_hard": alive}


def _task_for(protocol: dict[str, Any], drive: dict[str, Any]) -> str:
    route = protocol["routes"][drive["route"]]
    return str(route[f"{drive['direction']}_task"])


def _route_length(protocol: dict[str, Any], drive: dict[str, Any]) -> float:
    points = protocol["routes"][drive["route"]]["points"]
    return float(sum(math.dist(first, second) for first, second in zip(points, points[1:])))


def launch_command(
    protocol_path: Path,
    protocol: dict[str, Any],
    drive: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    detector = protocol["detector"]
    controller = protocol["controller"]
    # Corners cost time a straight-line estimate does not capture: the robot slows to turn
    # and the pure-pursuit path is slightly longer than the polyline.  The controller ends
    # the run at route completion, so this is only a backstop and generosity is free.
    turns = max(len(protocol["routes"][drive["route"]]["points"]) - 2, 0)
    timeout_after_first_cmd = math.ceil(
        _route_length(protocol, drive) / float(controller["nominal_cruise_mps"])
        + 2.0 * turns
        + 30.0
    )
    tasks_path = (REPO / protocol["launch"]["tasks_path"]).resolve()
    checkpoint = (REPO / detector["checkpoint_path"]).resolve()
    return [
        "ros2", "launch", "experiments", "warehouse_primary_comparison.launch.py",
        f"world:={protocol['world']['name']}",
        f"world_profiles:={(REPO / protocol['world']['profiles_path']).resolve()}",
        f"tasks_yaml:={tasks_path}",
        f"task:={_task_for(protocol, drive)}",
        "planner:=geometric_shortest_path",
        f"seed:={int(drive['seed'])}",
        f"log_dir:={output_dir / 'experiment_logs'}",
        f"outcome_journal_path:={output_dir / 'detector_outcomes.jsonl'}",
        f"manager_outcome_journal_path:={output_dir / 'manager_outcomes.jsonl'}",
        f"campaign_config_path:={protocol_path}",
        "enable_mission:=false",
        "auto_stop_on_goal:=false",
        f"run_timeout_after_first_cmd_s:={timeout_after_first_cmd}",
        "stuck_window_s:=9999.0",
        "perception_backend:=yolo",
        "multicam_belief:=true",
        f"yolo_model:={checkpoint}",
        f"yolo_device:={detector['device']}",
        f"yolo_imgsz:={int(detector['image_size_px'])}",
        f"yolo_conf_threshold:={float(detector['confidence_threshold'])}",
        f"yolo_predict_conf_floor:={float(detector['predict_confidence_floor'])}",
        f"yolo_iou_threshold:={float(detector['iou_threshold'])}",
        f"yolo_target_class:={detector['target_class']}",
        f"yolo_class_id:={int(detector['class_id'])}",
        "yolo_use_masks:=false",
        f"yolo_debug_crop_dir:={output_dir / 'selected_crops'}",
        "manager_require_gp_artifacts:=false",
        "manager_covariance_profile:=commissioned_sigma_px",
        "manager_commissioned_sigma_px:=1.0",
        "manager_observation_model:=raw_box",
        "manager_fusion_mode:=true",
        # Keep the existing estimator alive only as a shadow integrity consumer.
        # The reference controller owns the motion and the learning tables use the
        # pre-NIS detector/manager journals, but every published correction still
        # needs a terminal ledger outcome for an evidence-valid run.
        "state_correction_ekf:=true",
        "state_correction_mode:=fused",
        "require_state_correction_envelope:=true",
        "use_pixel_correction:=false",
        "v_max:=1.0",
        "use_command_noise:=true",
        "use_encoder_noise:=true",
        # Truth-derived geometric termination is forbidden by the logger contract.
        # Physical contact evidence remains active and is the authoritative abort.
        "terminate_on_geom_collision:=false",
        "enable_logging:=true",
        "headless:=true",
        "use_rviz:=false",
        "reset_world:=false",
    ]


def _find_run_summary(output_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    summaries = sorted((output_dir / "experiment_logs").glob("experiment_*/run_summary.json"))
    if len(summaries) != 1:
        return None, None
    return summaries[0], json.loads(summaries[0].read_text(encoding="utf-8"))


def run(protocol_path: Path, drive_id: str, output_root: Path, *, dry_run: bool) -> int:
    preflight = validate(protocol_path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    matches = [drive for drive in protocol["drives"] if drive["id"] == drive_id]
    if len(matches) != 1:
        raise ValueError(f"unknown drive id: {drive_id}")
    drive = matches[0]
    output_dir = output_root.resolve() / drive_id
    command = launch_command(protocol_path, protocol, drive, output_dir)
    if dry_run:
        print(json.dumps({
            "drive": drive,
            "task": _task_for(protocol, drive),
            "route_length_m": _route_length(protocol, drive),
            "output_dir": str(output_dir),
            "ros_domain_id": int(protocol["capture"]["ros_domain_id_base"]) + int(drive["order"]),
            "launch_command": command,
            "preflight": preflight,
        }, indent=2, sort_keys=True))
        return 0

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "ros_logs").mkdir()
    (output_dir / "ignition_logs").mkdir()
    _atomic_json(output_dir / "preflight.json", preflight)
    _atomic_json(output_dir / "drive_manifest.json", {
        "schema": "reference_controlled_commissioning_drive_manifest.v1",
        "drive": drive,
        "task": _task_for(protocol, drive),
        "route_points": protocol["routes"][drive["route"]]["points"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "launch_command": command,
    })
    domain_id = int(protocol["capture"]["ros_domain_id_base"]) + int(drive["order"])
    environment = _ros_environment(domain_id, output_dir / "ros_logs")

    launch_log = (output_dir / "launch_console.log").open("x", encoding="utf-8")
    controller_log = (output_dir / "controller_console.log").open("x", encoding="utf-8")
    recorder_log = (output_dir / "frame_recorder_console.log").open("x", encoding="utf-8")
    launch = controller = recorder = None
    timed_out = False
    simulator_reap: dict[str, Any] = {"attempted": False, "pids": []}
    try:
        launch = subprocess.Popen(command, cwd=REPO, env=environment, stdout=launch_log, stderr=subprocess.STDOUT)
        recorder = subprocess.Popen([
            sys.executable, str(HERE / "raw_frame_identity_recorder.py"),
            "--drive-id", drive_id,
            "--index-jsonl", str(output_dir / "frame_identity.jsonl"),
        ], cwd=REPO, env=environment, stdout=recorder_log, stderr=subprocess.STDOUT)
        controller = subprocess.Popen([
            sys.executable, str(HERE / "reference_pose_controller.py"),
            "--protocol", str(protocol_path),
            "--drive-id", drive_id,
            "--trace-jsonl", str(output_dir / "reference_controller_trace.jsonl"),
            "--status-json", str(output_dir / "reference_controller_status.json"),
        ], cwd=REPO, env=environment, stdout=controller_log, stderr=subprocess.STDOUT)

        deadline = time.monotonic() + float(protocol["capture"]["runner_wall_timeout_s"])
        settle_s = float(protocol["capture"].get("post_route_settle_s", 2.0))
        while launch.poll() is None and time.monotonic() < deadline:
            controller_state = controller.poll()
            if controller_state is not None:
                # The controller owns termination.  Whether it finished the route or
                # failed, stop the stack now: leaving it running logs the robot parked at
                # the goal, and those repeated renders of one pose are not commissioning
                # samples.  A short settle lets the terminal stop be acknowledged first.
                if controller_state == 0:
                    time.sleep(settle_s)
                _stop(launch)
                break
            time.sleep(0.2)
        if launch.poll() is None:
            timed_out = True
            _stop(launch)
    except KeyboardInterrupt:
        _stop(launch)
        raise
    finally:
        _stop(controller)
        _stop(recorder)
        _stop(launch)
        simulator_reap = _reap_simulator(environment)
        launch_log.close()
        controller_log.close()
        recorder_log.close()

    status_path = output_dir / "reference_controller_status.json"
    controller_status = (
        json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else None
    )
    summary_path, summary = _find_run_summary(output_dir)
    frame_index = output_dir / "frame_identity.jsonl"
    frame_rows = 0
    if frame_index.is_file():
        with frame_index.open("r", encoding="utf-8") as handle:
            frame_rows = sum(1 for line in handle if line.strip())
    table_integrity = None
    table_error = ""
    if (
        drive["partition"] != "audit"
        and controller_status
        and controller_status.get("complete")
        and summary
        and summary.get("completed")
        and frame_rows > 0
    ):
        try:
            table_integrity = build_tables(output_dir, protocol_path)
        except Exception as exc:
            table_error = f"{type(exc).__name__}: {exc}"
    summary_valid = bool(summary and summary.get("valid_run") is True)
    terminal_stop_verified = bool(
        summary and summary.get("terminal_stop_verified") is True
    )
    correction_ledger_valid = bool(
        summary
        and isinstance(summary.get("correction_ledger"), dict)
        and summary["correction_ledger"].get("valid") is True
    )
    passed = bool(
        not timed_out
        and controller_status
        and controller_status.get("complete")
        and summary
        and summary.get("completed")
        and summary_valid
        and terminal_stop_verified
        and correction_ledger_valid
        and frame_rows > 0
        and (drive["partition"] == "audit" or (
            table_integrity and table_integrity.get("passed")
        ))
    )
    result = {
        "schema": "reference_controlled_commissioning_drive_result.v1",
        "drive_id": drive_id,
        "passed": passed,
        "runner_wall_timeout": timed_out,
        "launch_returncode": None if launch is None else launch.returncode,
        "controller_returncode": None if controller is None else controller.returncode,
        "recorder_returncode": None if recorder is None else recorder.returncode,
        "controller_status": controller_status,
        "simulator_reap": simulator_reap,
        "run_summary_path": None if summary_path is None else str(summary_path),
        "run_completion_reason": None if summary is None else summary.get("completion_reason"),
        "valid_run": summary_valid,
        "invalid_reason": None if summary is None else summary.get("invalid_reason"),
        "terminal_stop_verified": terminal_stop_verified,
        "correction_ledger_valid": correction_ledger_valid,
        "frame_identity_rows": frame_rows,
        "table_integrity": table_integrity,
        "table_error": table_error,
        "audit_outcomes_sealed": drive["partition"] == "audit",
    }
    _atomic_json(output_dir / "drive_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=HERE / "campaign_v2.yaml")
    parser.add_argument("--drive-id", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "logs/commissioning/reference_controlled_v1",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run(args.protocol.resolve(), args.drive_id, args.output_root, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
