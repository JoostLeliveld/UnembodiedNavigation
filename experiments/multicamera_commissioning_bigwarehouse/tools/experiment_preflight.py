#!/usr/bin/env python3
"""Fail-closed host/static preflight for four-camera experiments.

This check intentionally runs before Gazebo or training.  It catches stale
drivers/recorders, resource contention, frozen-input drift, unsafe route
variants, protocol/analysis mismatches, and the known four-GPU-process OOM
configuration.  It never kills processes or mutates experiment artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[3]
for relative in ("src/unav_common",):
    location = str(REPO / relative)
    if location not in sys.path:
        sys.path.insert(0, location)

from unav_common.occlusion_geometry import (  # noqa: E402
    parse_collision_scene_from_world,
    signed_distance_to_union_xy,
)


STUDY_DIR = REPO / "experiments/multicamera_commissioning_bigwarehouse"
DEFAULT_STUDY = STUDY_DIR / "config/study.yaml"
DEFAULT_PROTOCOL = STUDY_DIR / "config/paper_protocol.yaml"
DEFAULT_ANALYSIS = STUDY_DIR / "config/paper_analysis_plan.yaml"
DEFAULT_DETECTOR_CONFIG = STUDY_DIR / "config/detector_4cam_v1.yaml"
DEFAULT_WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
STALE_PATTERNS = (
    "drive_study_route.py",
    "record_operational_logs.py",
    "record_evaluation_truth.py",
    "batched_four_camera_yolo_node",
    "yolo_robot_detector_node",
    "camera_manager_node",
    "ros2 launch experiments warehouse_full4cam_commissioning.launch.py",
    "gz sim",
    "ign gazebo",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _processes() -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    own_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        if not command or "experiment_preflight.py" in command:
            continue
        hits = [pattern for pattern in STALE_PATTERNS if pattern in command]
        if hits:
            matches.append({"pid": int(entry.name), "patterns": hits, "command": command})
    return sorted(matches, key=lambda item: item["pid"])


def _meminfo() -> dict[str, float]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0]) * 1024
    gib = 1024.0**3
    swap_total = values.get("SwapTotal", 0)
    swap_free = values.get("SwapFree", 0)
    return {
        "available_gib": values.get("MemAvailable", 0) / gib,
        "swap_total_gib": swap_total / gib,
        "swap_used_gib": (swap_total - swap_free) / gib,
        "swap_used_fraction": ((swap_total - swap_free) / swap_total) if swap_total else 0.0,
    }


def _vmstat_swap_pages(path: Path = Path("/proc/vmstat")) -> tuple[int, int]:
    """Return cumulative swap-in/out page counters from the kernel."""
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in {"pswpin", "pswpout"}:
            values[parts[0]] = int(parts[1])
    if "pswpin" not in values or "pswpout" not in values:
        raise ValueError(f"missing pswpin/pswpout counters in {path}")
    return values["pswpin"], values["pswpout"]


def _swap_activity(
    observation_s: float,
    *,
    read_pages: Any = _vmstat_swap_pages,
    sleep: Any = time.sleep,
    clock: Any = time.monotonic,
    page_size_bytes: int | None = None,
) -> dict[str, Any]:
    """Measure current swap I/O; occupied but idle swap is not memory pressure.

    Linux commonly leaves cold pages in swap after memory pressure has ended.
    Timing/training readiness therefore depends on current swap traffic together
    with MemAvailable, rather than on occupancy alone.
    """
    if observation_s <= 0.0:
        raise ValueError("swap observation interval must be positive")
    try:
        before_in, before_out = read_pages()
        start = clock()
        sleep(observation_s)
        elapsed = max(clock() - start, 1e-9)
        after_in, after_out = read_pages()
        page_size = int(page_size_bytes or os.sysconf("SC_PAGE_SIZE"))
        delta_in = max(0, after_in - before_in)
        delta_out = max(0, after_out - before_out)
        scale = page_size / (1024.0**2 * elapsed)
        return {
            "available": True,
            "observation_s": elapsed,
            "pages_in": delta_in,
            "pages_out": delta_out,
            "mib_per_s_in": delta_in * scale,
            "mib_per_s_out": delta_out * scale,
            "mib_per_s_total": (delta_in + delta_out) * scale,
        }
    except (OSError, ValueError) as exc:
        return {"available": False, "observation_s": observation_s, "error": str(exc)}


def _gpu() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}
    if result.returncode != 0:
        return {"available": False, "error": (result.stderr or result.stdout).strip()}
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {"available": bool(rows), "rows": rows}


def _route_clearances(study: dict[str, Any], world: Path) -> list[dict[str, Any]]:
    scene = parse_collision_scene_from_world(str(world))
    offsets = [float(value) for value in study["collection"]["lateral_offsets_m"]]
    rows: list[dict[str, Any]] = []
    for route in study["collection"]["routes"]:
        x0, y0 = float(route["start"]["x"]), float(route["start"]["y"])
        x1, y1 = float(route["goal"]["x"]), float(route["goal"]["y"])
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length <= 1.0e-9:
            rows.append({"route": route["name"], "error": "coincident endpoints"})
            continue
        left_x, left_y = -dy / length, dx / length
        samples = max(2, int(math.ceil(length / 0.02)) + 1)
        fraction = np.linspace(0.0, 1.0, samples)
        for offset in offsets:
            points = np.column_stack((
                x0 + offset * left_x + fraction * dx,
                y0 + offset * left_y + fraction * dy,
            ))
            distances = signed_distance_to_union_xy(scene.prisms, points, keep_in=False)
            index = int(np.argmin(distances))
            rows.append({
                "route": str(route["name"]),
                "offset_m": offset,
                "minimum_collision_clearance_m": float(distances[index]),
                "minimum_at_xy_m": [float(points[index, 0]), float(points[index, 1])],
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "confirmatory", "training"), required=True)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--analysis-plan", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--detector-config", type=Path, default=DEFAULT_DETECTOR_CONFIG)
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--projection-calibration", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-route-clearance-m", type=float, default=0.20)
    parser.add_argument("--min-available-ram-gib", type=float, default=3.0)
    parser.add_argument("--min-free-disk-gib", type=float, default=20.0)
    parser.add_argument("--swap-observation-s", type=float, default=2.0)
    parser.add_argument("--max-swap-io-mib-s", type=float, default=0.5)
    parser.add_argument("--allow-dirty-pilot", action="store_true")
    args = parser.parse_args()

    paths = {
        "study": args.study.expanduser().resolve(),
        "protocol": args.protocol.expanduser().resolve(),
        "analysis_plan": args.analysis_plan.expanduser().resolve(),
        "detector_config": args.detector_config.expanduser().resolve(),
        "world": args.world.expanduser().resolve(),
        "model": args.model.expanduser().resolve(),
    }
    if args.projection_calibration is not None:
        paths["projection_calibration"] = args.projection_calibration.expanduser().resolve()
    failures: list[str] = []
    warnings: list[str] = []
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        failures.append("missing inputs: " + ", ".join(missing))

    payloads: dict[str, dict[str, Any]] = {}
    for name in ("study", "protocol", "analysis_plan", "detector_config"):
        if paths[name].is_file():
            loaded = yaml.safe_load(paths[name].read_text(encoding="utf-8")) or {}
            payloads[name] = loaded if isinstance(loaded, dict) else {}
    protocol_id = payloads.get("protocol", {}).get("protocol_id")
    for name in ("analysis_plan", "detector_config"):
        if payloads.get(name, {}).get("protocol_id") != protocol_id:
            failures.append(f"{name} protocol_id does not match protocol")

    stale = _processes()
    if stale:
        failures.append(f"{len(stale)} stale/conflicting experiment processes are running")

    memory = _meminfo()
    if memory["available_gib"] < float(args.min_available_ram_gib):
        failures.append(
            f"available RAM {memory['available_gib']:.2f} GiB < {args.min_available_ram_gib:.2f} GiB"
        )
    swap_activity = _swap_activity(float(args.swap_observation_s))
    swap_active = bool(
        swap_activity.get("available")
        and float(swap_activity["mib_per_s_total"]) > float(args.max_swap_io_mib_s)
    )
    if swap_active:
        failures.append(
            "active swap I/O "
            f"{swap_activity['mib_per_s_total']:.2f} MiB/s > "
            f"{args.max_swap_io_mib_s:.2f} MiB/s"
        )
    if memory["swap_used_fraction"] >= 0.95:
        message = f"swap is {100.0 * memory['swap_used_fraction']:.1f}% used"
        if swap_activity.get("available") and not swap_active:
            warnings.append(
                message
                + f" but quiescent over {swap_activity['observation_s']:.1f}s; "
                "MemAvailable and live swap I/O are the gating signals"
            )
        elif swap_activity.get("available"):
            warnings.append(message + "; active swap I/O is a separate hard failure")
        elif args.mode == "pilot":
            warnings.append(message + "; swap activity could not be measured")
        else:
            failures.append(message + "; swap activity could not be measured")
    disk = shutil.disk_usage(REPO)
    disk_free_gib = disk.free / 1024.0**3
    if disk_free_gib < float(args.min_free_disk_gib):
        failures.append(f"free disk {disk_free_gib:.1f} GiB < {args.min_free_disk_gib:.1f} GiB")

    gpu = _gpu()
    if args.mode in {"pilot", "training"} and not gpu.get("available"):
        failures.append("NVIDIA GPU preflight failed: " + str(gpu.get("error", "unavailable")))

    route_rows: list[dict[str, Any]] = []
    if "study" in payloads and paths["world"].is_file():
        route_rows = _route_clearances(payloads["study"], paths["world"])
        unsafe = [
            row for row in route_rows
            if "minimum_collision_clearance_m" not in row
            or row["minimum_collision_clearance_m"] < float(args.min_route_clearance_m)
        ]
        if unsafe:
            failures.append(f"{len(unsafe)} route variants violate collision clearance")

    runtime_config = payloads.get("detector_config", {}).get("runtime_pilot", {})
    detector_mode = str(runtime_config.get("detector_mode", "separate_processes"))
    if detector_mode == "batched_four_camera":
        shared_device = str(runtime_config.get("device", "")).strip().lower()
        model_instances = int(runtime_config.get("model_instances", 0))
        batch_size = int(runtime_config.get("batch_size", 0))
        camera_order = list(runtime_config.get("camera_order", []))
        expected_camera_order = ["camera_A", "camera_B", "camera_C", "camera_D"]
        if (
            model_instances != 1
            or batch_size != 4
            or camera_order != expected_camera_order
            or runtime_config.get("runtime_executable") != "batched_four_camera_yolo_node"
            or runtime_config.get("model_format") != "native_ultralytics"
        ):
            failures.append(
                "batched detector allocation must be one native model and one ordered A-D batch"
            )
        if not shared_device:
            failures.append("batched detector device must be explicit for reproducible provenance")
        if runtime_config.get("fault_policy") != (
            "fatal_process_exit_and_launch_shutdown_no_synthetic_miss"
        ):
            failures.append("batched detector fault policy could contaminate availability evidence")
        gpu_processes = 0 if shared_device in {"", "cpu"} else model_instances
    else:
        devices = runtime_config.get("devices", {})
        if not devices:
            devices = runtime_config.get("separate_process_fallback", {}).get("devices", {})
        model_instances = int(runtime_config.get("model_instances", len(devices)))
        gpu_processes = sum(
            str(device).strip().lower() not in {"", "cpu"} for device in devices.values()
        )
    if gpu_processes > 3:
        failures.append(f"configured {gpu_processes} GPU detector processes; P2000 maximum is 3")

    git_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, check=False, capture_output=True, text=True
    ).stdout
    dirty = bool(git_status.strip())
    if dirty and not (args.mode == "pilot" and args.allow_dirty_pilot):
        failures.append("git worktree is dirty; freeze intentional source/config bytes before this mode")
    elif dirty:
        warnings.append("pilot uses a dirty worktree; all input hashes must remain diagnostic-only")

    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "pass": not failures,
        "failures": failures,
        "warnings": warnings,
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path) if path.is_file() else None}
            for name, path in paths.items()
        },
        "protocol_id": protocol_id,
        "conflicting_processes": stale,
        "memory": memory,
        "swap_activity": swap_activity,
        "disk_free_gib": disk_free_gib,
        "gpu": gpu,
        "detector_mode": detector_mode,
        "configured_model_instances": model_instances,
        "configured_gpu_detector_processes": gpu_processes,
        "route_clearance": route_rows,
        "git_dirty": dirty,
        "git_status_sha256": hashlib.sha256(git_status.encode("utf-8")).hexdigest(),
    }
    _atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
