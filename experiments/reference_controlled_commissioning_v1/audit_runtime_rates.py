#!/usr/bin/env python3
"""Audit source-time camera, detector, fusion, and assimilation rates per run.

The stages are deliberately reported separately.  A 5 Hz camera/detector stream
does not imply that the camera manager publishes, or that the estimator applies,
corrections at 5 Hz.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _cadence(stamps: Iterable[float]) -> dict:
    unique = sorted(set(float(value) for value in stamps if math.isfinite(float(value))))
    gaps = [right - left for left, right in zip(unique, unique[1:]) if right > left]
    duration = unique[-1] - unique[0] if len(unique) > 1 else 0.0
    return {
        "count": len(unique),
        "first_stamp_s": unique[0] if unique else None,
        "last_stamp_s": unique[-1] if unique else None,
        "span_rate_hz": ((len(unique) - 1) / duration if duration > 0 else None),
        "median_interval_s": statistics.median(gaps) if gaps else None,
        "p95_interval_s": _percentile(gaps, 0.95),
        "max_interval_s": max(gaps) if gaps else None,
    }


def _in_window(stamps: Iterable[float], start: float, stop: float) -> list[float]:
    return [stamp for stamp in stamps if start <= stamp <= stop]


def _audit(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    attempt_dir = run_dir.parent
    start = float(summary["first_cmd_stamp"])
    stop = float(summary["stop_stamp"])

    opportunities = _jsonl(run_dir / "camera_opportunities.jsonl")
    detector = _jsonl(attempt_dir / "detector_outcomes.jsonl")
    manager = _jsonl(attempt_dir / "manager_outcomes.jsonl")
    publications = _csv(run_dir / "correction_publications.csv")
    assimilations = _csv(run_dir / "correction_assimilations.csv")

    camera_stamps: dict[str, list[float]] = defaultdict(list)
    for row in opportunities:
        if row.get("duplicate") or not row.get("valid_contract"):
            continue
        obs = row.get("observation", {})
        stamp_ns = _number(obs.get("capture_stamp_ns"))
        camera = str(obs.get("camera_id") or row.get("topic_camera") or "unknown")
        if stamp_ns is not None:
            camera_stamps[camera].append(stamp_ns / 1e9)

    selected_stamps = []
    detector_status = Counter()
    for row in detector:
        detector_status[str(row.get("status"))] += 1
        if row.get("status") != "selected":
            continue
        members = row.get("members") or []
        member_stamps = {
            int(member["capture_stamp_ns"]) for member in members
            if member.get("capture_stamp_ns") is not None
        }
        if len(member_stamps) == 1:
            selected_stamps.append(next(iter(member_stamps)) / 1e9)

    manager_complete = []
    manager_decisions = []
    manager_status = Counter()
    for row in manager:
        status = str(row.get("status"))
        manager_status[status] += 1
        if status == "complete":
            source_batch = str(row.get("source_batch_id", ""))
            marker = source_batch.rsplit("@", 1)[-1].split(",", 1)[0]
            try:
                manager_complete.append(int(marker) / 1e9)
            except ValueError:
                pass
        elif status == "manager_decision":
            decision = row.get("decision") or {}
            stamp_ns = _number(decision.get("common_capture_stamp_ns"))
            # No-observation decisions do not contain a common capture stamp;
            # their manager clock is still the actual decision cadence.
            manager_decisions.append(
                stamp_ns / 1e9 if stamp_ns is not None else float(row["publish_stamp_s"])
            )

    publication_stamps = [
        value for row in publications
        if (value := _number(row.get("correction_stamp"))) is not None
    ]
    assimilation_stamps = [
        value for row in assimilations
        if (value := _number(row.get("apply_stamp"))) is not None
    ]
    assimilation_status = Counter(str(row.get("status")) for row in assimilations)

    mission_camera = {
        camera: _cadence(_in_window(stamps, start, stop))
        for camera, stamps in sorted(camera_stamps.items())
    }
    first_camera = next(iter(mission_camera.values()), {})
    all_camera_rates = [
        row["span_rate_hz"] for row in mission_camera.values()
        if row.get("span_rate_hz") is not None
    ]
    return {
        "run_dir": str(run_dir),
        "mission_window_s": {"first_cmd": start, "stop": stop, "duration": stop - start},
        "camera_capture_source_time_per_camera": mission_camera,
        "camera_capture_rate_range_hz": (
            [min(all_camera_rates), max(all_camera_rates)] if all_camera_rates else None
        ),
        "complete_detector_batches_source_time": _cadence(
            _in_window(selected_stamps, start, stop)
        ),
        "complete_manager_batches_source_time": _cadence(
            _in_window(manager_complete, start, stop)
        ),
        "manager_decisions": _cadence(_in_window(manager_decisions, start, stop)),
        "correction_publications": _cadence(_in_window(publication_stamps, start, stop)),
        "estimator_assimilation_attempts": _cadence(
            _in_window(assimilation_stamps, start, stop)
        ),
        "detector_status_counts": dict(sorted(detector_status.items())),
        "manager_status_counts": dict(sorted(manager_status.items())),
        "assimilation_status_counts": dict(sorted(assimilation_status.items())),
        "summary": {
            key: summary.get(key) for key in (
                "completed", "completion_reason", "valid_run", "invalid_reason",
                "collision_any", "collision_contact", "collision_geom",
                "mean_belief_error_gt_after_first_cmd_m", "longest_correction_gap_s",
            )
        },
        "reference_camera": next(iter(mission_camera), None),
        "reference_camera_count": first_camera.get("count"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Campaign root or one experiment_* directory")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.root / "run_summary.json").exists():
        run_dirs = [args.root]
    else:
        run_dirs = sorted(path.parent for path in args.root.rglob("run_summary.json"))
    if not run_dirs:
        raise RuntimeError(f"no run_summary.json below {args.root}")
    result = {"schema": "runtime_rate_audit.v1", "runs": [_audit(path) for path in run_dirs]}
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise RuntimeError("audit outputs are immutable; choose a new output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
