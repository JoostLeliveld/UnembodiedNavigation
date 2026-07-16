#!/usr/bin/env python3
"""Attach recorded simulation truth to per-camera CSVs, evaluation-only.

Reads the operational per-camera perception CSVs written by
``record_operational_logs.py`` plus the ``ground_truth.csv`` written by
``record_evaluation_truth.py`` and emits **copies** with ``true_x``, ``true_y``
and ``true_yaw`` columns (nearest-stamp join within a tolerance).  The
operational inputs are never modified; the output directory is an
evaluation-only artifact that ``reliability_tools export-multicamera`` turns
into populated ``evaluation_only/`` samples.

The tool also writes ``truth_alignment_summary.json`` with a per-camera
projection audit on detected rows (bias per axis, mean/median/p90 absolute
error).  This is the measurement that attributes a cross-camera disagreement
(e.g. the pilot's C-D +y offset) to a specific camera's calibration.

Example:

    python3 experiments/multicamera_commissioning_bigwarehouse/tools/attach_evaluation_truth.py \
      --raw-dir logs/multicamera_commissioning_bigwarehouse/run_001/raw \
      --truth-csv logs/multicamera_commissioning_bigwarehouse/run_001/evaluation_only/ground_truth.csv \
      --out-dir logs/multicamera_commissioning_bigwarehouse/run_001/evaluation_inputs

"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[3]
for relative in ("src/reliability", "src/unav_common"):
    location = str(REPO / relative)
    if location not in sys.path:
        sys.path.insert(0, location)

from reliability.projection import camera_model_from_world  # noqa: E402

TRUTH_COLUMNS = ("true_x", "true_y", "true_yaw")
DEFAULT_WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
DEFAULT_MODEL_INCLUDES = {
    "camera_A": "external_camera",
    "camera_B": "external_camera_b",
    "camera_C": "external_camera_c",
    "camera_D": "external_camera_d",
}


def _load_truth(path: Path) -> tuple[list[float], list[tuple[float, float, float]]]:
    stamps: list[float] = []
    poses: list[tuple[float, float, float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                stamp = float(row["stamp"])
                pose = (float(row["gt_x"]), float(row["gt_y"]), float(row["gt_yaw"]))
            except (KeyError, TypeError, ValueError):
                continue
            stamps.append(stamp)
            poses.append(pose)
    paired = sorted(zip(stamps, poses))
    return [item[0] for item in paired], [item[1] for item in paired]


def _nearest(
    stamps: list[float],
    poses: list[tuple[float, float, float]],
    stamp: float,
    tolerance_s: float,
) -> tuple[float, float, float] | None:
    if not stamps:
        return None
    index = bisect.bisect_left(stamps, stamp)
    best: tuple[float, int] | None = None
    for candidate in (index - 1, index):
        if 0 <= candidate < len(stamps):
            delta = abs(stamps[candidate] - stamp)
            if best is None or delta < best[0]:
                best = (delta, candidate)
    if best is None or best[0] > tolerance_s:
        return None
    return poses[best[1]]


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[position]


def attach_camera_csv(
    source: Path,
    destination: Path,
    stamps: list[float],
    poses: list[tuple[float, float, float]],
    tolerance_s: float,
    camera_ground_xy: tuple[float, float] | None = None,
) -> dict[str, object]:
    matched = 0
    total = 0
    error_x: list[float] = []
    error_y: list[float] = []
    error_norm: list[float] = []
    error_along_bearing: list[float] = []
    with source.open("r", newline="", encoding="utf-8") as in_handle:
        reader = csv.DictReader(in_handle)
        fieldnames = list(reader.fieldnames or [])
        for column in TRUTH_COLUMNS:
            if column not in fieldnames:
                fieldnames.append(column)
        with destination.open("w", newline="", encoding="utf-8") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                total += 1
                truth = None
                try:
                    stamp = float(row["diag_stamp"])
                except (KeyError, TypeError, ValueError):
                    stamp = math.nan
                if math.isfinite(stamp):
                    truth = _nearest(stamps, poses, stamp, tolerance_s)
                if truth is None:
                    row.update({column: "" for column in TRUTH_COLUMNS})
                else:
                    matched += 1
                    row["true_x"] = f"{truth[0]:.9f}"
                    row["true_y"] = f"{truth[1]:.9f}"
                    row["true_yaw"] = f"{truth[2]:.9f}"
                    if row.get("detected") == "1" and row.get("pred_world_x") and row.get("pred_world_y"):
                        px = float(row["pred_world_x"])
                        py = float(row["pred_world_y"])
                        dx = px - truth[0]
                        dy = py - truth[1]
                        error_x.append(dx)
                        error_y.append(dy)
                        error_norm.append(math.hypot(dx, dy))
                        if camera_ground_xy is not None:
                            bearing_x = px - camera_ground_xy[0]
                            bearing_y = py - camera_ground_xy[1]
                            norm = math.hypot(bearing_x, bearing_y)
                            if norm > 1.0e-9:
                                error_along_bearing.append(
                                    (dx * bearing_x + dy * bearing_y) / norm
                                )
                writer.writerow(row)
    audit: dict[str, object] = {
        "rows": total,
        "rows_with_truth": matched,
        "detected_rows_audited": len(error_norm),
    }
    if error_norm:
        audit.update(
            {
                "bias_x_m": statistics.fmean(error_x),
                "bias_y_m": statistics.fmean(error_y),
                "mean_abs_error_m": statistics.fmean(error_norm),
                "median_abs_error_m": statistics.median(error_norm),
                "p90_abs_error_m": _percentile(error_norm, 0.90),
                "max_abs_error_m": max(error_norm),
            }
        )
    if error_along_bearing:
        audit["along_bearing_bias_m"] = statistics.fmean(error_along_bearing)
        audit["along_bearing_std_m"] = (
            statistics.stdev(error_along_bearing) if len(error_along_bearing) > 1 else 0.0
        )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-time-delta-s", type=float, default=0.05)
    parser.add_argument(
        "--camera-glob",
        default="camera_*_perception.csv",
        help="Per-camera CSV pattern inside --raw-dir",
    )
    parser.add_argument("--world-sdf", type=Path, default=DEFAULT_WORLD)
    parser.add_argument(
        "--emit-projection-calibration",
        type=Path,
        default=None,
        help=(
            "Write per-camera along-bearing projection offsets (JSON) estimated "
            "from this run's truth-referenced bias. Commissioning-time "
            "calibration output; consumed via --projection-calibration on the "
            "operational recorder and the camera manager node."
        ),
    )
    args = parser.parse_args()
    if args.out_dir.resolve() == args.raw_dir.resolve():
        raise SystemExit("--out-dir must differ from --raw-dir: operational inputs are never modified")

    stamps, poses = _load_truth(args.truth_csv)
    if not stamps:
        raise SystemExit(f"no usable truth rows in {args.truth_csv}")

    sources = sorted(args.raw_dir.glob(args.camera_glob))
    if not sources:
        raise SystemExit(f"no camera CSVs matching {args.camera_glob!r} in {args.raw_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    audits: dict[str, dict[str, object]] = {}
    for source in sources:
        destination = args.out_dir / source.name
        camera_id = source.name.replace("_perception.csv", "")
        camera_ground_xy = None
        include = DEFAULT_MODEL_INCLUDES.get(camera_id)
        if include is not None and args.world_sdf.exists():
            model = camera_model_from_world(args.world_sdf, include_name=include)
            camera_ground_xy = (float(model.cam_pos[0]), float(model.cam_pos[1]))
        audits[camera_id] = attach_camera_csv(
            source,
            destination,
            stamps,
            poses,
            args.max_time_delta_s,
            camera_ground_xy=camera_ground_xy,
        )

    summary = {
        "evaluation_only": True,
        "contains_ground_truth": True,
        "truth_csv": str(args.truth_csv),
        "raw_dir": str(args.raw_dir),
        "max_time_delta_s": float(args.max_time_delta_s),
        "truth_samples": len(stamps),
        "cameras": audits,
        "note": (
            "Copies of operational CSVs with true_x/true_y/true_yaw attached for "
            "evaluation_only exports and per-camera calibration audits. The "
            "per-camera bias_x/bias_y attribute cross-camera disagreement to a "
            "specific calibration. Never feed these columns to a model or manager."
        ),
    }
    summary_path = args.out_dir / "truth_alignment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.emit_projection_calibration is not None:
        cameras = {}
        for camera_id, audit in sorted(audits.items()):
            if "along_bearing_bias_m" not in audit:
                continue
            cameras[camera_id] = {
                # The measured bias is toward the camera (negative along the
                # bearing); the corrective offset pushes away from the camera.
                "along_bearing_offset_m": -float(audit["along_bearing_bias_m"]),
                "along_bearing_std_m": float(audit["along_bearing_std_m"]),
                "samples": int(audit["detected_rows_audited"]),
            }
        calibration = {
            "kind": "projection_along_bearing_offsets",
            "method": (
                "per-camera mean along-bearing projection error vs simulation "
                "truth (near-edge box-bottom pull); commissioning-time constant, "
                "never refit during deployment"
            ),
            "source_run": str(args.raw_dir),
            "world_sdf": str(args.world_sdf),
            "cameras": cameras,
        }
        args.emit_projection_calibration.parent.mkdir(parents=True, exist_ok=True)
        args.emit_projection_calibration.write_text(
            json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote projection calibration for {len(cameras)} cameras -> {args.emit_projection_calibration}")
    for camera_id, audit in sorted(audits.items()):
        bias = (
            f" bias=({audit.get('bias_x_m', math.nan):+.3f}, {audit.get('bias_y_m', math.nan):+.3f}) m"
            if "bias_x_m" in audit
            else ""
        )
        print(f"{camera_id}: {audit['rows_with_truth']}/{audit['rows']} rows matched{bias}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
