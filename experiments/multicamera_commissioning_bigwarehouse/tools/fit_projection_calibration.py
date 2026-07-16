#!/usr/bin/env python3
"""Fit per-camera along-bearing projection calibrations from GT-attached runs.

Model: correction = intercept_m + slope_per_m * ground_distance, applied along
the camera->point bearing (positive away from the camera).  This captures both
the contact-height ray error (distance-proportional) and the near-edge
box-bottom pull (constant), which a single constant offset cannot absorb
across viewing distances.

Inputs are one or more ``evaluation_inputs/`` directories produced by
``attach_evaluation_truth.py`` (truth-attached per-camera CSVs).  Projections
are recomputed RAW from ``obs_u/obs_v`` so runs recorded with an older
calibration applied do not contaminate the fit.  This is a commissioning-time
procedure: the emitted constants are frozen and never refit during deployment.

Example:

    python3 .../fit_projection_calibration.py \
      --audit-dir logs/studies/.../gt_validation_smoke_20260716/evaluation_inputs \
      --audit-dir logs/studies/.../gt_validation_smoke2_20260716/evaluation_inputs \
      --output logs/studies/.../projection_calibration_v2.json
"""

from __future__ import annotations

import argparse
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

from reliability.contracts import CameraObservation  # noqa: E402
from reliability.projection import (  # noqa: E402
    camera_model_from_world,
    project_observation_to_world,
)

DEFAULT_WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
MODEL_INCLUDES = {
    "camera_A": "external_camera",
    "camera_B": "external_camera_b",
    "camera_C": "external_camera_c",
    "camera_D": "external_camera_d",
}
# Below this distance coverage a slope is not identifiable; fall back to a
# constant (slope 0) rather than extrapolate a noisy line.
MIN_SPAN_FOR_SLOPE_M = 3.0


def _collect_samples(
    audit_dir: Path,
    camera_id: str,
    model,
    contact_z_m: float,
) -> list[tuple[float, float]]:
    """Return (ground_distance, along_bearing_error) for raw re-projections."""

    source = audit_dir / f"{camera_id}_perception.csv"
    if not source.exists():
        return []
    cam_x, cam_y = float(model.cam_pos[0]), float(model.cam_pos[1])
    samples: list[tuple[float, float]] = []
    with source.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("detected") != "1" or not row.get("true_x") or not row.get("obs_u"):
                continue
            observation = CameraObservation(
                camera_id=camera_id,
                timestamp_s=float(row["diag_stamp"]),
                pixel_uv=(float(row["obs_u"]), float(row["obs_v"])),
                detection_valid=True,
            )
            point = project_observation_to_world(observation, model, contact_z_m=contact_z_m)
            if point is None:
                continue
            bearing_x = point[0] - cam_x
            bearing_y = point[1] - cam_y
            distance = math.hypot(bearing_x, bearing_y)
            if distance <= 1.0e-9:
                continue
            error_x = point[0] - float(row["true_x"])
            error_y = point[1] - float(row["true_y"])
            along = (error_x * bearing_x + error_y * bearing_y) / distance
            samples.append((distance, along))
    return samples


def _fit_line(samples: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Least-squares along = a + b*d; returns (a, b, residual_std)."""

    distances = [item[0] for item in samples]
    errors = [item[1] for item in samples]
    span = max(distances) - min(distances)
    if len(samples) < 8 or span < MIN_SPAN_FOR_SLOPE_M:
        mean_error = statistics.fmean(errors)
        residuals = [error - mean_error for error in errors]
        std = statistics.stdev(residuals) if len(residuals) > 1 else 0.0
        return mean_error, 0.0, std
    mean_d = statistics.fmean(distances)
    mean_e = statistics.fmean(errors)
    var_d = sum((d - mean_d) ** 2 for d in distances)
    cov_de = sum((d - mean_d) * (e - mean_e) for d, e in samples)
    slope = cov_de / var_d
    intercept = mean_e - slope * mean_d
    residuals = [e - (intercept + slope * d) for d, e in samples]
    std = statistics.stdev(residuals) if len(residuals) > 1 else 0.0
    return intercept, slope, std


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-sdf", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--contact-z-m", type=float, default=0.05)
    args = parser.parse_args()

    cameras: dict[str, dict[str, float | int]] = {}
    for camera_id, include in sorted(MODEL_INCLUDES.items()):
        model = camera_model_from_world(args.world_sdf, include_name=include)
        samples: list[tuple[float, float]] = []
        for audit_dir in args.audit_dir:
            samples.extend(_collect_samples(audit_dir, camera_id, model, args.contact_z_m))
        if not samples:
            print(f"{camera_id}: no samples, skipped")
            continue
        error_intercept, error_slope, residual_std = _fit_line(samples)
        distances = [item[0] for item in samples]
        cameras[camera_id] = {
            # The fit describes the ERROR (toward camera = negative); the
            # correction is its negation.
            "intercept_m": -error_intercept,
            "slope_per_m": -error_slope,
            "residual_std_m": residual_std,
            "samples": len(samples),
            "distance_min_m": min(distances),
            "distance_max_m": max(distances),
        }
        print(
            f"{camera_id}: n={len(samples)} d=[{min(distances):.1f},{max(distances):.1f}] m "
            f"correction = {-error_intercept:+.3f} {-error_slope:+.4f}*d  "
            f"residual std {residual_std:.3f} m"
        )

    payload = {
        "kind": "projection_along_bearing_offsets",
        "model": "correction_m = intercept_m + slope_per_m * ground_distance_m (along bearing, positive away from camera)",
        "method": (
            "least-squares fit of raw along-bearing projection error vs ground "
            "distance against simulation truth; commissioning-time constants, "
            "never refit during deployment"
        ),
        "source_audit_dirs": [str(item) for item in args.audit_dir],
        "world_sdf": str(args.world_sdf),
        "contact_z_m": float(args.contact_z_m),
        "cameras": cameras,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
