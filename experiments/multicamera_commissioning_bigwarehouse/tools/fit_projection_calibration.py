#!/usr/bin/env python3
"""Fit per-camera bearing-frame projection calibrations from GT-attached runs.

Model::

    p_corrected = p_raw + (intercept_m + slope_per_m * d) * e_along
                        + (cross_intercept_m + cross_slope_per_m * d) * e_cross

The along-bearing term (positive away from the camera) captures both the
contact-height ray error (distance-proportional) and the near-edge box-bottom
pull (constant), which a single constant offset cannot absorb across viewing
distances.

The cross-bearing term (positive to the left of the bearing) captures the lateral
offset the along-bearing model structurally cannot represent -- an
extrinsic/rotation signature rather than a contact-point one.  It is **gated**:
fitted only where the lateral bias is resolvable against that camera's own
scatter (``--cross-bias-sigma-gate``, default 1.2).  Below the gate the constant
is left at zero, because a lateral bias smaller than the scatter is estimated
mostly as noise and transfers badly to unseen regions -- on the 4-camera captures
forcing it on cost camera A 61 % of its held-out accuracy while camera C gained
46 %.  Evidence and the per-camera ratios:
``logs/studies/external_camera_bias_model/exp2_two_dof_bias/RESULTS.md``.

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


def _parse_camera_registry(values: list[str] | None) -> dict[str, str]:
    """Parse repeatable ``camera_ID=model_include`` overrides.

    No override preserves the frozen A--D commissioning contract. New worlds
    must pass their registry explicitly so camera identities cannot be inferred
    from include ordering.
    """

    if not values:
        return dict(MODEL_INCLUDES)
    registry: dict[str, str] = {}
    for raw in values:
        camera_id, separator, include = str(raw).partition("=")
        camera_id = camera_id.strip()
        include = include.strip()
        if not separator or not camera_id or not include:
            raise ValueError(
                f"Invalid --camera {raw!r}; expected camera_ID=model_include"
            )
        if camera_id in registry:
            raise ValueError(f"Duplicate --camera ID: {camera_id}")
        if include in registry.values():
            raise ValueError(f"Duplicate --camera model include: {include}")
        registry[camera_id] = include
    return registry


def _collect_samples(
    audit_dir: Path,
    camera_id: str,
    model,
    contact_z_m: float,
) -> list[tuple[float, float, float]]:
    """Return (ground_distance, along_bearing_error, cross_bearing_error).

    ``cross`` is positive to the LEFT of the bearing, matching the sign convention
    of ``reliability.projection._project_pixel_to_world``.
    """

    source = audit_dir / f"{camera_id}_perception.csv"
    if not source.exists():
        return []
    cam_x, cam_y = float(model.cam_pos[0]), float(model.cam_pos[1])
    samples: list[tuple[float, float, float]] = []
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
            cross = (-error_x * bearing_y + error_y * bearing_x) / distance
            samples.append((distance, along, cross))
    return samples


def _fit_line(samples: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Least-squares error = a + b*d; returns (a, b, residual_std)."""

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
    parser.add_argument(
        "--camera", action="append", default=None,
        help=(
            "Camera registry entry camera_ID=model_include; repeat once per camera. "
            "Omitting this option preserves the historical warehouse A--D registry."
        ),
    )
    parser.add_argument("--contact-z-m", type=float, default=0.05)
    parser.add_argument(
        "--cross-bias-sigma-gate", type=float, default=1.2,
        help="fit the cross-bearing term only where |mean cross error| / sigma_cross "
             "reaches this; 1.2 selected C and D and correctly rejected A on the "
             "4-camera captures (exp2_two_dof_bias). Set to 0 to force it on for all.",
    )
    args = parser.parse_args()
    camera_registry = _parse_camera_registry(args.camera)

    cameras: dict[str, dict[str, float | int | bool | str]] = {}
    for camera_id, include in camera_registry.items():
        model = camera_model_from_world(args.world_sdf, include_name=include)
        samples: list[tuple[float, float, float]] = []
        for audit_dir in args.audit_dir:
            samples.extend(_collect_samples(audit_dir, camera_id, model, args.contact_z_m))
        if not samples:
            print(f"{camera_id}: no samples, skipped")
            continue
        along_samples = [(d, along) for d, along, _ in samples]
        cross_samples = [(d, cross) for d, _, cross in samples]
        error_intercept, error_slope, residual_std = _fit_line(along_samples)
        distances = [item[0] for item in samples]

        # --- cross-bearing degree of freedom, GATED -------------------------
        # The along-bearing model cannot represent a lateral offset, and camera C
        # carries +0.078 m of one. But fitting a cross term where the lateral bias
        # is not resolvable against the camera's own scatter makes things WORSE:
        # camera A (ratio 0.16) lost 61 % on held-out data. See
        # logs/studies/external_camera_bias_model/exp2_two_dof_bias/RESULTS.md.
        cross_errors = [cross for _, cross in cross_samples]
        cross_mean = statistics.fmean(cross_errors)
        cross_std = statistics.stdev(cross_errors) if len(cross_errors) > 1 else 0.0
        cross_ratio = abs(cross_mean) / cross_std if cross_std > 0.0 else math.inf
        fit_cross = cross_ratio >= args.cross_bias_sigma_gate
        if fit_cross:
            cross_intercept, cross_slope, cross_residual_std = _fit_line(cross_samples)
        else:
            cross_intercept, cross_slope, cross_residual_std = 0.0, 0.0, cross_std

        cameras[camera_id] = {
            # The fit describes the ERROR (toward camera = negative); the
            # correction is its negation.
            "intercept_m": -error_intercept,
            "slope_per_m": -error_slope,
            "cross_intercept_m": -cross_intercept,
            "cross_slope_per_m": -cross_slope,
            "residual_std_m": residual_std,
            "cross_residual_std_m": cross_residual_std,
            "cross_bias_to_sigma": cross_ratio,
            "cross_term_fitted": fit_cross,
            "samples": len(samples),
            "distance_min_m": min(distances),
            "distance_max_m": max(distances),
        }
        gate = "FITTED" if fit_cross else f"GATED OFF (ratio {cross_ratio:.2f})"
        print(
            f"{camera_id}: n={len(samples)} d=[{min(distances):.1f},{max(distances):.1f}] m "
            f"along = {-error_intercept:+.3f} {-error_slope:+.4f}*d  "
            f"residual std {residual_std:.3f} m | cross {gate}: "
            f"{-cross_intercept:+.3f} {-cross_slope:+.4f}*d  "
            f"(raw cross bias {cross_mean:+.3f} m, sigma {cross_std:.3f} m)"
        )

    payload = {
        "kind": "projection_bearing_frame_offsets",
        "model": (
            "p_corrected = p_raw + (intercept_m + slope_per_m * d) * e_along "
            "+ (cross_intercept_m + cross_slope_per_m * d) * e_cross, "
            "d = ground distance, e_along positive away from the camera, "
            "e_cross positive to its left"
        ),
        "method": (
            "least-squares fit of raw bearing-frame projection error vs ground "
            "distance against simulation truth; commissioning-time constants, "
            "never refit during deployment. The cross-bearing term is fitted only "
            "where |mean cross error| / sigma_cross >= cross_bias_sigma_gate, "
            "because a lateral bias that is not resolvable against the camera's own "
            "scatter is estimated mostly as noise and transfers badly across "
            "regions (evidence: logs/studies/external_camera_bias_model/"
            "exp2_two_dof_bias/RESULTS.md)."
        ),
        "cross_bias_sigma_gate": float(args.cross_bias_sigma_gate),
        "source_audit_dirs": [str(item) for item in args.audit_dir],
        "world_sdf": str(args.world_sdf),
        "camera_model_includes": camera_registry,
        "contact_z_m": float(args.contact_z_m),
        "cameras": cameras,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
