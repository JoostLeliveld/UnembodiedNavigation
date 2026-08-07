#!/usr/bin/env python3
"""Turn a multicam grid capture into residuals in the schema the studies already read.

The existing `residuals.csv` came from three navigation captures: straight routes, two
headings, and a nearest-stamp join between detections and ground truth. That dataset
cannot separate range from image position from projection conditioning, because along a
straight line they are one variable wearing three names (rho >= 0.97).

A grid capture fixes the design and also removes a whole error source: the robot is
teleported to a commanded pose, so ground truth is recorded PER SAMPLE and there is no
timestamp join and no join tolerance.

Output columns match `external_camera_bias_model/exp1_residual_characterization/
residuals.csv` exactly, so `exp1_geometry_vs_detector.py` consumes this with a path
swap. Three columns are appended that the old file could not carry:

    theta        commanded robot heading -- the variable E6 says is missing
    yolo_score   detector confidence, so the 0.25-vs-0.05 threshold (U6) is a filter
                 applied at analysis time rather than a choice baked into the capture
    oracle_visible  geometric visibility, to separate "detector missed it" from
                 "it was never in frame"

Projection uses `reliability.projection._project_pixel_to_world` -- the runtime path,
not a reimplementation -- twice per detection: RAW, and through the deployed
projection_calibration_v2 constants, exactly as residual_audit.py does.

Ground truth is EVALUATION-ONLY here: it positions the robot and measures the residual.
It never enters a projection, a Jacobian, a fitted parameter or a covariance.

    python3 experiments/projection_amplification/build_grid_residuals.py \
        --capture logs/visibility_comparison/commissioning_grid_20260807
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
for _p in ("src/reliability", "src/unav_common", "experiments/external_camera_bias_model"):
    sys.path.insert(0, str(REPO / _p))

from reliability.projection import (  # noqa: E402
    _project_pixel_to_world,
    camera_model_from_world,
    load_projection_calibration,
)
import residual_audit as RA  # noqa: E402  (world, mounts, contact height, calibration)

#: perception_targets.csv names cameras by ROS frame; the studies key on camera_A..D.
FRAME_TO_CAMERA = {frame: camera for camera, frame in RA.MODEL_INCLUDES.items()}

COLUMNS = (
    "capture", "camera", "stamp", "true_x", "true_y", "range_m", "bearing_deg", "u", "v",
    "raw_ex", "raw_ey", "raw_along", "raw_cross", "raw_norm", "raw_px", "raw_py",
    "cor_ex", "cor_ey", "cor_along", "cor_cross", "cor_norm", "cor_px", "cor_py",
    # appended: what the route-based capture could not record
    "theta", "yolo_score", "oracle_visible",
)


def _f(value, default=math.nan) -> float:
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return out


def build(targets_csv: Path, conf_threshold: float) -> tuple[list[dict], dict]:
    calib = load_projection_calibration(RA.DEPLOYED_CALIB)
    models = {
        camera: camera_model_from_world(RA.WORLD_SDF, include_name=include)
        for camera, include in RA.MODEL_INCLUDES.items()
    }

    rows: list[dict] = []
    counts = {
        "input_rows": 0, "unknown_camera": 0, "no_detection": 0,
        "below_threshold": 0, "bad_pixel": 0, "projection_failed": 0, "kept": 0,
    }
    with targets_csv.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            counts["input_rows"] += 1
            camera = FRAME_TO_CAMERA.get(str(record.get("camera_frame", "")).strip())
            if camera is None:
                counts["unknown_camera"] += 1
                continue

            score = _f(record.get("yolo_score_raw"))
            u, v = _f(record.get("yolo_bottom_u")), _f(record.get("yolo_bottom_v"))
            if not (math.isfinite(u) and math.isfinite(v)):
                counts["no_detection"] += 1
                continue
            if math.isfinite(score) and score < conf_threshold:
                counts["below_threshold"] += 1
                continue

            tx, ty = _f(record.get("x")), _f(record.get("y"))
            if not (math.isfinite(tx) and math.isfinite(ty)):
                counts["bad_pixel"] += 1
                continue

            entry = calib.get(camera, {})
            raw = _project_pixel_to_world(
                u, v, models[camera], contact_z_m=RA.CONTACT_Z_M,
                along_bearing_offset_m=0.0, along_bearing_slope_per_m=0.0,
            )
            cor = _project_pixel_to_world(
                u, v, models[camera], contact_z_m=RA.CONTACT_Z_M,
                along_bearing_offset_m=float(entry.get("intercept_m", 0.0)),
                along_bearing_slope_per_m=float(entry.get("slope_per_m", 0.0)),
            )
            if raw is None or cor is None:
                counts["projection_failed"] += 1
                continue

            cam_x, cam_y = float(models[camera].cam_pos[0]), float(models[camera].cam_pos[1])
            bx, by = tx - cam_x, ty - cam_y
            rng_m = math.hypot(bx, by)
            # Bearing basis referenced to TRUTH, matching residual_audit: using the
            # projected point would rotate the basis by the very error under audit.
            ux, uy = (bx / rng_m, by / rng_m) if rng_m > 1e-9 else (1.0, 0.0)

            out = {
                "capture": targets_csv.parent.name,
                "camera": camera,
                "stamp": _f(record.get("timestamp"), 0.0),
                "true_x": tx, "true_y": ty, "range_m": rng_m,
                "bearing_deg": math.degrees(math.atan2(by, bx)),
                "u": u, "v": v,
                "theta": _f(record.get("theta")),
                "yolo_score": score,
                "oracle_visible": record.get("oracle_visible", ""),
            }
            for tag, point in (("raw", raw), ("cor", cor)):
                ex, ey = point[0] - tx, point[1] - ty
                out[f"{tag}_ex"] = ex
                out[f"{tag}_ey"] = ey
                out[f"{tag}_along"] = ex * ux + ey * uy
                out[f"{tag}_cross"] = -ex * uy + ey * ux
                out[f"{tag}_norm"] = math.hypot(ex, ey)
                out[f"{tag}_px"] = point[0]
                out[f"{tag}_py"] = point[1]
            rows.append(out)
            counts["kept"] += 1
    return rows, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True,
                        help="grid capture directory containing perception_targets.csv")
    parser.add_argument("--targets", default="",
                        help="explicit perception_targets.csv (defaults to <capture>/…)")
    parser.add_argument("--out", default="",
                        help="output residuals.csv (defaults beside the targets file)")
    parser.add_argument("--conf-threshold", type=float, default=0.05,
                        help="detector confidence floor. 0.05 is the RUNTIME value; the "
                             "offline gate contract says 0.25 (U6). Confidence is kept per "
                             "row either way, so this is a filter, not a commitment.")
    args = parser.parse_args()

    capture = Path(args.capture)
    targets = Path(args.targets) if args.targets else capture / "perception_targets.csv"
    if not targets.is_file():
        raise SystemExit(
            f"missing {targets}\n"
            "run scripts/visibility_comparison/extract_perception_targets.py first"
        )
    out_path = Path(args.out) if args.out else targets.parent / "grid_residuals.csv"

    rows, counts = build(targets, float(args.conf_threshold))
    if not rows:
        raise SystemExit(f"no usable detections; counts={counts}")

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    by_camera: dict[str, int] = {}
    headings: dict[str, set] = {}
    for row in rows:
        by_camera[row["camera"]] = by_camera.get(row["camera"], 0) + 1
        headings.setdefault(row["camera"], set()).add(round(math.degrees(row["theta"])))
    summary = {
        "targets_csv": str(targets),
        "conf_threshold": float(args.conf_threshold),
        "counts": counts,
        "per_camera": {c: {"n": n, "distinct_headings": sorted(headings[c])}
                       for c, n in sorted(by_camera.items())},
    }
    (out_path.parent / "grid_residuals_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(counts, indent=2))
    for camera, info in summary["per_camera"].items():
        print(f"  {camera}: {info['n']:6} detections, headings {info['distinct_headings']}")
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
