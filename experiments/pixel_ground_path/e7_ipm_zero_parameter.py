#!/usr/bin/env python3
"""e7 — does anything beyond plain IPM still earn its place?

The 2026-08-07 decision froze the measurement as inverse perspective mapping: box
bottom-centre intersected with the floor plane.  That left one loose end.  The v4
calibration artifact is *almost* plain IPM -- floor plane, no along-bearing term -- but it
still carries a gated per-camera CROSS-bearing offset for cameras C and D, fitted on a
different dataset (multicamera_commissioning_bigwarehouse).  Two fitted scalars survived the
cleanup, and nobody had ever checked them against this study's held-out evidence.

This script scores four arms on the SAME 1844 real detections e3 used.  It was first run
against the live ``reliability.projection._project_pixel_to_world``; that function was then
deleted *because of this result*, so the correction math is now inlined below and produces
bit-identical numbers to the run that justified the deletion:

  raw IPM        floor plane, zero corrections        <- candidate for "the whole method"
  v4             floor plane + gated cross-bearing    <- what is deployed today
  v3             0.05 m plane + along + cross         <- superseded
  v2             0.05 m plane + along                 <- superseded

If raw IPM ties or beats v4, the cross-bearing degree of freedom is dead weight in the
runtime and can be deleted along with the artifact plumbing that feeds it.

Run:  python3 experiments/pixel_ground_path/e7_ipm_zero_parameter.py
Out:  logs/studies/pixel_ground_path/e7_ipm_zero_parameter/
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
sys.path.insert(0, str(REPO / "src" / "reliability"))
sys.path.insert(0, str(REPO / "src" / "unav_common"))

from reliability.projection import camera_model_from_world  # noqa: E402

# The correction math and the artifact loader are INLINED here on purpose.  This script is
# the evidence that got them deleted from `reliability.projection` on 2026-08-07, so it must
# not import them -- an experiment whose conclusion is "delete X" cannot depend on X and
# still be re-runnable.  This is a faithful copy of the deleted implementation, kept only so
# the four arms below remain reproducible.


def _load_calibration(path):
    """Deleted `load_projection_calibration` + `load_projection_contact_z`, inlined."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cameras = payload.get("cameras", payload)
    calibrations = {}
    for camera_id, entry in cameras.items():
        calibrations[str(camera_id)] = {
            "along": float(entry.get("intercept_m", 0.0)),
            "along_slope": float(entry.get("slope_per_m", 0.0)),
            "cross": float(entry.get("cross_intercept_m", 0.0)),
            "cross_slope": float(entry.get("cross_slope_per_m", 0.0)),
        }
    return calibrations, float(payload.get("contact_z_m", 0.0))


def _project_pixel_to_world(u, v, camera, *, contact_z_m, along, along_slope, cross,
                            cross_slope):
    """Deleted `_project_pixel_to_world`, inlined.  Bearing basis from the RAW point."""

    if contact_z_m > 0.0:
        point = camera.pixel_to_world_at_z(u, v, contact_z_m)
    else:
        point = camera.pixel_to_world(u, v)
    if point is None or not (along or along_slope or cross or cross_slope):
        return point
    bx = point[0] - float(camera.cam_pos[0])
    by = point[1] - float(camera.cam_pos[1])
    norm = math.hypot(bx, by)
    if norm <= 1.0e-9:
        return point
    ux, uy = bx / norm, by / norm
    cx, cy = -uy, ux                      # left of the bearing
    a = along + along_slope * norm
    c = cross + cross_slope * norm
    return (point[0] + a * ux + c * cx, point[1] + a * uy + c * cy)

from dataset_paths import dataset_root  # noqa: E402

WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
DATASET = dataset_root(REPO)
DET_CACHE = (REPO / "logs/studies/pixel_ground_path/e2_detector_edge_characterisation"
             / "detector_boxes.csv")
CALIB = REPO / "logs/studies/multicamera_commissioning_bigwarehouse"
OUT = REPO / "logs/studies/pixel_ground_path/e7_ipm_zero_parameter"

CAMERA_INCLUDES = {
    "camera_A": "external_camera",   # NOT external_camera_a -- that include does not exist
    "camera_B": "external_camera_b",
    "camera_C": "external_camera_c",
    "camera_D": "external_camera_d",
}
IMG_W, IMG_H = 1280, 720


def load_rows() -> list[dict]:
    """Detector boxes joined to commanded poses.  Same join e3 scored."""

    det = {}
    with DET_CACHE.open(newline="", encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            if str(rec["detected"]) == "1" and rec["pu0"] != "":
                det[rec["sample_id"]] = rec
    rows = []
    with (DATASET / "localization_calibration_index.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for rec in csv.DictReader(handle):
            box = det.get(rec["sample_id"])
            if box is None:
                continue
            rows.append(
                dict(
                    camera=rec["camera_id"],
                    x=float(rec["robot_x"]),
                    y=float(rec["robot_y"]),
                    rng=float(rec["camera_range_m"]),
                    u0=float(box["pu0"]), v0=float(box["pv0"]),
                    u1=float(box["pu1"]), v1=float(box["pv1"]),
                )
            )
    return rows


def score(rows, models, kwargs_for) -> dict:
    """Radial / lateral error in the camera bearing frame, metres."""

    per_camera: dict[str, list[tuple[float, float, float, float]]] = {}
    radial, lateral, norms = [], [], []
    for row in rows:
        camera = models[row["camera"]]
        u = 0.5 * (row["u0"] + row["u1"])
        v = row["v1"]                      # box BOTTOM -- the contact-point statistic
        point = _project_pixel_to_world(u, v, camera, **kwargs_for(row["camera"]))
        if point is None:
            continue
        bx = row["x"] - float(camera.cam_pos[0])
        by = row["y"] - float(camera.cam_pos[1])
        dist = math.hypot(bx, by)
        ux, uy = bx / dist, by / dist
        ex, ey = point[0] - row["x"], point[1] - row["y"]
        rad = ex * ux + ey * uy
        lat = -ex * uy + ey * ux
        radial.append(rad)
        lateral.append(lat)
        norms.append(math.hypot(ex, ey))
        per_camera.setdefault(row["camera"], []).append(
            (rad, lat, math.hypot(ex, ey), row["rng"])
        )
    radial = np.asarray(radial)
    lateral = np.asarray(lateral)
    norms = np.asarray(norms)
    out = dict(
        n=int(norms.size),
        mean_m=float(norms.mean()),
        median_m=float(np.median(norms)),
        p95_m=float(np.percentile(norms, 95)),
        radial_bias_m=float(radial.mean()),
        radial_sd_m=float(radial.std()),
        lateral_bias_m=float(lateral.mean()),
        lateral_sd_m=float(lateral.std()),
    )
    out["per_camera"] = {
        cam: dict(
            n=len(vals),
            mean_range_m=float(np.mean([v[3] for v in vals])),
            mean_m=float(np.mean([v[2] for v in vals])),
            median_m=float(np.median([v[2] for v in vals])),
            p95_m=float(np.percentile([v[2] for v in vals], 95)),
            radial_bias_m=float(np.mean([v[0] for v in vals])),
            radial_sd_m=float(np.std([v[0] for v in vals])),
            lateral_bias_m=float(np.mean([v[1] for v in vals])),
            lateral_sd_m=float(np.std([v[1] for v in vals])),
        )
        for cam, vals in sorted(per_camera.items())
    }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    models = {
        cam: camera_model_from_world(WORLD, include_name=include)
        for cam, include in CAMERA_INCLUDES.items()
    }
    rows = load_rows()
    print(f"scored detections: {len(rows)}")

    arms: dict[str, dict] = {}

    # Arm 1: raw IPM.  Zero parameters of any kind.
    arms["raw IPM (floor, no correction)"] = dict(
        kwargs=lambda cam: dict(
            contact_z_m=0.0, along=0.0, along_slope=0.0, cross=0.0, cross_slope=0.0
        ),
        fitted=0,
    )

    # Arms 2-4: the real artifacts, loaded exactly the way the node loads them.
    for version, fitted in (("v4", 2), ("v3", 10), ("v2", 8)):
        path = CALIB / f"projection_calibration_{version}" / "projection_calibration.json"
        calibrations, contact_z = _load_calibration(path)
        arms[f"{version} (contact_z={contact_z})"] = dict(
            kwargs=lambda cam, c=calibrations, z=contact_z: dict(
                contact_z_m=z, **c.get(cam, dict(along=0.0, along_slope=0.0,
                                                 cross=0.0, cross_slope=0.0))
            ),
            fitted=fitted,
        )

    summary = {
        "schema_version": 2,
        "metric_scope": "open-loop per-detection camera measurement error",
        "runtime_information": [
            "camera_id", "RGB image", "YOLO bounding box",
            "camera intrinsics", "camera world pose",
        ],
        "evaluation_only_information": [
            "commanded ground-truth x/y", "robot yaw", "dataset stratum",
        ],
        "dataset": {
            "id": "warehouse_yolo_dataset_4cam_v3_20260724/localization_calibration",
            "protocol": "set-pose grid; four cameras; four cardinal robot yaws",
            "index": (
                "logs/perception_datasets/warehouse_yolo_dataset_4cam_v3_20260724/"
                "merged/localization_calibration_index.csv"
            ),
            "detector_boxes": str(DET_CACHE.relative_to(REPO)),
            "world": str(WORLD.relative_to(REPO)),
        },
        "n_detections": len(rows),
        "arms": {},
    }
    print(f"\n{'arm':38} {'fitted':>6} {'mean':>9} {'median':>9} "
          f"{'rad bias':>10} {'lat bias':>10}")
    for name, spec in arms.items():
        result = score(rows, models, spec["kwargs"])
        result["fitted_scalars"] = spec["fitted"]
        summary["arms"][name] = result
        print(f"  {name:36} {spec['fitted']:6d} {result['mean_m']*1000:7.1f}mm "
              f"{result['median_m']*1000:7.1f}mm {result['radial_bias_m']*1000:+8.1f}mm "
              f"{result['lateral_bias_m']*1000:+8.1f}mm")

    raw = summary["arms"]["raw IPM (floor, no correction)"]
    v4 = next(v for k, v in summary["arms"].items() if k.startswith("v4"))
    delta_mm = (v4["mean_m"] - raw["mean_m"]) * 1000.0
    summary["v4_minus_raw_mean_mm"] = delta_mm
    verdict = (
        "DELETE the cross-bearing DOF: v4 is not better than zero-parameter IPM"
        if delta_mm >= -1.0
        else "KEEP the cross-bearing DOF: v4 beats raw IPM"
    )
    summary["verdict"] = verdict
    print(f"\n  v4 - raw IPM = {delta_mm:+.1f} mm  ->  {verdict}")

    print("\ncurrent raw-IPM per-camera measurement error (same balanced dataset):")
    print(f"  {'camera':9} {'n':>5} {'range':>8} {'mean':>9} {'median':>9} "
          f"{'p95':>9} {'rad bias':>10} {'lat bias':>10}")
    for cam in sorted(raw["per_camera"]):
        value = raw["per_camera"][cam]
        print(f"  {cam:9} {value['n']:5d} {value['mean_range_m']:6.1f}m "
              f"{value['mean_m']*1000:7.1f}mm {value['median_m']*1000:7.1f}mm "
              f"{value['p95_m']*1000:7.1f}mm {value['radial_bias_m']*1000:+8.1f}mm "
              f"{value['lateral_bias_m']*1000:+8.1f}mm")

    print("\nhistorical v4 comparison -- lateral bias only (not current runtime):")
    print(f"  {'camera':9} {'raw IPM':>10} {'v4':>10}")
    for cam in sorted(raw["per_camera"]):
        print(f"  {cam:9} {raw['per_camera'][cam]['lateral_bias_m']*1000:+9.1f} "
              f"{v4['per_camera'][cam]['lateral_bias_m']*1000:+9.1f}")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
