"""Does the experiment-local candidate compute the path e3/e4 measured?

e3/e4/e5 each carry a private `estimate()` / `jac()` / `bearing_frame()` copy of the pixel to
ground math, and the candidate module was written afterwards. That is exactly the shape of
divergence this repository has been bitten by before -- `operational_residual_rcond/exp3` found
the detector node applying a projection with one fewer degree of freedom than the camera
manager, because the code had been hand-copied. So the candidate is not allowed to merely look
like the evidence path; it has to reproduce it numerically.

This script checks four things and writes nothing into `logs/`:

1. the four evidence cameras really do share the frozen reference mount;
2. the frozen constants still equal the values recorded in e4's `summary.json`;
3. the candidate's map-frame covariance is the exact frame change of e4's bearing-frame
   construction, over a synthetic box sweep on the real cameras;
4. the four NEES variants recomputed **through the candidate** on the real detections reproduce
   e4's recorded means.

Item 4 additionally quantifies the one deliberate difference between the two: e4 rotates
Sigma_yaw using the bearing to the *true* pose, and the candidate uses the bearing to the
*estimated* point, because that is the only one a deployed caller has.

    python3 experiments/pixel_ground_path/verify_candidate_matches_evidence.py
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

from dataset_paths import dataset_root  # noqa: E402
from box_projection import (  # noqa: E402
    BOX_STATISTIC_ALPHA,
    BOX_STATISTIC_PLANE_Z_M,
    BOX_STATISTIC_REFERENCE_MOUNT,
    BOX_STATISTIC_SIGMA_UV_PX,
    BOX_STATISTIC_SIGMA_YAW_M,
    box_statistic_mount_deviation,
    box_statistic_pixel,
    project_box_to_world_with_covariance,
)
from reliability.projection import camera_model_from_world  # noqa: E402

WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
E4_SUMMARY = REPO / "logs/studies/pixel_ground_path/e4_covariance_calibration/summary.json"
DET_CACHE = (REPO / "logs/studies/pixel_ground_path/e2_detector_edge_characterisation"
             / "detector_boxes.csv")
MODEL_INCLUDES = {
    "camera_A": "external_camera", "camera_B": "external_camera_b",
    "camera_C": "external_camera_c", "camera_D": "external_camera_d",
}
SD_U_DET, SD_V_DET = 0.63, 0.46      # e2, detector vs its own labels
RANGE_BINS = ((0, 5), (5, 8), (8, 12), (12, 16))

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        _failures.append(label)


def e4_covariance_in_map_frame(camera, box, sd_u, sd_v, sigma_r, sigma_l, *, bearing_xy):
    """e4's construction: build R in the bearing frame, then express it in map axes."""

    u, v = box_statistic_pixel(box, alpha=BOX_STATISTIC_ALPHA)
    step, jac = 0.5, np.zeros((2, 2))
    for axis in (0, 1):
        du, dv = (step, 0.0) if axis == 0 else (0.0, step)
        plus = camera.pixel_to_world_at_z(u + du, v + dv, BOX_STATISTIC_PLANE_Z_M)
        minus = camera.pixel_to_world_at_z(u - du, v - dv, BOX_STATISTIC_PLANE_Z_M)
        if plus is None or minus is None:
            return None
        jac[0, axis] = (plus[0] - minus[0]) / (2.0 * step)
        jac[1, axis] = (plus[1] - minus[1]) / (2.0 * step)
    bx = bearing_xy[0] - float(camera.cam_pos[0])
    by = bearing_xy[1] - float(camera.cam_pos[1])
    norm = math.hypot(bx, by)
    if norm <= 1.0e-9:
        return None
    ux, uy = bx / norm, by / norm
    rot = np.asarray(((ux, uy), (-uy, ux)))
    r_bearing = rot @ (jac @ np.diag((sd_u**2, sd_v**2)) @ jac.T) @ rot.T
    r_bearing[0, 0] += sigma_r**2
    r_bearing[1, 1] += sigma_l**2
    return rot.T @ r_bearing @ rot


def load_rows(dataset: Path) -> list[dict]:
    """e4's row set: index rows with a cached detection and a parseable label polygon."""

    detections = {}
    with DET_CACHE.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            detections[record["sample_id"]] = record
    rows = []
    with (dataset / "localization_calibration_index.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for record in csv.DictReader(handle):
            found = detections.get(record["sample_id"])
            if not found or str(found["detected"]) != "1" or found["pu0"] == "":
                continue
            label = dataset / record["label"]
            if not label.is_file() or not any(
                len(line.split()) >= 7 for line in label.read_text(encoding="utf-8").splitlines()
            ):
                continue
            rows.append(dict(
                camera=record["camera_id"], x=float(record["robot_x"]),
                y=float(record["robot_y"]), rng=float(record["camera_range_m"]),
                box=(float(found["pu0"]), float(found["pv0"]),
                     float(found["pu1"]), float(found["pv1"])),
            ))
    return rows


def main() -> int:
    summary = json.loads(E4_SUMMARY.read_text(encoding="utf-8"))
    models = {c: camera_model_from_world(WORLD, include_name=i)
              for c, i in MODEL_INCLUDES.items()}

    print("\n=== 1. the evidence cameras share the frozen reference mount ===")
    print(f"  reference: height {BOX_STATISTIC_REFERENCE_MOUNT[0]} m, "
          f"pitch {BOX_STATISTIC_REFERENCE_MOUNT[1]} rad")
    for name, camera in models.items():
        d_height, d_pitch = box_statistic_mount_deviation(camera)
        check(f"{name} on the reference mount",
              abs(d_height) < 1.0e-6 and abs(d_pitch) < 1.0e-3,
              f"dz {d_height:+.2e} m, dpitch {d_pitch:+.2e} rad")

    print("\n=== 2. frozen constants still equal the recorded evidence ===")
    check("alpha", abs(summary["alpha"] - BOX_STATISTIC_ALPHA) < 1e-12,
          f"{summary['alpha']} vs {BOX_STATISTIC_ALPHA}")
    check("plane z*", abs(summary["z_star_m"] - BOX_STATISTIC_PLANE_Z_M) < 1e-12,
          f"{summary['z_star_m']} vs {BOX_STATISTIC_PLANE_Z_M}")
    for axis, index in (("u", 0), ("v", 1)):
        recorded = summary["sigma_uv_px"][axis]
        check(f"sigma_uv {axis}", abs(recorded - BOX_STATISTIC_SIGMA_UV_PX[index]) < 5e-3,
              f"{recorded:.4f} px vs frozen {BOX_STATISTIC_SIGMA_UV_PX[index]}")
    for axis, index in (("radial_m", 0), ("lateral_m", 1)):
        recorded = summary["sigma_yaw"][axis]
        check(f"sigma_yaw {axis}", abs(recorded - BOX_STATISTIC_SIGMA_YAW_M[index]) < 5e-5,
              f"{recorded:.6f} m vs frozen {BOX_STATISTIC_SIGMA_YAW_M[index]}")

    print("\n=== 3. candidate covariance == e4 construction, expressed in map axes ===")
    rng = np.random.default_rng(20260806)
    worst_point, worst_cov, checked = 0.0, 0.0, 0
    for camera in models.values():
        for _ in range(400):
            u0, v0 = rng.uniform(0.0, 1150.0), rng.uniform(0.0, 620.0)
            box = (u0, v0, u0 + rng.uniform(10.0, 90.0), v0 + rng.uniform(10.0, 70.0))
            result = project_box_to_world_with_covariance(box, camera)
            if result is None:
                continue
            point, cov = result
            reference = e4_covariance_in_map_frame(
                camera, box, *BOX_STATISTIC_SIGMA_UV_PX, *BOX_STATISTIC_SIGMA_YAW_M,
                bearing_xy=point,
            )
            if reference is None:
                continue
            u, v = box_statistic_pixel(box)
            direct = camera.pixel_to_world_at_z(u, v, BOX_STATISTIC_PLANE_Z_M)
            worst_point = max(worst_point, abs(point[0] - direct[0]), abs(point[1] - direct[1]))
            worst_cov = max(worst_cov, float(np.max(np.abs(np.asarray(cov) - reference))))
            checked += 1
    check("point estimate identical", worst_point == 0.0, f"max |dx| {worst_point:.3e} m")
    check("covariance identical to rounding", worst_cov < 1e-15,
          f"max |dR| {worst_cov:.3e} m^2 over {checked} boxes")

    print("\n=== 4. e4's NEES variants, recomputed through the candidate ===")
    dataset = dataset_root(REPO)
    rows = load_rows(dataset)
    print(f"  {len(rows)} detections (e4 recorded n={summary['n']})")
    check("row set matches e4", len(rows) == int(summary["n"]))

    variants = {
        "detector px only, no Sigma_yaw": ((SD_U_DET, SD_V_DET), None),
        "detector px + Sigma_yaw": ((SD_U_DET, SD_V_DET), BOX_STATISTIC_SIGMA_YAW_M),
        "combined px, no Sigma_yaw": (BOX_STATISTIC_SIGMA_UV_PX, None),
        "combined px + Sigma_yaw (FULL)": (BOX_STATISTIC_SIGMA_UV_PX, BOX_STATISTIC_SIGMA_YAW_M),
    }
    full_values, full_ranges = [], []
    for name, (sigma_uv, sigma_yaw) in variants.items():
        values = []
        for row in rows:
            camera = models[row["camera"]]
            result = project_box_to_world_with_covariance(
                row["box"], camera, sigma_uv_px=sigma_uv, sigma_yaw_m=sigma_yaw,
            )
            if result is None:
                continue
            (px, py), cov = result
            error = np.asarray((px - row["x"], py - row["y"]))
            try:
                values.append(float(error @ np.linalg.solve(np.asarray(cov), error)))
            except np.linalg.LinAlgError:
                continue
        values = np.asarray(values)
        recorded = summary["nees"][name]
        check(f"{name:32} mean {values.mean():6.2f}",
              abs(values.mean() - recorded["mean"]) <= 0.02 * recorded["mean"],
              f"(e4 {recorded['mean']:.2f}, frac>9.21 {(values > 9.21).mean():.3f} "
              f"vs {recorded['frac_gt_gate']:.3f})")
        if name.endswith("(FULL)"):
            full_values = values
            full_ranges = np.asarray([r["rng"] for r in rows[: len(values)]])

    print("\n  NOTE: the two 'combined px' variants use the SHIPPED, rounded constants")
    print(f"  {BOX_STATISTIC_SIGMA_UV_PX} px against e4's "
          f"({summary['sigma_uv_px']['u']:.4f}, {summary['sigma_uv_px']['v']:.4f}).  That")
    print("  rounding is worth ~0.3 % of the pixel-term variance and is the whole of the")
    print("  no-Sigma_yaw gap above; the full model absorbs it into the heading term.")

    print("\n  FULL model per range stratum (e4 recorded a 2.54-3.18 spread overall):")
    for low, high in RANGE_BINS:
        mask = (full_ranges >= low) & (full_ranges < high)
        if mask.sum():
            print(f"    {f'{low}-{high} m':>10} n={int(mask.sum()):5d}  "
                  f"mean NEES {full_values[mask].mean():6.2f}")

    print("\n  the one deliberate difference: bearing from the ESTIMATE (candidate)")
    print("  versus from the TRUE pose (e4, evaluation-side).  Effect on the rotated block:")
    worst = 0.0
    for row in rows:
        camera = models[row["camera"]]
        result = project_box_to_world_with_covariance(row["box"], camera)
        if result is None:
            continue
        truth_referenced = e4_covariance_in_map_frame(
            camera, row["box"], *BOX_STATISTIC_SIGMA_UV_PX, *BOX_STATISTIC_SIGMA_YAW_M,
            bearing_xy=(row["x"], row["y"]),
        )
        if truth_referenced is None:
            continue
        worst = max(worst, float(np.max(np.abs(np.asarray(result[1]) - truth_referenced))))
    radial_variance = BOX_STATISTIC_SIGMA_YAW_M[0] ** 2
    check("estimate-referenced bearing is a second-order difference",
          worst < 0.05 * radial_variance,
          f"max |dR| {worst:.2e} m^2 = {100.0 * worst / radial_variance:.1f} % of sigma_r^2")

    print(f"\n{'ALL CHECKS PASSED' if not _failures else 'FAILURES: ' + ', '.join(_failures)}\n")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
