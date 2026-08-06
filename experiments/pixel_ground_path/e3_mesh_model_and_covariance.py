#!/usr/bin/env python3
"""e3 — the rendered-mesh object model, and a covariance that is actually calibrated.

e2 left two problems.
  (1) The residual was NOT detector-limited: the detector tracks its labels to +-0.5-0.9 px
      (~1.7 cm) but the end-to-end error was ~5 cm.  The gap is object-model mismatch --
      the semantic mask renders the VISUAL MESHES while the model used the collision
      primitives, which put the robot's top 6.8 mm too high among other differences.
  (2) NEES was 41-51 against a target of 2, i.e. the covariance was over-confident ~5x in
      sigma, because Sigma_model was assumed negligible when it dominates.

This script swaps in the mesh silhouette model, re-derives the statistic, decomposes the
error budget, and then sizes the covariance the way a deployment would have to: from
**inter-camera disagreement**, which needs no robot ground truth.  402 poses in this
dataset are seen by >=2 cameras.  Ground truth is used only to check that the GT-free
sizing agrees.

Run:  python3 experiments/pixel_ground_path/e3_mesh_model_and_covariance.py
Out:  logs/studies/pixel_ground_path/e3_mesh_model_and_covariance/
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
import robot_silhouette_model as RSM  # noqa: E402

WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
DATASET = Path(
    "/home/joostleliveld/Thesis/_archive/UnembodiedNavigation_paused_2026-08-05"
    "/perception_datasets/warehouse_yolo_dataset_4cam_v3_20260724/merged"
)
DET_CACHE = (REPO / "logs/studies/pixel_ground_path/e2_detector_edge_characterisation"
             / "detector_boxes.csv")
OUT = REPO / "logs/studies/pixel_ground_path/e3_mesh_model_and_covariance"

MODEL_INCLUDES = {
    "camera_A": "external_camera",
    "camera_B": "external_camera_b",
    "camera_C": "external_camera_c",
    "camera_D": "external_camera_d",
}
IMG_W, IMG_H = 1280, 720
SITE = (-11.20, 11.50, -8.60, 8.60)
RANGE_BINS = ((0, 5), (5, 8), (8, 12), (12, 16), (16, 20))
# Tie-break rule for the statistic, fixed before looking at any validation number:
# prefer a NAMED statistic (alpha = 0.5, the box centre) if it is within this of the
# design-time optimum.  e2 picked alpha = 0.6 on a 0.2 mm margin and it cost 2 mm on real
# data, which is exactly the over-fitting this rule exists to prevent.
PLATEAU_TOL_M = 0.002


def bearing_frame(camera, x, y):
    cx, cy = float(camera.cam_pos[0]), float(camera.cam_pos[1])
    bx, by = x - cx, y - cy
    d = math.hypot(bx, by)
    return d, bx / d, by / d


def load_rows():
    det = {}
    if DET_CACHE.exists():
        with DET_CACHE.open(newline="", encoding="utf-8") as h:
            for rec in csv.DictReader(h):
                det[rec["sample_id"]] = rec
    rows = []
    with (DATASET / "localization_calibration_index.csv").open(newline="", encoding="utf-8") as h:
        for rec in csv.DictReader(h):
            label = DATASET / rec["label"]
            if not label.exists():
                continue
            best = None
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 7:
                    continue
                xy = np.asarray([float(p) for p in parts[1:]], float).reshape(-1, 2)
                xy[:, 0] *= IMG_W
                xy[:, 1] *= IMG_H
                area = 0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1))
                                 - np.dot(xy[:, 1], np.roll(xy[:, 0], 1)))
                if best is None or area > best[0]:
                    best = (area, xy)
            if best is None:
                continue
            xy = best[1]
            row = dict(sample_id=rec["sample_id"], camera=rec["camera_id"],
                       split=rec["split"], x=float(rec["robot_x"]), y=float(rec["robot_y"]),
                       yaw=float(rec["robot_yaw"]), rng=float(rec["camera_range_m"]),
                       lu0=float(xy[:, 0].min()), lu1=float(xy[:, 0].max()),
                       lv0=float(xy[:, 1].min()), lv1=float(xy[:, 1].max()))
            d = det.get(rec["sample_id"])
            if d and str(d["detected"]) == "1" and d["pu0"] != "":
                row.update(pu0=float(d["pu0"]), pv0=float(d["pv0"]),
                           pu1=float(d["pu1"]), pv1=float(d["pv1"]))
            rows.append(row)
    return rows


def estimate(camera, u0, v0, u1, v1, alpha, z):
    uc = 0.5 * (u0 + u1)
    vv = v1 + alpha * (v0 - v1)
    return camera.pixel_to_world_at_z(uc, vv, z), uc, vv


def jac(camera, uc, vv, z, step=0.5):
    J = np.zeros((2, 2))
    for axis in (0, 1):
        du = step if axis == 0 else 0.0
        dv = step if axis == 1 else 0.0
        a = camera.pixel_to_world_at_z(uc + du, vv + dv, z)
        b = camera.pixel_to_world_at_z(uc - du, vv - dv, z)
        if a is None or b is None:
            return None
        J[0, axis] = (a[0] - b[0]) / (2 * step)
        J[1, axis] = (a[1] - b[1]) / (2 * step)
    return J


def main() -> int:
    models = {c: camera_model_from_world(WORLD, include_name=i)
              for c, i in MODEL_INCLUDES.items()}
    rows = load_rows()
    det_rows = [r for r in rows if "pu0" in r]
    print(f"{len(rows)} labelled samples, {len(det_rows)} with a detector box")
    print(f"mesh model: {len(RSM.MESH_LOCAL)} points, "
          f"{RSM.MESH_LOCAL[:,1].ptp()*1000:.1f} mm wide, "
          f"{RSM.MESH_LOCAL[:,2].max()*1000:.1f} mm tall "
          f"(collision primitives: {RSM.OVERALL_HEIGHT*1000:.1f} mm tall)\n")
    summary: dict[str, object] = dict(n_labelled=len(rows), n_detected=len(det_rows))

    # ---- A. mesh vs collision model against real labels ---------------------
    print("=== A. forward model vs real semantic-mask labels (px) ===")
    print(f"  {'model':12} {'camera':9} {'d u_left':>13} {'d u_right':>13} {'d v_top':>13} "
          f"{'d v_bot':>13}")
    fwd = {}
    for name, fn in (("collision", RSM.silhouette_bbox), ("MESH", RSM.mesh_silhouette_bbox)):
        for cam in sorted(models):
            g = [r for r in rows if r["camera"] == cam]
            e = {k: [] for k in ("ul", "ur", "vt", "vb")}
            for r in g:
                b = fn(models[cam], r["x"], r["y"], r["yaw"])
                if b is None:
                    continue
                e["ul"].append(r["lu0"] - b[0]); e["ur"].append(r["lu1"] - b[2])
                e["vt"].append(r["lv0"] - b[1]); e["vb"].append(r["lv1"] - b[3])
            fwd[f"{name}/{cam}"] = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v)))
                                    for k, v in e.items()}
            print(f"  {name:12} {cam:9} " + " ".join(
                f"{np.mean(e[k]):+6.2f}+-{np.std(e[k]):5.2f}" for k in ("ul", "ur", "vt", "vb")))
    summary["forward_edges_px"] = fwd

    # ---- B. design-time statistic with the mesh model -----------------------
    print("\n=== B. DESIGN TIME with the mesh model (CAD + calibration only) ===")
    xs = np.linspace(SITE[0] + 1.0, SITE[1] - 1.0, 12)
    ys = np.linspace(SITE[2] + 1.0, SITE[3] - 1.0, 12)
    yaws = np.linspace(-math.pi, math.pi, 16, endpoint=False)
    grid = []
    for cam, camera in models.items():
        for x in xs:
            for y in ys:
                d, _, _ = bearing_frame(camera, x, y)
                if not 2.0 <= d <= 17.0:
                    continue
                if not camera.world_to_pixel(x, y, 0.0)[2]:
                    continue
                for yaw in yaws:
                    b = RSM.mesh_silhouette_bbox(camera, x, y, yaw)
                    if b is None or not (0 <= b[0] and b[2] < IMG_W
                                         and 0 <= b[1] and b[3] < IMG_H):
                        continue
                    grid.append((cam, x, y, yaw, b))
    print(f"  grid: {len(grid)} (camera, position, yaw) combinations")

    def grid_cost(alpha, z):
        rad, lat = [], []
        for cam, x, y, _yaw, b in grid:
            camera = models[cam]
            p, _, _ = estimate(camera, b[0], b[1], b[2], b[3], alpha, z)
            if p is None:
                continue
            _, ux, uy = bearing_frame(camera, x, y)
            ex, ey = p[0] - x, p[1] - y
            rad.append(ex * ux + ey * uy); lat.append(-ex * uy + ey * ux)
        rad, lat = np.asarray(rad), np.asarray(lat)
        return math.sqrt(float(np.mean(rad**2) + np.mean(lat**2))), rad, lat

    curve = {}
    best = None
    print(f"  {'alpha':>6} {'z*(m)':>7} {'yaw-marginal RMS':>18} {'rad bias':>10} "
          f"{'rad sd':>8}")
    for alpha in np.linspace(0.0, 1.0, 11):
        inner = None
        for z in np.linspace(0.0, 0.20, 81):
            c, rad, lat = grid_cost(float(alpha), float(z))
            if inner is None or c < inner[0]:
                inner = (c, float(z), rad, lat)
        c, z, rad, lat = inner
        curve[f"{alpha:.1f}"] = dict(z_m=z, rms_m=c, radial_bias_m=float(rad.mean()),
                                     radial_sd_m=float(rad.std()))
        print(f"  {alpha:6.1f} {z:7.3f} {c*1000:16.1f}mm {rad.mean()*1000:+8.1f}mm "
              f"{rad.std()*1000:6.1f}mm")
        if best is None or c < best[0]:
            best = (c, float(alpha), z)
    opt_cost, opt_alpha, opt_z = best
    # apply the pre-registered plateau tie-break
    alpha_star, z_star, why = opt_alpha, opt_z, "design-time optimum"
    if abs(curve["0.5"]["rms_m"] - opt_cost) <= PLATEAU_TOL_M and opt_alpha != 0.5:
        alpha_star, z_star = 0.5, curve["0.5"]["z_m"]
        why = (f"box centre, within {PLATEAU_TOL_M*1000:.0f} mm of the optimum "
               f"(alpha={opt_alpha:.1f})")
    print(f"  optimum alpha={opt_alpha:.1f} z={opt_z:.3f} ({opt_cost*1000:.1f} mm); "
          f"CHOSEN alpha={alpha_star:.2f} z*={z_star:.3f} m -- {why}")
    summary["design_time"] = dict(curve=curve, optimum_alpha=opt_alpha,
                                  optimum_z_m=opt_z, chosen_alpha=alpha_star,
                                  chosen_z_m=z_star, rule=why,
                                  plateau_tol_m=PLATEAU_TOL_M)

    # ---- C. end to end on detector boxes -----------------------------------
    print("\n=== C. END TO END on detector boxes ===")

    def score(sub, alpha, z, src="p"):
        rad, lat, norm = [], [], []
        for r in sub:
            camera = models[r["camera"]]
            k = ("pu0", "pv0", "pu1", "pv1") if src == "p" else ("lu0", "lv0", "lu1", "lv1")
            if k[0] not in r:
                continue
            p, _, _ = estimate(camera, r[k[0]], r[k[1]], r[k[2]], r[k[3]], alpha, z)
            if p is None:
                continue
            _, ux, uy = bearing_frame(camera, r["x"], r["y"])
            ex, ey = p[0] - r["x"], p[1] - r["y"]
            rad.append(ex * ux + ey * uy); lat.append(-ex * uy + ey * ux)
            norm.append(math.hypot(ex, ey))
        return np.asarray(rad), np.asarray(lat), np.asarray(norm)

    arms = {
        "bottom @ 0.05 (DEPLOYED)": (0.0, 0.05),
        "bottom @ floor": (0.0, 0.0),
        "collision-model optimum": (0.5, 0.090),
        f"MESH model (a={alpha_star:.2f})": (alpha_star, z_star),
    }
    print(f"  {'arm':30} {'mean':>8} {'median':>8} {'p95':>8} {'rad bias':>10} "
          f"{'rad sd':>8} {'lat sd':>8}")
    e2e = {}
    for name, (a, z) in arms.items():
        rad, lat, norm = score(det_rows, a, z)
        e2e[name] = dict(n=int(norm.size), mean_m=float(norm.mean()),
                         median_m=float(np.median(norm)),
                         p95_m=float(np.percentile(norm, 95)),
                         radial_bias_m=float(rad.mean()), radial_sd_m=float(rad.std()),
                         lateral_bias_m=float(lat.mean()), lateral_sd_m=float(lat.std()))
        print(f"  {name:30} {norm.mean()*1000:6.1f}mm {np.median(norm)*1000:6.1f}mm "
              f"{np.percentile(norm,95)*1000:6.1f}mm {rad.mean()*1000:+8.1f}mm "
              f"{rad.std()*1000:6.1f}mm {lat.std()*1000:6.1f}mm")
    summary["end_to_end"] = e2e

    print("\n  chosen path, stratified (the uniformity tests):")
    print(f"  {'stratum':>12} {'n':>5} {'mean':>8} {'rad bias':>10} {'rad sd':>8} "
          f"{'lat bias':>10}")
    strata = {}
    groups = [(cam, [r for r in det_rows if r["camera"] == cam]) for cam in sorted(models)]
    groups += [(f"{lo}-{hi}m", [r for r in det_rows if lo <= r["rng"] < hi])
               for lo, hi in RANGE_BINS]
    groups += [(f"yaw {round(math.degrees(y))}",
                [r for r in det_rows if round(math.degrees(r["yaw"])) == round(math.degrees(y))])
               for y in sorted({r["yaw"] for r in det_rows})]
    for name, sub in groups:
        if len(sub) < 5:
            continue
        rad, lat, norm = score(sub, alpha_star, z_star)
        strata[name] = dict(n=int(norm.size), mean_m=float(norm.mean()),
                            radial_bias_m=float(rad.mean()), radial_sd_m=float(rad.std()),
                            lateral_bias_m=float(lat.mean()),
                            lateral_sd_m=float(lat.std()))
        print(f"  {name:>12} {norm.size:5d} {norm.mean()*1000:6.1f}mm "
              f"{rad.mean()*1000:+8.1f}mm {rad.std()*1000:6.1f}mm "
              f"{lat.mean()*1000:+8.1f}mm")
    cam_bias = [strata[c]["radial_bias_m"] for c in sorted(models) if c in strata]
    print(f"  per-camera radial bias spread: {(max(cam_bias)-min(cam_bias))*1000:.1f} mm")
    summary["strata"] = strata
    summary["per_camera_radial_bias_spread_m"] = float(max(cam_bias) - min(cam_bias))

    # ---- D. error budget ----------------------------------------------------
    print("\n=== D. error budget for the chosen path ===")
    rad_l, lat_l, _ = score(det_rows, alpha_star, z_star, src="l")   # perfect detector
    rad_p, lat_p, _ = score(det_rows, alpha_star, z_star, src="p")   # real detector
    geo_rad, _ = grid_cost(alpha_star, z_star)[1:]
    print(f"  {'term':38} {'radial sd':>11} {'lateral sd':>11}")
    print(f"  {'CAD yaw-marginal (design-time grid)':38} "
          f"{grid_cost(alpha_star, z_star)[1].std()*1000:9.1f}mm "
          f"{grid_cost(alpha_star, z_star)[2].std()*1000:9.1f}mm")
    print(f"  {'+ real silhouette vs mesh model (labels)':38} {rad_l.std()*1000:9.1f}mm "
          f"{lat_l.std()*1000:9.1f}mm")
    print(f"  {'+ detector vs label (full path)':38} {rad_p.std()*1000:9.1f}mm "
          f"{lat_p.std()*1000:9.1f}mm")
    det_only_r = math.sqrt(max(rad_p.std()**2 - rad_l.std()**2, 0.0))
    det_only_l = math.sqrt(max(lat_p.std()**2 - lat_l.std()**2, 0.0))
    print(f"  -> detector contributes {det_only_r*1000:.1f} mm radial / "
          f"{det_only_l*1000:.1f} mm lateral in quadrature")
    print(f"  -> object model + rendering contributes "
          f"{math.sqrt(max(rad_l.std()**2 - grid_cost(alpha_star,z_star)[1].std()**2,0))*1000:.1f}"
          f" mm radial")
    summary["error_budget"] = dict(
        cad_yaw_marginal_radial_sd_m=float(grid_cost(alpha_star, z_star)[1].std()),
        labels_radial_sd_m=float(rad_l.std()), detector_radial_sd_m=float(rad_p.std()),
        detector_only_radial_sd_m=float(det_only_r),
        labels_lateral_sd_m=float(lat_l.std()), detector_lateral_sd_m=float(lat_p.std()))

    # ---- E. GT-FREE covariance sizing from inter-camera disagreement -------
    print("\n=== E. sizing the covariance with NO ground truth: camera disagreement ===")
    by_pose: dict[tuple, list] = {}
    for r in det_rows:
        by_pose.setdefault((round(r["x"], 3), round(r["y"], 3), round(r["yaw"], 3)),
                           []).append(r)
    pairs = [(a, b) for group in by_pose.values() if len(group) >= 2
             for i, a in enumerate(group) for b in group[i + 1:]]
    print(f"  {sum(1 for g in by_pose.values() if len(g) >= 2)} poses seen by >=2 cameras "
          f"-> {len(pairs)} camera pairs")

    # unit covariance: R_c = k * J diag(1,1) J^T  (k in px^2), so
    # E[|p_i - p_j|^2] = k * (tr(J_i J_i^T) + tr(J_j J_j^T))
    num, den = 0.0, 0.0
    for a, b in pairs:
        ca, cb = models[a["camera"]], models[b["camera"]]
        pa, ua, va = estimate(ca, a["pu0"], a["pv0"], a["pu1"], a["pv1"], alpha_star, z_star)
        pb, ub, vb = estimate(cb, b["pu0"], b["pv0"], b["pu1"], b["pv1"], alpha_star, z_star)
        if pa is None or pb is None:
            continue
        Ja, Jb = jac(ca, ua, va, z_star), jac(cb, ub, vb, z_star)
        if Ja is None or Jb is None:
            continue
        num += (pa[0] - pb[0])**2 + (pa[1] - pb[1])**2
        den += float(np.trace(Ja @ Ja.T) + np.trace(Jb @ Jb.T))
    k_free = num / den
    print(f"  GT-FREE effective pixel variance  k = {k_free:.3f} px^2  "
          f"-> sigma = {math.sqrt(k_free):.2f} px")

    # the same quantity from ground truth, for the check only
    num_gt, den_gt = 0.0, 0.0
    for r in det_rows:
        camera = models[r["camera"]]
        p, uc, vv = estimate(camera, r["pu0"], r["pv0"], r["pu1"], r["pv1"],
                             alpha_star, z_star)
        J = jac(camera, uc, vv, z_star)
        if p is None or J is None:
            continue
        num_gt += (p[0] - r["x"])**2 + (p[1] - r["y"])**2
        den_gt += float(np.trace(J @ J.T))
    k_gt = num_gt / den_gt
    print(f"  ground-truth-derived              k = {k_gt:.3f} px^2  "
          f"-> sigma = {math.sqrt(k_gt):.2f} px")
    print(f"  ratio GT-free / GT = {k_free/k_gt:.2f}   "
          f"(1.0 would mean the deployable estimate is exact)")
    print("  caveat: the object-model term is a shared cause, so it is only partly visible")
    print("  in disagreement -- each camera is pulled toward its OWN nadir, so the")
    print("  directions differ and most of it does show up, but not all.")
    summary["covariance_sizing"] = dict(k_free_px2=float(k_free), k_gt_px2=float(k_gt),
                                        ratio=float(k_free / k_gt), n_pairs=len(pairs))

    # ---- F. NEES with each sizing ------------------------------------------
    print("\n=== F. NEES (target 2.0 for 2 DOF) ===")
    nees_block = {}
    for name, k in (("detector-only Sigma_uv (0.63/0.46 px)", None),
                    ("GT-free disagreement sizing", k_free),
                    ("GT-derived sizing", k_gt)):
        vals = []
        for r in det_rows:
            camera = models[r["camera"]]
            p, uc, vv = estimate(camera, r["pu0"], r["pv0"], r["pu1"], r["pv1"],
                                 alpha_star, z_star)
            J = jac(camera, uc, vv, z_star)
            if p is None or J is None:
                continue
            S = np.diag([0.63**2, 0.46**2]) if k is None else np.diag([k, k])
            R = J @ S @ J.T
            e = np.array([p[0] - r["x"], p[1] - r["y"]])
            try:
                vals.append(float(e @ np.linalg.solve(R, e)))
            except np.linalg.LinAlgError:
                continue
        vals = np.asarray(vals)
        nees_block[name] = dict(mean=float(vals.mean()), median=float(np.median(vals)),
                                frac_gt_9_21=float((vals > 9.21).mean()))
        print(f"  {name:38} mean {vals.mean():6.2f}  median {np.median(vals):6.2f}  "
              f"frac > 9.21 gate {(vals > 9.21).mean():.3f}")

    print("\n  per-stratum NEES under the GT-free sizing (is it uniform?):")
    for name, sub in groups:
        if len(sub) < 5:
            continue
        vals = []
        for r in sub:
            camera = models[r["camera"]]
            p, uc, vv = estimate(camera, r["pu0"], r["pv0"], r["pu1"], r["pv1"],
                                 alpha_star, z_star)
            J = jac(camera, uc, vv, z_star)
            if p is None or J is None:
                continue
            R = J @ np.diag([k_free, k_free]) @ J.T
            e = np.array([p[0] - r["x"], p[1] - r["y"]])
            try:
                vals.append(float(e @ np.linalg.solve(R, e)))
            except np.linalg.LinAlgError:
                continue
        nees_block[f"stratum/{name}"] = dict(mean=float(np.mean(vals)))
        print(f"    {name:>12} n={len(vals):5d}  mean NEES {np.mean(vals):6.2f}")
    summary["nees"] = nees_block

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                      encoding="utf-8")
    print(f"\nwrote {(OUT / 'summary.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
