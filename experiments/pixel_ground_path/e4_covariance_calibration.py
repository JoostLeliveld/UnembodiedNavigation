#!/usr/bin/env python3
"""e4 — a covariance that is calibrated, and an honest account of what sizes it.

e3 established the error budget for the chosen path (box centre, mesh object model,
z* = 0.085 m):

    irreducible yaw term          30.8 mm radial / 22.6 mm lateral  (CAD, design time)
    silhouette vs model           ~29.5 mm radial                   (rendering + labelling)
    detector                      ~0 mm                             (measured, negligible)

and two failures:
  * a single pixel-variance constant does not calibrate: NEES ran 26 at 0-5 m down to 7 at
    12-16 m, because the yaw term is roughly CONSTANT IN METRES while the pixel term grows
    with range.  The two must be separate terms.
  * the GT-free inter-camera-disagreement sizing underestimated the variance 4x (sigma 2x),
    because the dominant error is a shared cause that partly cancels between views.

This script builds the two-term covariance, checks its calibration per stratum, and then
asks the question that decides deployability: the disagreement estimator's shortfall is a
geometric property, so can it be PREDICTED at design time from CAD and corrected without
ever seeing a robot pose?

Run:  python3 experiments/pixel_ground_path/e4_covariance_calibration.py
Out:  logs/studies/pixel_ground_path/e4_covariance_calibration/
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
from dataset_paths import dataset_root  # noqa: E402

# Resolved, not hard-coded: the payload moved to cold storage on 2026-08-05.
DATASET = dataset_root(REPO)
DET_CACHE = (REPO / "logs/studies/pixel_ground_path/e2_detector_edge_characterisation"
             / "detector_boxes.csv")
OUT = REPO / "logs/studies/pixel_ground_path/e4_covariance_calibration"

MODEL_INCLUDES = {
    "camera_A": "external_camera", "camera_B": "external_camera_b",
    "camera_C": "external_camera_c", "camera_D": "external_camera_d",
}
IMG_W, IMG_H = 1280, 720
SITE = (-11.20, 11.50, -8.60, 8.60)
RANGE_BINS = ((0, 5), (5, 8), (8, 12), (12, 16))

ALPHA, Z_STAR = 0.50, 0.085        # frozen by e3
SD_U_DET, SD_V_DET = 0.63, 0.46    # e2, detector vs its own labels, GT-free


def bearing_frame(camera, x, y):
    cx, cy = float(camera.cam_pos[0]), float(camera.cam_pos[1])
    bx, by = x - cx, y - cy
    d = math.hypot(bx, by)
    return d, bx / d, by / d


def estimate(camera, u0, v0, u1, v1):
    uc = 0.5 * (u0 + u1)
    vv = v1 + ALPHA * (v0 - v1)
    return camera.pixel_to_world_at_z(uc, vv, Z_STAR), uc, vv


def jac(camera, uc, vv, step=0.5):
    J = np.zeros((2, 2))
    for axis in (0, 1):
        du = step if axis == 0 else 0.0
        dv = step if axis == 1 else 0.0
        a = camera.pixel_to_world_at_z(uc + du, vv + dv, Z_STAR)
        b = camera.pixel_to_world_at_z(uc - du, vv - dv, Z_STAR)
        if a is None or b is None:
            return None
        J[0, axis] = (a[0] - b[0]) / (2 * step)
        J[1, axis] = (a[1] - b[1]) / (2 * step)
    return J


def load_rows():
    det = {}
    with DET_CACHE.open(newline="", encoding="utf-8") as h:
        for rec in csv.DictReader(h):
            det[rec["sample_id"]] = rec
    rows = []
    with (DATASET / "localization_calibration_index.csv").open(newline="", encoding="utf-8") as h:
        for rec in csv.DictReader(h):
            d = det.get(rec["sample_id"])
            if not d or str(d["detected"]) != "1" or d["pu0"] == "":
                continue
            label = DATASET / rec["label"]
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
            rows.append(dict(sample_id=rec["sample_id"], camera=rec["camera_id"],
                             x=float(rec["robot_x"]), y=float(rec["robot_y"]),
                             yaw=float(rec["robot_yaw"]), rng=float(rec["camera_range_m"]),
                             pu0=float(d["pu0"]), pv0=float(d["pv0"]),
                             pu1=float(d["pu1"]), pv1=float(d["pv1"]),
                             lu0=float(xy[:, 0].min()), lu1=float(xy[:, 0].max()),
                             lv0=float(xy[:, 1].min()), lv1=float(xy[:, 1].max())))
    return rows


def cad_grid(models, n_pos=12, n_yaw=16):
    xs = np.linspace(SITE[0] + 1.0, SITE[1] - 1.0, n_pos)
    ys = np.linspace(SITE[2] + 1.0, SITE[3] - 1.0, n_pos)
    yaws = np.linspace(-math.pi, math.pi, n_yaw, endpoint=False)
    grid = []
    for cam, camera in models.items():
        for x in xs:
            for y in ys:
                d, _, _ = bearing_frame(camera, x, y)
                if not 2.0 <= d <= 17.0 or not camera.world_to_pixel(x, y, 0.0)[2]:
                    continue
                for yaw in yaws:
                    b = RSM.mesh_silhouette_bbox(camera, x, y, yaw)
                    if b is None or not (0 <= b[0] and b[2] < IMG_W
                                         and 0 <= b[1] and b[3] < IMG_H):
                        continue
                    grid.append((cam, x, y, yaw, b))
    return grid


def main() -> int:
    models = {c: camera_model_from_world(WORLD, include_name=i)
              for c, i in MODEL_INCLUDES.items()}
    rows = load_rows()
    print(f"{len(rows)} detections\n")
    summary: dict[str, object] = dict(n=len(rows), alpha=ALPHA, z_star_m=Z_STAR)

    # ---- 1. the irreducible yaw term, computed at DESIGN TIME from CAD -----
    print("=== 1. Sigma_yaw from CAD alone (no data): spread of the yaw-marginal error ===")
    grid = cad_grid(models)
    by_pose: dict[tuple, list] = {}
    for cam, x, y, yaw, b in grid:
        by_pose.setdefault((cam, round(x, 4), round(y, 4)), []).append((yaw, b))
    rad_sd, lat_sd, ranges = [], [], []
    for (cam, x, y), items in by_pose.items():
        camera = models[cam]
        r, l = [], []
        for yaw, b in items:
            p, _, _ = estimate(camera, b[0], b[1], b[2], b[3])
            if p is None:
                continue
            _, ux, uy = bearing_frame(camera, x, y)
            ex, ey = p[0] - x, p[1] - y
            r.append(ex * ux + ey * uy); l.append(-ex * uy + ey * ux)
        if len(r) < 8:
            continue
        rad_sd.append(np.std(r)); lat_sd.append(np.std(l))
        ranges.append(bearing_frame(camera, x, y)[0])
    rad_sd, lat_sd, ranges = np.asarray(rad_sd), np.asarray(lat_sd), np.asarray(ranges)
    print(f"  per-position spread over yaw: radial {rad_sd.mean()*1000:.1f} mm, "
          f"lateral {lat_sd.mean()*1000:.1f} mm  ({len(rad_sd)} positions)")
    print(f"  {'range':>10} {'radial':>9} {'lateral':>9}   is it constant in metres?")
    for lo, hi in RANGE_BINS:
        s = (ranges >= lo) & (ranges < hi)
        if s.sum():
            print(f"  {f'{lo}-{hi} m':>10} {rad_sd[s].mean()*1000:7.1f}mm "
                  f"{lat_sd[s].mean()*1000:7.1f}mm   n={int(s.sum())}")
    SIG_YAW_R, SIG_YAW_L = float(rad_sd.mean()), float(lat_sd.mean())
    summary["sigma_yaw"] = dict(radial_m=SIG_YAW_R, lateral_m=SIG_YAW_L,
                                n_positions=int(len(rad_sd)))
    print("  -> roughly constant in METRES, unlike the pixel term.  Two separate terms.")

    # ---- 2. the silhouette term, and what it costs to know it -------------
    print("\n=== 2. the silhouette term: real mask vs mesh model, per edge (px) ===")
    e = dict(
        ul=np.array([r["lu0"] - RSM.mesh_silhouette_bbox(models[r["camera"]], r["x"], r["y"],
                                                          r["yaw"])[0] for r in rows]),
        ur=np.array([r["lu1"] - RSM.mesh_silhouette_bbox(models[r["camera"]], r["x"], r["y"],
                                                          r["yaw"])[2] for r in rows]),
        vt=np.array([r["lv0"] - RSM.mesh_silhouette_bbox(models[r["camera"]], r["x"], r["y"],
                                                          r["yaw"])[1] for r in rows]),
        vb=np.array([r["lv1"] - RSM.mesh_silhouette_bbox(models[r["camera"]], r["x"], r["y"],
                                                          r["yaw"])[3] for r in rows]),
    )
    sd_u_sil = float(np.std(0.5 * (e["ul"] + e["ur"])))
    sd_v_sil = float(np.std(ALPHA * e["vt"] + (1 - ALPHA) * e["vb"]))
    print(f"  u centre {np.mean(0.5*(e['ul']+e['ur'])):+.2f} +- {sd_u_sil:.2f} px")
    print(f"  v statistic {np.mean(ALPHA*e['vt']+(1-ALPHA)*e['vb']):+.2f} +- {sd_v_sil:.2f} px")
    print(f"  detector, for comparison: sd_u {SD_U_DET:.2f} px, sd_v {SD_V_DET:.2f} px")
    print("  NOTE: measuring this needed robot poses.  It is a COMMISSIONING measurement,")
    print("  not a runtime one, but it is the one place the method is not pose-free.")
    summary["sigma_silhouette_px"] = dict(u=sd_u_sil, v=sd_v_sil)

    sd_u = math.hypot(SD_U_DET, sd_u_sil)
    sd_v = math.hypot(SD_V_DET, sd_v_sil)
    print(f"  combined Sigma_uv: sd_u {sd_u:.2f} px, sd_v {sd_v:.2f} px")
    summary["sigma_uv_px"] = dict(u=sd_u, v=sd_v)

    # ---- 3. NEES of the two-term covariance -------------------------------
    print("\n=== 3. NEES of R = J Sigma_uv J^T + Sigma_yaw   (target 2.0, 2 DOF) ===")

    def nees_for(sd_u_, sd_v_, sig_r, sig_l, sub=None):
        vals = []
        for r in (sub if sub is not None else rows):
            camera = models[r["camera"]]
            p, uc, vv = estimate(camera, r["pu0"], r["pv0"], r["pu1"], r["pv1"])
            J = jac(camera, uc, vv)
            if p is None or J is None:
                continue
            _, ux, uy = bearing_frame(camera, r["x"], r["y"])
            rot = np.array([[ux, uy], [-uy, ux]])
            R = rot @ (J @ np.diag([sd_u_**2, sd_v_**2]) @ J.T) @ rot.T
            R[0, 0] += sig_r**2
            R[1, 1] += sig_l**2
            ex, ey = p[0] - r["x"], p[1] - r["y"]
            err = np.array([ex * ux + ey * uy, -ex * uy + ey * ux])
            try:
                vals.append(float(err @ np.linalg.solve(R, err)))
            except np.linalg.LinAlgError:
                continue
        return np.asarray(vals)

    variants = {
        "detector px only, no Sigma_yaw": (SD_U_DET, SD_V_DET, 0.0, 0.0),
        "detector px + Sigma_yaw": (SD_U_DET, SD_V_DET, SIG_YAW_R, SIG_YAW_L),
        "combined px, no Sigma_yaw": (sd_u, sd_v, 0.0, 0.0),
        "combined px + Sigma_yaw (FULL)": (sd_u, sd_v, SIG_YAW_R, SIG_YAW_L),
    }
    for name, args in variants.items():
        v = nees_for(*args)
        print(f"  {name:32} mean {v.mean():6.2f}  median {np.median(v):6.2f}  "
              f"frac>9.21 {(v>9.21).mean():.3f}")
        summary.setdefault("nees", {})[name] = dict(mean=float(v.mean()),
                                                    median=float(np.median(v)),
                                                    frac_gt_gate=float((v > 9.21).mean()))

    print("\n  FULL model, per stratum -- uniformity is what makes a covariance usable:")
    groups = [(c, [r for r in rows if r["camera"] == c]) for c in sorted(models)]
    groups += [(f"{lo}-{hi}m", [r for r in rows if lo <= r["rng"] < hi])
               for lo, hi in RANGE_BINS]
    groups += [(f"yaw {round(math.degrees(y))}",
                [r for r in rows if abs(r["yaw"] - y) < 1e-6])
               for y in sorted({r["yaw"] for r in rows})]
    strat = {}
    for name, sub in groups:
        if len(sub) < 5:
            continue
        v = nees_for(sd_u, sd_v, SIG_YAW_R, SIG_YAW_L, sub)
        strat[name] = float(v.mean())
        print(f"    {name:>12} n={len(v):5d}  mean NEES {v.mean():6.2f}")
    print(f"  spread across strata: {min(strat.values()):.2f} .. {max(strat.values()):.2f}")
    summary["nees_strata"] = strat

    # ---- 4. can the GT-free sizing be corrected at design time? -----------
    print("\n=== 4. the disagreement estimator's shortfall: is it predictable from CAD? ===")
    print("  Simulate the estimator on the CAD grid: for each position seen by two cameras,")
    print("  compare what disagreement would report against the true yaw-marginal variance.")
    pos_map: dict[tuple, dict] = {}
    for cam, x, y, yaw, b in grid:
        pos_map.setdefault((round(x, 4), round(y, 4), round(yaw, 4)), {})[cam] = (x, y, b)
    num_dis, den_dis, num_true, den_true = 0.0, 0.0, 0.0, 0.0
    for _key, cams in pos_map.items():
        est = {}
        for cam, (x, y, b) in cams.items():
            camera = models[cam]
            p, uc, vv = estimate(camera, b[0], b[1], b[2], b[3])
            J = jac(camera, uc, vv)
            if p is None or J is None:
                continue
            est[cam] = (p, J, x, y)
        names = list(est)
        for i, a in enumerate(names):
            pa, Ja, x, y = est[a]
            num_true += (pa[0] - x)**2 + (pa[1] - y)**2
            den_true += float(np.trace(Ja @ Ja.T))
            for b_ in names[i + 1:]:
                pb, Jb, _, _ = est[b_]
                num_dis += (pa[0] - pb[0])**2 + (pa[1] - pb[1])**2
                den_dis += float(np.trace(Ja @ Ja.T) + np.trace(Jb @ Jb.T))
    k_dis_cad = num_dis / den_dis
    k_true_cad = num_true / den_true
    factor = k_true_cad / k_dis_cad
    print(f"  on CAD:  disagreement k = {k_dis_cad:.3f} px^2, true k = {k_true_cad:.3f} px^2")
    print(f"  design-time correction factor = {factor:.2f}")
    print("  measured on real data in e3: disagreement k = 0.835, GT k = 3.432 "
          f"-> factor {3.432/0.835:.2f}")
    print(f"  CAD predicts {factor:.2f}, reality wants {3.432/0.835:.2f} -- "
          f"{abs(factor/(3.432/0.835)-1)*100:.0f} % apart")
    summary["disagreement_correction"] = dict(k_disagreement_cad=float(k_dis_cad),
                                              k_true_cad=float(k_true_cad),
                                              cad_factor=float(factor),
                                              real_factor=float(3.432 / 0.835))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                      encoding="utf-8")
    print(f"\nwrote {(OUT / 'summary.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
