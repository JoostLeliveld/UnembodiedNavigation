#!/usr/bin/env python3
"""e1b — derive the pixel statistic from CAD, then validate it on real silhouettes.

e1 falsified the bounding-cylinder design: against real Gazebo semantic-mask silhouettes
the box centre at z = h/2 gave 56.5 mm, not the 1.9 mm that e0's self-consistent forward
check suggested, and the yaw cancellation collapsed from 30x to 1.2x.  The cause is the
object model: the Burger is not a constant-cross-section prism, and its body is offset
-0.032 m in x from the pose origin -- an offset that rotates with the unobserved yaw.

This script keeps the method's key property -- **no robot ground truth anywhere in the
estimator** -- by splitting the work:

  DESIGN TIME  (CAD + camera calibration only, no images, no poses from any run)
    Choose the vertical pixel statistic  v = v_bottom + alpha * (v_top - v_bottom)
    and the back-projection plane z*, to minimise the error of the inversion MARGINALISED
    OVER YAW.  Yaw is a nuisance parameter nobody observes, so marginalising is the correct
    treatment, and the leftover spread is a covariance term rather than a bias.

  VALIDATION  (real labels + poses)
    Score that fixed choice on 1849 real silhouettes.  Ground truth scores; it never
    selects alpha or z*.

Run:  python3 experiments/pixel_ground_path/e1b_cad_derived_statistic.py
Out:  logs/studies/pixel_ground_path/e1b_cad_derived_statistic/
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
OUT = REPO / "logs/studies/pixel_ground_path/e1b_cad_derived_statistic"

MODEL_INCLUDES = {
    "camera_A": "external_camera",
    "camera_B": "external_camera_b",
    "camera_C": "external_camera_c",
    "camera_D": "external_camera_d",
}
IMG_W, IMG_H = 1280, 720
SITE = (-11.20, 11.50, -8.60, 8.60)
RANGE_BINS = ((0, 5), (5, 8), (8, 12), (12, 16), (16, 20))


def bearing_frame(camera, x, y):
    cx, cy = float(camera.cam_pos[0]), float(camera.cam_pos[1])
    bx, by = x - cx, y - cy
    d = math.hypot(bx, by)
    return d, bx / d, by / d


def load_labels():
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
            rows.append(dict(camera=rec["camera_id"], split=rec["split"],
                             x=float(rec["robot_x"]), y=float(rec["robot_y"]),
                             yaw=float(rec["robot_yaw"]), rng=float(rec["camera_range_m"]),
                             u0=float(xy[:, 0].min()), u1=float(xy[:, 0].max()),
                             v0=float(xy[:, 1].min()), v1=float(xy[:, 1].max())))
    return rows


# ------------------------------------------------------------------ design-time grid


def design_grid(models, *, n_pos=14, n_yaw=24):
    """(camera, x, y, yaw) -> CAD silhouette bbox.  Uses NOTHING but CAD + calibration."""
    xs = np.linspace(SITE[0] + 1.0, SITE[1] - 1.0, n_pos)
    ys = np.linspace(SITE[2] + 1.0, SITE[3] - 1.0, n_pos)
    yaws = np.linspace(-math.pi, math.pi, n_yaw, endpoint=False)
    grid = []
    for cam, camera in models.items():
        for x in xs:
            for y in ys:
                d, _, _ = bearing_frame(camera, x, y)
                if not 2.0 <= d <= 17.0:
                    continue
                u, v, vis = camera.world_to_pixel(x, y, 0.0)
                if not vis:
                    continue
                for yaw in yaws:
                    b = RSM.silhouette_bbox(camera, x, y, yaw)
                    if not (0 <= b[0] and b[2] < IMG_W and 0 <= b[1] and b[3] < IMG_H):
                        continue
                    grid.append((cam, x, y, yaw, b))
    return grid


def score(models, items, alpha, z_plane):
    """Radial/lateral error of the (alpha, z_plane) estimator on a list of bbox items."""
    rad, lat = [], []
    for cam, x, y, _yaw, (u0, v0, u1, v1) in items:
        camera = models[cam]
        uc = 0.5 * (u0 + u1)
        vv = v1 + alpha * (v0 - v1)
        p = camera.pixel_to_world_at_z(uc, vv, z_plane)
        if p is None:
            continue
        _, ux, uy = bearing_frame(camera, x, y)
        ex, ey = p[0] - x, p[1] - y
        rad.append(ex * ux + ey * uy)
        lat.append(-ex * uy + ey * ux)
    rad, lat = np.asarray(rad), np.asarray(lat)
    return rad, lat


def main() -> int:
    models = {c: camera_model_from_world(WORLD, include_name=i)
              for c, i in MODEL_INCLUDES.items()}
    summary: dict[str, object] = {}

    print("object model, from the URDF (base_footprint frame):")
    print(f"  body   0.140 x 0.140 x 0.143 centred at x = {RSM.BODY_OFFSET_X:+.3f} m, "
          f"z = {RSM.BODY_CENTRE_Z:.3f} m")
    print(f"  wheels r = {RSM.WHEEL_RADIUS:.3f} m at y = +-{RSM.WHEEL_Y:.3f} m, "
          f"outer face +-{RSM.HALF_TRACK:.3f} m, top z = "
          f"{RSM.WHEEL_CENTRE_Z + RSM.WHEEL_RADIUS:.3f} m")
    print(f"  LiDAR  r = {RSM.LIDAR_RADIUS:.3f} m at x = {RSM.LIDAR_OFFSET_X:+.3f} m, "
          f"z = {RSM.LIDAR_CENTRE_Z:.3f} m, overall height {RSM.OVERALL_HEIGHT:.4f} m")
    print(f"  -> the body's {abs(RSM.BODY_OFFSET_X)*1000:.0f} mm x offset rotates with the "
          f"unobserved yaw.\n")

    rows = load_labels()
    print(f"{len(rows)} real silhouettes loaded\n")

    # --- A. does the CAD model predict the real silhouette? --------------------
    print("=== A. CAD silhouette model vs real semantic-mask labels (per edge, px) ===")
    print(f"  {'camera':9} {'n':>5} {'d u_left':>9} {'d u_right':>10} {'d v_top':>9} "
          f"{'d v_bot':>9} {'d width':>9} {'d height':>9}")
    block = {}
    for cam in sorted(models):
        g = [r for r in rows if r["camera"] == cam]
        e = {k: [] for k in ("ul", "ur", "vt", "vb", "w", "h")}
        for r in g:
            m0, m1, m2, m3 = RSM.silhouette_bbox(models[cam], r["x"], r["y"], r["yaw"])
            e["ul"].append(r["u0"] - m0); e["ur"].append(r["u1"] - m2)
            e["vt"].append(r["v0"] - m1); e["vb"].append(r["v1"] - m3)
            e["w"].append((r["u1"] - r["u0"]) - (m2 - m0))
            e["h"].append((r["v1"] - r["v0"]) - (m3 - m1))
        block[cam] = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v)))
                      for k, v in e.items()}
        print(f"  {cam:9} {len(g):5d} " + " ".join(
            f"{np.mean(e[k]):+9.2f}" for k in ("ul", "ur", "vt", "vb", "w", "h")))
    allh = [x for cam in block for x in [block[cam]["h"]["mean"]]]
    print(f"  height residual across cameras: {min(allh):+.2f} .. {max(allh):+.2f} px")
    print("  (the label polygon is approxPolyDP-simplified at epsilon = 1% of perimeter,")
    print("   which shrinks every edge slightly -- expect a small negative width/height)")
    summary["cad_forward_edges_px"] = block

    # --- B. design-time choice of (alpha, z*) ---------------------------------
    print("\n=== B. DESIGN TIME: choose (alpha, z*) from CAD + calibration alone ===")
    grid = design_grid(models)
    print(f"  grid: {len(grid)} (camera, position, yaw) combinations, no data involved")
    best = None
    coarse = {}
    for alpha in np.linspace(0.0, 1.0, 21):
        for z in np.linspace(0.0, 0.20, 41):
            rad, lat = score(models, grid, float(alpha), float(z))
            if rad.size == 0:
                continue
            cost = math.sqrt(float(np.mean(rad**2) + np.mean(lat**2)))
            coarse[f"{alpha:.2f}/{z:.3f}"] = cost
            if best is None or cost < best[0]:
                best = (cost, float(alpha), float(z))
    cost, alpha_star, z_star = best
    print(f"  optimum: alpha = {alpha_star:.2f}  z* = {z_star:.3f} m   "
          f"yaw-marginal RMS = {cost*1000:.1f} mm")
    rad, lat = score(models, grid, alpha_star, z_star)
    print(f"  at the optimum: radial bias {rad.mean()*1000:+.1f} mm, sd "
          f"{rad.std()*1000:.1f} mm;  lateral bias {lat.mean()*1000:+.1f} mm, sd "
          f"{lat.std()*1000:.1f} mm")
    print("  named reference points on the same grid:")
    for name, (a, z) in {
        "bottom edge @ floor": (0.0, 0.0),
        "bottom edge @ 0.05 (deployed)": (0.0, 0.05),
        "box centre @ h/2": (0.5, RSM.OVERALL_HEIGHT / 2),
        "CAD optimum": (alpha_star, z_star),
    }.items():
        r, l = score(models, grid, a, z)
        print(f"    {name:32} alpha={a:.2f} z={z:.3f}  RMS "
              f"{math.sqrt(float(np.mean(r**2)+np.mean(l**2)))*1000:6.1f} mm  "
              f"radial bias {r.mean()*1000:+7.1f} mm  radial sd {r.std()*1000:5.1f} mm")
    summary["design_time"] = dict(alpha=alpha_star, z_star_m=z_star,
                                  yaw_marginal_rms_m=cost,
                                  radial_bias_m=float(rad.mean()),
                                  radial_sd_m=float(rad.std()),
                                  lateral_sd_m=float(lat.std()),
                                  n_grid=len(grid))

    # --- C. validate on real silhouettes -------------------------------------
    print("\n=== C. VALIDATION on 1849 real silhouettes (choice above is frozen) ===")

    def real_score(sub, alpha, z):
        rad, lat, norm = [], [], []
        for r in sub:
            camera = models[r["camera"]]
            uc = 0.5 * (r["u0"] + r["u1"])
            vv = r["v1"] + alpha * (r["v0"] - r["v1"])
            p = camera.pixel_to_world_at_z(uc, vv, z)
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
        "box centre @ h/2": (0.5, RSM.OVERALL_HEIGHT / 2),
        "CAD optimum": (alpha_star, z_star),
    }
    print(f"  {'arm':26} {'mean':>8} {'median':>8} {'p95':>8} {'rad bias':>10} "
          f"{'rad sd':>8} {'lat sd':>8}")
    val = {}
    for name, (a, z) in arms.items():
        rad, lat, norm = real_score(rows, a, z)
        val[name] = dict(mean_m=float(norm.mean()), median_m=float(np.median(norm)),
                         p95_m=float(np.percentile(norm, 95)),
                         radial_bias_m=float(rad.mean()), radial_sd_m=float(rad.std()),
                         lateral_sd_m=float(lat.std()))
        print(f"  {name:26} {norm.mean()*1000:6.1f}mm {np.median(norm)*1000:6.1f}mm "
              f"{np.percentile(norm,95)*1000:6.1f}mm {rad.mean()*1000:+8.1f}mm "
              f"{rad.std()*1000:6.1f}mm {lat.std()*1000:6.1f}mm")

    print("\n  CAD optimum, stratified:")
    print(f"  {'stratum':>12} {'n':>5} {'mean':>8} {'rad bias':>10} {'rad sd':>8}")
    strat = {}
    for lo, hi in RANGE_BINS:
        sub = [r for r in rows if lo <= r["rng"] < hi]
        if len(sub) < 5:
            continue
        rad, lat, norm = real_score(sub, alpha_star, z_star)
        strat[f"{lo}-{hi}m"] = dict(n=len(sub), mean_m=float(norm.mean()),
                                    radial_bias_m=float(rad.mean()),
                                    radial_sd_m=float(rad.std()))
        print(f"  {f'{lo}-{hi} m':>12} {len(sub):5d} {norm.mean()*1000:6.1f}mm "
              f"{rad.mean()*1000:+8.1f}mm {rad.std()*1000:6.1f}mm")
    for cam in sorted(models):
        sub = [r for r in rows if r["camera"] == cam]
        rad, lat, norm = real_score(sub, alpha_star, z_star)
        strat[cam] = dict(n=len(sub), mean_m=float(norm.mean()),
                          radial_bias_m=float(rad.mean()), radial_sd_m=float(rad.std()))
        print(f"  {cam:>12} {len(sub):5d} {norm.mean()*1000:6.1f}mm "
              f"{rad.mean()*1000:+8.1f}mm {rad.std()*1000:6.1f}mm")

    print("\n  THE YAW TEST -- radial bias per yaw, CAD optimum vs the two edge arms:")
    print(f"  {'yaw':>6} {'n':>5} {'bottom@floor':>14} {'centre@h/2':>12} "
          f"{'CAD optimum':>13}")
    yaw_block = {}
    for yaw in sorted({round(math.degrees(r["yaw"])) for r in rows}):
        sub = [r for r in rows if round(math.degrees(r["yaw"])) == yaw]
        vals = {}
        for name, (a, z) in (("bottom", (0.0, 0.0)),
                             ("centre", (0.5, RSM.OVERALL_HEIGHT / 2)),
                             ("cad", (alpha_star, z_star))):
            rad, _, _ = real_score(sub, a, z)
            vals[name] = float(rad.mean())
        yaw_block[str(yaw)] = vals
        print(f"  {yaw:6d} {len(sub):5d} {vals['bottom']*1000:+12.1f}mm "
              f"{vals['centre']*1000:+10.1f}mm {vals['cad']*1000:+11.1f}mm")
    spreads = {k: max(v[k] for v in yaw_block.values()) - min(v[k] for v in yaw_block.values())
               for k in ("bottom", "centre", "cad")}
    print(f"  spread across yaw:  bottom {spreads['bottom']*1000:.1f} mm   "
          f"centre {spreads['centre']*1000:.1f} mm   CAD optimum "
          f"{spreads['cad']*1000:.1f} mm")
    print("  -> whatever survives here is IRREDUCIBLE: it varies with a state the runtime")
    print("     never observes, so it belongs in the covariance, not in a correction.")
    summary["validation"] = dict(arms=val, strata=strat, by_yaw=yaw_block,
                                 yaw_spread_m=spreads)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                      encoding="utf-8")
    print(f"\nwrote {(OUT / 'summary.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
