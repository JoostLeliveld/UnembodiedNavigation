#!/usr/bin/env python3
"""e5 — what does conditioning the observation model on heading buy?

e3/e4 established that the dominant error term is Sigma_yaw ~ 30 mm radial, and that it
exists for exactly one reason: the estimator is yaw-blind, while the robot's visual body is
offset -0.032 m in x from its pose origin, so the silhouette centre is not a fixed point of
the robot.

But the belief filter is not yaw-blind -- it carries a heading estimate.  A measurement
model h(x, y, theta) that depends on heading is ordinary for a nonlinear filter.  This
script measures the headroom, and how much heading accuracy is needed to collect it, by
solving for (x, y) with the yaw-aware mesh model at a supplied heading.

Run:  python3 experiments/pixel_ground_path/e5_yaw_aware_headroom.py
Out:  logs/studies/pixel_ground_path/e5_yaw_aware_headroom/
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
OUT = REPO / "logs/studies/pixel_ground_path/e5_yaw_aware_headroom"

MODEL_INCLUDES = {
    "camera_A": "external_camera", "camera_B": "external_camera_b",
    "camera_C": "external_camera_c", "camera_D": "external_camera_d",
}
ALPHA, Z_STAR = 0.50, 0.085
RANGE_BINS = ((0, 5), (5, 8), (8, 12), (12, 16))


def bearing_frame(camera, x, y):
    cx, cy = float(camera.cam_pos[0]), float(camera.cam_pos[1])
    bx, by = x - cx, y - cy
    d = math.hypot(bx, by)
    return d, bx / d, by / d


def load_rows():
    det = {r["sample_id"]: r for r in csv.DictReader(DET_CACHE.open(encoding="utf-8"))}
    rows = []
    with (DATASET / "localization_calibration_index.csv").open(newline="", encoding="utf-8") as h:
        for rec in csv.DictReader(h):
            d = det.get(rec["sample_id"])
            if not d or str(d["detected"]) != "1" or d["pu0"] == "":
                continue
            rows.append(dict(camera=rec["camera_id"], x=float(rec["robot_x"]),
                             y=float(rec["robot_y"]), yaw=float(rec["robot_yaw"]),
                             rng=float(rec["camera_range_m"]),
                             pu0=float(d["pu0"]), pv0=float(d["pv0"]),
                             pu1=float(d["pu1"]), pv1=float(d["pv1"])))
    return rows


def observed(r):
    return (0.5 * (r["pu0"] + r["pu1"]), r["pv1"] + ALPHA * (r["pv0"] - r["pv1"]))


def solve_yaw_aware(camera, r, yaw, *, iters=6, tol=1.0e-3, h=0.01):
    """Gauss-Newton on (x, y) so the modelled box statistic matches the observed one.

    Seeded with the yaw-blind plane inversion, which is already within a few cm, so this
    converges in two or three steps.
    """
    uo, vo = observed(r)
    seed = camera.pixel_to_world_at_z(uo, vo, Z_STAR)
    if seed is None:
        return None
    x, y = seed
    for _ in range(iters):
        b = RSM.mesh_silhouette_bbox(camera, x, y, yaw)
        if b is None:
            return None
        um, vm = 0.5 * (b[0] + b[2]), b[3] + ALPHA * (b[1] - b[3])
        res = np.array([uo - um, vo - vm])
        if np.abs(res).max() < tol:
            break
        J = np.zeros((2, 2))
        for k, (dx, dy) in enumerate(((h, 0.0), (0.0, h))):
            bp = RSM.mesh_silhouette_bbox(camera, x + dx, y + dy, yaw)
            if bp is None:
                return None
            J[0, k] = (0.5 * (bp[0] + bp[2]) - um) / h
            J[1, k] = ((bp[3] + ALPHA * (bp[1] - bp[3])) - vm) / h
        try:
            step = np.linalg.solve(J, res)
        except np.linalg.LinAlgError:
            return None
        x += float(step[0])
        y += float(step[1])
    return x, y


def main() -> int:
    models = {c: camera_model_from_world(WORLD, include_name=i)
              for c, i in MODEL_INCLUDES.items()}
    rows = load_rows()
    print(f"{len(rows)} detections\n")
    summary: dict[str, object] = dict(n=len(rows), alpha=ALPHA, z_star_m=Z_STAR)

    arms = [("yaw-BLIND (plane inversion)", None)]
    arms += [(f"yaw-AWARE, heading error {d:+3d} deg", math.radians(d))
             for d in (0, 5, 10, 20, 30, 45, 90)]

    print("=== headroom from conditioning on heading ===")
    print(f"  {'arm':34} {'mean':>8} {'median':>8} {'p95':>8} {'rad bias':>10} "
          f"{'rad sd':>8} {'lat sd':>8}")
    block = {}
    for name, offset in arms:
        rad, lat, norm = [], [], []
        for r in rows:
            camera = models[r["camera"]]
            if offset is None:
                uo, vo = observed(r)
                p = camera.pixel_to_world_at_z(uo, vo, Z_STAR)
            else:
                p = solve_yaw_aware(camera, r, r["yaw"] + offset)
            if p is None:
                continue
            _, ux, uy = bearing_frame(camera, r["x"], r["y"])
            ex, ey = p[0] - r["x"], p[1] - r["y"]
            rad.append(ex * ux + ey * uy)
            lat.append(-ex * uy + ey * ux)
            norm.append(math.hypot(ex, ey))
        rad, lat, norm = np.asarray(rad), np.asarray(lat), np.asarray(norm)
        block[name] = dict(n=int(norm.size), mean_m=float(norm.mean()),
                           median_m=float(np.median(norm)),
                           p95_m=float(np.percentile(norm, 95)),
                           radial_bias_m=float(rad.mean()), radial_sd_m=float(rad.std()),
                           lateral_sd_m=float(lat.std()))
        print(f"  {name:34} {norm.mean()*1000:6.1f}mm {np.median(norm)*1000:6.1f}mm "
              f"{np.percentile(norm,95)*1000:6.1f}mm {rad.mean()*1000:+8.1f}mm "
              f"{rad.std()*1000:6.1f}mm {lat.std()*1000:6.1f}mm")
    summary["arms"] = block

    blind = block["yaw-BLIND (plane inversion)"]["mean_m"]
    for name, v in block.items():
        if name.startswith("yaw-AWARE"):
            if v["mean_m"] > blind:
                print(f"  -> {name} is WORSE than yaw-blind: heading that bad is not usable")
                break
    print(f"\n  break-even heading accuracy: the yaw-aware arm beats yaw-blind "
          f"({blind*1000:.1f} mm) up to")
    worst_ok = [n for n, v in block.items()
                if n.startswith("yaw-AWARE") and v["mean_m"] < blind]
    print(f"  {worst_ok[-1] if worst_ok else 'no arm'}")

    print("\n=== yaw-aware (true heading), stratified ===")
    print(f"  {'stratum':>12} {'n':>5} {'mean':>8} {'rad sd':>8} {'lat sd':>8}")
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
        rad, lat, norm = [], [], []
        for r in sub:
            camera = models[r["camera"]]
            p = solve_yaw_aware(camera, r, r["yaw"])
            if p is None:
                continue
            _, ux, uy = bearing_frame(camera, r["x"], r["y"])
            ex, ey = p[0] - r["x"], p[1] - r["y"]
            rad.append(ex * ux + ey * uy)
            lat.append(-ex * uy + ey * ux)
            norm.append(math.hypot(ex, ey))
        rad, lat, norm = np.asarray(rad), np.asarray(lat), np.asarray(norm)
        strat[name] = dict(n=int(norm.size), mean_m=float(norm.mean()),
                           radial_sd_m=float(rad.std()), lateral_sd_m=float(lat.std()))
        print(f"  {name:>12} {norm.size:5d} {norm.mean()*1000:6.1f}mm "
              f"{rad.std()*1000:6.1f}mm {lat.std()*1000:6.1f}mm")
    summary["yaw_aware_strata"] = strat

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                      encoding="utf-8")
    print(f"\nwrote {(OUT / 'summary.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
