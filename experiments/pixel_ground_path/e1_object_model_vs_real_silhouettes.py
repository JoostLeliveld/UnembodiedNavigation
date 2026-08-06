#!/usr/bin/env python3
"""e1 — does the object model predict the REAL silhouette, and which statistic inverts?

e0 was forward geometry against a modelled silhouette.  e1 replaces the modelled
silhouette with the **Gazebo semantic-mask label** of every sample in the frozen 4-camera
detector dataset: 1849 samples, 372 distinct positions, 4 yaws, 4 cameras, 1.7-16.6 m.
The pose is the COMMANDED set_pose, so there is no timing join and no odometry anywhere --
this is the cleanest evidence in the repo for a projection question.

Still no detector: the label is what a perfect detector would emit.  e2 adds the detector.

Diagnostic first: the capture script spawns the robot at ``--robot-z 0.05``, so the
effective base height of the silhouette is unknown (the robot may have settled to the
floor under gravity).  Section 0 MEASURES it from the labels instead of assuming, because
the whole point of the design is that no plane height is a free choice.

Run:  python3 experiments/pixel_ground_path/e1_object_model_vs_real_silhouettes.py
Out:  logs/studies/pixel_ground_path/e1_object_model_vs_real_silhouettes/
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
sys.path.insert(0, str(REPO / "src" / "reliability"))
sys.path.insert(0, str(REPO / "src" / "unav_common"))

from reliability.projection import camera_model_from_world  # noqa: E402

WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
# Dataset payload lives in cold storage; see logs/perception_datasets/COLD_STORAGE.md.
# Read-only -- nothing is copied back into the workspace.
from dataset_paths import dataset_root  # noqa: E402

# Resolved, not hard-coded: the payload moved to cold storage on 2026-08-05.
DATASET = dataset_root(REPO)
OUT = REPO / "logs/studies/pixel_ground_path/e1_object_model_vs_real_silhouettes"

MODEL_INCLUDES = {
    "camera_A": "external_camera",
    "camera_B": "external_camera_b",
    "camera_C": "external_camera_c",
    "camera_D": "external_camera_d",
}
IMG_W, IMG_H = 1280, 720

HALF_BODY_X, HALF_TRACK, HEIGHT = 0.070, 0.089, 0.192
N_RIM = 64
_TH = np.linspace(0.0, 2.0 * math.pi, N_RIM, endpoint=False)
_COS, _SIN = np.cos(_TH), np.sin(_TH)

RANGE_BINS = ((0, 5), (5, 8), (8, 12), (12, 16), (16, 20))


def cylinder_bbox(camera, x, y, *, radius=HALF_TRACK, height=HEIGHT, z_base=0.0):
    us, vs = [], []
    for z in (z_base, z_base + height):
        for cx, sy in zip(_COS, _SIN):
            u, v, _ = camera.world_to_pixel(x + radius * cx, y + radius * sy, z)
            us.append(u)
            vs.append(v)
    return min(us), min(vs), max(us), max(vs)


def cuboid_bbox(camera, x, y, yaw, *, height=HEIGHT, z_base=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    us, vs = [], []
    for lx in (-HALF_BODY_X, HALF_BODY_X):
        for ly in (-HALF_TRACK, HALF_TRACK):
            for lz in (z_base, z_base + height):
                u, v, _ = camera.world_to_pixel(x + c * lx - s * ly, y + s * lx + c * ly, lz)
                us.append(u)
                vs.append(v)
    return min(us), min(vs), max(us), max(vs)


def bearing_frame(camera, x, y):
    cx, cy = float(camera.cam_pos[0]), float(camera.cam_pos[1])
    bx, by = x - cx, y - cy
    d = math.hypot(bx, by)
    return d, bx / d, by / d


def load_samples():
    """Every eligible sample with its REAL silhouette bbox from the semantic-mask label."""
    index = DATASET / "localization_calibration_index.csv"
    rows = []
    missing = 0
    with index.open(newline="", encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            label = DATASET / rec["label"]
            if not label.exists():
                missing += 1
                continue
            best = None
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 7:
                    continue
                xy = np.asarray([float(p) for p in parts[1:]], dtype=float).reshape(-1, 2)
                xy[:, 0] *= IMG_W
                xy[:, 1] *= IMG_H
                area = 0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1))
                                 - np.dot(xy[:, 1], np.roll(xy[:, 0], 1)))
                if best is None or area > best[0]:
                    best = (area, xy)
            if best is None:
                missing += 1
                continue
            xy = best[1]
            rows.append(
                dict(
                    sample_id=rec["sample_id"], camera=rec["camera_id"], split=rec["split"],
                    x=float(rec["robot_x"]), y=float(rec["robot_y"]),
                    yaw=float(rec["robot_yaw"]), rng=float(rec["camera_range_m"]),
                    u0=float(xy[:, 0].min()), u1=float(xy[:, 0].max()),
                    v0=float(xy[:, 1].min()), v1=float(xy[:, 1].max()),
                    poly_n=int(len(xy)),
                )
            )
    return rows, missing


def invert(camera, u, v, z_plane):
    return camera.pixel_to_world_at_z(u, v, z_plane)


def section0_base_height(models, rows, summary):
    """MEASURE the effective base height of the silhouette rather than assume it."""
    print("=== 0. diagnostic: what base height do the real silhouettes imply? ===")
    print("  (the capture spawned the robot at --robot-z 0.05; gravity may have settled it)")
    print(f"  {'z_base':>8} {'mean |dv_bottom|':>17} {'mean |dv_top|':>14} {'mean |dh|':>10}")
    table = {}
    for z_base in (0.0, 0.01, 0.02, 0.03, 0.05):
        db, dt, dh = [], [], []
        for rec in rows:
            camera = models[rec["camera"]]
            m0, m1, m2, m3 = cylinder_bbox(camera, rec["x"], rec["y"], z_base=z_base)
            db.append(rec["v1"] - m3)
            dt.append(rec["v0"] - m1)
            dh.append((rec["v1"] - rec["v0"]) - (m3 - m1))
        table[f"{z_base:.2f}"] = dict(bottom_px=float(np.mean(np.abs(db))),
                                     top_px=float(np.mean(np.abs(dt))),
                                     height_px=float(np.mean(np.abs(dh))))
        print(f"  {z_base:8.2f} {np.mean(np.abs(db)):15.2f}px {np.mean(np.abs(dt)):12.2f}px "
              f"{np.mean(np.abs(dh)):8.2f}px")
    best = min(table, key=lambda k: table[k]["bottom_px"] + table[k]["top_px"])
    print(f"  -> the labels are most consistent with z_base = {best} m")
    summary["base_height_diagnostic"] = dict(table=table, best_z_base_m=float(best))
    return float(best)


def section1_forward(models, rows, z_base, summary):
    print("\n=== 1. does the object model predict the real silhouette?  (per-edge, px) ===")
    print(f"  {'camera':9} {'n':>5} {'d u_left':>9} {'d u_right':>10} {'d v_top':>8} "
          f"{'d v_bottom':>11} {'d width':>8} {'d height':>9}")
    block = {}
    for cam in sorted(models):
        group = [r for r in rows if r["camera"] == cam]
        camera = models[cam]
        e = {k: [] for k in ("ul", "ur", "vt", "vb", "w", "h")}
        for rec in group:
            m0, m1, m2, m3 = cylinder_bbox(camera, rec["x"], rec["y"], z_base=z_base)
            e["ul"].append(rec["u0"] - m0)
            e["ur"].append(rec["u1"] - m2)
            e["vt"].append(rec["v0"] - m1)
            e["vb"].append(rec["v1"] - m3)
            e["w"].append((rec["u1"] - rec["u0"]) - (m2 - m0))
            e["h"].append((rec["v1"] - rec["v0"]) - (m3 - m1))
        block[cam] = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v)))
                      for k, v in e.items()}
        print(f"  {cam:9} {len(group):5d} " + " ".join(
            f"{np.mean(e[k]):+8.2f}" for k in ("ul", "ur", "vt", "vb", "w", "h")))
    print("  positive = the real silhouette extends further than the model predicts")
    print("  (the label polygon is approxPolyDP-simplified, which shrinks it slightly)")
    summary["forward_model_edges_px"] = block


def section2_inversion(models, rows, z_base, summary):
    print("\n=== 2. inverting the REAL silhouette: which statistic recovers the pose? ===")

    def errs(rows_sub, stat, z_plane):
        out = []
        for rec in rows_sub:
            camera = models[rec["camera"]]
            uc = 0.5 * (rec["u0"] + rec["u1"])
            vv = rec["v1"] if stat == "bottom" else 0.5 * (rec["v0"] + rec["v1"])
            p = invert(camera, uc, vv, z_plane)
            if p is None:
                continue
            d, ux, uy = bearing_frame(camera, rec["x"], rec["y"])
            ex, ey = p[0] - rec["x"], p[1] - rec["y"]
            out.append((math.hypot(ex, ey), ex * ux + ey * uy, -ex * uy + ey * ux))
        return np.asarray(out)

    arms = {
        "bottom @ z=0.05 (DEPLOYED)": ("bottom", 0.05),
        "bottom @ floor": ("bottom", 0.0),
        "centre @ z=h/2 (PROPOSED)": ("centre", z_base + HEIGHT / 2.0),
    }
    print(f"  {'arm':28} {'mean':>8} {'median':>8} {'p95':>8} {'max':>8} {'radial bias':>12}")
    block = {}
    for name, (stat, z) in arms.items():
        a = errs(rows, stat, z)
        block[name] = dict(mean_m=float(a[:, 0].mean()), median_m=float(np.median(a[:, 0])),
                           p95_m=float(np.percentile(a[:, 0], 95)), max_m=float(a[:, 0].max()),
                           radial_bias_m=float(a[:, 1].mean()),
                           radial_sd_m=float(a[:, 1].std()),
                           lateral_bias_m=float(a[:, 2].mean()),
                           lateral_sd_m=float(a[:, 2].std()))
        print(f"  {name:28} {a[:,0].mean()*1000:6.1f}mm {np.median(a[:,0])*1000:6.1f}mm "
              f"{np.percentile(a[:,0],95)*1000:6.1f}mm {a[:,0].max()*1000:6.1f}mm "
              f"{a[:,1].mean()*1000:+10.1f}mm")

    print("\n  stratified by range bin (mean position error):")
    print(f"  {'range':>10} {'n':>5} {'bottom @ floor':>16} {'centre @ h/2':>15}")
    strat = {}
    for lo, hi in RANGE_BINS:
        sub = [r for r in rows if lo <= r["rng"] < hi]
        if len(sub) < 5:
            continue
        b = errs(sub, "bottom", 0.0)
        c = errs(sub, "centre", z_base + HEIGHT / 2.0)
        strat[f"{lo}-{hi}"] = dict(n=len(sub), bottom_m=float(b[:, 0].mean()),
                                   centre_m=float(c[:, 0].mean()))
        print(f"  {f'{lo}-{hi} m':>10} {len(sub):5d} {b[:,0].mean()*1000:14.1f}mm "
              f"{c[:,0].mean()*1000:13.1f}mm")

    print("\n  stratified by camera (mean position error):")
    print(f"  {'camera':>10} {'n':>5} {'bottom @ floor':>16} {'centre @ h/2':>15}")
    per_cam = {}
    for cam in sorted(models):
        sub = [r for r in rows if r["camera"] == cam]
        b = errs(sub, "bottom", 0.0)
        c = errs(sub, "centre", z_base + HEIGHT / 2.0)
        per_cam[cam] = dict(n=len(sub), bottom_m=float(b[:, 0].mean()),
                            centre_m=float(c[:, 0].mean()))
        print(f"  {cam:>10} {len(sub):5d} {b[:,0].mean()*1000:14.1f}mm "
              f"{c[:,0].mean()*1000:13.1f}mm")

    print("\n  THE YAW TEST -- radial bias per yaw (the effect e0 predicted):")
    print(f"  {'yaw (deg)':>10} {'n':>5} {'bottom @ floor':>16} {'centre @ h/2':>15}")
    per_yaw = {}
    for yaw in sorted({round(math.degrees(r["yaw"])) for r in rows}):
        sub = [r for r in rows if round(math.degrees(r["yaw"])) == yaw]
        b = errs(sub, "bottom", 0.0)
        c = errs(sub, "centre", z_base + HEIGHT / 2.0)
        per_yaw[str(yaw)] = dict(n=len(sub), bottom_radial_m=float(b[:, 1].mean()),
                                 centre_radial_m=float(c[:, 1].mean()))
        print(f"  {yaw:10d} {len(sub):5d} {b[:,1].mean()*1000:+14.1f}mm "
              f"{c[:,1].mean()*1000:+13.1f}mm")
    spread_b = max(v["bottom_radial_m"] for v in per_yaw.values()) - \
        min(v["bottom_radial_m"] for v in per_yaw.values())
    spread_c = max(v["centre_radial_m"] for v in per_yaw.values()) - \
        min(v["centre_radial_m"] for v in per_yaw.values())
    print(f"  -> radial bias SPREAD across yaw: bottom {spread_b*1000:.1f} mm, "
          f"centre {spread_c*1000:.1f} mm  ({spread_b/max(spread_c,1e-9):.1f}x)")
    print("     This is the quantity a per-camera constant cannot represent, because it")
    print("     changes with a state nobody observes.")
    summary["inversion"] = dict(arms=block, by_range=strat, by_camera=per_cam,
                                by_yaw=per_yaw,
                                yaw_spread_bottom_m=float(spread_b),
                                yaw_spread_centre_m=float(spread_c))


def main() -> int:
    if not DATASET.exists():
        print(f"dataset payload not found at {DATASET}", file=sys.stderr)
        return 1
    models = {c: camera_model_from_world(WORLD, include_name=i)
              for c, i in MODEL_INCLUDES.items()}
    rows, missing = load_samples()
    print(f"{len(rows)} samples with real semantic-mask silhouettes "
          f"({missing} unusable), 4 cameras")
    print(f"ranges {min(r['rng'] for r in rows):.1f}-{max(r['rng'] for r in rows):.1f} m, "
          f"yaws {sorted({round(math.degrees(r['yaw'])) for r in rows})} deg, "
          f"{len({(r['x'], r['y']) for r in rows})} distinct positions\n")

    summary: dict[str, object] = dict(n_samples=len(rows), n_unusable=missing,
                                      dataset=str(DATASET))
    z_base = section0_base_height(models, rows, summary)
    section1_forward(models, rows, z_base, summary)
    section2_inversion(models, rows, z_base, summary)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                      encoding="utf-8")
    print(f"\nwrote {(OUT / 'summary.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
