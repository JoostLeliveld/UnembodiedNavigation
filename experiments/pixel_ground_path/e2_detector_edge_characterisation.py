#!/usr/bin/env python3
"""e2 — characterise the DETECTOR's box edges, with no robot ground truth.

This is the one per-camera step the design allows, and it is deliberately a *detector
evaluation*: run the frozen detector over its own labelled split and compare its predicted
box to the label box.  Labels only -- **robot poses are not read in sections 1-2 at all**,
which is what makes this reproducible in a real warehouse (any deployment labels a
validation set to accept a detector).

Section 3 then closes the loop that e1b left open.  e1b chose the vertical statistic
``v = v_bottom + alpha * (v_top - v_bottom)`` from CAD geometry alone and got alpha = 0.5,
but the CAD model tracks the bottom edge to +-0.7 px and the top edge only to +-1.6 px.
alpha is really a bias/variance trade-off: yaw-marginal geometry pulls it to 0.5, unequal
edge noise pulls it to 0.  With the detector's measured per-edge noise in hand, alpha
follows from CAD + a GT-free measurement, still with no robot pose in the estimator.

Section 4 validates on poses, and section 5 reports the resulting per-detection covariance.

Run:  python3 experiments/pixel_ground_path/e2_detector_edge_characterisation.py
Out:  logs/studies/pixel_ground_path/e2_detector_edge_characterisation/
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
WEIGHTS = REPO / "logs/perception_models/warehouse_yolo_detector_4cam_v3_960/model.pt"
OUT = REPO / "logs/studies/pixel_ground_path/e2_detector_edge_characterisation"
CACHE = OUT / "detector_boxes.csv"

MODEL_INCLUDES = {
    "camera_A": "external_camera",
    "camera_B": "external_camera_b",
    "camera_C": "external_camera_c",
    "camera_D": "external_camera_d",
}
IMG_W, IMG_H = 1280, 720
IMGSZ = 960          # matches ultralytics_run/args.yaml
CONF = 0.05          # deliberately low: we are characterising geometry, not screening
SITE = (-11.20, 11.50, -8.60, 8.60)
RANGE_BINS = ((0, 5), (5, 8), (8, 12), (12, 16), (16, 20))


def bearing_frame(camera, x, y):
    cx, cy = float(camera.cam_pos[0]), float(camera.cam_pos[1])
    bx, by = x - cx, y - cy
    d = math.hypot(bx, by)
    return d, bx / d, by / d


def load_index():
    rows = []
    with (DATASET / "localization_calibration_index.csv").open(newline="", encoding="utf-8") as h:
        for rec in csv.DictReader(h):
            label = DATASET / rec["label"]
            image = DATASET / rec["image"]
            if not (label.exists() and image.exists()):
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
            rows.append(dict(sample_id=rec["sample_id"], camera=rec["camera_id"],
                             split=rec["split"], image=str(image),
                             x=float(rec["robot_x"]), y=float(rec["robot_y"]),
                             yaw=float(rec["robot_yaw"]), rng=float(rec["camera_range_m"]),
                             lu0=float(xy[:, 0].min()), lu1=float(xy[:, 0].max()),
                             lv0=float(xy[:, 1].min()), lv1=float(xy[:, 1].max())))
    return rows


def run_detector(rows):
    """Predicted box per sample.  Cached, because this is the only GPU step."""
    if CACHE.exists():
        cached = {}
        with CACHE.open(newline="", encoding="utf-8") as h:
            for rec in csv.DictReader(h):
                cached[rec["sample_id"]] = rec
        if len(cached) >= len(rows):
            print(f"  using cached detector output ({len(cached)} rows) at "
                  f"{CACHE.relative_to(REPO)}")
            return cached
    from ultralytics import YOLO

    print(f"  running {WEIGHTS.name} at imgsz={IMGSZ}, conf={CONF} over {len(rows)} images")
    model = YOLO(str(WEIGHTS))
    out = {}
    batch = 16
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        results = model.predict([r["image"] for r in chunk], imgsz=IMGSZ, conf=CONF,
                                verbose=False, device=0)
        for rec, res in zip(chunk, results):
            boxes = getattr(res, "boxes", None)
            row = dict(sample_id=rec["sample_id"], detected=0, conf=0.0,
                       pu0="", pv0="", pu1="", pv1="")
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy()
                best = int(np.argmax(confs))
                row.update(detected=1, conf=float(confs[best]),
                           pu0=float(xyxy[best, 0]), pv0=float(xyxy[best, 1]),
                           pu1=float(xyxy[best, 2]), pv1=float(xyxy[best, 3]))
            out[rec["sample_id"]] = row
        if (start // batch) % 20 == 0:
            print(f"    {start + len(chunk)}/{len(rows)}")
    OUT.mkdir(parents=True, exist_ok=True)
    with CACHE.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["sample_id", "detected", "conf",
                                          "pu0", "pv0", "pu1", "pv1"])
        w.writeheader()
        for row in out.values():
            w.writerow(row)
    return out


def main() -> int:
    models = {c: camera_model_from_world(WORLD, include_name=i)
              for c, i in MODEL_INCLUDES.items()}
    rows = load_index()
    print(f"{len(rows)} samples with both an image and a label")
    preds = run_detector(rows)

    merged = []
    for r in rows:
        p = preds.get(r["sample_id"])
        if p is None or str(p["detected"]) != "1" or p["pu0"] == "":
            continue
        merged.append(r | dict(pu0=float(p["pu0"]), pv0=float(p["pv0"]),
                               pu1=float(p["pu1"]), pv1=float(p["pv1"]),
                               conf=float(p["conf"])))
    print(f"{len(merged)} detected ({len(merged)/len(rows)*100:.1f} %)\n")
    summary: dict[str, object] = dict(n_samples=len(rows), n_detected=len(merged),
                                      weights=WEIGHTS.name, imgsz=IMGSZ, conf=CONF)

    # ---- 1. detector vs label: per-edge, NO ROBOT POSE USED -----------------
    print("=== 1. detector box vs LABEL box (px).  No robot pose is read here. ===")
    edge_stats = {}
    for split in ("val", "train"):
        print(f"  --- {split} split ---")
        print(f"  {'camera':9} {'n':>5} {'d u_left':>13} {'d u_right':>13} "
              f"{'d v_top':>13} {'d v_bottom':>13}")
        for cam in sorted(models):
            g = [r for r in merged if r["camera"] == cam and r["split"] == split]
            if not g:
                continue
            e = dict(
                ul=np.array([r["pu0"] - r["lu0"] for r in g]),
                ur=np.array([r["pu1"] - r["lu1"] for r in g]),
                vt=np.array([r["pv0"] - r["lv0"] for r in g]),
                vb=np.array([r["pv1"] - r["lv1"] for r in g]),
            )
            edge_stats[f"{split}/{cam}"] = {k: dict(n=len(g), mean=float(v.mean()),
                                                    sd=float(v.std()))
                                            for k, v in e.items()}
            print(f"  {cam:9} {len(g):5d} " + " ".join(
                f"{e[k].mean():+6.2f}+-{e[k].std():5.2f}" for k in ("ul", "ur", "vt", "vb")))
    print("  positive = the detector's edge sits further right / further down than the label")
    summary["detector_vs_label_px"] = edge_stats

    print("\n  pooled over cameras, val split -- the numbers that set alpha and Sigma_uv:")
    gv = [r for r in merged if r["split"] == "val"]
    pooled = {}
    for k, (a, b) in {"vt": ("pv0", "lv0"), "vb": ("pv1", "lv1"),
                      "ul": ("pu0", "lu0"), "ur": ("pu1", "lu1")}.items():
        v = np.array([r[a] - r[b] for r in gv])
        pooled[k] = dict(mean=float(v.mean()), sd=float(v.std()))
        print(f"    {k}: {v.mean():+.2f} +- {v.std():.2f} px")
    uc = np.array([0.5 * (r["pu0"] + r["pu1"]) - 0.5 * (r["lu0"] + r["lu1"]) for r in gv])
    print(f"    u centre: {uc.mean():+.2f} +- {uc.std():.2f} px")
    pooled["u_centre"] = dict(mean=float(uc.mean()), sd=float(uc.std()))
    summary["pooled_val_px"] = pooled

    # ---- 2. alpha from CAD geometry + the measured edge noise ---------------
    print("\n=== 2. choose alpha: CAD yaw-marginal geometry + measured edge noise ===")
    print("  (still no robot pose: the geometry term is CAD, the noise term is label-only)")
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
                _, _, vis = camera.world_to_pixel(x, y, 0.0)
                if not vis:
                    continue
                for yaw in yaws:
                    b = RSM.silhouette_bbox(camera, x, y, yaw)
                    if not (0 <= b[0] and b[2] < IMG_W and 0 <= b[1] and b[3] < IMG_H):
                        continue
                    grid.append((cam, x, y, yaw, b))
    print(f"  CAD grid: {len(grid)} combinations")

    sd_vt, sd_vb = pooled["vt"]["sd"], pooled["vb"]["sd"]
    print(f"  detector edge noise: v_top +-{sd_vt:.2f} px, v_bottom +-{sd_vb:.2f} px")

    def total_cost(alpha, z):
        rad = []
        for cam, x, y, _yaw, (u0, v0, u1, v1) in grid:
            camera = models[cam]
            p = camera.pixel_to_world_at_z(0.5 * (u0 + u1), v1 + alpha * (v0 - v1), z)
            if p is None:
                continue
            _, ux, uy = bearing_frame(camera, x, y)
            rad.append((p[0] - x) * ux + (p[1] - y) * uy)
        rad = np.asarray(rad)
        # geometry term: yaw-marginal bias + spread, in metres
        geom = float(np.mean(rad**2))
        # noise term: sd of v = alpha*v_top + (1-alpha)*v_bottom, pushed through dRange/dv
        sd_v = math.hypot(alpha * sd_vt, (1.0 - alpha) * sd_vb)
        mpp = []
        for cam, x, y, _yaw, (u0, v0, u1, v1) in grid[::7]:
            camera = models[cam]
            _, ux, uy = bearing_frame(camera, x, y)
            vv = v1 + alpha * (v0 - v1)
            a = camera.pixel_to_world_at_z(0.5 * (u0 + u1), vv + 0.5, z)
            b = camera.pixel_to_world_at_z(0.5 * (u0 + u1), vv - 0.5, z)
            if a is None or b is None:
                continue
            mpp.append(abs((a[0] - b[0]) * ux + (a[1] - b[1]) * uy))
        noise = (sd_v * float(np.mean(mpp)))**2
        return math.sqrt(geom + noise), math.sqrt(geom), math.sqrt(noise)

    print(f"  {'alpha':>6} {'z*(m)':>7} {'total':>8} {'geometry':>9} {'noise':>8}")
    best = None
    curve = {}
    for alpha in np.linspace(0.0, 1.0, 11):
        inner = None
        for z in np.linspace(0.0, 0.20, 41):
            t, g, n = total_cost(float(alpha), float(z))
            if inner is None or t < inner[0]:
                inner = (t, float(z), g, n)
        t, z, g, n = inner
        curve[f"{alpha:.1f}"] = dict(z_m=z, total_m=t, geom_m=g, noise_m=n)
        mark = ""
        if best is None or t < best[0]:
            best = (t, float(alpha), z)
            mark = ""
        print(f"  {alpha:6.1f} {z:7.3f} {t*1000:6.1f}mm {g*1000:7.1f}mm {n*1000:6.1f}mm")
    tot, alpha_star, z_star = best
    print(f"  -> alpha* = {alpha_star:.1f}, z* = {z_star:.3f} m, predicted total "
          f"{tot*1000:.1f} mm")
    summary["alpha_selection"] = dict(curve=curve, alpha=alpha_star, z_star_m=z_star,
                                      predicted_total_m=tot,
                                      sd_vt_px=sd_vt, sd_vb_px=sd_vb)

    # ---- 3. validate the END-TO-END path on real detections ----------------
    print("\n=== 3. END TO END on real detections: detector box -> position ===")

    def score(sub, alpha, z):
        rad, lat, norm = [], [], []
        for r in sub:
            camera = models[r["camera"]]
            uc = 0.5 * (r["pu0"] + r["pu1"])
            vv = r["pv1"] + alpha * (r["pv0"] - r["pv1"])
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
        "e1b CAD optimum": (0.5, 0.090),
        f"e2 optimum (a={alpha_star:.1f})": (alpha_star, z_star),
    }
    for split in ("val", "train"):
        sub = [r for r in merged if r["split"] == split]
        print(f"  --- {split} ({len(sub)} detections) ---")
        print(f"  {'arm':28} {'mean':>8} {'median':>8} {'p95':>8} {'rad bias':>10} "
              f"{'rad sd':>8} {'lat sd':>8}")
        for name, (a, z) in arms.items():
            rad, lat, norm = score(sub, a, z)
            if norm.size == 0:
                continue
            summary.setdefault("end_to_end", {})[f"{split}/{name}"] = dict(
                n=int(norm.size), mean_m=float(norm.mean()),
                median_m=float(np.median(norm)), p95_m=float(np.percentile(norm, 95)),
                radial_bias_m=float(rad.mean()), radial_sd_m=float(rad.std()),
                lateral_bias_m=float(lat.mean()), lateral_sd_m=float(lat.std()))
            print(f"  {name:28} {norm.mean()*1000:6.1f}mm {np.median(norm)*1000:6.1f}mm "
                  f"{np.percentile(norm,95)*1000:6.1f}mm {rad.mean()*1000:+8.1f}mm "
                  f"{rad.std()*1000:6.1f}mm {lat.std()*1000:6.1f}mm")

    print("\n  chosen path, per camera (val + train, the per-camera uniformity test):")
    print(f"  {'camera':9} {'n':>5} {'mean':>8} {'rad bias':>10} {'rad sd':>8} "
          f"{'lat bias':>10} {'lat sd':>8}")
    per_cam = {}
    for cam in sorted(models):
        sub = [r for r in merged if r["camera"] == cam]
        rad, lat, norm = score(sub, alpha_star, z_star)
        per_cam[cam] = dict(n=int(norm.size), mean_m=float(norm.mean()),
                            radial_bias_m=float(rad.mean()), radial_sd_m=float(rad.std()),
                            lateral_bias_m=float(lat.mean()),
                            lateral_sd_m=float(lat.std()))
        print(f"  {cam:9} {norm.size:5d} {norm.mean()*1000:6.1f}mm "
              f"{rad.mean()*1000:+8.1f}mm {rad.std()*1000:6.1f}mm "
              f"{lat.mean()*1000:+8.1f}mm {lat.std()*1000:6.1f}mm")
    spread_r = max(v["radial_bias_m"] for v in per_cam.values()) - \
        min(v["radial_bias_m"] for v in per_cam.values())
    spread_l = max(v["lateral_bias_m"] for v in per_cam.values()) - \
        min(v["lateral_bias_m"] for v in per_cam.values())
    print(f"  per-camera bias SPREAD: radial {spread_r*1000:.1f} mm, "
          f"lateral {spread_l*1000:.1f} mm")
    print("  -> this is the number that decides whether any per-camera parameter is needed.")

    print("\n  chosen path, per range bin and per yaw:")
    for lo, hi in RANGE_BINS:
        sub = [r for r in merged if lo <= r["rng"] < hi]
        if len(sub) < 5:
            continue
        rad, lat, norm = score(sub, alpha_star, z_star)
        per_cam[f"{lo}-{hi}m"] = dict(n=int(norm.size), mean_m=float(norm.mean()),
                                      radial_bias_m=float(rad.mean()),
                                      radial_sd_m=float(rad.std()),
                                      lateral_sd_m=float(lat.std()))
        print(f"  {f'{lo}-{hi} m':>9} {norm.size:5d} {norm.mean()*1000:6.1f}mm "
              f"{rad.mean()*1000:+8.1f}mm {rad.std()*1000:6.1f}mm")
    for yaw in sorted({round(math.degrees(r["yaw"])) for r in merged}):
        sub = [r for r in merged if round(math.degrees(r["yaw"])) == yaw]
        rad, lat, norm = score(sub, alpha_star, z_star)
        per_cam[f"yaw{yaw}"] = dict(n=int(norm.size), mean_m=float(norm.mean()),
                                    radial_bias_m=float(rad.mean()),
                                    radial_sd_m=float(rad.std()))
        print(f"  {f'yaw {yaw}':>9} {norm.size:5d} {norm.mean()*1000:6.1f}mm "
              f"{rad.mean()*1000:+8.1f}mm {rad.std()*1000:6.1f}mm")
    summary["chosen_path_strata"] = per_cam
    summary["per_camera_bias_spread_m"] = dict(radial=float(spread_r), lateral=float(spread_l))

    # ---- 4. the covariance, and its calibration ---------------------------
    print("\n=== 4. covariance: R = J Sigma_uv J^T + Sigma_model, and its NEES ===")
    sd_u = pooled["u_centre"]["sd"]
    sd_v = math.hypot(alpha_star * sd_vt, (1.0 - alpha_star) * sd_vb)
    print(f"  Sigma_uv from the val split: sd_u = {sd_u:.2f} px, sd_v = {sd_v:.2f} px")
    yaw_rad_sd = 0.0
    yb = [summary["chosen_path_strata"][k]["radial_bias_m"]
          for k in summary["chosen_path_strata"] if k.startswith("yaw")]
    if yb:
        yaw_rad_sd = float(np.std(yb))
    print(f"  Sigma_model radial sd from the yaw sweep: {yaw_rad_sd*1000:.1f} mm")

    for name, model_term in (("pixel term only", 0.0), ("with Sigma_model", yaw_rad_sd)):
        nees = []
        for r in merged:
            camera = models[r["camera"]]
            uc = 0.5 * (r["pu0"] + r["pu1"])
            vv = r["pv1"] + alpha_star * (r["pv0"] - r["pv1"])
            p = camera.pixel_to_world_at_z(uc, vv, z_star)
            if p is None:
                continue
            J = np.zeros((2, 2))
            for axis in (0, 1):
                du = 0.5 if axis == 0 else 0.0
                dv = 0.5 if axis == 1 else 0.0
                a = camera.pixel_to_world_at_z(uc + du, vv + dv, z_star)
                b = camera.pixel_to_world_at_z(uc - du, vv - dv, z_star)
                if a is None or b is None:
                    J = None
                    break
                J[0, axis] = (a[0] - b[0]) / 1.0
                J[1, axis] = (a[1] - b[1]) / 1.0
            if J is None:
                continue
            R = J @ np.diag([sd_u**2, sd_v**2]) @ J.T
            _, ux, uy = bearing_frame(camera, r["x"], r["y"])
            rot = np.array([[ux, uy], [-uy, ux]])
            R = rot @ R @ rot.T
            R[0, 0] += model_term**2
            e = np.array([(p[0] - r["x"]) * ux + (p[1] - r["y"]) * uy,
                          -(p[0] - r["x"]) * uy + (p[1] - r["y"]) * ux])
            try:
                nees.append(float(e @ np.linalg.solve(R, e)))
            except np.linalg.LinAlgError:
                continue
        nees = np.asarray(nees)
        print(f"  {name:18} mean NEES = {nees.mean():6.2f}  median "
              f"{np.median(nees):6.2f}   (target 2.0 for 2 DOF)")
        summary.setdefault("nees", {})[name] = dict(mean=float(nees.mean()),
                                                    median=float(np.median(nees)))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                      encoding="utf-8")
    print(f"\nwrote {(OUT / 'summary.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
