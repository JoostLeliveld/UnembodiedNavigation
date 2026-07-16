#!/usr/bin/env python3
"""Ablation for the mono-depth occlusion prior: model size x rasterization robustness.

Edge-bleed diagnosis: mono depth is blurry at object boundaries; back-projected
edge pixels land mid-ray as phantom tall points, and per-cell MAX height turns those
sparse streaks into fat phantom occluders. Ablate:
  raster: h_max (baseline)  vs  robust (per-cell 90th percentile, min 3 points)
  model:  DA-V2 metric-indoor Small  vs  Large
  correction: raw vs floor-affine (deployment-legit RANSAC on drivable-floor pixels)
All scored as held-out usability AUROC on the uniform teleport grid (stack_capture2).
"""
from __future__ import annotations
import csv, json, pathlib, sys
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
REPO = _HERE.parents[1]
sys.path.insert(0, str(REPO / "src" / "unav_common"))
sys.path.insert(0, str(_HERE))
import geometry_visibility as gv
from depth_occlusion_prior import auroc, brier, _raycast_hm, STACK, Z, ART, SAMP, TGT
from mono_depth_occlusion_prior import backproject, floor_calibration_mask, ransac_affine

FRAMES = REPO / "logs/geometry_visibility_prior/frames"
CFG = REPO / "scripts/visibility_comparison/warehouse_visibility_campaign.yaml"


def robust_height_map(points, xs, ys, q=90.0, min_count=3):
    """Per-cell qth-percentile height with a minimum point count — prunes sparse
    edge-bleed streaks that h_max amplifies."""
    pts = np.asarray(points, float)
    ny, nx = len(ys), len(xs)
    ix = np.clip(np.searchsorted(xs, pts[:, 0]), 0, nx - 1)
    iy = np.clip(np.searchsorted(ys, pts[:, 1]), 0, ny - 1)
    cell = iy * nx + ix
    order = np.argsort(cell, kind="stable")
    cell_s, z_s = cell[order], np.maximum(pts[order, 2], 0.0)
    uniq, start = np.unique(cell_s, return_index=True)
    h = np.zeros(ny * nx); counts = np.diff(np.append(start, len(cell_s)))
    for u, s, c in zip(uniq, start, counts):
        if c >= min_count:
            h[u] = np.percentile(z_s[s:s + c], q)
    return h.reshape(ny, nx)


def main():
    meta = gv.load_gp_artifact_geometry(ART); xs, ys = meta["xs"], meta["ys"]; cam = gv.make_camera(meta)
    import yaml
    drv = gv.prisms_from_json(json.loads(yaml.safe_load(open(CFG))["driveable_geometry_json"]))
    depth_true = np.load(FRAMES / "depth_stack.npy")

    fov = gv.fov_projection_grid(cam, xs, ys, Z); jac = gv.projection_jacobian_scale(cam, xs, ys, Z)
    def prior_from(hmap):
        clear = _raycast_hm(cam, xs, ys, hmap)
        return gv.compute_visibility(fov_mask=fov["fov_mask"], min_clearance=clear, px_per_m_min=jac["px_per_m_min"],
                                     u=fov["u"], v=fov["v"], img_w=cam.img_width, img_h=cam.img_height,
                                     tau_clearance=0.10)["visibility_score"]

    det = {r["sample_id"]: (1 if str(r.get("yolo_detected_after_threshold", "")).strip() in ("1", "1.0", "True", "true") else 0)
           for r in csv.DictReader(open(TGT))}
    X, Y = [], []
    for r in csv.DictReader(open(SAMP)):
        if r["sample_id"] in det: X.append((float(r["x"]), float(r["y"]))); Y.append(det[r["sample_id"]])
    X = np.array(X); Y = np.array(Y)
    at = lambda f: f[np.clip(np.searchsorted(ys, X[:, 1]), 0, len(ys)-1), np.clip(np.searchsorted(xs, X[:, 0]), 0, len(xs)-1)]
    def score(name, hmap):
        s = at(prior_from(hmap)); print(f"  {name:44s}  AUROC {auroc(s, Y):.3f}  Brier {brier(s, Y):.3f}")

    print(f"uniform grid: {len(Y)} samples, hit-rate {Y.mean():.2f}\n")
    for tag, fn in [("small", "monodepth_stack.npy"), ("large", "monodepth_stack_large.npy")]:
        p = FRAMES / fn
        if not p.exists():
            print(f"  [skip {tag}: {fn} missing]"); continue
        dm = np.load(p)
        ui, vi, d_floor = floor_calibration_mask(cam, dm.shape, drv)
        a, b, inl = ransac_affine(dm[vi, ui], d_floor)
        dc = a * dm + b
        err = np.abs(dc - depth_true)
        print(f"[{tag}] affine a={a:.3f} b={b:.2f} (inl {100*inl:.0f}%)  corrected MAE {np.nanmean(err):.3f} m  "
              f"AbsRel {np.nanmean(err/depth_true):.3f}")
        pts = backproject(cam, dc)
        score(f"{tag} + affine + h_max", gv.height_map_from_points(pts, xs, ys)["h_max"])
        score(f"{tag} + affine + robust(q90,min3)", robust_height_map(pts, xs, ys))
        score(f"{tag} + affine + robust(q75,min5)", robust_height_map(pts, xs, ys, q=75, min_count=5))
    # references
    pts_true = backproject(cam, depth_true)
    score("real depth + h_max (ref)", gv.height_map_from_points(pts_true, xs, ys)["h_max"])
    score("CAD (eval ref)", gv.build_height_map(xs, ys, meta["prisms"] + [STACK])["h_max"])


if __name__ == "__main__":
    main()
