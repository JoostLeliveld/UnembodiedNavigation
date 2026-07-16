#!/usr/bin/env python3
"""[REAL SENSING — ZERO HARDWARE] Occlusion prior from MONOCULAR depth on the fixed
camera's own RGB frame (Depth Anything V2 metric-indoor), vs the real depth sensor.

This is the survey's primary recommendation validated: no new hardware, deployment
inputs only (camera calibration + 2D drivable map + the camera's RGB). The monocular
metric depth is affine-corrected with a deployment-legit trick: pixels whose ray
lands inside the DRIVABLE region have analytically known floor depth (z=0 plane via
calibration), so we RANSAC-fit d_true ~ a*d_pred + b on those pixels (robust to the
minority of drivable pixels that are actually occluded by racks).

Pipeline after correction is identical to depth_occlusion_prior.py:
corrected depth -> back-project -> height map -> raycast occlusion -> visibility
prior -> held-out usability AUROC on the uniform teleport grid (stack_capture2).

RESULT (2026-07-13, one frame / one world / sim imagery — indicative): DA-V2
Metric-Indoor-LARGE + floor-affine + h_max = AUROC 0.973 ~= real depth sensor
0.968 > CAD 0.890 > camera-only 0.782. Model size and the floor-affine correction
are both essential (Small 0.841; Large uncorrected 0.822, AbsRel 15.5%->3.2%).
Robust (percentile) rasterization HURTS (h_max's conservative over-marking is
viewpoint-matched); see mono_depth_ablation.py.
Output: logs/geometry_visibility_prior/demo/mono_depth_occlusion_prior.png
"""
from __future__ import annotations
import csv, pathlib, sys
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
REPO = _HERE.parents[1]
sys.path.insert(0, str(REPO / "src" / "unav_common"))
sys.path.insert(0, str(_HERE))
import geometry_visibility as gv
from depth_occlusion_prior import auroc, brier, _raycast_hm, STACK, Z, ART, SAMP, TGT
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

FRAMES = REPO / "logs/geometry_visibility_prior/frames"
OUT = REPO / "logs/geometry_visibility_prior/demo"; OUT.mkdir(parents=True, exist_ok=True)
CFG = REPO / "scripts/visibility_comparison/warehouse_visibility_campaign.yaml"


def backproject(cam, depth, step=3, dmax=60.0):
    H, W = depth.shape
    vv, uu = np.mgrid[0:H:step, 0:W:step]
    d = depth[vv, uu].ravel(); uu = uu.ravel().astype(float); vv = vv.ravel().astype(float)
    ok = np.isfinite(d) & (d > 0.2) & (d < dmax)
    uu, vv, d = uu[ok], vv[ok], d[ok]
    rays = np.linalg.inv(cam.K) @ np.vstack([uu, vv, np.ones_like(uu)])
    world = (cam.R.T @ (rays * d)) + np.asarray(cam.cam_pos)[:, None]
    return world.T


def floor_calibration_mask(cam, depth_shape, drivable_prisms, step=3):
    """Deployment-legit: analytic z=0 depth for pixels whose floor ray lands in the
    drivable region. Returns (u,v index arrays, analytic optical-axis depth)."""
    H, W = depth_shape
    vv, uu = np.mgrid[0:H:step, 0:W:step]
    uu = uu.ravel().astype(float); vv = vv.ravel().astype(float)
    rays = np.linalg.inv(cam.K) @ np.vstack([uu, vv, np.ones_like(uu)])
    wdir = cam.R.T @ rays                      # world direction per unit optical depth
    cz = float(np.asarray(cam.cam_pos)[2])
    with np.errstate(divide="ignore"):
        d_floor = -cz / wdir[2]                # optical-axis depth to the z=0 plane
    ok = np.isfinite(d_floor) & (d_floor > 0.5) & (d_floor < 60)
    hit = np.asarray(cam.cam_pos)[:, None] + wdir * d_floor
    inside = np.zeros(uu.shape, bool)
    for p in drivable_prisms:
        inside |= (hit[0] >= p.xmin) & (hit[0] <= p.xmax) & (hit[1] >= p.ymin) & (hit[1] <= p.ymax)
    m = ok & inside
    return uu[m].astype(int), vv[m].astype(int), d_floor[m]


def ransac_affine(d_pred, d_true, iters=500, tol=0.25, seed=0):
    """Robust d_true ~ a*d_pred + b (occluded drivable pixels are outliers)."""
    rng = np.random.default_rng(seed)
    best = (1.0, 0.0, -1)
    n = len(d_pred)
    for _ in range(iters):
        i, j = rng.integers(0, n, 2)
        if abs(d_pred[i] - d_pred[j]) < 1e-3: continue
        a = (d_true[i] - d_true[j]) / (d_pred[i] - d_pred[j]); b = d_true[i] - a * d_pred[i]
        inl = np.abs(a * d_pred + b - d_true) < tol
        if inl.sum() > best[2]: best = (a, b, inl.sum())
    a, b, _ = best
    inl = np.abs(a * d_pred + b - d_true) < tol
    A = np.vstack([d_pred[inl], np.ones(inl.sum())]).T
    a, b = np.linalg.lstsq(A, d_true[inl], rcond=None)[0]
    return float(a), float(b), float(inl.mean())


def main():
    meta = gv.load_gp_artifact_geometry(ART); xs, ys = meta["xs"], meta["ys"]; cam = gv.make_camera(meta)
    depth_true = np.load(FRAMES / "depth_stack.npy")
    depth_mono = np.load(FRAMES / "monodepth_stack_large.npy")  # DA-V2 Metric-Indoor-Large (Small ablation: mono_depth_ablation.py)
    import yaml
    drv = gv.prisms_from_json(json.loads(yaml.safe_load(open(CFG))["driveable_geometry_json"]))

    # --- floor-plane affine correction (calibration + drivable map only) -------
    ui, vi, d_floor = floor_calibration_mask(cam, depth_mono.shape, drv)
    d_pred_floor = depth_mono[vi, ui]
    a, b, inlier_frac = ransac_affine(d_pred_floor, d_floor)
    depth_corr = a * depth_mono + b
    print(f"floor calibration: {len(d_floor)} drivable-floor pixels, affine a={a:.3f} b={b:.2f} m, "
          f"RANSAC inliers {100*inlier_frac:.0f}%")

    # honesty metric vs the real sensor (evaluation-only use of true depth)
    for name, dm in [("raw mono", depth_mono), ("corrected mono", depth_corr)]:
        err = np.abs(dm - depth_true)
        print(f"  {name:15s} vs true depth: MAE {np.nanmean(err):.3f} m, p95 {np.nanpercentile(err,95):.2f} m, "
              f"AbsRel {np.nanmean(err/depth_true):.3f}")

    # --- height maps + priors ---------------------------------------------------
    fov = gv.fov_projection_grid(cam, xs, ys, Z); jac = gv.projection_jacobian_scale(cam, xs, ys, Z)
    def prior_from(hmap):
        clear = _raycast_hm(cam, xs, ys, hmap)
        return gv.compute_visibility(fov_mask=fov["fov_mask"], min_clearance=clear, px_per_m_min=jac["px_per_m_min"],
                                     u=fov["u"], v=fov["v"], img_w=cam.img_width, img_h=cam.img_height,
                                     tau_clearance=0.10)["visibility_score"]
    cam_only = gv.compute_visibility(fov_mask=fov["fov_mask"], min_clearance=np.full((len(ys), len(xs)), 10.0),
                                     px_per_m_min=jac["px_per_m_min"], u=fov["u"], v=fov["v"],
                                     img_w=cam.img_width, img_h=cam.img_height, tau_clearance=0.10)["visibility_score"]
    hm_mono = gv.height_map_from_points(backproject(cam, depth_corr), xs, ys)["h_max"]
    hm_mono_raw = gv.height_map_from_points(backproject(cam, depth_mono), xs, ys)["h_max"]
    hm_true = gv.height_map_from_points(backproject(cam, depth_true), xs, ys)["h_max"]
    h_cad = gv.build_height_map(xs, ys, meta["prisms"] + [STACK])["h_max"]
    priors = {
        "camera-only": cam_only,
        "mono raw (no correction)": prior_from(hm_mono_raw),
        "mono + floor-affine": prior_from(hm_mono),
        "real depth sensor": prior_from(hm_true),
        "CAD (eval ref)": prior_from(h_cad),
    }

    # --- held-out usability on the uniform teleport grid ------------------------
    det = {r["sample_id"]: (1 if str(r.get("yolo_detected_after_threshold", "")).strip() in ("1", "1.0", "True", "true") else 0)
           for r in csv.DictReader(open(TGT))}
    X, Y = [], []
    for r in csv.DictReader(open(SAMP)):
        if r["sample_id"] in det: X.append((float(r["x"]), float(r["y"]))); Y.append(det[r["sample_id"]])
    X = np.array(X); Y = np.array(Y)
    at = lambda f: f[np.clip(np.searchsorted(ys, X[:, 1]), 0, len(ys)-1), np.clip(np.searchsorted(xs, X[:, 0]), 0, len(xs)-1)]
    print(f"\nheld-out usability on UNIFORM teleport grid ({len(Y)} samples, hit-rate {Y.mean():.2f}):")
    stats = {}
    for k, f in priors.items():
        s = at(f); stats[k] = (auroc(s, Y), brier(s, Y))
        print(f"  {k:28s}  AUROC {stats[k][0]:.3f}  Brier {stats[k][1]:.3f}")

    # --- figure ------------------------------------------------------------------
    occ = meta["prisms"] + [STACK]; ext = [xs.min(), xs.max(), ys.min(), ys.max()]
    base = gv.in_any_prism(xs, ys, drv) & fov["fov_mask"]
    fig, ax = plt.subplots(1, 4, figsize=(22, 5.4), constrained_layout=True); fig.patch.set_facecolor("white")
    im0 = ax[0].imshow(np.where(depth_true > 0, depth_corr - depth_true, np.nan), cmap="coolwarm", vmin=-1, vmax=1)
    ax[0].set_title("A. Corrected mono − true depth [m]\n(affine from drivable-floor pixels only)", fontsize=11, loc="left")
    fig.colorbar(im0, ax=ax[0], shrink=0.75); ax[0].set_xticks([]); ax[0].set_yticks([])
    im1 = ax[1].imshow(np.where(hm_mono > 0.05, hm_mono, np.nan), origin="lower", extent=ext, cmap="cividis", vmin=0, vmax=2.6)
    for p in occ: ax[1].add_patch(Rectangle((p.xmin, p.ymin), p.xmax-p.xmin, p.ymax-p.ymin, fill=False, ec="k", lw=0.6))
    ax[1].set_aspect("equal"); ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].set_title("B. Height map from MONO depth\n(black = true CAD outlines)", fontsize=11, loc="left")
    fig.colorbar(im1, ax=ax[1], shrink=0.75).set_label("sensed height [m]", fontsize=8)
    ax[2].imshow(np.where(base, priors["mono + floor-affine"], np.nan), origin="lower", extent=ext, cmap="RdYlGn", vmin=0, vmax=1)
    for p in occ: ax[2].add_patch(Rectangle((p.xmin, p.ymin), p.xmax-p.xmin, p.ymax-p.ymin, fill=False, ec="k", lw=0.6))
    ax[2].set_aspect("equal"); ax[2].set_xticks([]); ax[2].set_yticks([])
    ax[2].set_title("C. Prior with MONO-sensed occlusion", fontsize=11, loc="left")
    ks = list(priors)
    colors = ["#3a6ea5", "#c98a2b", "#2a9d3a", "#207f8f", "#999999"]
    ax[3].bar(range(len(ks)), [stats[k][0] for k in ks], color=colors)
    ax[3].set_xticks(range(len(ks))); ax[3].set_xticklabels(["camera\nonly", "mono\nraw", "mono+\nfloor-affine", "real\ndepth", "CAD\n(eval)"], fontsize=8)
    for i, k in enumerate(ks): ax[3].text(i, stats[k][0], f"{stats[k][0]:.3f}", ha="center", va="bottom", fontsize=9)
    ax[3].set_ylim(0.6, 1.0); ax[3].set_ylabel("held-out usability AUROC (uniform grid)")
    ax[3].set_title("D. Zero-hardware mono depth vs sensor", fontsize=11, loc="left")
    fig.suptitle("[REAL SENSING, ZERO HARDWARE] Monocular-depth occlusion prior from the fixed camera's own RGB",
                 fontsize=12.5, fontweight="bold")
    fig.savefig(OUT / "mono_depth_occlusion_prior.png", dpi=130, facecolor="white")
    print(f"\nwrote {OUT/'mono_depth_occlusion_prior.png'}")


if __name__ == "__main__":
    main()
