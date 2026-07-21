#!/usr/bin/env python3
"""[REAL SENSING] Occlusion prior from a REAL depth camera (RGB-D), not CAD.

A gz depth_camera at the fixed external-camera pose produced one depth frame
(external_camera/depth, 32FC1). We back-project it to a point cloud (the surfaces
the camera can actually see; occluded backs return nothing), build a 2.5-D height
map (height_map_from_points), and derive occlusion by raycast. Then validate on the
UNIFORM teleport grid (stack_capture2, 35% occluded): does depth-sensed occlusion
recover the +0.10 AUROC that the exact CAD occlusion gets? CAD stays EVAL-ONLY ref.

Depth frame: scratchpad/depth.npy (captured live). Verifies the back-projection
(floor≈0, rack tops at their heights) before using it.
Output: logs/geometry_visibility_prior/demo/depth_occlusion_prior.png
"""
from __future__ import annotations
import csv, json, pathlib, sys
import numpy as np
from scipy.stats import rankdata

_HERE = pathlib.Path(__file__).resolve().parent
REPO = _HERE.parents[1]
sys.path.insert(0, str(REPO / "src" / "unav_common"))
sys.path.insert(0, str(_HERE))
import geometry_visibility as gv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ART = REPO / "paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz"
CFG = REPO / "scripts/visibility_comparison/warehouse_visibility_campaign.yaml"
DEPTH = REPO / "logs/geometry_visibility_prior/frames/depth_stack.npy"  # persistent recapture (capture_camera_frames.py)
SAMP = REPO / "logs/visibility_comparison/stack_capture2/samples.csv"
TGT = REPO / "logs/visibility_comparison/stack_targets2/perception_targets.csv"
OUT = REPO / "logs/geometry_visibility_prior/demo"; OUT.mkdir(parents=True, exist_ok=True)
Z = 0.35
STACK = gv.Prism("dropped_pallet_A2", xmin=-1.425, xmax=-0.525, ymin=-0.35, ymax=0.20, zmin=0.0, zmax=2.6)


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y); pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0: return np.nan
    r = rankdata(s); return float((r[y == 1].sum() - pos*(pos+1)/2) / (pos*neg))
def brier(p, y): return float(np.mean((np.clip(p, 1e-4, 1-1e-4) - y)**2))


def main():
    meta = gv.load_gp_artifact_geometry(ART); xs, ys = meta["xs"], meta["ys"]; cam = gv.make_camera(meta)
    depth = np.load(DEPTH)
    print(f"[REAL depth] frame {depth.shape}, depth {np.nanmin(depth):.2f}-{np.nanmax(depth):.2f} m")

    # --- back-project depth -> world points (verify convention) ---------------
    H, W = depth.shape; step = 3
    vv, uu = np.mgrid[0:H:step, 0:W:step]
    d = depth[vv, uu].ravel(); uu = uu.ravel().astype(float); vv = vv.ravel().astype(float)
    ok = np.isfinite(d) & (d > 0.2) & (d < 60)
    uu, vv, d = uu[ok], vv[ok], d[ok]
    Kinv = np.linalg.inv(cam.K)
    rays = Kinv @ np.vstack([uu, vv, np.ones_like(uu)])          # (3,N), z-row = 1
    cam_pt = rays * d                                            # cam_pt[2] = depth (Z along optical axis)
    world = (cam.R.T @ cam_pt) + np.asarray(cam.cam_pos)[:, None]
    pts = world.T                                               # (N,3)
    print(f"  back-projected {len(pts)} points; world z: p1 {np.percentile(pts[:,2],1):.2f}  "
          f"p50 {np.percentile(pts[:,2],50):.2f}  p99 {np.percentile(pts[:,2],99):.2f} m")
    floor = pts[np.abs(pts[:, 2]) < 0.15]
    print(f"  VERIFY: {100*len(floor)/len(pts):.0f}% of points near floor (|z|<0.15); "
          f"floor xy extent x[{floor[:,0].min():.1f},{floor[:,0].max():.1f}] y[{floor[:,1].min():.1f},{floor[:,1].max():.1f}]")

    # --- sensed height map -> sensed occlusion --------------------------------
    hm = gv.height_map_from_points(pts, xs, ys)["h_max"]
    h_cad = gv.build_height_map(xs, ys, meta["prisms"] + [STACK])["h_max"]
    m = (h_cad > 0.1)
    print(f"  sensed vs CAD height on obstacle cells: median sensed {np.median(hm[m]):.2f} vs CAD {np.median(h_cad[m]):.2f} m")

    fov = gv.fov_projection_grid(cam, xs, ys, Z); jac = gv.projection_jacobian_scale(cam, xs, ys, Z)
    def prior(hmap):
        clear = _raycast_hm(cam, xs, ys, hmap)
        return gv.compute_visibility(fov_mask=fov["fov_mask"], min_clearance=clear, px_per_m_min=jac["px_per_m_min"],
                                     u=fov["u"], v=fov["v"], img_w=cam.img_width, img_h=cam.img_height, tau_clearance=0.10)["visibility_score"]
    cam_only = gv.compute_visibility(fov_mask=fov["fov_mask"], min_clearance=np.full((len(ys), len(xs)), 10.0),
                                     px_per_m_min=jac["px_per_m_min"], u=fov["u"], v=fov["v"],
                                     img_w=cam.img_width, img_h=cam.img_height, tau_clearance=0.10)["visibility_score"]
    sensed = prior(hm)                                          # DEPTH-SENSED occlusion (deployable)
    cad = prior(h_cad)                                          # CAD occlusion (EVAL-ONLY ref)

    # --- validate on uniform teleport grid ------------------------------------
    det = {r["sample_id"]: (1 if str(r.get("yolo_detected_after_threshold", "")).strip() in ("1", "1.0", "True", "true") else 0)
           for r in csv.DictReader(open(TGT))}
    X, Y = [], []
    for r in csv.DictReader(open(SAMP)):
        if r["sample_id"] in det: X.append((float(r["x"]), float(r["y"]))); Y.append(det[r["sample_id"]])
    X = np.array(X); Y = np.array(Y)
    at = lambda f: f[np.clip(np.searchsorted(ys, X[:, 1]), 0, len(ys)-1), np.clip(np.searchsorted(xs, X[:, 0]), 0, len(xs)-1)]
    res = {"camera-only": at(cam_only), "camera + DEPTH occlusion (sensed)": at(sensed), "camera + CAD occlusion (eval ref)": at(cad)}
    print(f"\nheld-out usability on UNIFORM teleport grid ({len(Y)} samples, hit {Y.mean():.2f}):")
    stats = {}
    for k, s in res.items():
        stats[k] = (auroc(s, Y), brier(s, Y)); print(f"  {k:34s}  AUROC {stats[k][0]:.3f}  Brier {stats[k][1]:.3f}")

    # --- figure ---------------------------------------------------------------
    occ = meta["prisms"] + [STACK]; ext = [xs.min(), xs.max(), ys.min(), ys.max()]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True); fig.patch.set_facecolor("white")
    im = ax[0].imshow(np.where(hm > 0.05, hm, np.nan), origin="lower", extent=ext, cmap="cividis", vmin=0, vmax=2.6)
    for p in occ: ax[0].add_patch(Rectangle((p.xmin, p.ymin), p.xmax-p.xmin, p.ymax-p.ymin, fill=False, ec="k", lw=0.6))
    ax[0].plot([cam.cam_pos[0]], [cam.cam_pos[1]], "*", color="r", ms=12); ax[0].set_aspect("equal"); ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].set_title("A. Height map from the REAL depth frame\n(black = true CAD outlines)", fontsize=11, loc="left")
    fig.colorbar(im, ax=ax[0], shrink=0.8).set_label("sensed height [m]", fontsize=8)
    base = gv.in_any_prism(xs, ys, gv.prisms_from_json(json.loads(__import__("yaml").safe_load(open(CFG))["driveable_geometry_json"]))) & fov["fov_mask"]
    ax[1].imshow(np.where(base, sensed, np.nan), origin="lower", extent=ext, cmap="RdYlGn", vmin=0, vmax=1)
    for p in occ: ax[1].add_patch(Rectangle((p.xmin, p.ymin), p.xmax-p.xmin, p.ymax-p.ymin, fill=False, ec="k", lw=0.6))
    ax[1].plot([cam.cam_pos[0]], [cam.cam_pos[1]], "*", color="#1f5fd0", ms=12); ax[1].set_aspect("equal"); ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].set_title("B. Prior with DEPTH-sensed occlusion\ndark = shadows from the real depth map", fontsize=11, loc="left")
    ks = list(res); ax[2].bar(["camera\nonly", "camera+\nDEPTH occ\n(sensed)", "camera+\nCAD occ\n(eval ref)"],
                              [stats[k][0] for k in ks], color=["#3a6ea5", "#2a9d3a", "#999999"])
    ax[2].set_ylim(0.7, 1.0); ax[2].set_ylabel("held-out usability AUROC (uniform grid)")
    for i, k in enumerate(ks): ax[2].text(i, stats[k][0], f"{stats[k][0]:.3f}", ha="center", va="bottom", fontsize=9)
    ax[2].set_title("C. Does DEPTH-sensed occlusion recover CAD's gain?", fontsize=11, loc="left")
    fig.suptitle("[REAL SENSING] Occlusion prior from a real depth camera (RGB-D) — validated on the uniform teleport grid",
                 fontsize=12.5, fontweight="bold")
    fig.savefig(OUT / "depth_occlusion_prior.png", dpi=130, facecolor="white")
    print(f"\nwrote {OUT/'depth_occlusion_prior.png'}")


def _raycast_hm(cam, xs, ys, hm, n=40):
    gx, gy = np.meshgrid(xs, ys); tgt = np.stack([gx, gy, np.full_like(gx, Z)], -1).reshape(-1, 3)
    cp = np.asarray(cam.cam_pos); t = np.linspace(0.02, 0.98, n).reshape(1, -1, 1)
    s = cp.reshape(1, 1, 3) + t * (tgt[:, None, :] - cp.reshape(1, 1, 3))
    ix = np.clip(np.searchsorted(xs, s[..., 0]), 0, len(xs)-1); iy = np.clip(np.searchsorted(ys, s[..., 1]), 0, len(ys)-1)
    return (s[..., 2] - hm[iy, ix]).min(axis=1).reshape(gx.shape)


if __name__ == "__main__":
    main()
