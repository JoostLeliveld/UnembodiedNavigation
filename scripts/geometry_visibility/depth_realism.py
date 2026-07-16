#!/usr/bin/env python3
"""[REAL depth frame + honest sensor model] How much does occlusion degrade with a
REALISTIC depth sensor, and how do sensed height maps compare to the COMPLETE-CAD
ground truth?

The gz depth_camera gives PERFECT depth (no noise/holes/range limit). Real RGB-D has
range-growing noise, dropouts (dark/specular/oblique), and a ~10 m usable range (< the
~13 m warehouse diagonal). We degrade the captured perfect frame accordingly.

Ground truth = COMPLETE CAD occlusion: ALL rendered occluders (racks + WALLS + visual
shelf boxes/rails, parsed from the world) + the stack — not the 18-box rack subset.

Compare, predicting real YOLO hit/miss on the uniform teleport grid:
  camera-only · realistic-depth occ · perfect-depth occ · COMPLETE-CAD occ (ground truth)
Output: logs/geometry_visibility_prior/demo/depth_realism.png
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
from unav_common.occlusion_geometry import parse_occlusion_scene_from_world, AxisAlignedPrism
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ART = REPO / "paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz"
CFG = REPO / "scripts/visibility_comparison/warehouse_visibility_campaign.yaml"
WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_aws.world.sdf"
DEPTH = pathlib.Path("/tmp/claude-1000/-home-joostleliveld-Thesis/2e584162-8150-4091-9318-d7f33ec3aeeb/scratchpad/depth.npy")
SAMP = REPO / "logs/visibility_comparison/stack_capture2/samples.csv"
TGT = REPO / "logs/visibility_comparison/stack_targets2/perception_targets.csv"
OUT = REPO / "logs/geometry_visibility_prior/demo"; OUT.mkdir(parents=True, exist_ok=True)
Z, RMAX = 0.35, 10.0
STACK = gv.Prism("stk", -1.425, -0.525, -0.35, 0.20, 0.0, 2.6)


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y); pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0: return np.nan
    r = rankdata(s); return float((r[y == 1].sum() - pos*(pos+1)/2) / (pos*neg))
def brier(p, y): return float(np.mean((np.clip(p, 1e-4, 1-1e-4) - y)**2))


def complete_cad_prisms():
    """ALL rendered occluders: walls + rack occluders (VISUAL geom, so shelf boxes/rails
    included) + the stack. This is the ground-truth geometry the camera actually sees."""
    sc = parse_occlusion_scene_from_world(str(WORLD), model_name=("warehouse_walls", "warehouse_rack_occluders"),
                                          geometry_tags=("visual",))
    prisms = [gv.Prism(p.name, p.xmin, p.xmax, p.ymin, p.ymax, p.zmin, p.zmax) for p in sc.prisms]
    return prisms + [STACK]


def backproject(depth, cam, step=3):
    H, W = depth.shape; vv, uu = np.mgrid[0:H:step, 0:W:step]
    d = depth[vv, uu].ravel(); uu = uu.ravel().astype(float); vv = vv.ravel().astype(float)
    ok = np.isfinite(d) & (d > 0.2) & (d < 60)
    rays = np.linalg.inv(cam.K) @ np.vstack([uu[ok], vv[ok], np.ones(ok.sum())])
    return ((cam.R.T @ (rays * d[ok])) + np.asarray(cam.cam_pos)[:, None]).T


def realistic(depth, seed=0):
    """Degrade perfect depth to a plausible RGB-D: range-growing noise, ~10 m limit,
    range-dependent dropouts (approximates dark/oblique/textureless failures)."""
    rng = np.random.RandomState(seed); d = depth.astype(float).copy()
    d = d + rng.normal(0, 1, d.shape) * (0.005 * d + 0.004 * d ** 2)   # mm near, cm+ far
    d[d > RMAX] = np.nan                                               # beyond usable range -> no return
    p_drop = np.clip(0.03 + 0.05 * (d - 3.0), 0, 0.6)                  # more holes with range
    d[rng.random(d.shape) < p_drop] = np.nan
    return d


def main():
    meta = gv.load_gp_artifact_geometry(ART); xs, ys = meta["xs"], meta["ys"]; cam = gv.make_camera(meta)
    depth = np.load(DEPTH)
    cad = complete_cad_prisms()
    print(f"COMPLETE-CAD ground truth: {len(cad)} occluder prisms (was 18 rack-only); "
          f"heights to {max(p.zmax for p in cad):.1f} m (walls)")

    hm_perfect = gv.height_map_from_points(backproject(depth, cam), xs, ys)["h_max"]
    d_real = realistic(depth); print(f"realistic sensor: {100*np.isnan(d_real).mean():.0f}% pixels dropped "
                                     f"(range>{RMAX}m + holes)")
    hm_real = gv.height_map_from_points(backproject(d_real, cam), xs, ys)["h_max"]
    h_gt = gv.build_height_map(xs, ys, cad)["h_max"]

    fov = gv.fov_projection_grid(cam, xs, ys, Z); jac = gv.projection_jacobian_scale(cam, xs, ys, Z)
    def rc(hm):
        gx, gy = np.meshgrid(xs, ys); tgt = np.stack([gx, gy, np.full_like(gx, Z)], -1).reshape(-1, 3)
        cp = np.asarray(cam.cam_pos); t = np.linspace(0.02, 0.98, 40).reshape(1, -1, 1)
        s = cp.reshape(1, 1, 3) + t * (tgt[:, None, :] - cp.reshape(1, 1, 3))
        ix = np.clip(np.searchsorted(xs, s[..., 0]), 0, len(xs)-1); iy = np.clip(np.searchsorted(ys, s[..., 1]), 0, len(ys)-1)
        return (s[..., 2] - hm[iy, ix]).min(axis=1).reshape(gx.shape)
    def vis(min_clear):
        return gv.compute_visibility(fov_mask=fov["fov_mask"], min_clearance=min_clear, px_per_m_min=jac["px_per_m_min"],
                                     u=fov["u"], v=fov["v"], img_w=cam.img_width, img_h=cam.img_height, tau_clearance=0.10)["visibility_score"]
    priors = {
        "camera-only": vis(np.full((len(ys), len(xs)), 10.0)),
        "realistic-depth occ": vis(rc(hm_real)),
        "perfect-depth occ": vis(rc(hm_perfect)),
        "COMPLETE-CAD occ (GT)": vis(gv.raycast_min_clearance(cam, xs, ys, cad, Z)),
    }
    # height-map accuracy vs GT (on cells the GT says are occluders in view)
    m = (h_gt > 0.3) & fov["fov_mask"]
    print(f"\nsensed height vs COMPLETE-CAD on obstacle-in-view cells (n={int(m.sum())}):")
    print(f"  perfect-depth   covers {100*(hm_perfect[m]>0.3).mean():.0f}% of them")
    print(f"  realistic-depth covers {100*(hm_real[m]>0.3).mean():.0f}% of them")

    det = {r["sample_id"]: (1 if str(r.get("yolo_detected_after_threshold", "")).strip() in ("1", "1.0", "True", "true") else 0) for r in csv.DictReader(open(TGT))}
    X, Y = [], []
    for r in csv.DictReader(open(SAMP)):
        if r["sample_id"] in det: X.append((float(r["x"]), float(r["y"]))); Y.append(det[r["sample_id"]])
    X = np.array(X); Y = np.array(Y)
    at = lambda f: f[np.clip(np.searchsorted(ys, X[:, 1]), 0, len(ys)-1), np.clip(np.searchsorted(xs, X[:, 0]), 0, len(xs)-1)]
    print(f"\nheld-out usability on uniform teleport grid ({len(Y)} samples, hit {Y.mean():.2f}):")
    st = {}
    for k, f in priors.items():
        st[k] = (auroc(at(f), Y), brier(at(f), Y)); print(f"  {k:26s}  AUROC {st[k][0]:.3f}  Brier {st[k][1]:.3f}")

    # figure
    occ = cad; ext = [xs.min(), xs.max(), ys.min(), ys.max()]
    fig, ax = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True); fig.patch.set_facecolor("white")
    for a, hmap, ttl in [(ax[0], h_gt, "GT height (complete CAD:\nracks+walls+boxes+stack)"),
                         (ax[1], hm_perfect, "perfect depth sensor\n(idealized)"),
                         (ax[2], hm_real, f"realistic depth sensor\n({100*np.isnan(d_real).mean():.0f}% dropped, noisy, {RMAX:.0f}m)")]:
        im = a.imshow(np.where(hmap > 0.05, hmap, np.nan), origin="lower", extent=ext, cmap="cividis", vmin=0, vmax=4.2)
        a.plot([cam.cam_pos[0]], [cam.cam_pos[1]], "*", color="r", ms=10); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(ttl, fontsize=10, loc="left")
    ks = list(priors); ax[3].bar(range(len(ks)), [st[k][0] for k in ks], color=["#3a6ea5", "#e5893a", "#2a9d3a", "#555"])
    ax[3].set_xticks(range(len(ks))); ax[3].set_xticklabels(["camera\nonly", "realistic\ndepth", "perfect\ndepth", "complete\nCAD (GT)"], fontsize=8)
    ax[3].set_ylim(0.7, 1.0); ax[3].set_ylabel("held-out usability AUROC")
    for i, k in enumerate(ks): ax[3].text(i, st[k][0], f"{st[k][0]:.3f}", ha="center", va="bottom", fontsize=9)
    ax[3].set_title("occlusion prior vs YOLO detection", fontsize=10, loc="left")
    fig.suptitle("[REAL depth + honest sensor model] realistic vs perfect depth vs COMPLETE-CAD ground truth",
                 fontsize=13, fontweight="bold")
    fig.savefig(OUT / "depth_realism.png", dpi=125, facecolor="white")
    print(f"\nwrote {OUT/'depth_realism.png'}")


if __name__ == "__main__":
    main()
