#!/usr/bin/env python3
"""Level 3, warehouse-implementable: build the observability prior from a SENSED
height map (a point cloud), not CAD.

Story: the fixed infrastructure cameras (or a robot LiDAR/RGB-D pass) sense the 3-D
surfaces they can see; occluded backs return nothing -> 'unknown' cells. From that
sensed point cloud we build a 2.5-D height map (`height_map_from_points`) and the
same camera raycast reliability prior. We then check how well the sensed prior
recovers the CAD-based prior, and where the sensor left blind (unknown) regions.

In sim the point cloud is synthesised by sampling the surfaces each camera can see
(this stands in for real stereo/LiDAR returns, with range-dependent noise); in
deployment the *same* `height_map_from_points` consumes the real sensor's cloud.

Output: logs/geometry_visibility_prior/demo/sensed_height_prior.png
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np

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
OUT = REPO / "logs" / "geometry_visibility_prior" / "demo"; OUT.mkdir(parents=True, exist_ok=True)
Z_MARKER, TAU_OCC = 0.35, 0.10


def raycast_hm(cam, xs, ys, hm, n=40):
    gx, gy = np.meshgrid(xs, ys)
    tgt = np.stack([gx, gy, np.full_like(gx, Z_MARKER)], -1).reshape(-1, 3)
    cp = np.asarray(cam.cam_pos); t = np.linspace(0.02, 0.98, n).reshape(1, -1, 1)
    s = cp.reshape(1, 1, 3) + t * (tgt[:, None, :] - cp.reshape(1, 1, 3))
    ix = np.clip(np.searchsorted(xs, s[..., 0]), 0, len(xs)-1); iy = np.clip(np.searchsorted(ys, s[..., 1]), 0, len(ys)-1)
    return (s[..., 2] - hm[iy, ix]).min(axis=1).reshape(gx.shape)


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    ra, rb = np.argsort(np.argsort(a[m])), np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    meta = gv.load_gp_artifact_geometry(ART)
    xs, ys = meta["xs"], meta["ys"]
    cam1 = gv.make_camera(meta)                                    # south localization camera
    cam2 = gv.ObliqueCameraModel(cam_pos=(0.0, 5.6, 4.8), look_at=(0.0, 1.8, 0.0),
                                 img_width=cam1.img_width, img_height=cam1.img_height, fov_h_rad=cam1.fov_h_rad)
    prisms = meta["prisms"]                                        # CAD: only to synthesise sensor returns
    drive = gv.prisms_from_json(json.loads(__import__("yaml").safe_load(open(CFG))["driveable_geometry_json"]))
    dmask = gv.in_any_prism(xs, ys, drive)
    fov1 = gv.fov_projection_grid(cam1, xs, ys, Z_MARKER)
    jac1 = gv.projection_jacobian_scale(cam1, xs, ys, Z_MARKER)
    base = dmask & fov1["fov_mask"]
    ext = [xs.min(), xs.max(), ys.min(), ys.max()]

    # --- synthesise the sensed point cloud (what infra stereo would return) ------
    rng = np.random.RandomState(0)
    pts = []
    fy, fx = np.where(base)                                         # floor candidates at every driveable+FOV cell
    pts.append(np.stack([xs[fx], ys[fy], np.zeros(fx.size)], 1))
    for p in prisms:                                               # occluder TOP faces
        pxr = np.arange(p.xmin, p.xmax + 1e-6, 0.10); pyr = np.arange(p.ymin, p.ymax + 1e-6, 0.10)
        if len(pxr) and len(pyr):
            tx, ty = np.meshgrid(pxr, pyr)
            pts.append(np.stack([tx.ravel(), ty.ravel(), np.full(tx.size, p.zmax)], 1))
    cand = np.concatenate(pts, 0)
    vis = gv.points_visible_from_camera(cam1, cand, prisms) | gv.points_visible_from_camera(cam2, cand, prisms)
    sensed = cand[vis].copy()
    # stereo range noise on z (grows with range^2 / baseline)
    r = np.linalg.norm(sensed - np.asarray(cam1.cam_pos), axis=1)
    sensed[:, 2] += rng.normal(0, 1, len(sensed)) * (r ** 2 / (1.5 * 640.0) * 0.4)
    print(f"candidate surface points: {len(cand)}; sensed (visible from >=1 cam): {len(sensed)}")

    # --- height map from the sensed cloud + priors -------------------------------
    hm = gv.height_map_from_points(sensed, xs, ys)
    h_sensed = hm["h_max"]
    def prior_from(clear):
        return gv.compute_visibility(fov_mask=fov1["fov_mask"], min_clearance=clear, px_per_m_min=jac1["px_per_m_min"],
                                     u=fov1["u"], v=fov1["v"], img_w=cam1.img_width, img_h=cam1.img_height,
                                     tau_clearance=TAU_OCC)["visibility_score"]
    prior_sensed = prior_from(raycast_hm(cam1, xs, ys, h_sensed))
    prior_cad = prior_from(gv.raycast_min_clearance(cam1, xs, ys, prisms, Z_MARKER))
    sp = spearman(prior_sensed[base], prior_cad[base])
    print(f"Spearman(sensed prior, CAD prior) over driveable+FOV cells: {sp:.3f}")

    # unknown cells: driveable+FOV floor cells the sensors never returned a point for
    obs_floor = hm["observed"]
    unknown = base & ~obs_floor
    print(f"driveable+FOV cells: {int(base.sum())}; unknown (no sensor return): {int(unknown.sum())} "
          f"({100*unknown.sum()/max(base.sum(),1):.0f}%)")

    # ---------------- figure ----------------
    occ = prisms
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6), constrained_layout=True)
    fig.patch.set_facecolor("white")

    def scene(a2, ec="#9a9a9a"):
        for p in occ:
            a2.add_patch(Rectangle((p.xmin, p.ymin), p.xmax-p.xmin, p.ymax-p.ymin, fill=False, ec=ec, lw=0.6, zorder=4))
        for c, lbl in ((cam1, "CAM 1"), (cam2, "CAM 2")):
            a2.plot([c.cam_pos[0]], [c.cam_pos[1]], "*", ms=13, color="#1f5fd0", mec="w", mew=0.7, zorder=6)
        a2.set_xlim(ext[0], ext[1]); a2.set_ylim(ext[2], ext[3]); a2.set_aspect("equal"); a2.set_xticks([]); a2.set_yticks([])

    scene(ax[0], ec="k")
    im = ax[0].imshow(np.where(h_sensed > 0.05, h_sensed, np.nan), origin="lower", extent=ext, cmap="cividis", vmin=0, vmax=1.9, zorder=2)
    ax[0].scatter(sensed[::7, 0], sensed[::7, 1], s=1, c="#3a6ea5", alpha=0.25, zorder=3)
    ax[0].set_title("Sensed height map (infra stereo point cloud)", fontsize=12, fontweight="bold", loc="left")
    ax[0].set_xlabel("built from what the cameras can see — no CAD", fontsize=8.5, color="#666")
    fig.colorbar(im, ax=ax[0], shrink=0.82).set_label("sensed h_max [m]", fontsize=8)

    scene(ax[1])
    im = ax[1].imshow(np.where(base, prior_sensed, np.nan), origin="lower", extent=ext, cmap="RdYlGn", vmin=0, vmax=1, zorder=2)
    # hatch the unknown (unsensed) driveable cells
    ax[1].contourf(xs, ys, unknown.astype(float), levels=[0.5, 1.5], colors="none", hatches=["xxx"], zorder=5)
    ax[1].set_title("Reliability prior from SENSED geometry", fontsize=12, fontweight="bold", loc="left")
    ax[1].set_xlabel("hatched = unknown (no sensor return) → handle conservatively", fontsize=8.5, color="#666")
    fig.colorbar(im, ax=ax[1], shrink=0.82).set_label("P(reliable)", fontsize=8)

    scene(ax[2])
    d = np.where(base, prior_sensed - prior_cad, np.nan)
    m = float(np.nanmax(np.abs(d)))
    im = ax[2].imshow(d, origin="lower", extent=ext, cmap="RdBu", vmin=-m, vmax=m, zorder=2)
    ax[2].set_title("Sensed − CAD prior (recovery error)", fontsize=12, fontweight="bold", loc="left")
    ax[2].set_xlabel(f"Spearman(sensed, CAD) = {sp:.2f} — stereo recovers the geometry prior", fontsize=8.5, color="#666")
    fig.colorbar(im, ax=ax[2], shrink=0.82).set_label("sensed − CAD", fontsize=8)

    fig.suptitle("Level 3 (deployable): observability prior from a SENSED height map — cameras build it themselves",
                 fontsize=14, fontweight="bold")
    fig.savefig(OUT / "sensed_height_prior.png", dpi=130, facecolor="white")
    print(f"wrote {OUT/'sensed_height_prior.png'}")


if __name__ == "__main__":
    main()
