#!/usr/bin/env python3
"""Which depth source best INITIALIZES the observability GP?

Each real sensing modality is emulated as its characteristic corruption of the
ground-truth (SDF) obstacle height map, then run through the SAME chain
(reconstructed heights -> raycast occlusion -> x camera terms -> visibility) and
scored against the empirical YOLO GP (Stage-1 Spearman / affine R^2).

Modalities:
  TRUE        upper bound (perfect depth / SDF)
  FREESPACE   footprints from driveable-map holes, assumed height (no depth)
  RGBD        measured depth + noise, but range-limited (dropout beyond R_max -> freespace fallback)
  MONOCULAR   dense, full range, but height-compressed + blurred + noisy (OOD oblique view)
  STEREO      dense metric, noise grows with range^2/baseline (triangulation)
Stochastic modalities are run over several seeds -> mean +/- std (robustness).
"""
from __future__ import annotations
import json, pathlib, sys
from collections import deque
import numpy as np
from scipy.ndimage import gaussian_filter

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "unav_common"))
sys.path.insert(0, str(REPO / "scripts" / "geometry_visibility"))
import geometry_visibility as gv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = REPO / "paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz"
CFG = REPO / "scripts/visibility_comparison/warehouse_visibility_campaign.yaml"
OUT = REPO / "logs" / "geometry_visibility_prior" / "demo"
OUT.mkdir(parents=True, exist_ok=True)
Z_MARKER, TAU, H0 = 0.35, 0.10, 1.5
SEEDS = range(6)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def affine_r2(g, e):
    A = np.vstack([g, np.ones_like(g)]).T
    coef, *_ = np.linalg.lstsq(A, e, rcond=None)
    pred = A @ coef
    return 1.0 - np.sum((e - pred) ** 2) / np.sum((e - e.mean()) ** 2)


def interior_holes(free_mask):
    nd = ~free_mask; ny, nx = nd.shape
    seen = np.zeros_like(nd); dq = deque()
    for i in range(ny):
        for j in (0, nx - 1):
            if nd[i, j] and not seen[i, j]: seen[i, j] = True; dq.append((i, j))
    for j in range(nx):
        for i in (0, ny - 1):
            if nd[i, j] and not seen[i, j]: seen[i, j] = True; dq.append((i, j))
    while dq:
        i, j = dq.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < ny and 0 <= nj < nx and nd[ni, nj] and not seen[ni, nj]:
                seen[ni, nj] = True; dq.append((ni, nj))
    return nd & ~seen


def raycast_hm(cam, xs, ys, hm, z_marker=Z_MARKER, n=48):
    gx, gy = np.meshgrid(xs, ys)
    tgt = np.stack([gx, gy, np.full_like(gx, z_marker)], -1).reshape(-1, 3)
    cp = np.asarray(cam.cam_pos)
    t = np.linspace(0.02, 0.98, n).reshape(1, -1, 1)
    s = cp.reshape(1, 1, 3) + t * (tgt[:, None, :] - cp.reshape(1, 1, 3))
    ix = np.clip(np.searchsorted(xs, s[..., 0]), 0, len(xs) - 1)
    iy = np.clip(np.searchsorted(ys, s[..., 1]), 0, len(ys) - 1)
    clear = (s[..., 2] - hm[iy, ix]).min(axis=1)
    return clear.reshape(gx.shape)


def main():
    meta = gv.load_gp_artifact_geometry(ART)
    import yaml
    drive = gv.prisms_from_json(json.loads(yaml.safe_load(open(CFG))["driveable_geometry_json"]))
    xs, ys = meta["xs"], meta["ys"]
    cam = gv.make_camera(meta)
    drive_mask = gv.in_any_prism(xs, ys, drive)
    fov = gv.fov_projection_grid(cam, xs, ys, Z_MARKER)
    jac = gv.projection_jacobian_scale(cam, xs, ys, Z_MARKER)
    emp = meta["P_mean_map"]
    base = drive_mask & fov["fov_mask"] & np.isfinite(emp)
    h_true = gv.build_height_map(xs, ys, meta["prisms"])["h_max"]
    holes = interior_holes(drive_mask)
    gx, gy = np.meshgrid(xs, ys)
    rng_map = np.sqrt((gx - cam.cam_pos[0]) ** 2 + (gy - cam.cam_pos[1]) ** 2 + cam.cam_pos[2] ** 2)

    def visibility(hm):
        clear = raycast_hm(cam, xs, ys, hm)
        return gv.compute_visibility(
            fov_mask=fov["fov_mask"], min_clearance=clear, px_per_m_min=jac["px_per_m_min"],
            u=fov["u"], v=fov["v"], img_w=cam.img_width, img_h=cam.img_height,
            tau_clearance=TAU, use_range=True, use_boundary=True)["visibility_score"]

    def score(hm):
        s = visibility(hm)
        return spearman(s[base], emp[base]), affine_r2(s[base], emp[base])

    # --- sensor emulators returning an estimated height map -----------------
    def em_true(seed=0):
        return h_true

    def em_freespace(seed=0):
        return holes.astype(float) * H0

    def em_rgbd(seed=0, r_max=8.0, sigma=0.04):
        rs = np.random.RandomState(seed)
        hm = h_true + rs.normal(0, sigma, h_true.shape) * (h_true > 0)
        hm = np.clip(hm, 0, None)
        out_of_range = rng_map > r_max          # sensor gets no return -> fallback
        hm = np.where(out_of_range, holes.astype(float) * H0, hm)
        return hm

    def em_monocular(seed=0, k=0.6, sigma=0.12, blur=1.5):
        rs = np.random.RandomState(seed)
        hm = k * h_true + rs.normal(0, sigma, h_true.shape)   # height-compressed + noisy
        hm = gaussian_filter(hm, blur)                        # OOD -> smooth, soft edges
        return np.clip(hm, 0, None)

    def em_stereo(seed=0, baseline=1.5, focal=640.0, sigma_disp=0.4, texture_p=0.9):
        rs = np.random.RandomState(seed)
        sig_z = (rng_map ** 2) / (baseline * focal) * sigma_disp   # triangulation error grows w/ range^2
        hm = h_true + rs.normal(0, 1, h_true.shape) * sig_z * (h_true > 0)
        textured = rs.rand(*h_true.shape) < texture_p
        hm = np.where(textured, hm, holes.astype(float) * H0)      # textureless -> fallback
        return np.clip(hm, 0, None)

    modalities = [
        ("TRUE (SDF/perfect)", em_true, False),
        ("FREESPACE (no depth)", em_freespace, False),
        ("RGBD range=6m", lambda s: em_rgbd(s, r_max=6.0), True),
        ("RGBD range=8m", lambda s: em_rgbd(s, r_max=8.0), True),
        ("RGBD range=10m", lambda s: em_rgbd(s, r_max=10.0), True),
        ("MONOCULAR", em_monocular, True),
        ("STEREO b=1.5m", em_stereo, True),
        ("STEREO b=0.3m", lambda s: em_stereo(s, baseline=0.3), True),
    ]

    print(f"{'depth source':24s}  Spearman        affineR2")
    print("-" * 62)
    rows = {}
    for name, em, stoch in modalities:
        seeds = SEEDS if stoch else [0]
        sp, r2 = zip(*[score(em(s)) for s in seeds])
        rows[name] = (np.mean(sp), np.std(sp), np.mean(r2), np.std(r2))
        pm = f"{np.mean(sp):.3f}" + (f" +/- {np.std(sp):.3f}" if stoch else "        ")
        rm = f"{np.mean(r2):.3f}" + (f" +/- {np.std(r2):.3f}" if stoch else "")
        print(f"{name:24s}  {pm:16s}  {rm}")

    # --- figure: estimated height + resulting visibility per modality -------
    show = ["TRUE (SDF/perfect)", "FREESPACE (no depth)", "RGBD range=8m", "MONOCULAR", "STEREO b=1.5m"]
    emap = dict((n, em) for n, em, _ in modalities)
    ext = [xs.min(), xs.max(), ys.min(), ys.max()]
    fig, ax = plt.subplots(2, len(show), figsize=(4 * len(show), 8))
    for j, name in enumerate(show):
        hm = emap[name](0)
        vs = np.where(base, visibility(hm), np.nan)
        ax[0, j].imshow(np.where(hm > 0.05, hm, np.nan), origin="lower", extent=ext, cmap="cividis", vmin=0, vmax=1.9)
        ax[0, j].set_title(f"{name}\nest. height", fontsize=9)
        ax[1, j].imshow(vs, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1)
        ax[1, j].set_title("visibility", fontsize=9)
        for a in (ax[0, j], ax[1, j]):
            a.plot([cam.cam_pos[0]], [cam.cam_pos[1]], "*", color="r", markersize=9)
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Depth-source emulation: estimated height (top) -> visibility (bottom)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "depth_source_comparison.png", dpi=115)
    print(f"\nwrote {OUT/'depth_source_comparison.png'}")

    # --- multi-camera demo (qualitative; not scoreable vs single-cam GP) ----
    cam2 = gv.ObliqueCameraModel(cam_pos=(0.0, 5.6, 4.8), look_at=(0.0, 1.8, 0.0),
                                 img_width=cam.img_width, img_height=cam.img_height, fov_h_rad=cam.fov_h_rad)
    fov2 = gv.fov_projection_grid(cam2, xs, ys, Z_MARKER)
    jac2 = gv.projection_jacobian_scale(cam2, xs, ys, Z_MARKER)
    clear2 = gv.raycast_min_clearance(cam2, xs, ys, meta["prisms"], Z_MARKER)
    vis1 = visibility(h_true)
    vis2 = gv.compute_visibility(fov_mask=fov2["fov_mask"], min_clearance=clear2, px_per_m_min=jac2["px_per_m_min"],
                                 u=fov2["u"], v=fov2["v"], img_w=cam.img_width, img_h=cam.img_height,
                                 tau_clearance=TAU)["visibility_score"]
    vis_union = 1.0 - (1.0 - vis1) * (1.0 - vis2)   # observed if either camera sees it
    fig2, ax2 = plt.subplots(1, 3, figsize=(15, 5))
    for a, f, t in [(ax2[0], vis1, "cam1 (south) only"), (ax2[1], vis2, "cam2 (north) only"),
                    (ax2[2], vis_union, "2-cam union (fills shadows)")]:
        a.imshow(np.where(drive_mask, f, np.nan), origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1)
        a.set_title(t); a.set_xlabel("x"); a.set_ylabel("y")
    fig2.suptitle("Multi-camera: observability is the UNION over cameras", fontsize=12)
    fig2.tight_layout(); fig2.savefig(OUT / "multicam_union.png", dpi=115)
    cov1 = float((vis1[drive_mask] > 0.5).mean()); covU = float((vis_union[drive_mask] > 0.5).mean())
    print(f"multi-cam: driveable cells with visibility>0.5  1-cam {cov1:.2f} -> 2-cam {covU:.2f}")
    print(f"wrote {OUT/'multicam_union.png'}")


if __name__ == "__main__":
    main()
