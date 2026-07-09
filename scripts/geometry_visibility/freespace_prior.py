#!/usr/bin/env python3
"""What can the geometry prior recover from ONLY a driveable-region map?

Three levels of geometry knowledge, same camera + same chain:
  TRUE       : real SDF obstacle boxes with true heights (Stage-0 baseline)
  FREESPACE  : obstacles = interior holes of the driveable map, assumed height h0
               (no obstacle presence/height given; inferred from free-space negative space)
  NONE       : flat floor, no occlusion knowledge (camera terms only)

Reports how well each explains the empirical YOLO GP (Stage-1 Spearman / affine R^2).
"""
from __future__ import annotations
import json, pathlib, sys
from collections import deque
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "unav_common"))
sys.path.insert(0, str(REPO / "scripts" / "geometry_visibility"))
import geometry_visibility as gv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ART = REPO / "paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz"
CFG = REPO / "scripts/visibility_comparison/warehouse_visibility_campaign.yaml"
OUT = REPO / "logs" / "geometry_visibility_prior" / "demo"
OUT.mkdir(parents=True, exist_ok=True)
Z_MARKER, TAU = 0.35, 0.10


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def affine_r2(g, e):
    A = np.vstack([g, np.ones_like(g)]).T
    coef, *_ = np.linalg.lstsq(A, e, rcond=None)
    pred = A @ coef
    ss_res = np.sum((e - pred) ** 2); ss_tot = np.sum((e - e.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def interior_holes(free_mask):
    """Non-driveable cells NOT reachable from the grid border through non-driveable
    space -> bounded 'holes' surrounded by free space = inferred obstacle footprints."""
    nd = ~free_mask
    ny, nx = nd.shape
    seen = np.zeros_like(nd); dq = deque()
    for i in range(ny):
        for j in (0, nx - 1):
            if nd[i, j] and not seen[i, j]:
                seen[i, j] = True; dq.append((i, j))
    for j in range(nx):
        for i in (0, ny - 1):
            if nd[i, j] and not seen[i, j]:
                seen[i, j] = True; dq.append((i, j))
    while dq:
        i, j = dq.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < ny and 0 <= nj < nx and nd[ni, nj] and not seen[ni, nj]:
                seen[ni, nj] = True; dq.append((ni, nj))
    return nd & ~seen


def main():
    meta = gv.load_gp_artifact_geometry(ART)
    import yaml
    cfg = yaml.safe_load(open(CFG))
    drive = gv.prisms_from_json(json.loads(cfg["driveable_geometry_json"]))
    xs, ys = meta["xs"], meta["ys"]
    cam = gv.make_camera(meta)
    drive_mask = gv.in_any_prism(xs, ys, drive)

    # obstacle-independent terms (identical across all three)
    fov = gv.fov_projection_grid(cam, xs, ys, Z_MARKER)
    jac = gv.projection_jacobian_scale(cam, xs, ys, Z_MARKER)
    emp = meta["P_mean_map"]
    base = drive_mask & fov["fov_mask"] & np.isfinite(emp)

    # TRUE height map (SDF)
    h_true = gv.build_height_map(xs, ys, meta["prisms"])["h_max"]
    # FREESPACE inferred obstacle footprints (holes of the driveable map)
    holes = interior_holes(drive_mask)

    def score_for(height_map, use_occ=True):
        clear = _raycast_from_heightmap(cam, xs, ys, height_map, Z_MARKER) if use_occ else None
        min_clear = clear if use_occ else np.full_like(h_true, 10.0)
        comp = gv.compute_visibility(
            fov_mask=fov["fov_mask"], min_clearance=min_clear,
            px_per_m_min=jac["px_per_m_min"], u=fov["u"], v=fov["v"],
            img_w=cam.img_width, img_h=cam.img_height, tau_clearance=TAU,
            use_range=True, use_boundary=True)
        return comp["visibility_score"], min_clear

    results = {}
    variants = [("TRUE (SDF heights)", h_true, True)]
    for h0 in (1.0, 1.5, 1.9):
        variants.append((f"FREESPACE holes @ h0={h0}m", holes.astype(float) * h0, True))
    variants.append(("NONE (camera-only, no occ)", np.zeros_like(h_true), False))

    for name, hmap, use_occ in variants:
        score, _ = score_for(hmap, use_occ=use_occ)
        g, e = score[base], emp[base]
        results[name] = (spearman(g, e), affine_r2(g, e))

    print(f"{'geometry source':32s}  Spearman   affineR2")
    print("-" * 56)
    for name, (sp, r2) in results.items():
        print(f"{name:32s}   {sp:5.3f}     {r2:5.3f}")

    # coverage of inferred holes vs true obstacles
    true_obst = h_true > 0
    inter = (holes & true_obst).sum()
    print(f"\ninferred-hole cells: {holes.sum()}, true-obstacle cells: {true_obst.sum()}, "
          f"overlap: {inter} ({100*inter/max(true_obst.sum(),1):.0f}% of true recovered, "
          f"{100*inter/max(holes.sum(),1):.0f}% of holes are real)")

    # figure: true heights vs inferred footprints
    ext = [xs.min(), xs.max(), ys.min(), ys.max()]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(np.where(h_true > 0, h_true, np.nan), origin="lower", extent=ext,
                 cmap="cividis"); ax[0].set_title("TRUE obstacle heights (SDF)")
    ax[1].imshow(np.where(holes, 1.0, np.nan), origin="lower", extent=ext, cmap="Reds",
                 vmin=0, vmax=1); ax[1].set_title("FREESPACE-inferred footprints (holes)")
    ax[2].imshow(drive_mask.astype(float), origin="lower", extent=ext, cmap="Greens")
    ax[2].imshow(np.where(holes, 1.0, np.nan), origin="lower", extent=ext, cmap="Reds", alpha=0.6)
    ax[2].set_title("driveable map (green) + inferred obstacles (red)")
    for a in ax:
        a.plot([meta["camera_pos"][0]], [meta["camera_pos"][1]], "*", color="b", markersize=12)
        a.set_xlabel("x [m]"); a.set_ylabel("y [m]")
    fig.tight_layout(); fig.savefig(OUT / "freespace_vs_true.png", dpi=120)
    print(f"\nwrote {OUT/'freespace_vs_true.png'}")


def _raycast_from_heightmap(cam, xs, ys, height_map, z_marker, n=48):
    """Raycast against a rasterised height map (nearest-cell lookup) instead of prisms."""
    gx, gy = np.meshgrid(xs, ys)
    targets = np.stack([gx, gy, np.full_like(gx, z_marker)], -1).reshape(-1, 3)
    cam_pos = np.asarray(cam.cam_pos)
    t = np.linspace(0.02, 0.98, n).reshape(1, -1, 1)
    s = cam_pos.reshape(1, 1, 3) + t * (targets[:, None, :] - cam_pos.reshape(1, 1, 3))
    ix = np.clip(np.searchsorted(xs, s[..., 0]) - 0, 0, len(xs) - 1)
    iy = np.clip(np.searchsorted(ys, s[..., 1]) - 0, 0, len(ys) - 1)
    obst = height_map[iy, ix]
    clear = (s[..., 2] - obst).min(axis=1)
    return clear.reshape(gx.shape)


if __name__ == "__main__":
    main()
