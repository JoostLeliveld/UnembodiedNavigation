#!/usr/bin/env python3
"""Build the offline geometry-derived observability prior for the warehouse.

Runs the full chain against the packaged GP artifact + campaign config, with no
ROS / Gazebo / YOLO / simulator. Emits maps, a per-cell CSV, validation figures,
and a VALIDATION.md that quantifies how well geometry explains the empirically
learned GP (Stage 1) and fuses the two into a gap-filled posterior (Stage 2).

Example:
    python3 scripts/geometry_visibility/build_geometry_visibility_prior.py \
        --gp-artifact paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz \
        --campaign-config scripts/visibility_comparison/warehouse_visibility_campaign.yaml \
        --out logs/geometry_visibility_prior/warehouse_aws_v0
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO / "src" / "unav_common"))
sys.path.insert(0, str(_HERE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import geometry_visibility as gv  # noqa: E402


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 3:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return _pearson(ra, rb)


def load_campaign_config(path: str):
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    driveable = json.loads(cfg["driveable_geometry_json"])
    return {
        "driveable_prisms": gv.prisms_from_json(driveable),
        "r_visible_uv": float(cfg.get("r_visible_uv", 2.5)),
        "r_miss_uv": float(cfg.get("r_miss_uv", 40.0)),
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _extent(xs, ys):
    return [float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())]


def _overlay_prisms(ax, prisms, edgecolor, lw=1.0, fill=False, facecolor="none", alpha=1.0):
    for p in prisms:
        ax.add_patch(
            Rectangle(
                (p.xmin, p.ymin), p.xmax - p.xmin, p.ymax - p.ymin,
                fill=fill, facecolor=facecolor, edgecolor=edgecolor, lw=lw, alpha=alpha,
            )
        )


def _field_fig(path, field, xs, ys, title, cmap, meta, driveable=None, occ=None,
               vmin=None, vmax=None, cbar_label="", mask=None, points=None):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    plot = np.array(field, dtype=float)
    if mask is not None:
        plot = np.where(mask, plot, np.nan)
    im = ax.imshow(plot, origin="lower", extent=_extent(xs, ys), cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
    if occ is not None:
        _overlay_prisms(ax, occ, edgecolor="k", lw=1.1)
    if driveable is not None:
        _overlay_prisms(ax, driveable, edgecolor="#00b050", lw=0.9)
    cam = meta["camera_pos"]
    ax.plot([cam[0]], [cam[1]], marker="*", color="#ff2d2d", markersize=16,
            markeredgecolor="k", label="camera", zorder=5)
    if points is not None:
        for (px, py, lbl) in points:
            ax.plot([px], [py], "o", color="w", markeredgecolor="k", markersize=7, zorder=6)
            ax.annotate(lbl, (px, py), textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color="k",
                        bbox=dict(boxstyle="round,pad=0.15", fc="w", ec="k", alpha=0.7))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.colorbar(im, ax=ax, shrink=0.85, label=cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------
def build(args):
    out = pathlib.Path(args.out)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    # --- Step 1: load artifact + config ------------------------------------
    meta = gv.load_gp_artifact_geometry(args.gp_artifact)
    cfg = load_campaign_config(args.campaign_config)
    xs, ys = meta["xs"], meta["ys"]
    occ = meta["prisms"]
    drive = cfg["driveable_prisms"]
    r_vis, r_miss = cfg["r_visible_uv"], cfg["r_miss_uv"]
    z_marker = float(args.z_marker)

    assert np.all(np.diff(xs) > 0) and np.all(np.diff(ys) > 0), "grid must be increasing"
    assert r_vis < r_miss, "r_visible_uv must be < r_miss_uv"
    cam = gv.make_camera(meta)

    drive_mask = gv.in_any_prism(xs, ys, drive)
    log = {"artifact": str(args.gp_artifact), "z_marker": z_marker,
           "geometry_sha256": meta["geometry_sha256"],
           "n_occ_prisms": len(occ), "n_drive_prisms": len(drive),
           "r_visible_uv": r_vis, "r_miss_uv": r_miss,
           "grid": [len(xs), len(ys)], "driveable_cells": int(drive_mask.sum())}

    # Fig 01: geometry overlay
    _field_fig(figs / "01_geometry_overlay.png", drive_mask.astype(float), xs, ys,
               "01 Geometry overlay (green=driveable, black=occluders, star=camera)",
               "Greens", meta, driveable=drive, occ=occ, cbar_label="driveable")

    # --- Step 2: height map ------------------------------------------------
    hm = gv.build_height_map(xs, ys, occ)
    np.savez(out / "height_map.npz", xs=xs, ys=ys, h_max=hm["h_max"], known=hm["known"],
             occ_conf=hm["occ_conf"], resolution=hm["resolution"], origin_xy=hm["origin_xy"],
             geometry_json_sha256=meta["geometry_sha256"])
    _field_fig(figs / "02_height_map.png", hm["h_max"], xs, ys,
               f"02 2.5D height map  (max {hm['h_max'].max():.2f} m)", "cividis", meta,
               occ=occ, cbar_label="h_max [m]")
    log["h_max_max"] = float(hm["h_max"].max())
    log["obstacle_cells"] = int((hm["h_max"] > 0).sum())

    # --- Step 3: FOV + projection ------------------------------------------
    fov = gv.fov_projection_grid(cam, xs, ys, z_marker)
    _field_fig(figs / "03_fov_mask.png", (fov["fov_mask"] & drive_mask).astype(float), xs, ys,
               "03 In-FOV driveable cells", "Blues", meta, driveable=drive, occ=occ,
               cbar_label="in FOV")
    log["driveable_in_fov"] = int((fov["fov_mask"] & drive_mask).sum())

    # --- Step 4: raycast clearance -----------------------------------------
    clear = gv.raycast_min_clearance(cam, xs, ys, occ, z_marker, n_samples=args.ray_samples)
    _field_fig(figs / "04_clearance_map.png", clear, xs, ys,
               "04 Raycast min clearance (m): <0 occluded", "RdBu", meta,
               driveable=drive, occ=occ, vmin=-1.0, vmax=1.0,
               cbar_label="min clearance [m]", mask=fov["fov_mask"])

    # --- Step 5: components + visibility -----------------------------------
    jac = gv.projection_jacobian_scale(cam, xs, ys, z_marker)
    comp = gv.compute_visibility(
        fov_mask=fov["fov_mask"], min_clearance=clear, px_per_m_min=jac["px_per_m_min"],
        u=fov["u"], v=fov["v"], img_w=cam.img_width, img_h=cam.img_height,
        tau_clearance=args.tau_clearance, use_range=True, use_boundary=True,
    )
    score = comp["visibility_score"]
    _field_fig(figs / "05_raw_visibility.png", score, xs, ys,
               "05 Geometry visibility score [0,1]", "viridis", meta,
               driveable=drive, occ=occ, vmin=0, vmax=1,
               cbar_label="visibility", mask=drive_mask)

    # Fig 06: components side by side
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, key, ttl, cm in [
        (axes[0, 0], "f_occ", "f_occ (occlusion)", "magma"),
        (axes[0, 1], "f_range", "f_range (range/obliquity)", "magma"),
        (axes[1, 0], "f_boundary", "f_boundary (image edge)", "magma"),
        (axes[1, 1], "visibility_score", "visibility_score (product)", "viridis"),
    ]:
        f = np.where(drive_mask, comp[key], np.nan)
        im = ax.imshow(f, origin="lower", extent=_extent(xs, ys), cmap=cm, vmin=0, vmax=1,
                       aspect="equal", interpolation="nearest")
        _overlay_prisms(ax, occ, edgecolor="k", lw=0.8)
        ax.plot([meta["camera_pos"][0]], [meta["camera_pos"][1]], "*", color="#ff2d2d",
                markersize=12, markeredgecolor="k")
        ax.set_title(ttl, fontsize=10); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("06 Visibility components (driveable cells)", fontsize=12)
    fig.tight_layout()
    fig.savefig(figs / "06_visibility_components.png", dpi=120)
    plt.close(fig)

    # --- Step 6: label noise -----------------------------------------------
    label_noise = gv.compute_label_noise(clear, comp["f_boundary"], tau_clearance=args.tau_clearance)

    # --- Step 7: trust -> R_plan -------------------------------------------
    r_std, r_var = gv.trust_to_r_plan(score, r_vis, r_miss)
    _field_fig(figs / "07_geometry_r_plan_std.png", r_std, xs, ys,
               f"07 R_plan std [px]  (visible {r_vis:.1f} -> miss {r_miss:.0f})", "inferno_r",
               meta, driveable=drive, occ=occ, vmin=r_vis, vmax=r_miss,
               cbar_label="R_plan std [px]", mask=drive_mask)

    # Fig 08: explicit 2x2 R_plan matrices at three example cells
    examples = _pick_examples(xs, ys, score, drive_mask & fov["fov_mask"])
    _plot_r_plan_examples(figs / "08_r_plan_matrix_examples.png", examples, score, r_std, r_vis, r_miss)

    # --- Step 8 / Stage 1: compare to empirical GP -------------------------
    stage1 = _stage1_explanatory(meta, score, drive_mask, fov["fov_mask"], figs, log)

    # --- Stage 2: fusion ----------------------------------------------------
    stage2 = _stage2_fusion(meta, score, drive_mask, stage1, figs, args)

    # --- Save prior npz + CSV ----------------------------------------------
    np.savez(
        out / "geometry_visibility_prior.npz",
        xs=xs, ys=ys, visibility_score_map=score, label_noise_map=label_noise,
        clearance_map=clear, unknown_fraction_map=np.zeros_like(score),
        fov_mask=fov["fov_mask"], r_plan_std_map=r_std, r_plan_var_map=r_var,
        current_gp_trust_map=meta.get("P_mean_map", np.full_like(score, np.nan)),
        # The three fields a planning figure needs, so it does not have to refit anything:
        # what the survey learned, what geometry predicts on the same scale, and the two
        # combined. NaN-filled when there is no GP to compare against.
        empirical_gp_prob=meta.get("P_mean_map", np.full_like(score, np.nan)),
        calibrated_geometry_prob=stage1.get("calibrated_geometry_prob",
                                            np.full_like(score, np.nan)),
        fused_posterior_prob=stage2.get("posterior_prob", np.full_like(score, np.nan)),
        geometry_weight_map=stage2.get("geometry_weight", np.full_like(score, np.nan)),
        px_per_m_min=jac["px_per_m_min"], driveable_mask=drive_mask,
        z_marker=z_marker, geometry_sha256=meta["geometry_sha256"],
    )
    _write_csv(out / "raycast_visibility.csv", xs, ys, drive_mask, fov, clear, comp,
               jac, label_noise, r_std, r_var)

    _write_validation(out, log, stage1, stage2, meta, xs, ys, score, clear, drive_mask,
                      fov["fov_mask"], r_std, r_vis, r_miss)

    print(f"[done] wrote outputs to {out}")
    return log, stage1, stage2


def _pick_examples(xs, ys, score, valid):
    s = np.where(valid, score, np.nan)
    flat = s.ravel()
    order = np.argsort(np.nan_to_num(flat, nan=-1))
    good = order[-1]
    finite = np.where(np.isfinite(flat))[0]
    vals = flat[finite]
    med_idx = finite[np.argsort(np.abs(vals - np.nanmedian(vals)))[0]]
    low_idx = finite[np.argsort(vals)[0]]
    out = []
    for idx, lbl in [(good, "high"), (med_idx, "medium"), (low_idx, "low")]:
        iy, ix = np.unravel_index(idx, s.shape)
        out.append((lbl, float(xs[ix]), float(ys[iy]), float(s[iy, ix])))
    return out


def _plot_r_plan_examples(path, examples, score, r_std, r_vis, r_miss):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    for ax, (lbl, px, py, sc) in zip(axes, examples):
        std, _ = gv.trust_to_r_plan(sc, r_vis, r_miss)
        M = gv.r_plan_matrix(std)
        ax.axis("off")
        ax.set_title(f"{lbl}  @({px:.2f},{py:.2f})\ntrust={sc:.2f}, std={float(std):.1f}px", fontsize=10)
        txt = (f"R_plan =\n[[{M[0,0]:.1f}, {M[0,1]:.1f}],\n [{M[1,0]:.1f}, {M[1,1]:.1f}]]  px^2")
        ax.text(0.5, 0.5, txt, ha="center", va="center", family="monospace", fontsize=12,
                bbox=dict(boxstyle="round,pad=0.4", fc="#eef", ec="k"))
    fig.suptitle("08 R_plan measurement covariance at example cells", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _stage1_explanatory(meta, score, drive_mask, fov_mask, figs, log):
    """Quantify how well geometry visibility explains the empirical detection-rate GP."""
    out = {"available": "P_mean_map" in meta}
    if not out["available"]:
        return out
    emp = meta["P_mean_map"]  # detection rate in [0,1]
    base = drive_mask & fov_mask & np.isfinite(emp)

    # Well-sampled restriction: keep only cells the GP determines confidently
    # (low posterior std), so the correlation is measured against real detector
    # evidence rather than GP prior-fallback. Falls back to distance if no std.
    near = base.copy()
    fstd_thr = None
    if "F_std_map" in meta:
        fstd = meta["F_std_map"]
        fstd_thr = float(np.percentile(fstd[base], 60.0))
        near = base & (fstd <= fstd_thr)
    elif "X_train" in meta:
        gx, gy = np.meshgrid(meta["xs"], meta["ys"])
        d2min = np.full(gx.shape, np.inf)
        for (tx, ty) in meta["X_train"]:
            d2min = np.minimum(d2min, (gx - tx) ** 2 + (gy - ty) ** 2)
        near = base & (d2min <= (0.35 ** 2))

    def stats(mask):
        g = score[mask]; e = emp[mask]
        return {"n": int(mask.sum()), "pearson": _pearson(g, e), "spearman": _spearman(g, e)}

    all_stats = stats(base)
    near_stats = stats(near)
    near_stats["fstd_threshold"] = fstd_thr

    # Affine calibration geometry->empirical over near-data cells, residual map.
    g = score[near]; e = emp[near]
    A = np.vstack([g, np.ones_like(g)]).T
    coef, *_ = np.linalg.lstsq(A, e, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    pred_full = np.clip(a * score + b, 0.0, 1.0)
    ss_res = float(np.sum((e - (a * g + b)) ** 2))
    ss_tot = float(np.sum((e - e.mean()) ** 2)) if e.size else float("nan")
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    residual = np.where(base, emp - pred_full, np.nan)

    out.update({"all": all_stats, "near_data": near_stats, "affine_a": a, "affine_b": b,
                "affine_r2": r2, "calibrated_geometry_prob": pred_full})
    log["stage1"] = {"all": all_stats, "near_data": near_stats, "affine_r2": r2}

    # Fig 09: geometry vs empirical vs residual
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ext = _extent(meta["xs"], meta["ys"])
    for ax, f, ttl, cm, vmm in [
        (axes[0], np.where(base, pred_full, np.nan), "geometry prior (calibrated prob)", "viridis", (0, 1)),
        (axes[1], np.where(base, emp, np.nan), "current YOLO GP P_mean", "viridis", (0, 1)),
        (axes[2], residual, "residual (empirical - geometry)", "RdBu", (-0.6, 0.6)),
    ]:
        im = ax.imshow(f, origin="lower", extent=ext, cmap=cm, vmin=vmm[0], vmax=vmm[1],
                       aspect="equal", interpolation="nearest")
        _overlay_prisms(ax, meta["prisms"], edgecolor="k", lw=0.7)
        ax.plot([meta["camera_pos"][0]], [meta["camera_pos"][1]], "*", color="#ff2d2d",
                markersize=12, markeredgecolor="k")
        ax.set_title(ttl, fontsize=10); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(
        f"09 Geometry prior vs YOLO GP  |  Spearman(all)={all_stats['spearman']:.2f} "
        f"well-sampled={near_stats['spearman']:.2f}  affine R^2={r2:.2f}",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(figs / "09_current_yolo_gp_vs_geometry_prior.png", dpi=120)
    plt.close(fig)
    return out


def _stage2_fusion(meta, score, drive_mask, stage1, figs, args):
    """Precision-weighted fusion: geometry prior + empirical GP -> posterior."""
    out = {"available": stage1.get("available", False) and "F_mean_map" in meta}
    if not out["available"]:
        return out
    xs, ys = meta["xs"], meta["ys"]
    # Geometry prior in logit space, calibrated to the empirical detection-rate scale.
    prior_prob = stage1["calibrated_geometry_prob"]
    prior_logit = _logit(prior_prob)
    sigma_prior = float(args.sigma_prior)  # logit-space prior std
    tau_prior = np.full_like(score, 1.0 / sigma_prior ** 2)

    # Empirical likelihood already lives in logit space (F_mean_map), precision 1/std^2.
    emp_logit = meta["F_mean_map"]
    emp_std = np.clip(meta["F_std_map"], 1e-3, None)
    tau_emp = 1.0 / emp_std ** 2

    post_logit = (tau_prior * prior_logit + tau_emp * emp_logit) / (tau_prior + tau_emp)
    post_prob = gv._sigmoid(post_logit)
    geom_weight = tau_prior / (tau_prior + tau_emp)  # how much geometry contributes

    ext = _extent(xs, ys)
    mask = drive_mask
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, f, ttl, cm, vmm in [
        (axes[0], np.where(mask, post_prob, np.nan), "fused posterior detection prob", "viridis", (0, 1)),
        (axes[1], np.where(mask, geom_weight, np.nan), "geometry weight (gap-fill)", "cividis", (0, 1)),
        (axes[2], np.where(mask, post_prob - meta["P_mean_map"], np.nan),
         "posterior - empirical (geometry correction)", "RdBu", (-0.5, 0.5)),
    ]:
        im = ax.imshow(f, origin="lower", extent=ext, cmap=cm, vmin=vmm[0], vmax=vmm[1],
                       aspect="equal", interpolation="nearest")
        _overlay_prisms(ax, meta["prisms"], edgecolor="k", lw=0.7)
        ax.plot([meta["camera_pos"][0]], [meta["camera_pos"][1]], "*", color="#ff2d2d",
                markersize=12, markeredgecolor="k")
        ax.set_title(ttl, fontsize=10); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("10 Stage 2 fusion: geometry fills where the empirical GP is uncertain",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(figs / "10_fusion_posterior.png", dpi=120)
    plt.close(fig)

    out.update({
        "posterior_prob": post_prob,
        "geometry_weight": geom_weight,
        "sigma_prior": sigma_prior,
        "median_geom_weight": float(np.nanmedian(np.where(mask, geom_weight, np.nan))),
        "max_geom_weight": float(np.nanmax(np.where(mask, geom_weight, np.nan))),
        "gap_cells_geom_dominant": int(np.sum((geom_weight > 0.5) & mask)),
    })
    return out


def _write_csv(path, xs, ys, drive_mask, fov, clear, comp, jac, label_noise, r_std, r_var):
    gx, gy = np.meshgrid(xs, ys)
    rows = ["x,y,in_driveable,in_fov,u,v,min_clearance_m,f_fov,f_occ,f_range,f_boundary,"
            "visibility_score,label_noise,r_plan_std_px,r_plan_var_px2,reason"]
    ny, nx = gx.shape
    for iy in range(ny):
        for ix in range(nx):
            if not drive_mask[iy, ix]:
                continue
            infov = bool(fov["fov_mask"][iy, ix])
            reason = "ok" if infov else "outside_fov"
            if infov and clear[iy, ix] < 0:
                reason = "occluded"
            rows.append(
                f"{gx[iy,ix]:.3f},{gy[iy,ix]:.3f},1,{int(infov)},"
                f"{fov['u'][iy,ix]:.1f},{fov['v'][iy,ix]:.1f},{clear[iy,ix]:.3f},"
                f"{comp['f_fov'][iy,ix]:.3f},{comp['f_occ'][iy,ix]:.3f},"
                f"{comp['f_range'][iy,ix]:.3f},{comp['f_boundary'][iy,ix]:.3f},"
                f"{comp['visibility_score'][iy,ix]:.4f},{label_noise[iy,ix]:.4f},"
                f"{r_std[iy,ix]:.2f},{r_var[iy,ix]:.2f},{reason}"
            )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_validation(out, log, stage1, stage2, meta, xs, ys, score, clear, drive_mask,
                      fov_mask, r_std, r_vis, r_miss):
    # Five hand-picked probe points across regimes.
    probes = [
        ("open apron", 0.0, -3.1),
        ("lower aisle A1", -3.0, -1.0),
        ("mid aisle A3", 1.0, 1.0),
        ("upper cross aisle", 0.0, 4.5),
        ("east far corner", 4.9, 4.0),
    ]
    lines = ["# Geometry Visibility Prior — VALIDATION", "",
             f"Artifact: `{log['artifact']}`  ",
             f"geometry sha256: `{log['geometry_sha256'][:16]}...`  ",
             f"z_marker: {log['z_marker']} m  | grid {log['grid'][0]}x{log['grid'][1]}"
             f"  | driveable cells {log['driveable_cells']} ({log['driveable_in_fov']} in FOV)",
             "", "## Geometry sanity", "",
             f"- occluder prisms: {log['n_occ_prisms']}, max height {log['h_max_max']:.2f} m",
             f"- obstacle cells: {log['obstacle_cells']}", "",
             "## Hand-labelled probe points", "",
             "| region | x | y | in FOV | min clearance [m] | visibility | R_plan std [px] |",
             "|---|---|---|---|---|---|---|"]
    for name, px, py in probes:
        iy, ix = gv.world_to_grid(xs, ys, px, py)
        lines.append(
            f"| {name} | {px:.2f} | {py:.2f} | {int(bool(fov_mask[iy,ix]))} | "
            f"{clear[iy,ix]:.3f} | {score[iy,ix]:.2f} | {r_std[iy,ix]:.1f} |")

    if stage1.get("available"):
        s = stage1
        lines += ["", "## Stage 1 — does geometry explain the learned GP?", "",
                  f"- all driveable in-FOV cells (n={s['all']['n']}): "
                  f"Pearson {s['all']['pearson']:.3f}, Spearman {s['all']['spearman']:.3f}",
                  f"- well-sampled cells (GP F_std ≤ {s['near_data'].get('fstd_threshold', float('nan')):.2f}, "
                  f"n={s['near_data']['n']}): Pearson {s['near_data']['pearson']:.3f}, "
                  f"Spearman {s['near_data']['spearman']:.3f}",
                  f"- affine calibration p ≈ {s['affine_a']:.3f}·score + {s['affine_b']:.3f}, "
                  f"R² = {s['affine_r2']:.3f}", "",
                  "Interpretation: a strong rank correlation means geometry *orders* cells the",
                  "way the detector reliability does. Residuals (fig 09, right) mark where",
                  "appearance/range/calibration diverge from pure geometry — documented, not tuned away."]
        # agreement / disagreement regions from the residual sign
        emp = meta["P_mean_map"]
        base = drive_mask & fov_mask & np.isfinite(emp)
        pred = np.clip(s["affine_a"] * score + s["affine_b"], 0, 1)
        resid = np.where(base, emp - pred, np.nan)
        lines += ["", "### Largest agreement / disagreement cells", ""]
        flat = resid.ravel()
        order = np.argsort(np.nan_to_num(np.abs(flat), nan=-1))
        gx, gy = np.meshgrid(xs, ys)
        agree = order[np.isfinite(flat[order])][:3]
        disagree = order[np.isfinite(flat[order])][-3:][::-1]
        lines.append("Agreement (|residual| smallest):")
        for idx in agree:
            iy, ix = np.unravel_index(idx, resid.shape)
            lines.append(f"- ({gx.ravel()[idx]:.2f},{gy.ravel()[idx]:.2f}) residual {flat[idx]:+.3f}")
        lines.append("")
        lines.append("Disagreement (|residual| largest):")
        for idx in disagree:
            lines.append(f"- ({gx.ravel()[idx]:.2f},{gy.ravel()[idx]:.2f}) residual {flat[idx]:+.3f}")

    if stage2.get("available"):
        lines += ["", "## Stage 2 — geometry+GP fusion (gap filling)", "",
                  f"- prior logit std σ = {stage2['sigma_prior']}",
                  f"- median geometry weight over driveable cells: {stage2['median_geom_weight']:.2f}",
                  f"- cells where geometry dominates (weight>0.5): {stage2['gap_cells_geom_dominant']}",
                  "",
                  "Where the empirical GP is data-starved (high F_std), the posterior leans on",
                  "geometry; where YOLO has dense evidence, the data dominates (fig 10)."]

    lines += ["", "## Downstream contract", "",
              "- This artifact is an *observability prior*, not a planner result.",
              "- No planner code consumes it yet.",
              "- R_plan is pixel-space measurement covariance; endpoints recover",
              f"  r_visible_uv={r_vis} px (trust 1) and r_miss_uv={r_miss} px (trust 0)."]
    (out / "VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gp-artifact",
                    default="paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz")
    ap.add_argument("--campaign-config",
                    default="scripts/visibility_comparison/warehouse_visibility_campaign.yaml")
    ap.add_argument("--world-profile", default="src/experiments/config/world_profiles.yaml")
    ap.add_argument("--world", default="warehouse_aws.world.sdf")
    ap.add_argument("--out", default="logs/geometry_visibility_prior/warehouse_aws_v0")
    ap.add_argument("--z-marker", type=float, default=0.35)
    ap.add_argument("--tau-clearance", type=float, default=0.10)
    ap.add_argument("--ray-samples", type=int, default=48)
    ap.add_argument("--sigma-prior", type=float, default=1.5)
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
