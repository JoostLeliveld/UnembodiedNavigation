#!/usr/bin/env python3
"""Audit the DEPLOYED along-bearing projection correction on real captures.

Loads every capture that carries both a camera pixel observation and evaluation
ground truth, reprojects each pixel through the exact runtime path
(``reliability.projection._project_pixel_to_world``) twice — once RAW and once
through the deployed ``projection_calibration_v2`` constants — and characterises
what REMAINS after the deployed correction.

The along-bearing bias model itself is pre-existing deployed work and is NOT
refit here; it is loaded and applied.  See this study's README.

Ground truth is evaluation-only: it is used to *measure* residuals and never
enters a projection, a correction, or a covariance.

Outputs -> logs/studies/external_camera_bias_model/exp1_residual_characterization/
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
sys.path.insert(0, str(REPO / "src" / "reliability"))
sys.path.insert(0, str(REPO / "src" / "unav_common"))
sys.path.insert(0, str(REPO / "scripts" / "shared"))
sys.path.insert(
    0, str(REPO / "experiments" / "multicamera_commissioning_bigwarehouse" / "tools")
)

from metrics import spearman, binned  # noqa: E402  (THE shared scoring library)
from reliability.projection import (  # noqa: E402
    camera_model_from_world,
    load_projection_calibration,
    _project_pixel_to_world,
)
import attach_evaluation_truth as AET  # noqa: E402  (canonical nearest-stamp join)

# ---------------------------------------------------------------- configuration

WORLD_SDF = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
DEPLOYED_CALIB = (
    REPO
    / "logs/studies/multicamera_commissioning_bigwarehouse/projection_calibration_v2"
    / "projection_calibration.json"
)
OUT = REPO / "logs/studies/external_camera_bias_model/exp1_residual_characterization"

CAMERAS = ("camera_A", "camera_B", "camera_C", "camera_D")
MODEL_INCLUDES = {
    "camera_A": "external_camera",
    "camera_B": "external_camera_b",
    "camera_C": "external_camera_c",
    "camera_D": "external_camera_d",
}
# Verified by reprojection: reproduces every recorded pred_world_x/y to 0.000000 m.
CONTACT_Z_M = 0.05
TRUTH_TOL_S = 0.05
# scripts/visibility_comparison/*.yaml -> pixel_max_correction_jump_m
JUMP_LIMIT_M = 0.5
# scripts/geometry_visibility/make_warehouse_full.py SITE_X0/X1/Y0/Y1
SITE = (-11.20, 11.50, -8.60, 8.60)
MIN_SPAN_FOR_SLOPE_M = 3.0  # fit_projection_calibration.MIN_SPAN_FOR_SLOPE_M

CAPTURES = {
    "smoke1_20260716": {
        "dir": REPO
        / "logs/studies/multicamera_commissioning_bigwarehouse/gt_validation_smoke_20260716",
        "csv_subdir": "evaluation_inputs",
        "truth": "evaluation_only/ground_truth.csv",
    },
    "smoke2_20260716": {
        "dir": REPO
        / "logs/studies/multicamera_commissioning_bigwarehouse/gt_validation_smoke2_20260716",
        "csv_subdir": "evaluation_inputs",
        "truth": "evaluation_only/ground_truth.csv",
    },
    "fusion_handover_20260721": {
        "dir": REPO
        / "logs/studies/multicamera_fusion_extension/fusion_handover_real_20260721/data",
        "csv_subdir": "raw",
        "truth": "evaluation_only/ground_truth.csv",
    },
}

RNG = np.random.default_rng(20260730)
N_BOOT = 4000


# ---------------------------------------------------------------------- loading


def _load_truth(path: Path):
    """Nearest-stamp truth table. Tolerates GT files without a gt_yaw column."""
    stamps, poses = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                stamp = float(row["stamp"])
                pose = (
                    float(row["gt_x"]),
                    float(row["gt_y"]),
                    float(row.get("gt_yaw") or "nan"),
                )
            except (KeyError, TypeError, ValueError):
                continue
            stamps.append(stamp)
            poses.append(pose)
    paired = sorted(zip(stamps, poses))
    return [p[0] for p in paired], [p[1] for p in paired]


def load_samples(models, calib):
    """One record per detected, truth-matched observation, raw and corrected."""
    rows = []
    inventory = []
    for cap_name, cap in CAPTURES.items():
        truth_path = cap["dir"] / cap["truth"]
        stamps, poses = _load_truth(truth_path)
        for cam in CAMERAS:
            src = cap["dir"] / cap["csv_subdir"] / f"{cam}_perception.csv"
            n_rows = n_det = n_match = 0
            if not src.exists():
                inventory.append(
                    dict(
                        capture=cap_name, camera=cam, file=str(src.relative_to(REPO)),
                        rows=0, detected=0, truth_matched=0, status="missing",
                    )
                )
                continue
            cam_x = float(models[cam].cam_pos[0])
            cam_y = float(models[cam].cam_pos[1])
            cal = calib.get(cam, {"intercept_m": 0.0, "slope_per_m": 0.0})
            with src.open(newline="", encoding="utf-8") as handle:
                for rec in csv.DictReader(handle):
                    n_rows += 1
                    if rec.get("detected") != "1" or not rec.get("obs_u"):
                        continue
                    try:
                        u, v = float(rec["obs_u"]), float(rec["obs_v"])
                        stamp = float(rec["diag_stamp"])
                    except (TypeError, ValueError):
                        continue
                    if not (math.isfinite(u) and math.isfinite(v)):
                        continue
                    n_det += 1
                    truth = AET._nearest(stamps, poses, stamp, TRUTH_TOL_S)
                    if truth is None:
                        continue
                    raw = _project_pixel_to_world(
                        u, v, models[cam], contact_z_m=CONTACT_Z_M,
                        along_bearing_offset_m=0.0, along_bearing_slope_per_m=0.0,
                    )
                    cor = _project_pixel_to_world(
                        u, v, models[cam], contact_z_m=CONTACT_Z_M,
                        along_bearing_offset_m=cal["intercept_m"],
                        along_bearing_slope_per_m=cal["slope_per_m"],
                    )
                    if raw is None or cor is None:
                        continue
                    n_match += 1
                    tx, ty = truth[0], truth[1]
                    # Bearing basis referenced to TRUTH (evaluation convention of
                    # attach_evaluation_truth): using the projected point would
                    # rotate the basis by the very error under audit.
                    bx, by = tx - cam_x, ty - cam_y
                    rng_m = math.hypot(bx, by)
                    ux, uy = (bx / rng_m, by / rng_m) if rng_m > 1e-9 else (1.0, 0.0)
                    rec_out = dict(
                        capture=cap_name, camera=cam, stamp=stamp,
                        true_x=tx, true_y=ty, range_m=rng_m,
                        bearing_deg=math.degrees(math.atan2(by, bx)),
                        u=u, v=v,
                    )
                    for tag, pt in (("raw", raw), ("cor", cor)):
                        ex, ey = pt[0] - tx, pt[1] - ty
                        rec_out[f"{tag}_ex"] = ex
                        rec_out[f"{tag}_ey"] = ey
                        rec_out[f"{tag}_along"] = ex * ux + ey * uy
                        rec_out[f"{tag}_cross"] = -ex * uy + ey * ux
                        rec_out[f"{tag}_norm"] = math.hypot(ex, ey)
                        rec_out[f"{tag}_px"] = pt[0]
                        rec_out[f"{tag}_py"] = pt[1]
                    rows.append(rec_out)
            inventory.append(
                dict(
                    capture=cap_name, camera=cam, file=str(src.relative_to(REPO)),
                    rows=n_rows, detected=n_det, truth_matched=n_match,
                    status="ok" if n_match else ("no_detections" if not n_det else "no_truth"),
                )
            )
    return rows, inventory


# ------------------------------------------------------------------- statistics


def moving_block_bootstrap(values, n_boot=N_BOOT, block=None):
    """Block bootstrap of the mean of a (n,k) time-ordered array.

    Consecutive detections along a trajectory are strongly autocorrelated, so an
    i.i.d. bootstrap would understate the CI. Block length defaults to n**(1/3).
    """
    values = np.asarray(values, float)
    n = len(values)
    if n < 2:
        return np.full((n_boot, values.shape[1]), np.nan)
    block = block or max(1, int(round(n ** (1.0 / 3.0))))
    n_blocks = int(math.ceil(n / block))
    starts_max = max(1, n - block + 1)
    out = np.empty((n_boot, values.shape[1]))
    for i in range(n_boot):
        starts = RNG.integers(0, starts_max, size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        out[i] = values[idx].mean(axis=0)
    return out


def lag1_autocorr(x):
    x = np.asarray(x, float)
    if len(x) < 3:
        return math.nan
    x = x - x.mean()
    denom = float(np.dot(x, x))
    return float(np.dot(x[:-1], x[1:]) / denom) if denom > 0 else math.nan


def cov_summary(ex, ey):
    """Conditional covariance about the mean + eigen-decomposition."""
    pts = np.column_stack([ex, ey])
    if len(pts) < 2:
        return dict(sxx=math.nan, sxy=math.nan, syy=math.nan, eig_major=math.nan,
                    eig_minor=math.nan, major_deg=math.nan, sigma_iso=math.nan)
    cov = np.cov(pts, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    return dict(
        sxx=float(cov[0, 0]), sxy=float(cov[0, 1]), syy=float(cov[1, 1]),
        eig_major=float(max(vals[0], 0.0)), eig_minor=float(max(vals[1], 0.0)),
        major_deg=float(math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))),
        sigma_iso=float(math.sqrt(max(0.5 * (cov[0, 0] + cov[1, 1]), 0.0))),
    )


def describe(group, tag):
    """Bias / conditional covariance / bias-to-noise ratio, with block-bootstrap CI."""
    ex = np.array([r[f"{tag}_ex"] for r in group])
    ey = np.array([r[f"{tag}_ey"] for r in group])
    al = np.array([r[f"{tag}_along"] for r in group])
    cr = np.array([r[f"{tag}_cross"] for r in group])
    nrm = np.array([r[f"{tag}_norm"] for r in group])
    n = len(group)
    world = cov_summary(ex, ey)
    bear = cov_summary(al, cr)
    bias = np.array([ex.mean(), ey.mean()])
    bias_bear = np.array([al.mean(), cr.mean()])
    boot_w = moving_block_bootstrap(np.column_stack([ex, ey]))
    boot_b = moving_block_bootstrap(np.column_stack([al, cr]))
    boot_wn = np.linalg.norm(boot_w, axis=1)

    def ci(arr):
        return (float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5)))

    ratio_w = float(np.linalg.norm(bias) / world["sigma_iso"]) if world["sigma_iso"] else math.nan
    boot_ratio = boot_wn / world["sigma_iso"] if world["sigma_iso"] else np.full(N_BOOT, np.nan)
    return dict(
        n=n,
        bias_x=float(bias[0]), bias_y=float(bias[1]),
        bias_norm=float(np.linalg.norm(bias)),
        bias_x_ci=ci(boot_w[:, 0]), bias_y_ci=ci(boot_w[:, 1]),
        bias_norm_ci=ci(boot_wn),
        along_bias=float(bias_bear[0]), cross_bias=float(bias_bear[1]),
        along_bias_ci=ci(boot_b[:, 0]), cross_bias_ci=ci(boot_b[:, 1]),
        along_std=float(al.std(ddof=1)) if n > 1 else math.nan,
        cross_std=float(cr.std(ddof=1)) if n > 1 else math.nan,
        std_x=float(ex.std(ddof=1)) if n > 1 else math.nan,
        std_y=float(ey.std(ddof=1)) if n > 1 else math.nan,
        rmse=float(np.sqrt(np.mean(nrm ** 2))),
        mean_abs=float(nrm.mean()), median_abs=float(np.median(nrm)),
        p90_abs=float(np.percentile(nrm, 90)), max_abs=float(nrm.max()),
        ratio_world=ratio_w, ratio_world_ci=ci(boot_ratio),
        ratio_bearing=(
            float(abs(bias_bear[0]) / bear["sigma_iso"]) if bear["sigma_iso"] else math.nan
        ),
        lag1_along=lag1_autocorr(al),
        block_len=max(1, int(round(n ** (1.0 / 3.0)))),
        **{f"world_{k}": v for k, v in world.items()},
        **{f"bear_{k}": v for k, v in bear.items()},
    )


# ------------------------------------------------------- model comparison (audit)


def _fit_line(dist, err):
    """along = a + b*d, mirroring fit_projection_calibration._fit_line gating."""
    dist, err = np.asarray(dist, float), np.asarray(err, float)
    if len(dist) < 8 or (dist.max() - dist.min()) < MIN_SPAN_FOR_SLOPE_M:
        return float(err.mean()), 0.0
    b, a = np.polyfit(dist, err, 1)
    return float(a), float(b)


def blocked_folds(n, k=5):
    """Contiguous-in-time folds: random folds would leak across autocorrelation."""
    edges = np.linspace(0, n, k + 1).astype(int)
    for i in range(k):
        test = np.arange(edges[i], edges[i + 1])
        if len(test) == 0:
            continue
        train = np.concatenate([np.arange(0, edges[i]), np.arange(edges[i + 1], n)])
        if len(train) >= 4:
            yield train, test


def model_comparison(group, models, calib):
    """Held-out RMS residual for candidate correction models vs the deployed one.

    All fitted models are applied through the SAME runtime reprojection path so
    the comparison is exact rather than an approximation in residual space.
    """
    cam = group[0]["camera"]
    model = models[cam]
    cam_x, cam_y = float(model.cam_pos[0]), float(model.cam_pos[1])
    cal = calib.get(cam, {"intercept_m": 0.0, "slope_per_m": 0.0})
    n = len(group)
    # raw along-bearing error in the RUNTIME basis (projected point), as the
    # deployed fitter uses -- this is what the deployed constants were fit to.
    fit_d, fit_e = [], []
    for r in group:
        bx, by = r["raw_px"] - cam_x, r["raw_py"] - cam_y
        d = math.hypot(bx, by)
        fit_d.append(d)
        fit_e.append(((r["raw_px"] - r["true_x"]) * bx + (r["raw_py"] - r["true_y"]) * by) / d)
    fit_d, fit_e = np.array(fit_d), np.array(fit_e)

    def reproject(idx, intercept, slope):
        out = []
        for i in idx:
            r = group[i]
            p = _project_pixel_to_world(
                r["u"], r["v"], model, contact_z_m=CONTACT_Z_M,
                along_bearing_offset_m=intercept, along_bearing_slope_per_m=slope,
            )
            out.append(math.hypot(p[0] - r["true_x"], p[1] - r["true_y"]))
        return np.array(out)

    scores = {k: [] for k in ("M0_raw", "M1_world_const", "M2_bearing_const",
                              "M3_bearing_affine", "MD_deployed")}
    for train, test in blocked_folds(n):
        scores["M0_raw"].append(reproject(test, 0.0, 0.0))
        scores["MD_deployed"].append(reproject(test, cal["intercept_m"], cal["slope_per_m"]))
        # M1: constant 2D world-frame bias fitted on train, subtracted on test
        bx = np.mean([group[i]["raw_ex"] for i in train])
        by = np.mean([group[i]["raw_ey"] for i in train])
        scores["M1_world_const"].append(
            np.array([
                math.hypot(group[i]["raw_ex"] - bx, group[i]["raw_ey"] - by) for i in test
            ])
        )
        a_c = float(fit_e[train].mean())
        scores["M2_bearing_const"].append(reproject(test, -a_c, 0.0))
        a, b = _fit_line(fit_d[train], fit_e[train])
        scores["M3_bearing_affine"].append(reproject(test, -a, -b))
    return {
        k: float(np.sqrt(np.mean(np.concatenate(v) ** 2))) if v else math.nan
        for k, v in scores.items()
    }


# ------------------------------------------------------------ extrapolation risk


def fov_range_coverage(models, calib_json, step=0.15):
    """Fraction of each camera's on-site ground FOV outside its fitted range window."""
    x = np.arange(SITE[0], SITE[1] + step, step)
    y = np.arange(SITE[2], SITE[3] + step, step)
    gx, gy = np.meshgrid(x, y)
    out = {}
    for cam in CAMERAS:
        model = models[cam]
        entry = calib_json.get("cameras", {}).get(cam, {})
        dmin = entry.get("distance_min_m")
        dmax = entry.get("distance_max_m")
        if dmin is None or dmax is None:
            continue
        # in-FOV mask: world point projects back inside the image
        inside = np.zeros(gx.shape, bool)
        dist = np.hypot(gx - float(model.cam_pos[0]), gy - float(model.cam_pos[1]))
        for j in range(gx.shape[0]):
            for i in range(gx.shape[1]):
                _, _, visible = model.world_to_pixel(
                    float(gx[j, i]), float(gy[j, i]), CONTACT_Z_M
                )
                inside[j, i] = bool(visible)
        n_in = int(inside.sum())
        if not n_in:
            continue
        d_in = dist[inside]
        outside = (d_in < dmin) | (d_in > dmax)
        out[cam] = dict(
            fitted_min_m=float(dmin), fitted_max_m=float(dmax),
            fov_cells=n_in, cell_area_m2=float(step * step),
            fov_area_m2=float(n_in * step * step),
            frac_outside_fitted_range=float(outside.mean()),
            frac_beyond_max=float((d_in > dmax).mean()),
            frac_below_min=float((d_in < dmin).mean()),
            fov_range_min_m=float(d_in.min()), fov_range_max_m=float(d_in.max()),
        )
    return out


# ------------------------------------------------------------------------- main


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    models = {c: camera_model_from_world(WORLD_SDF, include_name=i)
              for c, i in MODEL_INCLUDES.items()}
    calib = load_projection_calibration(DEPLOYED_CALIB)
    calib_json = json.loads(DEPLOYED_CALIB.read_text())

    rows, inventory = load_samples(models, calib)
    print(f"loaded {len(rows)} truth-matched detections")

    # ---- artifacts: tidy residuals + inventory
    fields = list(rows[0].keys())
    with (OUT / "residuals.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with (OUT / "inventory.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(inventory[0].keys()))
        w.writeheader()
        w.writerows(inventory)

    # ---- per-camera decomposition, raw vs corrected, pooled and per capture
    stats = {}
    for cam in CAMERAS:
        for cap in list(CAPTURES) + ["POOLED"]:
            grp = [r for r in rows if r["camera"] == cam
                   and (cap == "POOLED" or r["capture"] == cap)]
            if len(grp) < 2:
                continue
            stats[f"{cam}|{cap}"] = {
                tag: describe(grp, tag) for tag in ("raw", "cor")
            }

    # ---- model comparison (pooled per camera)
    comparison = {}
    for cam in CAMERAS:
        grp = [r for r in rows if r["camera"] == cam]
        if len(grp) >= 20:
            comparison[cam] = model_comparison(grp, models, calib)

    # ---- leftover structure after the deployed correction
    leftover = {}
    for cam in CAMERAS:
        grp = [r for r in rows if r["camera"] == cam]
        if len(grp) < 10:
            continue
        rng = np.array([r["range_m"] for r in grp])
        along = np.array([r["cor_along"] for r in grp])
        cross = np.array([r["cor_cross"] for r in grp])
        nrm = np.array([r["cor_norm"] for r in grp])
        rho_a, n_a = spearman(rng, along)
        rho_n, _ = spearman(rng, nrm)
        rho_c, _ = spearman(rng, cross)
        edges = np.linspace(rng.min(), rng.max(), 6)
        c_r, c_along, c_n = binned(rng, along, edges, fn=np.nanmean)
        _, c_norm, _ = binned(rng, nrm, edges, fn=np.nanmean)
        boot_a = moving_block_bootstrap(np.column_stack([along, cross]))
        leftover[cam] = dict(
            n=len(grp),
            along_bias=float(along.mean()),
            along_bias_ci=[float(np.percentile(boot_a[:, 0], 2.5)),
                           float(np.percentile(boot_a[:, 0], 97.5))],
            cross_bias=float(cross.mean()),
            cross_bias_ci=[float(np.percentile(boot_a[:, 1], 2.5)),
                           float(np.percentile(boot_a[:, 1], 97.5))],
            spearman_range_vs_along=rho_a, spearman_range_vs_cross=rho_c,
            spearman_range_vs_norm=rho_n, n_finite=n_a,
            range_bin_centers=[float(v) for v in c_r],
            range_bin_along=[float(v) for v in c_along],
            range_bin_norm=[float(v) for v in c_norm],
            range_bin_counts=[int(v) for v in c_n],
        )

    # ---- handover step matrix (post-correction), pooled per camera
    handover = {}
    for tag in ("raw", "cor"):
        mat = {}
        for a in CAMERAS:
            ga = [r for r in rows if r["camera"] == a]
            if not ga:
                continue
            ba = np.array([np.mean([r[f"{tag}_ex"] for r in ga]),
                           np.mean([r[f"{tag}_ey"] for r in ga])])
            for b in CAMERAS:
                gb = [r for r in rows if r["camera"] == b]
                if not gb:
                    continue
                bb = np.array([np.mean([r[f"{tag}_ex"] for r in gb]),
                               np.mean([r[f"{tag}_ey"] for r in gb])])
                mat[f"{a}|{b}"] = float(np.linalg.norm(ba - bb))
        handover[tag] = mat

    # ---- REAL concurrent-observation fusion (not simulated)
    fusion = fusion_analysis(rows)

    extrapolation = fov_range_coverage(models, calib_json)

    payload = dict(
        deployed_calibration=str(DEPLOYED_CALIB.relative_to(REPO)),
        deployed_constants=calib_json.get("cameras", {}),
        contact_z_m=CONTACT_Z_M, truth_tolerance_s=TRUTH_TOL_S,
        jump_limit_m=JUMP_LIMIT_M, n_bootstrap=N_BOOT,
        inventory=inventory, stats=stats, model_comparison=comparison,
        leftover_structure=leftover, handover=handover, fusion=fusion,
        extrapolation=extrapolation,
    )
    (OUT / "audit.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(f"wrote {OUT/'audit.json'}")
    return 0


def fusion_analysis(rows, window_s=0.35):
    """Fuse cameras that observed CONCURRENTLY. Real residuals only.

    Groups detections into time clusters per capture; where >=2 cameras are
    present, the fused estimate is the unweighted mean of their corrected
    world projections. Errors are compared on the SAME clusters so the
    single-vs-fused comparison is paired.
    """
    out = {}
    for cap in CAPTURES:
        grp = sorted([r for r in rows if r["capture"] == cap], key=lambda r: r["stamp"])
        if not grp:
            continue
        clusters, cur = [], []
        for r in grp:
            if cur and r["stamp"] - cur[0]["stamp"] > window_s:
                clusters.append(cur)
                cur = []
            cur.append(r)
        if cur:
            clusters.append(cur)
        multi = []
        for cl in clusters:
            by_cam = {}
            for r in cl:
                by_cam.setdefault(r["camera"], []).append(r)
            if len(by_cam) < 2:
                continue
            picks = {c: v[0] for c, v in by_cam.items()}
            tx = np.mean([p["true_x"] for p in picks.values()])
            ty = np.mean([p["true_y"] for p in picks.values()])
            fx = np.mean([p["cor_px"] for p in picks.values()])
            fy = np.mean([p["cor_py"] for p in picks.values()])
            entry = dict(
                cameras=sorted(picks),
                fused_err=float(math.hypot(fx - tx, fy - ty)),
                per_cam={c: float(math.hypot(p["cor_px"] - p["true_x"],
                                             p["cor_py"] - p["true_y"]))
                         for c, p in picks.items()},
            )
            multi.append(entry)
        if not multi:
            out[cap] = dict(n_multi_camera_clusters=0)
            continue
        fused = np.array([m["fused_err"] for m in multi])
        best = np.array([min(m["per_cam"].values()) for m in multi])
        worst = np.array([max(m["per_cam"].values()) for m in multi])
        mean_single = np.array([np.mean(list(m["per_cam"].values())) for m in multi])
        out[cap] = dict(
            n_multi_camera_clusters=len(multi),
            fused_rmse=float(np.sqrt(np.mean(fused ** 2))),
            fused_mean=float(fused.mean()), fused_p90=float(np.percentile(fused, 90)),
            best_single_rmse=float(np.sqrt(np.mean(best ** 2))),
            worst_single_rmse=float(np.sqrt(np.mean(worst ** 2))),
            mean_single_rmse=float(np.sqrt(np.mean(mean_single ** 2))),
            fused_beats_best_frac=float((fused < best).mean()),
            errors=[dict(m) for m in multi],
        )
    return out


if __name__ == "__main__":
    raise SystemExit(main())
