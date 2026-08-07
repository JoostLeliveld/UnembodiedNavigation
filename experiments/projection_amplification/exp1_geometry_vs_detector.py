#!/usr/bin/env python3
"""exp1: is the apparent spatial camera noise detector variation, or projection geometry?

The reliability story so far treats "camera accuracy varies with position" as a
property to be *learned* per camera. But an oblique ground-plane camera amplifies a
FIXED pixel error into a position error that grows strongly and anisotropically
with range and image row: the same detector, equally good everywhere in the image,
would still look position-dependent on the floor. Before fitting any
heteroscedastic R_cond(x), that null model has to be priced.

Three nested variance models are compared on held-out data, all on the SAME
deployed-corrected residuals (bias handled by centering, so this is purely about
variance):

    R1-iso   R_xy = s^2 I                       (one number per camera)
    R1-full  R_xy = Sigma_c                     (one 2x2 per camera; the deployed
                                                 conditional-covariance form)
    R2-geom  R_xy = sigma_pix^2 J_g J_g'        (ONE pixel-noise number per camera,
                                                 all spatial structure comes from
                                                 the projection Jacobian)

R2-geom has the same number of free parameters as R1-iso (one), so if it wins on
held-out likelihood, the spatial structure was geometry, not learning.

J_g is the Jacobian of the exact deployed projection path, obtained from
``reliability.projection._projection_derivative`` — the same finite-difference
derivative the runtime covariance propagation uses. No reimplementation.

Ground truth is EVALUATION-ONLY: it measures residuals and never enters a
projection, a Jacobian, a fitted parameter or a covariance.

Inputs  <- logs/studies/external_camera_bias_model/exp1_residual_characterization/residuals.csv
Outputs -> logs/studies/projection_amplification/exp1_geometry_vs_detector/
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
sys.path.insert(0, str(REPO / "src" / "reliability"))
sys.path.insert(0, str(REPO / "src" / "unav_common"))
sys.path.insert(0, str(REPO / "experiments" / "external_camera_bias_model"))

from metrics import binned, spearman  # noqa: E402  (THE shared scoring library)
from reliability.conditional_covariance import chi2_coverage, matrix_nll  # noqa: E402
from reliability.projection import (  # noqa: E402
    _project_pixel_to_world,
    _projection_derivative,
    camera_model_from_world,
    load_projection_calibration,
)

# Study constants (world SDF, camera includes, contact height, deployed calibration,
# site extent) are owned by the residual-audit study; imported, never re-declared.
import residual_audit as RA  # noqa: E402

RESIDUALS_CSV = (
    REPO / "logs/studies/external_camera_bias_model/exp1_residual_characterization"
    / "residuals.csv"
)
OUT = REPO / "logs/studies/projection_amplification/exp1_geometry_vs_detector"

CAMERAS = RA.CAMERAS
JACOBIAN_STEP_PX = 0.5  # reliability.projection default
N_SPATIAL_FOLDS = 4
MIN_FOLD_TRAIN = 20
MIN_CAMERA_SAMPLES = 60
RANGE_EDGES = np.arange(3.0, 16.01, 1.0)

C_ISO = "#E69F00"
C_FULL = "#D55E00"
C_GEOM = "#0072B2"
C_RANGE = "#009E73"
MODEL_COLORS = {"R1-iso": C_ISO, "R1-full": C_FULL, "R2-geom": C_GEOM, "R1-range": C_RANGE}
#: Okabe-Ito, one per camera, for the per-detection scatter in fig_g2.
CAMERA_COLORS = {"camera_A": "#0072B2", "camera_B": "#E69F00",
                 "camera_C": "#009E73", "camera_D": "#CC79A7"}


def _rho(a, b) -> float:
    """Spearman from the shared library, tolerating its (rho, p) return shape."""

    value = spearman(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return float(value[0] if isinstance(value, tuple) else value)


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "axes.grid": True, "grid.color": "#CCCCCC",
        "grid.alpha": 0.3, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "font.size": 9,
    })


# --------------------------------------------------------------------- loading


def camera_models():
    return {
        camera: camera_model_from_world(RA.WORLD_SDF, include_name=include)
        for camera, include in RA.MODEL_INCLUDES.items()
    }


def projection_kwargs(calib: dict, camera: str) -> dict:
    entry = calib.get(camera, {})
    return {
        "contact_z_m": RA.CONTACT_Z_M,
        "along_bearing_offset_m": float(entry.get("intercept_m", 0.0)),
        "along_bearing_slope_per_m": float(entry.get("slope_per_m", 0.0)),
    }


def jacobian_at(camera_model, u: float, v: float, kwargs: dict):
    """2x2 dP_world/d(u, v) of the complete deployed projection, or None at an edge."""

    centre = _project_pixel_to_world(u, v, camera_model, **kwargs)
    if centre is None:
        return None
    du = _projection_derivative(
        camera_model, u, v, axis=0, step=JACOBIAN_STEP_PX, centre=centre, kwargs=kwargs
    )
    dv = _projection_derivative(
        camera_model, u, v, axis=1, step=JACOBIAN_STEP_PX, centre=centre, kwargs=kwargs
    )
    if du is None or dv is None:
        return None
    return np.array([[du[0], dv[0]], [du[1], dv[1]]], dtype=float)


def load_samples(models, calib) -> dict[str, dict]:
    """Per-camera arrays of corrected residuals plus their projection Jacobians."""

    per_camera: dict[str, dict] = {
        camera: {"cor": [], "raw": [], "jac": [], "range": [], "xy": [], "uv": [],
                 "capture": []}
        for camera in CAMERAS
    }
    skipped = {camera: 0 for camera in CAMERAS}
    with RESIDUALS_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            camera = row["camera"]
            if camera not in per_camera:
                continue
            u, v = float(row["u"]), float(row["v"])
            jac = jacobian_at(models[camera], u, v, projection_kwargs(calib, camera))
            if jac is None or not np.all(np.isfinite(jac)) or abs(np.linalg.det(jac)) < 1e-12:
                skipped[camera] += 1
                continue
            bucket = per_camera[camera]
            bucket["cor"].append([float(row["cor_ex"]), float(row["cor_ey"])])
            bucket["raw"].append([float(row["raw_ex"]), float(row["raw_ey"])])
            bucket["jac"].append(jac)
            bucket["range"].append(float(row["range_m"]))
            bucket["xy"].append([float(row["true_x"]), float(row["true_y"])])
            bucket["uv"].append([u, v])
            bucket["capture"].append(row["capture"])
    for camera, bucket in per_camera.items():
        for key in ("cor", "raw", "jac", "range", "xy", "uv"):
            bucket[key] = np.asarray(bucket[key], dtype=float)
        bucket["skipped"] = skipped[camera]
    return per_camera


# ----------------------------------------------------------------- fold scheme


def spatial_folds(xy: np.ndarray, n_folds: int = N_SPATIAL_FOLDS):
    """Leave-region-out folds: contiguous bands along the dominant spread axis.

    A *spatial* variance model must be tested on positions it never saw, otherwise
    temporally adjacent frames (0.2 s apart, essentially the same pose) leak the
    answer across the split. Bands are cut along whichever world axis the samples
    spread over more, at sample quantiles so folds are balanced in count.
    """

    axis = 0 if xy[:, 0].ptp() >= xy[:, 1].ptp() else 1
    coordinate = xy[:, axis]
    edges = np.quantile(coordinate, np.linspace(0.0, 1.0, n_folds + 1))
    edges[0] -= 1.0
    edges[-1] += 1.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        test = (coordinate >= lo) & (coordinate < hi)
        if test.sum() == 0 or (~test).sum() < MIN_FOLD_TRAIN:
            continue
        yield ~test, test


# ---------------------------------------------------------------- the 3 models


def geometric_shapes(jac: np.ndarray) -> np.ndarray:
    """G = J J' per sample: the world-space covariance of ONE unit of pixel noise."""

    return np.einsum("nij,nkj->nik", jac, jac)


def range_shapes(ranges: np.ndarray) -> np.ndarray:
    """R1-range: isotropic shape growing as range^2, normalised to unit mean scale.

    The reviewer question this answers is "is J J' doing anything beyond encoding
    distance from the camera?". To answer it the range model must cost the SAME one
    free parameter as R2-geom, so the range dependence enters as a fixed per-sample
    *shape* and a single scalar is fitted by the same closed-form ML. Only the shape
    differs between the two models, which is exactly the comparison of interest.
    """

    scale = (ranges / float(np.mean(ranges))) ** 2
    return scale[:, None, None] * np.eye(2)[None, :, :]


def stated_rms_m(covariances) -> float:
    """The error magnitude the model CLAIMS, in metres: sqrt(E[|e|^2]) = sqrt(E[tr C]).

    Pairs with the measured ``rms_m`` so a reader can compare "the model says 4 cm"
    against "reality is 5 cm" without converting nats.
    """

    traces = [float(np.trace(np.asarray(c))) for c in covariances]
    return float(np.sqrt(np.mean(traces))) if traces else float("nan")


def fit_iso(res: np.ndarray) -> np.ndarray:
    """R1-iso: s^2 I, ML over the centred residuals (s^2 = mean of e'e / 2)."""

    var = float(np.mean(np.sum(res**2, axis=1)) / 2.0)
    return np.eye(2) * max(var, 1e-12)


def fit_full(res: np.ndarray) -> np.ndarray:
    """R1-full: one 2x2 sample covariance per camera (the deployed R_cond form)."""

    cov = np.cov(res.T, bias=True)
    return _psd_floor(np.atleast_2d(cov))


def fit_geometric_pixel_variance(res: np.ndarray, shapes: np.ndarray) -> float:
    """R2-geom: ML pixel variance under R_xy = sigma_pix^2 G_n, G_n = J_n J_n'.

    With the shape fixed per sample, the Gaussian ML solution is closed form:
    sigma^2 = (1 / 2n) * sum_n e_n' G_n^-1 e_n.
    """

    total = 0.0
    for residual, shape in zip(res, shapes):
        total += float(residual @ np.linalg.solve(shape, residual))
    return max(total / (2.0 * len(res)), 1e-12)


def _psd_floor(matrix: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, floor)
    return vectors @ np.diag(values) @ vectors.T


# ------------------------------------------------------------------ evaluation


def evaluate(per_camera: dict, kind: str = "cor") -> dict:
    """Held-out NLL + coverage per camera per model, over leave-region-out folds.

    ``kind`` selects the residual the models must explain: ``"raw"`` (bare
    projection) or ``"cor"`` (after the deployed along-bearing correction). Both
    matter here: the deployed correction is itself a function of range, so it has
    already absorbed range-dependent structure that a geometric variance model
    would otherwise be credited with.
    """

    results: dict[str, dict] = {}
    for camera, bucket in per_camera.items():
        res_all = bucket[kind]
        if len(res_all) < MIN_CAMERA_SAMPLES:
            results[camera] = {"status": "insufficient_samples", "n": int(len(res_all))}
            continue
        shapes_all = geometric_shapes(bucket["jac"])
        range_all = range_shapes(bucket["range"])
        collected = {
            (name, centring): {"res": [], "cov": []}
            for name in ("R1-iso", "R1-full", "R1-range", "R2-geom")
            for centring in ("train", "oracle")
        }
        folds = 0
        sigma_pix = []
        bias_shift_m = []
        for train, test in spatial_folds(bucket["xy"]):
            folds += 1
            # Bias is a separate model (see external_camera_bias_model); centre on
            # the TRAINING mean so this compares variance models only, and so the
            # centring itself never sees held-out data.
            mean = res_all[train].mean(axis=0)
            train_res = res_all[train] - mean
            test_res = res_all[test] - mean
            # Diagnostic only: the same variance models scored against residuals
            # centred on the TEST-fold mean. The gap between the two isolates
            # "the bias does not transfer across regions" from "the variance model
            # is wrong". Oracle centring uses held-out data and is NOT deployable.
            oracle_res = res_all[test] - res_all[test].mean(axis=0)
            bias_shift_m.append(float(np.linalg.norm(res_all[test].mean(axis=0) - mean)))

            iso = fit_iso(train_res)
            full = fit_full(train_res)
            pixel_var = fit_geometric_pixel_variance(train_res, shapes_all[train])
            # Same closed-form ML, same one free parameter, different fixed shape.
            range_var = fit_geometric_pixel_variance(train_res, range_all[train])
            sigma_pix.append(math.sqrt(pixel_var))

            for name, covariances in (
                ("R1-iso", [iso] * int(test.sum())),
                ("R1-full", [full] * int(test.sum())),
                ("R1-range", [range_var * shape for shape in range_all[test]]),
                ("R2-geom", [pixel_var * shape for shape in shapes_all[test]]),
            ):
                floored = [_psd_floor(np.asarray(c)).tolist() for c in covariances]
                for centring, residuals in (("train", test_res), ("oracle", oracle_res)):
                    collected[(name, centring)]["res"].extend(residuals.tolist())
                    collected[(name, centring)]["cov"].extend(floored)

        entry = {"status": "ok", "n": int(len(res_all)), "folds": folds,
                 "skipped_edge_samples": int(bucket["skipped"]),
                 "sigma_pix_px": {"mean": float(np.mean(sigma_pix)),
                                  "min": float(np.min(sigma_pix)),
                                  "max": float(np.max(sigma_pix))},
                 "train_to_test_bias_shift_m": {"mean": float(np.mean(bias_shift_m)),
                                                "max": float(np.max(bias_shift_m))}}
        for (name, centring), payload in collected.items():
            residuals, covariances = payload["res"], payload["cov"]
            key = name if centring == "train" else f"{name} (oracle-centred)"
            entry[key] = {
                "nll": matrix_nll(residuals, covariances),
                "coverage_50": chi2_coverage(residuals, covariances, 0.50),
                "coverage_90": chi2_coverage(residuals, covariances, 0.90),
                "coverage_95": chi2_coverage(residuals, covariances, 0.95),
                "rms_m": float(np.sqrt(np.mean(np.sum(np.asarray(residuals) ** 2, axis=1)))),
                # What the model CLAIMS, in metres, so the score is readable without nats.
                "stated_rms_m": stated_rms_m(covariances),
            }
        entry["nll_gain_geom_vs_iso"] = entry["R1-iso"]["nll"] - entry["R2-geom"]["nll"]
        entry["nll_gain_geom_vs_full"] = entry["R1-full"]["nll"] - entry["R2-geom"]["nll"]
        entry["nll_gain_geom_vs_iso_oracle"] = (
            entry["R1-iso (oracle-centred)"]["nll"] - entry["R2-geom (oracle-centred)"]["nll"]
        )
        results[camera] = entry
    return results


# --------------------------------------------------------------------- figures


def fig_g1(models, calib) -> dict:
    """Geometry only: metres of ground displacement per pixel of image error."""

    x0, x1, y0, y1 = RA.SITE
    xs = np.linspace(x0, x1, 220)
    ys = np.linspace(y0, y1, 180)
    fig, axes = plt.subplots(1, len(CAMERAS), figsize=(4.1 * len(CAMERAS), 4.0), sharey=True)
    stats = {}
    for ax, camera in zip(axes, CAMERAS):
        model = models[camera]
        kwargs = projection_kwargs(calib, camera)
        grid = np.full((ys.size, xs.size), np.nan)
        for j, x in enumerate(xs):
            for i, y in enumerate(ys):
                u, v, visible = model.world_to_pixel(x, y, RA.CONTACT_Z_M)
                if not visible:
                    continue
                jac = jacobian_at(model, u, v, kwargs)
                if jac is None:
                    continue
                # Largest singular value: worst-case metres per pixel.
                grid[i, j] = float(np.linalg.svd(jac, compute_uv=False)[0])
        finite = grid[np.isfinite(grid)]
        stats[camera] = {
            "m_per_px_p05": float(np.percentile(finite, 5)) if finite.size else None,
            "m_per_px_median": float(np.median(finite)) if finite.size else None,
            "m_per_px_p95": float(np.percentile(finite, 95)) if finite.size else None,
            "amplification_p95_over_p05": (
                float(np.percentile(finite, 95) / np.percentile(finite, 5))
                if finite.size else None
            ),
            "fov_cells": int(finite.size),
        }
        im = ax.pcolormesh(xs, ys, grid * 1000.0, cmap="viridis", shading="auto",
                           vmin=0.0, vmax=60.0)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_title(camera.replace("camera_", "camera "), fontweight="bold", fontsize=10.5)
        ax.text(
            0.03, 0.03,
            f"median {1000 * stats[camera]['m_per_px_median']:.1f} mm/px\n"
            f"p95/p05 = {stats[camera]['amplification_p95_over_p05']:.1f}$\\times$",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2.0),
        )
        ax.grid(False)
    axes[0].set_ylabel("y [m]")
    cb = fig.colorbar(im, ax=axes.tolist(), shrink=0.85, pad=0.02)
    cb.set_label("ground error per pixel of image error  [mm/px]\n(worst-case direction)")
    spread = [s["amplification_p95_over_p05"] for s in stats.values() if s["fov_cells"]]
    amplification = (
        f"{min(spread):.1f}$\\times$" if max(spread) - min(spread) < 0.05
        else f"{min(spread):.1f}–{max(spread):.1f}$\\times$"
    )
    fig.suptitle(
        f"A fixed detector error is amplified "
        f"{amplification} across a single camera's footprint\n"
        r"(pure geometry: the largest singular value of $\partial(x,y)/\partial(u,v)$ "
        "through the deployed projection — no data involved)",
        fontsize=12.5, fontweight="bold", y=1.06)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_g1_pixel_to_ground_amplification.{ext}", bbox_inches="tight")
    plt.close(fig)
    return stats


def fig_g2(per_camera: dict, results_by_kind: dict) -> dict:
    """Does projection geometry explain the residual we actually observe?

    One point per detection, never a range bin. The previous version of this figure
    reduced each camera to 3-7 range-binned medians and reported a Spearman over those
    bins; with 3 bins that statistic is +-1 almost regardless of the data, and it
    reported rho = -1.00 for a camera whose observed curve ran OPPOSITE to the
    prediction. Range was also the wrong axis: J = J(x, y), not J(r).

    Left/middle: observed |e| against the geometric prediction sigma_pix*sqrt(tr G),
    raw and after the deployed correction. Right: what each model CLAIMS versus what
    actually happened, in millimetres, with held-out 90 % coverage -- the same
    comparison fig_g3 scores in nats, in units a reader can check by eye.

    Scope, stated on the figure because it is the binding limit: the robot drove
    straight routes, so sigma_max(J) and range are collinear at rho >= 0.97 and this
    data CANNOT separate them.
    """

    kinds = (("raw", "raw projection"), ("cor", "after the deployed correction"))
    fig = plt.figure(figsize=(15.6, 5.4))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 1.25), wspace=0.28)
    stats: dict = {}

    for col, (kind, kind_label) in enumerate(kinds):
        ax = fig.add_subplot(grid[0, col])
        ax.grid(True, zorder=0)
        allx, ally = [], []
        for camera in CAMERAS:
            bucket = per_camera[camera]
            entry = results_by_kind[kind].get(camera, {})
            if entry.get("status") != "ok":
                continue
            sigma_pix = entry["sigma_pix_px"]["mean"]
            shapes = geometric_shapes(bucket["jac"])
            pred_mm = 1000.0 * sigma_pix * np.sqrt(np.trace(shapes, axis1=1, axis2=2))
            obs_mm = 1000.0 * np.linalg.norm(bucket[kind], axis=1)
            ax.scatter(pred_mm, obs_mm, s=7, alpha=0.35, linewidths=0,
                       color=CAMERA_COLORS[camera], label=camera.replace("camera_", ""),
                       zorder=3)
            allx.extend(pred_mm.tolist())
            ally.extend(obs_mm.tolist())
            stats.setdefault(kind, {})[camera] = {
                "n": int(len(obs_mm)),
                "spearman_obs_vs_pred": float(_rho(obs_mm, pred_mm)),
                "median_obs_mm": float(np.median(obs_mm)),
                "median_pred_mm": float(np.median(pred_mm)),
            }
        if allx:
            # Percentile, not max: a handful of large residuals otherwise compress the
            # bulk of the detections into the bottom-left corner and hide the structure.
            hi = float(np.percentile(np.concatenate([allx, ally]), 99.0)) * 1.08
            ax.plot([0, hi], [0, hi], color="#444444", ls="--", lw=1.2, zorder=4,
                    label="identity")
            ax.set_xlim(0, hi)
            ax.set_ylim(0, hi)
        ax.set_xlabel(r"geometry prediction  $\sigma_{pix}\sqrt{\mathrm{tr}\,JJ^{T}}$  [mm]")
        if col == 0:
            ax.set_ylabel("observed residual $|e|$  [mm]")
        rhos = "\n".join(
            f"  {c.replace('camera_', '')}  {stats[kind][c]['spearman_obs_vs_pred']:+.2f}"
            for c in CAMERAS if c in stats.get(kind, {})
        )
        ax.set_title(f"({'ab'[col]})  {kind_label}", fontsize=10, fontweight="bold")
        ax.text(0.97, 0.03, "Spearman per detection\n" + rhos, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.8, family="monospace",
                bbox=dict(fc="white", ec="#cccccc", alpha=0.93, pad=3), zorder=9)
        ax.legend(fontsize=7.5, loc="upper left", ncol=2, markerscale=2.0)

    # ---- (c) stated vs actual, in millimetres ----
    ax = fig.add_subplot(grid[0, 2])
    ax.grid(True, axis="y", zorder=0)
    models = ("R1-iso", "R1-range", "R2-geom")
    labels = {"R1-iso": "constant", "R1-range": "range-only", "R2-geom": r"geometry $JJ^{T}$"}
    width = 0.26
    xs = np.arange(len(CAMERAS), dtype=float)
    for mi, model in enumerate(models):
        stated, actual, cov90 = [], [], []
        for camera in CAMERAS:
            entry = results_by_kind["cor"].get(camera, {})
            row = entry.get(model, {}) if entry.get("status") == "ok" else {}
            stated.append(1000.0 * row.get("stated_rms_m", np.nan))
            actual.append(1000.0 * row.get("rms_m", np.nan))
            cov90.append(row.get("coverage_90", np.nan))
        off = (mi - 1) * width
        ax.bar(xs + off, stated, width * 0.9, color=MODEL_COLORS.get(model, "#888888"),
               edgecolor="white", linewidth=0.7, zorder=3, label=f"{labels[model]} — claimed")
        for x, a, c in zip(xs + off, actual, cov90):
            if np.isfinite(c):
                ax.text(x, 2, f"{100 * c:.0f}%", ha="center", va="bottom", fontsize=6.5,
                        rotation=90, color="white", zorder=6)
    actual_cor = [1000.0 * results_by_kind["cor"].get(c, {}).get("R1-iso", {}).get("rms_m", np.nan)
                  for c in CAMERAS]
    ax.plot(xs, actual_cor, "k_", markersize=26, markeredgewidth=2.2, zorder=7,
            label="actual RMS (measured)")
    ax.set_xticks(xs)
    ax.set_xticklabels([c.replace("camera_", "") for c in CAMERAS])
    ax.set_ylabel("error magnitude [mm]")
    ax.set_title("(c)  what each model CLAIMS vs what happened\n"
                 "bar = claimed RMS, dash = measured RMS, % = held-out 90 % coverage",
                 fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, max(np.nanmax(actual_cor), 80.0) * 1.32)
    ax.legend(fontsize=7, loc="upper center", ncol=2, framealpha=0.95)

    fig.suptitle("Does projection geometry explain the observed residual?  "
                 "One point per detection, no range binning",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.text(0.5, -0.10,
             "1424 detections from THREE captures (smoke1 90 deg, smoke2 0 deg, "
             "fusion_handover tangent-derived): two headings only. The robot drove straight "
             "routes, so the sampled footprint is a thin ribbon -- image column spans ~100 px "
             "of ~1280 on cameras A and B -- and sigma_max(J) is collinear with range at "
             "rho = 0.976/0.969/0.999/0.996. This data therefore CANNOT test whether JJ^T "
             "adds anything beyond distance from camera; the range-only bar is shown so that "
             "limit is visible rather than assumed. Folds are leave-region-out, not "
             "train/val/test by run. Ground truth measures the residual and never enters a "
             "projection, Jacobian or covariance.",
             ha="center", va="top", fontsize=7.4, color="#333333", wrap=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_g2_observed_vs_geometric.{ext}", bbox_inches="tight")
    plt.close(fig)
    return stats


def fig_g3(results: dict, kind_label: str) -> None:
    """Held-out model comparison, as differences against the one-parameter baseline.

    Absolute NLL is dominated by camera A (whose held-out bias shift is larger than
    its noise), which hides every other camera. Plotting NLL *relative to R1-iso*
    removes the per-camera scale and shows the only thing being compared: which
    variance form wins, and by how much. Symlog keeps A on the same axis as the
    sub-nat differences elsewhere.
    """

    usable = [c for c in CAMERAS if results.get(c, {}).get("status") == "ok"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    width = 0.34
    positions = np.arange(len(usable))
    for ax, centring, panel in (
        (axes[0], "", "deployable  (train-fold centring)"),
        (axes[1], " (oracle-centred)", "variance model isolated  (oracle centring)"),
    ):
        for offset, name in ((-width / 2, "R1-full"), (width / 2, "R2-geom")):
            deltas = [
                results[c][name + centring]["nll"] - results[c]["R1-iso" + centring]["nll"]
                for c in usable
            ]
            ax.bar(positions + offset, deltas, width, color=MODEL_COLORS[name], label=name)
        ax.axhline(0.0, color="#333333", lw=1.2)
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_xticks(positions, [c.replace("camera_", "") for c in usable])
        ax.set_ylabel("held-out NLL relative to R1-iso  [nats]\n(negative = beats the "
                      "one-parameter baseline)")
        ax.set_title(panel, fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)
    ax2 = axes[2]
    for offset, name in zip((-width * 0.75, 0.0, width * 0.75),
                            ("R1-iso", "R1-full", "R2-geom")):
        ax2.bar(positions + offset, [100 * results[c][name]["coverage_90"] for c in usable],
                width * 0.7, color=MODEL_COLORS[name], label=name)
    ax2.axhline(90.0, color="#333333", lw=1.2, ls="--", label="nominal 90 %")
    ax2.set_xticks(positions, [c.replace("camera_", "") for c in usable])
    ax2.set_ylabel("empirical 90 % coverage  [%]")
    ax2.set_title("Calibration, deployable centring\n(read with the likelihood, never alone)",
                  fontweight="bold", fontsize=10)
    ax2.legend(fontsize=8)
    fig.suptitle(f"R2-geom ($\\sigma_{{pix}}^2 J J^T$) has the SAME parameter count as R1-iso "
                 f"— residuals {kind_label}",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_g3_heldout_model_comparison.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    if not RESIDUALS_CSV.is_file():
        print(f"missing input: {RESIDUALS_CSV}\n"
              "run experiments/external_camera_bias_model/residual_audit.py first")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    calib = load_projection_calibration(RA.DEPLOYED_CALIB)
    models = camera_models()
    per_camera = load_samples(models, calib)
    results_by_kind = {kind: evaluate(per_camera, kind) for kind in ("raw", "cor")}
    payload = {
        "inputs": {
            "residuals_csv": str(RESIDUALS_CSV.relative_to(REPO)),
            "world_sdf": str(RA.WORLD_SDF.relative_to(REPO)),
            "deployed_calibration": str(RA.DEPLOYED_CALIB.relative_to(REPO)),
        },
        "config": {
            "jacobian_step_px": JACOBIAN_STEP_PX,
            "spatial_folds": N_SPATIAL_FOLDS,
            "min_camera_samples": MIN_CAMERA_SAMPLES,
            "residual_kinds": {"raw": "bare projection",
                               "cor": "after deployed projection_calibration_v2"},
        },
        "heldout": results_by_kind,
        "geometry": fig_g1(models, calib),
        "range_profile": fig_g2(per_camera, results_by_kind),
    }
    fig_g3(results_by_kind["cor"], "after the deployed correction")
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for kind, results in results_by_kind.items():
        print(f"===== residuals: {kind} =====")
        for camera in CAMERAS:
            entry = results[camera]
            if entry.get("status") != "ok":
                print(f"{camera}: {entry}")
                continue
            print(f"{camera}: n={entry['n']} folds={entry['folds']} "
                  f"sigma_pix={entry['sigma_pix_px']['mean']:.2f}px "
                  f"bias_shift={entry['train_to_test_bias_shift_m']['mean']:.3f}m")
            for centring in ("", " (oracle-centred)"):
                print(f"    {centring or ' train-centred':>18}: NLL "
                      f"iso={entry['R1-iso' + centring]['nll']:8.3f} "
                      f"full={entry['R1-full' + centring]['nll']:8.3f} "
                      f"geom={entry['R2-geom' + centring]['nll']:8.3f}   cov90 "
                      f"iso={entry['R1-iso' + centring]['coverage_90']:.2f} "
                      f"full={entry['R1-full' + centring]['coverage_90']:.2f} "
                      f"geom={entry['R2-geom' + centring]['coverage_90']:.2f}")
    print("wrote ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
