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

import argparse
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

# --- fig_g4 spatial-cell contract (frozen; do not tune for presentation) ---------
#: Cell edge for aggregating repeated detections before comparing RMS to RMS.
CELL_SIZE_M = 0.5
#: A cell must hold this many detections before its measured RMS is trustworthy.
MIN_CELL_DETECTIONS = 5
#: Below this many populated cells a camera cannot establish a trend; say so on the plot.
MIN_CELLS_FOR_TREND = 10

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
                 "theta": [], "capture": []}
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
            # Grid captures record the commanded heading; route captures do not.
            try:
                bucket["theta"].append(float(row.get("theta", "nan")))
            except (TypeError, ValueError):
                bucket["theta"].append(float("nan"))
    for camera, bucket in per_camera.items():
        for key in ("cor", "raw", "jac", "range", "xy", "uv", "theta"):
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


def heldout_sigma_pix(bucket: dict, kind: str) -> np.ndarray:
    """Per-detection sigma_pix fitted WITHOUT that detection's spatial region.

    Each detection is predicted using the pixel-noise scalar fitted on the folds where
    it was held out, so the cell scatter is a genuine out-of-region prediction rather
    than a fit evaluated on itself.
    """

    res_all = bucket[kind]
    shapes_all = geometric_shapes(bucket["jac"])
    out = np.full(len(res_all), np.nan)
    for train, test in spatial_folds(bucket["xy"]):
        train_res = res_all[train] - res_all[train].mean(axis=0)
        out[test] = math.sqrt(fit_geometric_pixel_variance(train_res, shapes_all[train]))
    return out


def spatial_cells(bucket: dict, kind: str) -> dict:
    """Aggregate detections into fixed ground cells and compare RMS against RMS.

    A covariance model does not claim "this detection will be 42 mm wrong"; it claims a
    spread for observations made around a place. Comparing a predicted spread to a single
    realised residual makes the identity line meaningless, so both axes are RMS over
    repeated detections in the same cell.
    """

    xy = bucket["xy"]
    residual = bucket[kind]
    sigma_pix = heldout_sigma_pix(bucket, kind)
    shapes = geometric_shapes(bucket["jac"])
    # E[|e|^2] = tr(Sigma) with Sigma = sigma_pix^2 J J', so the predicted RMS magnitude
    # is sigma_pix * sqrt(tr(J J')). sigma_max would answer a different question (G1's).
    pred = sigma_pix * np.sqrt(np.trace(shapes, axis1=1, axis2=2))
    obs = np.linalg.norm(residual, axis=1)

    buckets: dict[tuple[int, int], list[int]] = {}
    for index, (x, y) in enumerate(xy):
        if not np.isfinite(pred[index]):
            continue
        buckets.setdefault(
            (int(np.floor(x / CELL_SIZE_M)), int(np.floor(y / CELL_SIZE_M))), []
        ).append(index)

    cells = []
    deviation = np.full(len(obs), np.nan)  # |e_i - mean_k|, the conditional part
    for (ix, iy), members in sorted(buckets.items()):
        if len(members) < MIN_CELL_DETECTIONS:
            continue
        idx = np.asarray(members)
        # Split the cell into a deterministic part and a variable part. A projection
        # covariance should be judged against the SECOND: bounding-box geometry, heading
        # and calibration all leave a deterministic offset here, and folding that into
        # "noise" would credit the covariance model for explaining a bias.
        mean_e = residual[idx].mean(axis=0)
        centred = residual[idx] - mean_e
        deviation[idx] = np.linalg.norm(centred, axis=1)
        cov = centred.T @ centred / max(len(idx) - 1, 1)
        cells.append({
            "x_m": (ix + 0.5) * CELL_SIZE_M,
            "y_m": (iy + 0.5) * CELL_SIZE_M,
            "n": int(idx.size),
            "pred_rms_m": float(np.sqrt(np.mean(pred[idx] ** 2))),
            "obs_rms_m": float(np.sqrt(np.mean(obs[idx] ** 2))),
            "bias_m": float(np.linalg.norm(mean_e)),
            "cond_rms_m": float(np.sqrt(np.trace(cov))),
        })

    payload = {
        "cell_size_m": CELL_SIZE_M,
        "min_cell_detections": MIN_CELL_DETECTIONS,
        "n_detections": int(len(obs)),
        "n_cells": len(cells),
        "cells": cells,
    }
    payload["deviation_m"] = deviation
    if len(cells) >= 3:
        q = np.array([c["pred_rms_m"] for c in cells])
        for label, key in (("", "obs_rms_m"), ("cond_", "cond_rms_m")):
            o = np.array([c[key] for c in cells])
            payload[f"{label}r2_identity"] = float(
                1.0 - ((o - q) ** 2).sum() / ((o - o.mean()) ** 2).sum())
            payload[f"{label}spearman_cells"] = _rho(o, q)
            payload[f"{label}median_ratio_obs_over_pred"] = float(np.median(o / q))
        payload["median_bias_m"] = float(np.median([c["bias_m"] for c in cells]))
    return payload


def footprint_mask(model, kwargs, nx: int = 90, ny: int = 74):
    """Coarse visible-floor mask for the camera, so the sampled ribbon can be seen
    against the footprint it fails to cover."""

    x0, x1, y0, y1 = RA.SITE
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    mask = np.zeros((ys.size, xs.size), dtype=float)
    for j, x in enumerate(xs):
        for i, y in enumerate(ys):
            _u, _v, visible = model.world_to_pixel(x, y, RA.CONTACT_Z_M)
            mask[i, j] = 1.0 if visible else np.nan
    return xs, ys, mask


def fig_g4(per_camera: dict, models, calib, kind: str = "cor") -> dict:
    """One figure per camera: where the data is, then whether geometry predicts it.

    Deliberately the DEPLOYED-corrected residual only. Raw-versus-corrected is the
    ablation in fig_g2; mixing both here would double the panels and bury the question
    that actually matters downstream, which is whether the uncertainty REMAINING after
    the deployed correction follows projection geometry.
    """

    out: dict = {}
    per_cam_cells = {c: spatial_cells(per_camera[c], kind) for c in CAMERAS
                     if len(per_camera[c][kind]) >= MIN_CAMERA_SAMPLES}
    # One colour scale across A-D, otherwise each map silently rescales its own errors.
    all_dev_mm = np.concatenate([
        1000.0 * per_cam_cells[c]["deviation_m"][np.isfinite(per_cam_cells[c]["deviation_m"])]
        for c in per_cam_cells
    ])
    vmax = float(np.percentile(all_dev_mm, 95))

    for camera, cells in per_cam_cells.items():
        bucket = per_camera[camera]
        fig, (ax_map, ax_fit) = plt.subplots(1, 2, figsize=(11.4, 4.9))

        xs, ys, mask = footprint_mask(models[camera], projection_kwargs(calib, camera))
        ax_map.pcolormesh(xs, ys, mask, cmap="Greys", vmin=0.0, vmax=6.0, shading="auto",
                          zorder=1)
        dev_mm = 1000.0 * cells["deviation_m"]
        finite = np.isfinite(dev_mm)
        ax_map.scatter(bucket["xy"][~finite, 0], bucket["xy"][~finite, 1], s=6,
                       color="#cccccc", linewidths=0, zorder=2,
                       label="below cell minimum")
        scatter = ax_map.scatter(bucket["xy"][finite, 0], bucket["xy"][finite, 1],
                                 c=dev_mm[finite], s=11, cmap="viridis", vmin=0.0,
                                 vmax=vmax, linewidths=0, zorder=3)
        cam_xy = getattr(models[camera], "cam_pos", None)
        if cam_xy is not None:
            ax_map.plot(cam_xy[0], cam_xy[1], marker="v", color="#D55E00", markersize=11,
                        markeredgecolor="white", linestyle="none", zorder=5, label="camera")
            # Keep the mount in frame: the oblique viewing direction is half the story of
            # why the amplification field looks the way it does.
            ax_map.set_xlim(min(ax_map.get_xlim()[0], cam_xy[0] - 1.0),
                            max(ax_map.get_xlim()[1], cam_xy[0] + 1.0))
            ax_map.set_ylim(min(ax_map.get_ylim()[0], cam_xy[1] - 1.0),
                            max(ax_map.get_ylim()[1], cam_xy[1] + 1.0))
            ax_map.legend(fontsize=8, loc="upper right")
        fig.colorbar(scatter, ax=ax_map,
                     label=f"conditional |e - mean$_k$| [mm]  (shared, p95={vmax:.0f})")
        ax_map.set_aspect("equal")
        ax_map.set_xlabel("x [m]")
        ax_map.set_ylabel("y [m]")
        ax_map.set_title("(a)  where the robot was actually observed\n"
                         "colour = error AFTER removing the local mean (bias split out)",
                         fontsize=9.5, fontweight="bold")

        pred = np.array([c["pred_rms_m"] for c in cells["cells"]]) * 1000.0
        obs = np.array([c["cond_rms_m"] for c in cells["cells"]]) * 1000.0
        counts = np.array([c["n"] for c in cells["cells"]], dtype=float)
        ax_fit.grid(True, zorder=0)
        if pred.size:
            ax_fit.scatter(pred, obs, s=18 + 4.0 * counts, alpha=0.8, zorder=3,
                           color=CAMERA_COLORS[camera], edgecolor="white", linewidth=0.8)
            hi = float(max(pred.max(), obs.max())) * 1.15
            ax_fit.plot([0, hi], [0, hi], ls="--", lw=1.3, color="#444444", zorder=4,
                        label="$y=x$")
            ax_fit.set_xlim(0, hi)
            ax_fit.set_ylim(0, hi)
            ax_fit.legend(fontsize=8, loc="upper left")
        ax_fit.set_xlabel(r"geometry-predicted RMS  $\sigma_{pix}\sqrt{\mathrm{tr}\,JJ^{T}}$  [mm]")
        ax_fit.set_ylabel(r"measured CONDITIONAL RMS $\sqrt{\mathrm{tr}\,\hat\Sigma_k}$ [mm]")
        note = (f"N = {cells['n_detections']} detections · M = {cells['n_cells']} cells "
                f"({CELL_SIZE_M:g} m, $\geq${MIN_CELL_DETECTIONS} each)")
        if "cond_r2_identity" in cells:
            note += (f"\nconditional: $R^2$ vs identity = {cells['cond_r2_identity']:+.2f} · "
                     f"cell Spearman = {cells['cond_spearman_cells']:+.2f} · "
                     f"median obs/pred = {cells['cond_median_ratio_obs_over_pred']:.2f}"
                     f"\nmedian per-cell BIAS removed first = "
                     f"{1000 * cells['median_bias_m']:.0f} mm")
        if cells["n_cells"] < MIN_CELLS_FOR_TREND:
            note += (f"\nONLY {cells['n_cells']} CELLS — too few to establish a trend; "
                     "read as a scale check, not a fit")
        ax_fit.set_title("(b)  does geometry predict the local SPREAD?\n" + note,
                         fontsize=9.0, fontweight="bold")

        fig.suptitle(f"{camera.replace('camera_', 'Camera ')} — deployed-corrected residual",
                     fontsize=12.5, fontweight="bold", y=1.03)
        # The coverage caveat is MEASURED, never hardcoded: the same code renders both the
        # thin-ribbon route capture and the 2-D grid, and stating the route limitation on a
        # grid figure would be a straight factual error.
        x_span = float(bucket["xy"][:, 0].ptp())
        y_span = float(bucket["xy"][:, 1].ptp())
        rho_u = abs(_rho(bucket["uv"][:, 0], bucket["range"]))
        theta = np.asarray(bucket.get("theta", []), dtype=float)
        theta = theta[np.isfinite(theta)]  # route captures carry no heading column
        n_head = len({round(math.degrees(t)) % 360 for t in theta})
        covered = (x_span >= 6.0 and y_span >= 6.0 and rho_u < 0.6)
        if covered:
            limit = (f"COVERAGE: {x_span:.1f} x {y_span:.1f} m sampled in two dimensions"
                     + (f" across {n_head} headings" if n_head else "")
                     + f"; image column is decorrelated from range (|rho| = {rho_u:.2f}), so "
                     "this dataset CAN separate image position from distance. It still cannot "
                     "separate sigma_max(J) from range: those stay collinear at |rho| ~ 0.98 "
                     "because for an oblique ground-plane camera the amplification is set by "
                     "depth, which is what range measures. That is a property of the geometry, "
                     "not of the sampling.")
        else:
            limit = (f"READ THE LIMIT BEFORE THE RESULT: only {x_span:.1f} x {y_span:.1f} m is "
                     f"sampled and image column tracks range at |rho| = {rho_u:.2f}, so range, "
                     "image row and projection conditioning are collinear here. A weak or "
                     "negative R^2 does NOT show that geometry fails -- it shows this dataset "
                     "cannot test it.")
        fig.text(0.5, -0.06,
                 "Point area in (b) scales with detections in the cell. sigma_pix is fitted on "
                 "leave-region-out folds, so each cell's prediction is out-of-region. Ground "
                 "truth positions the dots in (a) and measures the residual; it never enters a "
                 "projection, Jacobian or covariance. " + limit,
                 ha="center", va="top", fontsize=7.4, color="#333333", wrap=True)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"fig_g4_{camera}_geometry_check.{ext}", bbox_inches="tight")
        plt.close(fig)
        # deviation_m is a per-detection plotting intermediate, not evidence.
        out[camera] = {k: v for k, v in cells.items() if k != "deviation_m"}
    return out


def fig_g3(results: dict, kind_label: str) -> None:
    """Held-out model comparison as ABSOLUTE scores, in the units each metric is in.

    The previous version plotted NLL *relative to R1-iso* on a symlog axis. That hides
    the two things a reader needs: what the numbers actually are, and whether any model
    is calibrated. A relative bar of "-0.04 nats" is unreadable; "the model claims 41 mm,
    reality is 56 mm, and its 90 % interval caught 82 %" is not.

    Left: absolute held-out NLL per camera per model, lowest wins, annotated.
    Right: calibration -- claimed RMS against measured RMS, with 90 % coverage, in mm.

    R1-range carries the same single free parameter as R2-geom and is fitted by the same
    closed-form ML; only the shape differs. It is here because a reviewer is entitled to
    ask whether JJ^T does anything beyond encoding distance from the camera.
    """

    usable = [c for c in CAMERAS if results.get(c, {}).get("status") == "ok"]
    if not usable:
        return
    models = ("R1-iso", "R1-range", "R2-geom", "R1-full")
    labels = {"R1-iso": "constant", "R1-range": "range-only",
              "R2-geom": r"geometry $JJ^{T}$", "R1-full": "full 2x2"}

    fig, (ax_nll, ax_cal) = plt.subplots(1, 2, figsize=(14.2, 4.8))
    width = 0.2
    positions = np.arange(len(usable), dtype=float)

    for index, name in enumerate(models):
        values = [results[c][name]["nll"] for c in usable]
        offset = (index - (len(models) - 1) / 2.0) * width
        ax_nll.bar(positions + offset, values, width * 0.92, zorder=3,
                   color=MODEL_COLORS[name], edgecolor="white", linewidth=0.7,
                   label=labels[name])
        for x, v in zip(positions + offset, values):
            ax_nll.text(x, v, f"{v:.2f}", ha="center", fontsize=6.8, rotation=90,
                        va="bottom" if v >= 0 else "top", zorder=6)
    # Mark the winner per camera so "lower is better" needs no decoding.
    for x, camera in zip(positions, usable):
        best = min(models, key=lambda m: results[camera][m]["nll"])
        ax_nll.text(x, ax_nll.get_ylim()[1], f"best: {labels[best]}", ha="center",
                    va="top", fontsize=7.5, fontweight="bold", color="#333333")
    ax_nll.grid(True, axis="y", zorder=0)
    ax_nll.axhline(0.0, color="#333333", lw=1.0)
    ax_nll.set_xticks(positions)
    ax_nll.set_xticklabels([c.replace("camera_", "") for c in usable])
    ax_nll.set_ylabel("held-out NLL  [nats]   (lower is better)")
    ax_nll.set_title("(a)  absolute held-out score, not a difference",
                     fontweight="bold", fontsize=10)
    ax_nll.legend(fontsize=7.5, ncol=2)

    for index, name in enumerate(models):
        claimed = [1000.0 * results[c][name]["stated_rms_m"] for c in usable]
        offset = (index - (len(models) - 1) / 2.0) * width
        ax_cal.bar(positions + offset, claimed, width * 0.92, zorder=3,
                   color=MODEL_COLORS[name], edgecolor="white", linewidth=0.7,
                   label=labels[name])
        for x, c_name in zip(positions + offset, usable):
            cov = results[c_name][name]["coverage_90"]
            ax_cal.text(x, 1.0, f"{100 * cov:.0f}%", ha="center", va="bottom",
                        fontsize=6.5, rotation=90, color="white", zorder=6)
    measured = [1000.0 * results[c]["R1-iso"]["rms_m"] for c in usable]
    ax_cal.plot(positions, measured, "k_", markersize=30, markeredgewidth=2.4, zorder=7,
                label="measured RMS")
    ax_cal.grid(True, axis="y", zorder=0)
    ax_cal.set_xticks(positions)
    ax_cal.set_xticklabels([c.replace("camera_", "") for c in usable])
    ax_cal.set_ylabel("error magnitude [mm]")
    ax_cal.set_title("(b)  calibration: claimed RMS vs measured, % = held-out 90 % coverage\n"
                     "a model can buy coverage by claiming a huge sigma -- read both bars",
                     fontweight="bold", fontsize=9.5)
    ax_cal.legend(fontsize=7.5, ncol=2)

    fig.suptitle(f"Held-out variance-model comparison — {kind_label}",
                 fontweight="bold", fontsize=12.5, y=1.02)
    fig.text(0.5, -0.04,
             "All four models are fitted on leave-region-out folds and scored on the held-out "
             "fold. R1-iso, R1-range and R2-geom each carry ONE free parameter and differ only "
             "in the per-sample shape; R1-full carries three. Nominal coverage is 90 %. "
             "Ground truth scores the outcome and never enters a fit.",
             ha="center", va="top", fontsize=7.4, color="#333333", wrap=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_g3_heldout_model_comparison.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residuals", default="",
                        help="residual CSV to analyse; defaults to the route-based "
                             "external_camera_bias_model file. Point at a grid capture's "
                             "grid_residuals.csv to use 2-D coverage.")
    parser.add_argument("--out", default="", help="output directory override")
    args = parser.parse_args()
    global RESIDUALS_CSV, OUT
    if args.residuals:
        RESIDUALS_CSV = Path(args.residuals).resolve()
    if args.out:
        OUT = Path(args.out).resolve()
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
        "spatial_cells": fig_g4(per_camera, models, calib, kind="cor"),
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
