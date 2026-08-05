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
MODEL_COLORS = {"R1-iso": C_ISO, "R1-full": C_FULL, "R2-geom": C_GEOM}


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
        collected = {
            (name, centring): {"res": [], "cov": []}
            for name in ("R1-iso", "R1-full", "R2-geom")
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
            sigma_pix.append(math.sqrt(pixel_var))

            for name, covariances in (
                ("R1-iso", [iso] * int(test.sum())),
                ("R1-full", [full] * int(test.sum())),
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
    """Does the geometric prediction track the observed residual growth with range?

    Two rows, because the answer differs: the RAW residual is what pure projection
    geometry should explain, while the deployed correction is itself a function of
    range and so removes part of that structure before the variance model sees it.
    """

    kinds = (("raw", "raw projection"), ("cor", "after the deployed correction"))
    fig, axes = plt.subplots(len(kinds), len(CAMERAS),
                             figsize=(3.9 * len(CAMERAS), 3.5 * len(kinds)),
                             sharex="col")
    stats: dict[str, dict] = {}
    for row, (kind, kind_label) in enumerate(kinds):
        results = results_by_kind[kind]
        stats[kind] = {}
        for column, camera in enumerate(CAMERAS):
            ax = axes[row][column]
            bucket = per_camera[camera]
            entry = results.get(camera, {})
            if entry.get("status") != "ok":
                ax.set_title(f"{camera} — insufficient samples", fontsize=9)
                continue
            res = bucket[kind] - bucket[kind].mean(axis=0)
            shapes = geometric_shapes(bucket["jac"])
            sigma_pix = entry["sigma_pix_px"]["mean"]
            observed = np.sum(res**2, axis=1)
            predicted = (sigma_pix**2) * np.trace(shapes, axis1=1, axis2=2)
            centres, obs_med, counts = binned(bucket["range"], observed, RANGE_EDGES)
            _, pred_med, _ = binned(bucket["range"], predicted, RANGE_EDGES)
            keep = counts >= 5
            ax.plot(centres[keep], np.sqrt(obs_med[keep]) * 1000.0, "o-", color="#333333",
                    lw=1.8, ms=4, label="observed (median)")
            ax.plot(centres[keep], np.sqrt(pred_med[keep]) * 1000.0, "s--", color=C_GEOM,
                    lw=1.8, ms=4, label=r"geometric $\sigma_{pix}\|J\|$")
            if row == len(kinds) - 1:
                ax.set_xlabel("range [m]")
            ax.set_title(f"{camera.replace('camera_', 'camera ')} · {kind_label}\n"
                         f"$\\sigma_{{pix}}$ = {sigma_pix:.2f} px",
                         fontweight="bold", fontsize=9)
            if row == 0 and column == 0:
                ax.legend(fontsize=7.5)
            ratio = np.sqrt(obs_med[keep] / pred_med[keep])
            rho = spearman(pred_med[keep], obs_med[keep])[0] if keep.sum() >= 3 else float("nan")
            ax.text(
                0.03, 0.03,
                f"shape $\\rho$ = {rho:+.2f}  ({int(keep.sum())} bins)\n"
                f"obs/pred = {np.median(ratio):.2f}",
                transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2.0),
            )
            # Spearman of observed-vs-predicted across bins says whether the SHAPE
            # of the range dependence matches, independent of overall scale.
            stats[kind][camera] = {
                "sigma_pix_px": sigma_pix,
                "observed_over_predicted_std_median": float(np.median(ratio)),
                "observed_over_predicted_std_range": [float(np.min(ratio)),
                                                      float(np.max(ratio))],
                "shape_spearman_binned": (
                    float(spearman(pred_med[keep], obs_med[keep])[0])
                    if keep.sum() >= 3 else None
                ),
                "range_bins_used": int(keep.sum()),
            }
        axes[row][0].set_ylabel(f"residual magnitude [mm]\n({kind_label})")
    fig.suptitle("Observed residual growth with range versus the one-parameter geometric "
                 "prediction\n"
                 r"($\rho$ = Spearman of binned observed vs predicted; obs/pred = median std "
                 "ratio. A/B cover only 3 range bins)",
                 fontsize=12.0, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
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
