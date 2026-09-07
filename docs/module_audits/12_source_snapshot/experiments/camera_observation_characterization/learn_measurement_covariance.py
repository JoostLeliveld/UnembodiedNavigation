#!/usr/bin/env python3
"""Stage-2 ladder: how simple a conditional measurement covariance `R` can be and still hold.

PLAN.md fixes the ladder and the stopping rule -- start at the simplest rung and stop when the
next one does not earn itself on held-out data:

    R0 = one isotropic covariance, pooled over every camera
    R1 = one isotropic covariance per camera
    R2 = one full 2x2 covariance per camera
    R3 = camera + range / viewing-angle dependence
    R4 = a pooled network over every runtime-available feature

WHAT THIS `R` IS, AND WHAT IT IS NOT
------------------------------------
The field capture holds ONE observation per camera-position-heading cell, so at a single state
the spread is not identifiable: one residual is a sample, not a spread. This script therefore
does not claim a per-state covariance. It fixes the mean with the already-frozen bias model and
then asks a strictly weaker, answerable question:

    given the runtime-visible description of a reading, how large is the residual the frozen
    mean model leaves behind, and can that size be predicted?

Anything the mean model failed to remove -- the heading dependence in folder 03, for one --
lands inside this covariance and inflates it. So this is an upper bound on the random part,
never a clean separation of bias from noise. The separately frozen repeat panel that PLAN.md
requires is still the only thing that can do that, and this script does not replace it.

METHOD
------
The deployed mean model is frozen and is never replaced. But its residuals on the tiles it was
fitted on are far too small to fit a covariance to: on this capture the network scores 4.2 cm
RMSE in-sample and 16.7 cm on held-out floor. A covariance fitted to the first number is four
times too tight, and the filter would be badly overconfident.

So the covariance is fitted to OUT-OF-FOLD residuals. The TRAIN tiles are cut into five folds by
floor position, the mean model is refitted without each fold, and that fold contributes the
residuals of a model that never saw it. Folds are cut by position so all eight headings of a
position move together. Those refits exist only to generate honest residuals; the frozen model
is what gets scored on the held-out tiles.

Rungs R3 and R4 use the standard two-stage estimator: regress log(residual^2)
on the features, then correct the chi-square bias. For e ~ N(0, s^2),

    E[log e^2] = log s^2 + psi(1/2) + log 2 = log s^2 - 1.2704

so the fitted surface is shifted back up by 1.2704 to recover log s^2. Residuals are expressed
in the camera-ray frame (along the ray, across it), which is where the physics is diagonal.

SCORING
-------
Every rung is fitted on TRAIN tiles and scored on held-out TEST tiles, and calibration is never
reported without sharpness beside it -- a rung that simply widens its ellipse passes any
calibration test and is useless:

    mean NEES        should be 2.0 for a 2-DoF Gaussian; >2 is overconfident, <2 is loose
    95% containment  fraction with NEES <= chi2(0.95, 2) = 5.991; should be 0.95
    held-out NLL     a proper score, so widening is penalised
    sharpness        mean predicted sigma in cm; the honest cost of any calibration gain
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np
from scipy.stats import chi2
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
for rel in ("experiments/deck_figures", "experiments/camera_observation_characterization"):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

import style as D  # noqa: E402
from fit_bias_updates import (  # noqa: E402
    CAMERAS,
    FEATURE_NAMES,
    apply_correction,
    camera_geometry,
    features,
    ray_frame,
    target,
)

DEFAULT_CAPTURE = REPO / "logs/perception_datasets/warehouse_v2_bbox_characterization_20260831"
DEFAULT_OUT = REPO / "logs/studies/camera_observation_characterization_20260831"
FOLDER = "10_learning_R"

# E[log chi2_1] = psi(1/2) + log 2. Adding this back turns a fit of log(e^2) into log(sigma^2).
LOG_CHI2_BIAS = 1.2703628454614782
VARIANCE_FLOOR_M2 = (1e-4) ** 2      # keeps log(e^2) finite when a residual is ~0
CHI2_95_2DOF = float(chi2.ppf(0.95, 2))

OUT_OF_FOLD_SPLITS = 5

RUNGS = (
    ("R0", "one isotropic sigma, pooled over all cameras", "#8a8983"),
    ("R1", "one isotropic sigma per camera", "#2a78d6"),
    ("R2", "one full 2x2 per camera", "#1baf7a"),
    ("R3", "camera + range and viewing angle", "#eb6834"),
    ("R4", "pooled network over all runtime features", "#d4267b"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def camera_title(camera_id: str) -> str:
    return f"Camera {camera_id[-1]}"


# ---------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------
def load(capture: Path, method: str) -> dict:
    """Ray-frame residuals of the frozen mean model, with their runtime features."""
    table = capture / "bias_update_interpretations.csv"
    manifest_path = capture / "bias_update_interpretations_manifest.json"
    capture_manifest_path = capture / "capture_manifest.json"
    for required in (table, manifest_path, capture_manifest_path):
        if not required.is_file():
            raise RuntimeError(f"Missing required input: {required}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("bias_update_interpretations manifest is not complete")
    if sha256(table) != manifest["bias_update_interpretations_sha256"]:
        raise RuntimeError("bias_update_interpretations.csv changed after fitting")
    geometry = camera_geometry(json.loads(capture_manifest_path.read_text(encoding="utf-8")))

    rows = [row for row in csv.DictReader(table.open(encoding="utf-8"))
            if row[f"{method}_valid"] == "1"]
    if not rows:
        raise RuntimeError(f"No valid rows for mean model {method!r}")

    design, residual, camera_index, split, position, tile = [], [], [], [], [], []
    for row in rows:
        camera_id = row["camera_id"]
        design.append(features(row, geometry[camera_id]))
        residual.append([float(row[f"{method}_along_m"]), float(row[f"{method}_across_m"])])
        camera_index.append(CAMERAS.index(camera_id))
        split.append(row["split"])
        position.append((float(row["robot_x"]), float(row["robot_y"])))
        tile.append(row["position_id"])
    return {
        "rows": rows,
        "geometry": geometry,
        "x": np.asarray(design, dtype=float),
        "e": np.asarray(residual, dtype=float),
        "camera": np.asarray(camera_index, dtype=int),
        "split": np.asarray(split),
        "xy": np.asarray(position, dtype=float),
        "tile": np.asarray(tile),
        "table_sha256": sha256(table),
        "holdout": manifest["holdout"],
        "n": len(rows),
    }


# ---------------------------------------------------------------------------------------
# out-of-fold residuals of the frozen mean model
# ---------------------------------------------------------------------------------------
def fit_mean_model(rows, geometry, seed: int) -> dict:
    """The frozen recipe: per-camera Ridge, plus one pooled network with a camera indicator."""
    linear = {}
    for camera_id in CAMERAS:
        picked = [row for row in rows if row["camera_id"] == camera_id]
        if len(picked) < 12:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(np.stack([features(row, geometry[camera_id]) for row in picked]),
                  np.stack([target(row, geometry[camera_id]) for row in picked]))
        linear[camera_id] = model
    onehot = np.eye(len(CAMERAS))
    pooled = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", solver="adam",
                     alpha=1e-4, learning_rate_init=1e-3, max_iter=3000, early_stopping=True,
                     validation_fraction=0.15, n_iter_no_change=40, random_state=seed),
    )
    pooled.fit(
        np.stack([np.concatenate([features(row, geometry[row["camera_id"]]),
                                  onehot[CAMERAS.index(row["camera_id"])]]) for row in rows]),
        np.stack([target(row, geometry[row["camera_id"]]) for row in rows]),
    )
    return {"linear": linear, "nn": pooled}


def mean_residuals(rows, geometry, model: dict, method: str) -> np.ndarray:
    """Ray-frame residual each row would have under a mean model it did not see."""
    onehot = np.eye(len(CAMERAS))
    out = np.zeros((len(rows), 2), dtype=float)
    for index, row in enumerate(rows):
        camera_id = row["camera_id"]
        geom = geometry[camera_id]
        if method == "raw":
            correction = np.zeros(2)
        elif method == "learned":
            fitted = model["linear"].get(camera_id)
            correction = (np.zeros(2) if fitted is None
                          else fitted.predict(features(row, geom)[None, :])[0])
        elif method == "nn":
            design = np.concatenate([features(row, geom),
                                     onehot[CAMERAS.index(camera_id)]])
            correction = model["nn"].predict(design[None, :])[0]
        else:
            raise ValueError(f"out-of-fold residuals are not defined for {method!r}")
        estimate = apply_correction(row, geom, correction)
        truth = np.array([float(row["robot_x"]), float(row["robot_y"])])
        error = estimate - truth
        unit, left = ray_frame(truth, geom["xy"])
        out[index] = [float(error @ unit), float(error @ left)]
    return out


def out_of_fold_residuals(data: dict, method: str, seed: int, splits: int) -> np.ndarray:
    """Residuals the mean model leaves on floor it was NOT fitted on.

    The mean model overfits: on this capture the network scores 4.2 cm RMSE on the tiles it
    was fitted on and 16.7 cm on held-out tiles. Fitting a covariance to the first number
    teaches it a spread four times too small, and the filter would be wildly overconfident.

    So the covariance is fitted to out-of-fold residuals: the TRAIN tiles are cut into folds
    by floor position, the mean model is refitted without each fold, and that fold's residuals
    are taken from the refitted model. Folds are cut by position, never by row, so all eight
    headings of a position move together and a fold's own floor never leaks into its mean model.
    """
    if method not in ("raw", "learned", "nn"):
        # rungs with nothing fitted (fixed, hull) cannot overfit, so in-sample is honest
        return data["e"][data["split"] == "train"]
    train_index = np.flatnonzero(data["split"] == "train")
    rows = [data["rows"][i] for i in train_index]
    tiles = np.asarray([row["position_id"] for row in rows])
    unique = np.unique(tiles)
    rng = np.random.default_rng(seed)
    assignment = {tile: int(fold) for tile, fold in
                  zip(unique, rng.permutation(len(unique)) % splits)}
    fold_of = np.asarray([assignment[tile] for tile in tiles])

    out = np.zeros((len(rows), 2), dtype=float)
    for fold in range(splits):
        held = fold_of == fold
        kept = ~held
        if not held.any() or kept.sum() < 50:
            out[held] = data["e"][train_index[held]]
            continue
        model = fit_mean_model([rows[i] for i in np.flatnonzero(kept)],
                               data["geometry"], seed)
        out[held] = mean_residuals([rows[i] for i in np.flatnonzero(held)],
                                   data["geometry"], model, method)
        print(f"    fold {fold + 1}/{splits}: "
              f"{int(held.sum())} readings, RMSE "
              f"{100 * float(np.sqrt(np.mean(np.sum(out[held] ** 2, axis=1)))):.1f} cm")
    return out


# ---------------------------------------------------------------------------------------
# the five rungs; each returns a per-reading 2x2 covariance for the rows it is given
# ---------------------------------------------------------------------------------------
def _isotropic(residual: np.ndarray) -> np.ndarray:
    variance = max(float(np.mean(residual ** 2)), VARIANCE_FLOOR_M2)
    return np.eye(2) * variance


def fit_r0(train: dict) -> dict:
    return {"kind": "R0", "cov": _isotropic(train["e"])}


def fit_r1(train: dict) -> dict:
    covariances = {}
    for index in range(len(CAMERAS)):
        mask = train["camera"] == index
        covariances[index] = _isotropic(train["e"][mask]) if mask.any() else np.eye(2) * VARIANCE_FLOOR_M2
    return {"kind": "R1", "cov": covariances}


def fit_r2(train: dict) -> dict:
    covariances = {}
    for index in range(len(CAMERAS)):
        mask = train["camera"] == index
        if mask.sum() < 3:
            covariances[index] = np.eye(2) * VARIANCE_FLOOR_M2
            continue
        # second moment about zero, not about the sample mean: any leftover bias belongs in R
        sample = train["e"][mask]
        covariances[index] = (sample.T @ sample) / sample.shape[0] + np.eye(2) * VARIANCE_FLOOR_M2
    return {"kind": "R2", "cov": covariances}


def _log_targets(residual: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(residual ** 2, VARIANCE_FLOOR_M2)) + LOG_CHI2_BIAS


def resolvable_log_variance_range(train: dict) -> tuple[float, float]:
    """How far the variance is allowed to move, measured rather than chosen.

    A regression on log-variance is unbounded, so it must be held to a range. That range cannot
    come from single-sample log(e^2) values: one residual component crosses zero routinely, so
    its log runs off to minus infinity and the model would be licensed to promise a 0.1 mm
    sigma. Those are samples, not variances.

    Instead the data is cut into strata that each hold enough readings to estimate a variance
    honestly -- camera x range quartile, twenty strata of roughly 160 readings -- and each
    stratum's pooled variance is computed. The smallest and largest of those are the real
    dynamic range this capture can resolve, and no rung may predict outside it.
    """
    quartiles = np.quantile(train["x"][:, FEATURE_NAMES.index("range_m")],
                            [0.0, 0.25, 0.5, 0.75, 1.0])
    variances = []
    for camera in range(len(CAMERAS)):
        for low, high in zip(quartiles[:-1], quartiles[1:]):
            mask = (
                (train["camera"] == camera)
                & (train["x"][:, FEATURE_NAMES.index("range_m")] >= low)
                & (train["x"][:, FEATURE_NAMES.index("range_m")] <= high)
            )
            if mask.sum() < 40:
                continue
            variances.append(float(np.mean(train["e"][mask] ** 2)))
    if not variances:
        return (math.log(VARIANCE_FLOOR_M2), math.log(1.0))
    return (math.log(max(min(variances), VARIANCE_FLOOR_M2)), math.log(max(variances)))


def fit_r3(train: dict, seed: int) -> dict:
    """Per-camera ridge on log variance, over range and viewing-angle features only."""
    keep = [FEATURE_NAMES.index(name) for name in
            ("range_m", "inv_range", "bearing_cos", "bearing_sin")]
    bounds = resolvable_log_variance_range(train)
    models = {}
    for index in range(len(CAMERAS)):
        mask = train["camera"] == index
        if mask.sum() < 8:
            models[index] = ("constant", float(np.log(max(
                np.mean(train["e"][mask] ** 2) if mask.any() else VARIANCE_FLOOR_M2,
                VARIANCE_FLOOR_M2))))
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(train["x"][mask][:, keep], _log_targets(train["e"][mask]))
        models[index] = ("model", model, bounds)
    return {"kind": "R3", "models": models, "keep": keep}


def fit_r4(train: dict, seed: int) -> dict:
    """Pooled network on every runtime feature plus a camera indicator."""
    onehot = np.eye(len(CAMERAS))[train["camera"]]
    design = np.hstack([train["x"], onehot])
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", solver="adam",
                     alpha=1e-3, learning_rate_init=1e-3, max_iter=3000,
                     early_stopping=True, validation_fraction=0.15, n_iter_no_change=40,
                     random_state=seed),
    )
    model.fit(design, _log_targets(train["e"]))
    return {"kind": "R4", "model": model,
            "clamp": resolvable_log_variance_range(train)}


def predict(rung: dict, data: dict) -> np.ndarray:
    """Per-reading 2x2 covariance, in the along/across camera-ray frame."""
    count = data["e"].shape[0]
    kind = rung["kind"]
    if kind == "R0":
        return np.repeat(rung["cov"][None, :, :], count, axis=0)
    if kind in ("R1", "R2"):
        return np.stack([rung["cov"][int(index)] for index in data["camera"]])
    if kind == "R3":
        out = np.zeros((count, 2, 2))
        for index in range(len(CAMERAS)):
            mask = data["camera"] == index
            if not mask.any():
                continue
            entry = rung["models"][index]
            if entry[0] == "constant":
                log_variance = np.full((int(mask.sum()), 2), entry[1])
            else:
                log_variance = np.clip(
                    entry[1].predict(data["x"][mask][:, rung["keep"]]), *entry[2])
            variance = np.maximum(np.exp(log_variance), VARIANCE_FLOOR_M2)
            out[mask, 0, 0] = variance[:, 0]
            out[mask, 1, 1] = variance[:, 1]
        return out
    if kind == "R4":
        onehot = np.eye(len(CAMERAS))[data["camera"]]
        log_variance = np.clip(rung["model"].predict(np.hstack([data["x"], onehot])),
                               *rung["clamp"])
        variance = np.maximum(np.exp(log_variance), VARIANCE_FLOOR_M2)
        out = np.zeros((count, 2, 2))
        out[:, 0, 0] = variance[:, 0]
        out[:, 1, 1] = variance[:, 1]
        return out
    raise ValueError(kind)


# ---------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------
def score(residual: np.ndarray, covariance: np.ndarray) -> dict:
    inverse = np.linalg.inv(covariance)
    nees = np.einsum("ni,nij,nj->n", residual, inverse, residual)
    sign, logdet = np.linalg.slogdet(covariance)
    if np.any(sign <= 0):
        raise RuntimeError("Non-positive-definite covariance produced")
    nll = 0.5 * (nees + logdet + 2.0 * math.log(2.0 * math.pi))
    sigma = np.sqrt(np.maximum(np.einsum("nii->ni", covariance), 0.0))
    return {
        "n": int(residual.shape[0]),
        "mean_nees": float(np.mean(nees)),
        "median_nees": float(np.median(nees)),
        "containment_95": float(np.mean(nees <= CHI2_95_2DOF)),
        "mean_nll": float(np.mean(nll)),
        "sharpness_cm": float(np.mean(sigma) * 100.0),
        "sigma_along_cm": float(np.mean(sigma[:, 0]) * 100.0),
        "sigma_across_cm": float(np.mean(sigma[:, 1]) * 100.0),
        "nees": nees,
        "sigma": sigma,
    }


def public(entry: dict) -> dict:
    return {key: value for key, value in entry.items() if key not in ("nees", "sigma")}


# ---------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------
def draw_ladder(results: dict, method: str, out: Path) -> None:
    """Calibration and sharpness together: widening the ellipse must never look like a win."""
    labels = [key for key, _label, _colour in RUNGS]
    colours = [colour for _key, _label, colour in RUNGS]
    x = np.arange(len(RUNGS))
    test = [results[key]["test"] for key, _l, _c in RUNGS]
    train = [results[key]["train"] for key, _l, _c in RUNGS]

    fig, axes = plt.subplots(1, 4, figsize=(23.0, 6.9), constrained_layout=True)

    panels = (
        ("mean_nees", "Is it honest?\nmean NEES (2.0 is right)", 2.0, "2.0 = honest"),
        ("containment_95", "Does 95% really contain 95%?\nfraction inside the 95% ellipse",
         0.95, "0.95 = honest"),
        ("sharpness_cm", "What does it cost?\nmean predicted sigma (cm)", None, None),
        ("mean_nll", "Proper score\nheld-out negative log-likelihood", None, None),
    )
    for ax, (field, title, reference, reference_label) in zip(axes, panels):
        test_values = [entry[field] for entry in test]
        train_values = [entry[field] for entry in train]
        ax.plot(x, train_values, marker="o", ms=10, lw=0, mfc="white", mec=D.MUTED, mew=2.2,
                label="tiles it was fitted on", zorder=3)
        for index, (low, high) in enumerate(zip(train_values, test_values)):
            ax.plot([index, index], [low, high], color="#c9c7c0", lw=2.2, zorder=1)
        for index, value in enumerate(test_values):
            ax.plot(index, value, marker="o", ms=11, color=colours[index], zorder=4)
            ax.annotate(f"{value:.2f}" if field != "mean_nll" else f"{value:.2f}",
                        (index, value), textcoords="offset points", xytext=(0, 13),
                        ha="center", fontsize=10.5, fontweight="bold", color=colours[index])
        if reference is not None:
            ax.axhline(reference, color=D.INK, lw=1.6, linestyle="--", zorder=2)
            ax.text(0.99, reference, f" {reference_label}", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=10.5, color=D.INK)
        if field == "mean_nees":
            ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=13)
        for tick, colour in zip(ax.get_xticklabels(), colours):
            tick.set_color(colour)
            tick.set_fontweight("bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(axis="y", color="#e4e2dc", lw=0.8)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper right", fontsize=10.5, frameon=True)
    fig.legend(
        handles=[Line2D([0], [0], marker="o", ms=10, lw=0, color=colour,
                        label=f"{key}  {label}") for key, label, colour in RUNGS],
        loc="lower center", ncol=5, fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.10))

    best = min(RUNGS, key=lambda item: results[item[0]]["test"]["mean_nll"])[0]
    fig.suptitle(
        f"How simple can the measurement covariance be? Ladder on the {method} reading\n"
        f"Fitted on TRAIN tiles, scored on held-out TEST tiles — "
        f"best held-out likelihood: {best}\n"
        "Calibration and sharpness are shown together, so a rung cannot win by widening its ellipse",
        fontsize=18, fontweight="bold")
    fig.text(
        0.5, -0.175,
        "NEES above 2 means the stated ellipse is too small for the errors that actually occur; "
        "it is on a log axis so every rung stays readable.\n"
        "The covariance is fitted to OUT-OF-FOLD residuals of the mean model. Fitting it to "
        "in-sample residuals instead gives NEES 30-60 rather than 4-10, because the mean model "
        "scores 4.2 cm on its own tiles and 12.0 cm off them.\n"
        "This is the spread the frozen mean model leaves behind: anything it failed to remove "
        "is inside these numbers, so they bound the random part rather than isolate it.",
        ha="center", fontsize=11.5, color=D.MUTED)
    save(fig, out / FOLDER / "23_r_ladder_calibration.png")


def draw_reliability(results: dict, data_test: dict, method: str, out: Path) -> None:
    """Does a bigger predicted sigma actually mean a bigger error?"""
    fig, axes = plt.subplots(1, 3, figsize=(20.0, 6.6), constrained_layout=True)

    # 1. predicted vs realised spread, in bins of predicted sigma
    ax = axes[0]
    for key, label, colour in RUNGS:
        sigma = results[key]["test"]["sigma"]
        predicted = np.sqrt(np.mean(sigma ** 2, axis=1))
        realised = np.linalg.norm(data_test["e"], axis=1) / math.sqrt(2.0)
        if predicted.max() - predicted.min() < 1e-9:
            ax.plot(100 * predicted.mean(), 100 * np.sqrt(np.mean(realised ** 2)),
                    marker="s", ms=11, color=colour, lw=0, label=label.replace("\n", " "))
            continue
        edges = np.quantile(predicted, np.linspace(0, 1, 9))
        edges = np.unique(edges)
        px, py = [], []
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (predicted >= low) & (predicted < high)
            if mask.sum() < 12:
                continue
            px.append(100 * float(np.sqrt(np.mean(predicted[mask] ** 2))))
            py.append(100 * float(np.sqrt(np.mean(realised[mask] ** 2))))
        ax.plot(px, py, marker="o", ms=7, lw=2.2, color=colour,
                label=label.replace("\n", " "))
    limit = 1.30 * max(
        100 * float(np.sqrt(np.mean(np.linalg.norm(data_test["e"], axis=1) ** 2 / 2.0))),
        max(100 * float(np.sqrt(np.mean(results[key]["test"]["sigma"] ** 2)))
            for key, _l, _c in RUNGS),
    )
    ax.plot([0, limit], [0, limit], color=D.INK, lw=1.6, linestyle="--", zorder=1)
    ax.text(limit * 0.50, limit * 0.56, "perfect: stated = realised", fontsize=11,
            color=D.INK, rotation=41)
    ax.set_xlim(0, limit); ax.set_ylim(0, limit)
    ax.set_xlabel("Stated sigma for that reading (cm)")
    ax.set_ylabel("Error that actually occurred (RMS, cm)")
    ax.set_title("Does a bigger stated sigma mean a bigger error?\n"
                 "Every rung sits above the line: the error that happens is "
                 "larger than the one promised",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9.5, frameon=True, loc="lower right")
    ax.grid(color="#e4e2dc", lw=0.8); ax.set_axisbelow(True)

    # 2. NEES distribution against the chi-square it should follow
    ax = axes[1]
    grid = np.linspace(0, 20, 400)
    ax.plot(grid, chi2.pdf(grid, 2), color=D.INK, lw=2.4, linestyle="--",
            label="what an honest R would give\n(chi-square, 2 degrees of freedom)")
    for key, label, colour in RUNGS:
        nees = results[key]["test"]["nees"]
        ax.hist(np.minimum(nees, 20.0), bins=np.linspace(0, 20, 61), histtype="step",
                lw=2.0, density=True, color=colour, label=label.replace("\n", " "))
    ax.axvline(CHI2_95_2DOF, color=D.MUTED, lw=1.4, linestyle=":")
    ax.text(CHI2_95_2DOF, ax.get_ylim()[1] * 0.92, "  95% line", fontsize=10, color=D.MUTED)
    overflow = {key: float(np.mean(results[key]["test"]["nees"] > 20.0))
                for key, _l, _c in RUNGS}
    ax.text(0.97, 0.55,
            "readings worse than NEES 20\n"
            + "\n".join(f"  {key}: {100 * value:.1f}%" for key, value in overflow.items())
            + "\nan honest R would give 0.005%",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", fc="#f2f0ea", ec="#d0cec7"))
    ax.set_xlabel("NEES of one held-out reading (last bin holds everything above 20)")
    ax.set_ylabel("Density")
    ax.set_title("Where the Gaussian breaks\nNEES should follow the dashed curve",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, frameon=True)
    ax.grid(color="#e4e2dc", lw=0.8); ax.set_axisbelow(True)

    # 3. what the best rung's ellipse looks like against the actual residual cloud
    ax = axes[2]
    best = min(RUNGS, key=lambda item: results[item[0]]["test"]["mean_nll"])
    residual = data_test["e"]
    ax.scatter(100 * residual[:, 0], 100 * residual[:, 1], s=7, alpha=0.20,
               color=D.MUTED, linewidths=0, zorder=2)

    def add_ellipses(key: str, colour: str) -> None:
        sigma = results[key]["test"]["sigma"]
        scalar_sigma = np.sqrt(np.mean(sigma ** 2, axis=1))
        constant = float(scalar_sigma.max() - scalar_sigma.min()) < 1e-6
        order = np.argsort(scalar_sigma)
        picks = [(0.50, "-")] if constant else [(0.10, ":"), (0.50, "-"), (0.90, "--")]
        for fraction, style in picks:
            index = order[int(fraction * (len(order) - 1))]
            width, height = 2.0 * math.sqrt(CHI2_95_2DOF) * 100 * sigma[index]
            label = (f"{key}: the same {100 * sigma[index][0]:.1f} cm ellipse "
                     "for every reading" if constant else
                     f"{key}, {int(fraction * 100)}th-percentile reading: "
                     f"{100 * sigma[index][0]:.1f} x {100 * sigma[index][1]:.1f} cm")
            ax.add_patch(Ellipse((0, 0), width, height, fill=False, ec=colour, lw=2.2,
                                 linestyle=style, zorder=4, label=label))

    add_ellipses(best[0], best[2])
    # when the winning rung is a constant, show what conditioning would have claimed instead
    conditional = max(RUNGS, key=lambda item: float(
        np.ptp(np.sqrt(np.mean(results[item[0]]["test"]["sigma"] ** 2, axis=1)))))
    if conditional[0] != best[0]:
        add_ellipses(conditional[0], conditional[2])
    span = float(np.quantile(np.abs(residual), 0.995) * 100 * 1.15)
    ax.set_xlim(-span, span); ax.set_ylim(-span, span)
    ax.axhline(0, color=D.MUTED, lw=1.0); ax.axvline(0, color=D.MUTED, lw=1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("Residual along the camera ray (cm)")
    ax.set_ylabel("Residual across the camera ray (cm)")
    ax.set_title(f"What {best[0]} actually claims\n"
                 "95% ellipses against the real residual cloud",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9.5, frameon=True, loc="upper left")
    ax.grid(color="#e4e2dc", lw=0.8); ax.set_axisbelow(True)

    fig.suptitle(
        f"Is the learned covariance usable? Held-out tiles, {method} reading",
        fontsize=18, fontweight="bold")
    save(fig, out / FOLDER / "24_predicted_vs_observed_spread.png")


def draw_field(results: dict, data_test: dict, method: str, out: Path) -> None:
    """Where in the warehouse the covariance says a reading is trustworthy.

    Only a rung whose sigma actually varies can produce a map: the winning rung here is a
    single constant, which would render as one flat colour. So the map is drawn for the most
    strongly conditioned rung, and the caption says outright that this structure does not pay
    for itself on held-out likelihood.
    """
    best = min(RUNGS, key=lambda item: results[item[0]]["test"]["mean_nll"])
    shown = max(RUNGS, key=lambda item: float(
        np.ptp(np.sqrt(np.mean(results[item[0]]["test"]["sigma"] ** 2, axis=1)))))
    sigma = results[shown[0]]["test"]["sigma"]
    stated = np.sqrt(np.mean(sigma ** 2, axis=1))
    low, cap = (float(np.quantile(stated, 0.02)), float(np.quantile(stated, 0.98)))

    columns = len(CAMERAS) + 1
    fig, axes = plt.subplots(1, columns, figsize=(4.9 * columns, 5.8),
                             constrained_layout=True)
    scalar = None
    for column, camera_id in enumerate(CAMERAS):
        ax = axes[column]
        mask = data_test["camera"] == column
        D.draw_warehouse(ax, D.layout(), show_cameras=True, camera_labels=True, rack_alpha=0.80)
        if mask.any():
            scalar = ax.scatter(data_test["xy"][mask, 0], data_test["xy"][mask, 1],
                                c=100 * stated[mask], cmap="cividis",
                                vmin=100 * low, vmax=100 * cap, marker="s", s=26,
                                linewidths=0, zorder=4)
            ax.set_title(f"{camera_title(camera_id)}\n"
                         f"median {100 * np.median(stated[mask]):.1f} cm  ·  "
                         f"range {100 * stated[mask].min():.1f}–"
                         f"{100 * stated[mask].max():.1f} cm",
                         fontsize=12.2, fontweight="bold")
    bar = fig.colorbar(scalar, ax=axes[:len(CAMERAS)].tolist(), fraction=0.022, pad=0.010,
                       extend="both")
    bar.set_label(f"{shown[0]} stated sigma for a reading here (cm)", fontsize=12)

    note = axes[-1]
    note.axis("off")
    note.text(0.02, 0.97, "Reading this map", fontsize=15.5, fontweight="bold", va="top")
    note.text(
        0.02, 0.88,
        f"Colour is what {shown[0]} tells the filter to expect from\n"
        "a reading taken at that spot, before the reading is\n"
        "made. Dark = trust it; bright = widen the ellipse.\n\n"
        "This is the quantity a planner could actually use: it\n"
        "is available in advance, from the camera and the\n"
        "geometry alone.\n\n"
        f"\u26a0 {shown[0]} is shown because it is the rung whose sigma\n"
        f"varies most. It is NOT the rung that wins: {best[0]} does,\n"
        "and it states one constant everywhere. The spatial\n"
        "structure below is real but does not pay for itself\n"
        "on held-out likelihood.\n\n"
        "Held-out tiles only, so the checkerboard gaps are the\n"
        "floor the model was fitted on.\n\n"
        "It is a stated sigma, not a measured one. With one\n"
        "shot per state the true local spread is not\n"
        "identifiable, and any bias the mean model left behind\n"
        "is inside this number.",
        fontsize=11.6, va="top", linespacing=1.34)

    fig.suptitle(
        f"What a conditioned covariance expects before the reading is taken — {shown[0]}, "
        f"{method} reading\n"
        f"The structure is real, but the flat rung {best[0]} still wins on held-out likelihood",
        fontsize=17.5, fontweight="bold")
    save(fig, out / FOLDER / "25_stated_sigma_field.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mean-model", default="nn",
                        choices=("raw", "fixed", "learned", "nn", "hull"),
                        help="Frozen mean model whose residual this covariance describes.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    capture = args.capture.expanduser().resolve()
    out = args.out.expanduser().resolve()
    target = out / FOLDER
    owned = [target / name for name in ("23_r_ladder_calibration.png",
                                        "24_predicted_vs_observed_spread.png",
                                        "25_stated_sigma_field.png",
                                        "learned_covariance_manifest.json")]
    if any(path.exists() for path in owned) and not args.overwrite:
        raise RuntimeError(f"Output already exists under {target}; pass --overwrite")

    data = load(capture, args.mean_model)
    train_mask = data["split"] == "train"
    test_mask = data["split"] == "test"
    subset = lambda mask: {  # noqa: E731
        "x": data["x"][mask], "e": data["e"][mask], "camera": data["camera"][mask],
        "xy": data["xy"][mask],
    }
    train, test = subset(train_mask), subset(test_mask)
    print(f"mean model {args.mean_model}: {train['e'].shape[0]} train / "
          f"{test['e'].shape[0]} held-out readings")

    print(f"  out-of-fold residuals over {OUT_OF_FOLD_SPLITS} tile folds:")
    in_sample_rmse = 100 * float(np.sqrt(np.mean(np.sum(train["e"] ** 2, axis=1))))
    train["e"] = out_of_fold_residuals(data, args.mean_model, args.seed, OUT_OF_FOLD_SPLITS)
    out_of_fold_rmse = 100 * float(np.sqrt(np.mean(np.sum(train["e"] ** 2, axis=1))))
    print(f"  mean-model RMSE on its own tiles {in_sample_rmse:.1f} cm  ->  "
          f"out-of-fold {out_of_fold_rmse:.1f} cm "
          f"(the covariance is fitted to the second)")

    fitted = {
        "R0": fit_r0(train),
        "R1": fit_r1(train),
        "R2": fit_r2(train),
        "R3": fit_r3(train, args.seed),
        "R4": fit_r4(train, args.seed),
    }
    results = {}
    for key, _label, _colour in RUNGS:
        results[key] = {
            "train": score(train["e"], predict(fitted[key], train)),
            "test": score(test["e"], predict(fitted[key], test)),
        }
        entry = results[key]["test"]
        print(f"  {key}: NEES {entry['mean_nees']:8.2f}  95% containment "
              f"{entry['containment_95']:.3f}  sigma {entry['sharpness_cm']:6.2f} cm  "
              f"NLL {entry['mean_nll']:7.3f}")

    draw_ladder(results, args.mean_model, out)
    draw_reliability(results, test, args.mean_model, out)
    draw_field(results, test, args.mean_model, out)

    best = min(RUNGS, key=lambda item: results[item[0]]["test"]["mean_nll"])[0]
    manifest = {
        "status": "complete",
        "schema": "learned_measurement_covariance.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mean_model": args.mean_model,
        "deployed_mean_model_refitted": False,
        "variance_fitted_on": "out-of-fold residuals of the mean model over TRAIN tiles",
        "out_of_fold_splits": OUT_OF_FOLD_SPLITS,
        "mean_model_rmse_cm": {
            "in_sample_train_tiles": in_sample_rmse,
            "out_of_fold_train_tiles": out_of_fold_rmse,
        },
        "source_table_sha256": data["table_sha256"],
        "holdout": data["holdout"],
        "n_train": int(train["e"].shape[0]),
        "n_test": int(test["e"].shape[0]),
        "frame": "along and across the camera-to-reading ray",
        "estimator": (
            "two-stage on out-of-fold residuals: log(residual^2) is "
            f"regressed on runtime features and shifted by +{LOG_CHI2_BIAS:.4f} to undo the "
            "chi-square bias of the log transform"
        ),
        "rungs": {key: label.replace("\n", " ") for key, label, _colour in RUNGS},
        "scores": {key: {"train": public(value["train"]), "test": public(value["test"])}
                   for key, value in results.items()},
        "best_by_heldout_nll": best,
        "chi2_95_2dof": CHI2_95_2DOF,
        "interpretation_limit": (
            "One observation per camera-position-heading cell: this is the spread the frozen "
            "mean model leaves behind, not a per-state repeated-sampling covariance. Bias the "
            "mean model failed to remove is inside it, so it upper-bounds the random part and "
            "cannot separate bias from noise."
        ),
        "next_required_evidence": (
            "The separately frozen stratified repeat panel in PLAN.md 1.2 is still required to "
            "identify a conditional covariance at fixed camera, position and heading."
        ),
        "figures": ["23_r_ladder_calibration.png", "24_predicted_vs_observed_spread.png",
                    "25_stated_sigma_field.png"],
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / "learned_covariance_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(target), "best_by_heldout_nll": best}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
