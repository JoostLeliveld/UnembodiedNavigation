#!/usr/bin/env python3
"""Choose the stochastic model for the commissioned camera-measurement error.

The post-correction residual is strongly heavy-tailed: on the fit drives the excess
kurtosis is 18 (along-ray) and 52 (across-ray), the worst two percent of readings carry
about half of the total squared error, and the sample standard deviation is roughly twice
the robust (MAD) scale.  A single Gaussian therefore cannot describe it: fitted by moments
its variance is set by the tail, so the ellipse is too wide through the bulk and still too
narrow at the extremes, which is precisely the containment pattern the Gaussian ladder
shows (50 percent coverage near 0.46-0.68, 95 percent coverage never reaching 0.95).

Two model families are compared, both fitted only on fit drives and scored only on
held-out development drives.

  Gaussian     N(0, R(s))                     -- the incumbent
  Mixture      pi(s) N(0, R_good(s)) + (1-pi(s)) N(0, R_bad(s))

For the estimator, a mixture is summarised by the covariance a Kalman update should use.
Two summaries are scored, because they answer different questions:

  moment        R = pi R_good + (1-pi) R_bad     -- matches the second moment
  gated         R = R_good, and readings whose responsibility for the degraded component
                exceeds a threshold are refused                 -- robust-estimator view

Scoring uses held-out predictive log likelihood (a proper score), chi-square containment
at 50/90/95/99 percent, ellipse area (sharpness), and the frequency of the extreme tail.
The replication unit is the complete drive.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE)]

import analyze_commissioned_model as base  # noqa: E402

CAMERAS = base.CAMERAS
CHI2_2 = base.CHI2_2


def spd(matrix: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    return base.spd(matrix, floor)


def gaussian_logpdf(residual: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Log N(residual; 0, covariance) for a stack of 2-vectors and matching 2x2 matrices."""
    determinant = np.maximum(
        covariance[:, 0, 0] * covariance[:, 1, 1] - covariance[:, 0, 1] ** 2, 1e-16)
    inverse00 = covariance[:, 1, 1] / determinant
    inverse11 = covariance[:, 0, 0] / determinant
    inverse01 = -covariance[:, 0, 1] / determinant
    quadratic = (inverse00 * residual[:, 0] ** 2
                 + 2 * inverse01 * residual[:, 0] * residual[:, 1]
                 + inverse11 * residual[:, 1] ** 2)
    return -0.5 * (2 * math.log(2 * math.pi) + np.log(determinant) + quadratic)


def mahalanobis(residual: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    determinant = np.maximum(
        covariance[:, 0, 0] * covariance[:, 1, 1] - covariance[:, 0, 1] ** 2, 1e-16)
    return ((covariance[:, 1, 1] * residual[:, 0] ** 2
             - 2 * covariance[:, 0, 1] * residual[:, 0] * residual[:, 1]
             + covariance[:, 0, 0] * residual[:, 1] ** 2) / determinant)


def score(residual: np.ndarray, covariance: np.ndarray, drives: np.ndarray,
          loglik: np.ndarray | None = None) -> dict:
    """Proper score, calibration and sharpness, aggregated with the drive as the unit."""
    if loglik is None:
        loglik = gaussian_logpdf(residual, covariance)
    d2 = mahalanobis(residual, covariance)
    determinant = np.maximum(
        covariance[:, 0, 0] * covariance[:, 1, 1] - covariance[:, 0, 1] ** 2, 1e-16)
    areas = math.pi * CHI2_2["95"] * np.sqrt(determinant)
    per_drive = {d: float(np.mean(-loglik[drives == d])) for d in sorted(set(drives.tolist()))}
    values = np.asarray(list(per_drive.values()))
    return {
        "equal_drive_mean_nll": float(values.mean()),
        "drive_nll_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "per_drive_nll": per_drive,
        "containment": {k: float(np.mean(d2 <= v)) for k, v in CHI2_2.items()},
        "calibration_error": float(sum(abs(np.mean(d2 <= v) - float(k) / 100.0)
                                       for k, v in CHI2_2.items())),
        "median_95_ellipse_area_m2": float(np.median(areas)),
        "above_99_fraction": float(np.mean(d2 > CHI2_2["99"])),
        "n": int(len(d2)),
    }


# --------------------------------------------------------------------------------------
# conditioning features available to the planner and at runtime
# --------------------------------------------------------------------------------------

def strata(records: list[dict], index: np.ndarray, edges: dict | None) -> tuple[np.ndarray, dict]:
    """Camera x range-tertile strata.

    Range is the conditioning variable of choice: it is the geometric driver of projection
    error, it is observable at runtime, and unlike detected box width it is also available
    to the planner from a predicted pose before any image exists.
    """
    camera = np.asarray([records[i]["camera"] for i in index])
    ranges = np.asarray([records[i]["range_m"] for i in index])
    if edges is None:
        edges = {c: np.quantile(ranges[camera == c], [1 / 3, 2 / 3]).tolist()
                 if (camera == c).sum() >= 30 else None for c in CAMERAS}
    labels = np.empty(len(index), dtype=object)
    for position, (c, r) in enumerate(zip(camera, ranges)):
        entry = edges.get(c)
        if entry is None:
            labels[position] = f"{c}|all"
        else:
            bin_index = int(np.searchsorted(entry, r, side="left"))
            labels[position] = f"{c}|{bin_index}"
    return labels, edges


# --------------------------------------------------------------------------------------
# model fitting
# --------------------------------------------------------------------------------------

def fit_gaussian(residual: np.ndarray, labels: np.ndarray) -> dict:
    pooled = base.sample_covariance(residual)
    model = {"pooled": pooled.tolist(), "strata": {}}
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        if mask.sum() >= 25:
            n = int(mask.sum())
            weight = n / (n + 20.0)
            model["strata"][label] = spd(
                weight * base.sample_covariance(residual[mask]) + (1 - weight) * pooled).tolist()
    return model


def predict_gaussian(model: dict, labels: np.ndarray, scale: float = 1.0) -> np.ndarray:
    pooled = np.asarray(model["pooled"])
    out = np.empty((len(labels), 2, 2))
    for position, label in enumerate(labels):
        out[position] = np.asarray(model["strata"].get(label, pooled))
    return np.stack([spd(scale * m) for m in out])


def fit_mixture_em(residual: np.ndarray, iterations: int = 200, seed: int = 0) -> dict:
    """Two zero-mean Gaussians by expectation maximisation.

    Both components are zero-mean: the conditional mean is already removed by the
    commissioned correction, so any residual offset belongs in the mean model, not here.
    The wide component is initialised from the sample covariance and the narrow one from
    the robust (MAD) scale, which separates them from the outset.
    """
    total = base.sample_covariance(residual)
    robust = np.diag([
        (1.4826 * np.median(np.abs(residual[:, j] - np.median(residual[:, j])))) ** 2
        for j in (0, 1)])
    good = spd(robust)
    bad = spd(4.0 * total)
    weight = 0.9
    previous = -np.inf
    for _ in range(iterations):
        stacked_good = np.repeat(good[None], len(residual), axis=0)
        stacked_bad = np.repeat(bad[None], len(residual), axis=0)
        log_good = np.log(max(weight, 1e-9)) + gaussian_logpdf(residual, stacked_good)
        log_bad = np.log(max(1 - weight, 1e-9)) + gaussian_logpdf(residual, stacked_bad)
        peak = np.maximum(log_good, log_bad)
        total_log = peak + np.log(np.exp(log_good - peak) + np.exp(log_bad - peak))
        objective = float(total_log.mean())
        responsibility = np.exp(log_good - total_log)
        weight = float(np.clip(responsibility.mean(), 0.30, 0.995))
        for target, share in ((0, responsibility), (1, 1.0 - responsibility)):
            mass = float(share.sum())
            if mass < 5.0:
                continue
            matrix = spd((residual * share[:, None]).T @ residual / mass)
            if target == 0:
                good = matrix
            else:
                bad = matrix
        # keep the components ordered and separated so the labels stay meaningful
        if np.linalg.det(good) > np.linalg.det(bad):
            good, bad = bad, good
            weight = 1.0 - weight
        if abs(objective - previous) < 1e-9:
            break
        previous = objective
    return {"weight_good": weight, "good": good.tolist(), "bad": bad.tolist(),
            "train_mean_loglik": previous}


def fit_mixture(residual: np.ndarray, labels: np.ndarray, seed: int) -> dict:
    pooled = fit_mixture_em(residual, seed=seed)
    model = {"pooled": pooled, "strata": {}}
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        if mask.sum() >= 120:
            model["strata"][label] = fit_mixture_em(residual[mask], seed=seed)
    return model


def mixture_components(model: dict, labels: np.ndarray):
    weights = np.empty(len(labels))
    good = np.empty((len(labels), 2, 2))
    bad = np.empty((len(labels), 2, 2))
    for position, label in enumerate(labels):
        entry = model["strata"].get(label, model["pooled"])
        weights[position] = entry["weight_good"]
        good[position] = np.asarray(entry["good"])
        bad[position] = np.asarray(entry["bad"])
    return weights, good, bad


def mixture_loglik(residual, weights, good, bad) -> np.ndarray:
    log_good = np.log(np.clip(weights, 1e-9, 1)) + gaussian_logpdf(residual, good)
    log_bad = np.log(np.clip(1 - weights, 1e-9, 1)) + gaussian_logpdf(residual, bad)
    peak = np.maximum(log_good, log_bad)
    return peak + np.log(np.exp(log_good - peak) + np.exp(log_bad - peak))


def calibrate_scale(residual: np.ndarray, covariance: np.ndarray) -> float:
    """One scalar chosen so the 90 and 95 percent ellipses contain their nominal share."""
    d2 = mahalanobis(residual, covariance)
    return max(1.0,
               float(np.quantile(d2, 0.90) / CHI2_2["90"]),
               float(np.quantile(d2, 0.95) / CHI2_2["95"]))


def route_of(drive_id: str) -> str:
    """Route identity of a drive: replicates of one lap share it."""
    return drive_id.rsplit("_r", 1)[0]


def nested_calibration_scale(records: list[dict], fit_index: np.ndarray,
                             drive_of: np.ndarray, target: np.ndarray,
                             build, seed: int) -> float:
    """Calibrate on residuals the mean model did not fit.

    The commissioned mean is a flexible regressor, so its residuals on the drives that
    trained it are smaller than on a drive it has never seen: here the held-out median is
    2.40 cm against 1.84 cm out of fold on the fit drives.  A covariance calibrated on the
    latter is therefore optimistic by construction, no matter which family it belongs to.

    Each fit ROUTE is held out in turn -- not each drive, because replicates of one lap
    share their geometry and the mean model effectively memorises it, which is why an
    earlier drive-level version returned a scale of 1.0 while the held-out partition needed
    inflation.  The mean model, the mixture and the regime classifier are rebuilt without
    the held-out route, and the scale that route then requires is recorded.  The median of
    those is the deployed scale, so calibration sees the same route-to-route generalisation
    gap the deployed model will meet.
    """
    route = np.asarray([route_of(d) for d in drive_of])
    scales: list[float] = []
    for held in sorted(set(route[fit_index].tolist())):
        inner = fit_index[route[fit_index] != held]
        outer = fit_index[route[fit_index] == held]
        if len(inner) < 200 or len(outer) < 100:
            continue
        inner_corrected = np.zeros((len(records), 2))
        for inner_held in sorted(set(drive_of[inner].tolist())):
            deeper = inner[drive_of[inner] != inner_held]
            deeper_out = inner[drive_of[inner] == inner_held]
            inner_corrected[deeper_out] = base.predict_mean(
                records, deeper_out, "box_mlp",
                base.fit_mean(records, deeper, "box_mlp", seed))
        fitted = base.fit_mean(records, inner, "box_mlp", seed)
        inner_corrected[outer] = base.predict_mean(records, outer, "box_mlp", fitted)
        inner_residual = target[inner] - inner_corrected[inner]
        outer_residual = target[outer] - inner_corrected[outer]
        covariance = build(inner, inner_residual, outer)
        scales.append(calibrate_scale(outer_residual, covariance))
    return float(np.median(scales)) if scales else 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=260911)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)

    _, records, _ = base.load_capture(arguments.capture_root.resolve())
    partition = np.asarray([r["partition"] for r in records])
    drive_of = np.asarray([r["drive"] for r in records])
    fit_index = np.flatnonzero(partition == "fit")
    dev_index = np.flatnonzero(partition == "development")
    fit_drives = sorted(set(drive_of[fit_index].tolist()))
    target = np.stack([r["target_ray"] for r in records])

    # ---- frozen mean correction, leave one drive out on the fit drives ----------------
    corrected = np.zeros((len(records), 2))
    for held in fit_drives:
        inner = fit_index[drive_of[fit_index] != held]
        outer = fit_index[drive_of[fit_index] == held]
        corrected[outer] = base.predict_mean(
            records, outer, "box_mlp",
            base.fit_mean(records, inner, "box_mlp", arguments.seed))
    corrected[dev_index] = base.predict_mean(
        records, dev_index, "box_mlp",
        base.fit_mean(records, fit_index, "box_mlp", arguments.seed))
    residual = target - corrected

    fit_residual, dev_residual = residual[fit_index], residual[dev_index]
    fit_labels, edges = strata(records, fit_index, None)
    dev_labels, _ = strata(records, dev_index, edges)
    dev_drives = drive_of[dev_index]

    diagnostics = {
        "excess_kurtosis": {
            "along": float(np.mean((fit_residual[:, 0] / fit_residual[:, 0].std()) ** 4) - 3.0),
            "across": float(np.mean((fit_residual[:, 1] / fit_residual[:, 1].std()) ** 4) - 3.0),
        },
        "sd_over_robust_scale": {
            axis: float(fit_residual[:, j].std()
                        / (1.4826 * np.median(np.abs(fit_residual[:, j]
                                                     - np.median(fit_residual[:, j])))))
            for j, axis in ((0, "along"), (1, "across"))
        },
        "squared_error_share_of_worst_2_percent": float(
            np.sum(np.sort(np.linalg.norm(fit_residual, axis=1))[::-1][
                :int(0.02 * len(fit_residual))] ** 2) / np.sum(np.linalg.norm(fit_residual, axis=1) ** 2)),
    }

    # The generalisation gap this design exists to measure: residual size on a route the
    # mean model trained on, against a route it has never seen.
    route_labels = np.asarray([route_of(d) for d in drive_of])
    gap = {}
    for label, index in (("fit_routes_out_of_fold", fit_index),
                         ("development_routes_unseen", dev_index)):
        magnitude = np.linalg.norm(residual[index], axis=1)
        gap[label] = {
            "median_m": float(np.median(magnitude)),
            "p90_m": float(np.percentile(magnitude, 90)),
            "routes": sorted(set(route_labels[index].tolist())),
        }
    gap["median_ratio_unseen_over_fit"] = (
        gap["development_routes_unseen"]["median_m"] / gap["fit_routes_out_of_fold"]["median_m"])

    results: dict[str, Any] = {}

    # ---- Gaussian incumbents ---------------------------------------------------------
    gaussian = fit_gaussian(fit_residual, fit_labels)
    results["G_pooled"] = score(dev_residual,
                                np.repeat(np.asarray(gaussian["pooled"])[None], len(dev_index), 0),
                                dev_drives)
    results["G_strata"] = score(dev_residual, predict_gaussian(gaussian, dev_labels), dev_drives)
    gaussian_scale = calibrate_scale(fit_residual, predict_gaussian(gaussian, fit_labels))
    results["G_strata_calibrated"] = score(
        dev_residual, predict_gaussian(gaussian, dev_labels, gaussian_scale), dev_drives)
    results["G_strata_calibrated"]["calibration_scale"] = gaussian_scale

    # ---- mixture ---------------------------------------------------------------------
    mixture = fit_mixture(fit_residual, fit_labels, arguments.seed)
    weights, good, bad = mixture_components(mixture, dev_labels)
    loglik = mixture_loglik(dev_residual, weights, good, bad)
    moment = weights[:, None, None] * good + (1.0 - weights)[:, None, None] * bad
    results["M_mixture_moment"] = score(dev_residual, moment, dev_drives, loglik=loglik)
    results["M_mixture_moment"]["note"] = (
        "held-out score is the true mixture likelihood; containment and area describe the "
        "moment-matched covariance an unmodified Kalman update would use")

    fit_weights, fit_good, fit_bad = mixture_components(mixture, fit_labels)
    fit_moment = (fit_weights[:, None, None] * fit_good
                  + (1.0 - fit_weights)[:, None, None] * fit_bad)
    moment_scale = calibrate_scale(fit_residual, fit_moment)
    results["M_mixture_moment_calibrated"] = score(
        dev_residual, np.stack([spd(moment_scale * m) for m in moment]), dev_drives, loglik=loglik)
    results["M_mixture_moment_calibrated"]["calibration_scale"] = moment_scale

    # ---- M1b: mixture weight conditioned on runtime-observable features ---------------
    # The pooled and stratified mixtures assign responsibility using the residual, which
    # needs ground truth and so cannot be evaluated at runtime.  A deployable model must
    # predict the regime from what the detector and geometry supply.  The fit-drive
    # responsibilities become labels for a classifier over exactly those features, and the
    # covariance a reading receives is its predicted mixture.
    from sklearn.ensemble import GradientBoostingClassifier

    def regime_features(index: np.ndarray) -> np.ndarray:
        return np.asarray([[
            records[i]["range_m"], 1.0 / max(records[i]["range_m"], 1e-6),
            records[i]["bbox_w"], records[i]["bbox_h"],
            records[i]["bbox_w"] / max(records[i]["bbox_h"], 1e-6),
            records[i]["bottom_v"] / 720.0, records[i]["bottom_u"] / 1280.0,
            records[i]["confidence"],
            *[float(records[i]["camera"] == c) for c in CAMERAS],
        ] for i in index], dtype=float)

    fit_responsibility = np.exp(
        np.log(np.clip(fit_weights, 1e-9, 1)) + gaussian_logpdf(fit_residual, fit_good)
        - mixture_loglik(fit_residual, fit_weights, fit_good, fit_bad))
    degraded = (fit_responsibility < 0.5).astype(int)
    classifier = GradientBoostingClassifier(random_state=arguments.seed)
    classifier.fit(regime_features(fit_index), degraded)
    predicted_good = 1.0 - classifier.predict_proba(regime_features(dev_index))[:, 1]
    predicted_good = np.clip(predicted_good, 0.02, 0.98)

    learned_loglik = mixture_loglik(dev_residual, predicted_good, good, bad)
    learned_moment = (predicted_good[:, None, None] * good
                      + (1.0 - predicted_good)[:, None, None] * bad)
    results["M1b_learned_weight"] = score(dev_residual, learned_moment, dev_drives,
                                          loglik=learned_loglik)
    def build_learned(inner: np.ndarray, inner_residual: np.ndarray,
                      outer: np.ndarray) -> np.ndarray:
        """Rebuild the whole learned-weight model on `inner` and apply it to `outer`."""
        inner_labels, inner_edges = strata(records, inner, None)
        outer_labels, _ = strata(records, outer, inner_edges)
        inner_mixture = fit_mixture(inner_residual, inner_labels, arguments.seed)
        inner_w, inner_g, inner_b = mixture_components(inner_mixture, inner_labels)
        inner_resp = np.exp(
            np.log(np.clip(inner_w, 1e-9, 1)) + gaussian_logpdf(inner_residual, inner_g)
            - mixture_loglik(inner_residual, inner_w, inner_g, inner_b))
        inner_classifier = GradientBoostingClassifier(random_state=arguments.seed)
        inner_classifier.fit(regime_features(inner), (inner_resp < 0.5).astype(int))
        _, outer_g, outer_b = mixture_components(inner_mixture, outer_labels)
        outer_good = np.clip(
            1.0 - inner_classifier.predict_proba(regime_features(outer))[:, 1], 0.02, 0.98)
        return (outer_good[:, None, None] * outer_g
                + (1.0 - outer_good)[:, None, None] * outer_b)

    learned_scale = nested_calibration_scale(
        records, fit_index, drive_of, target, build_learned, arguments.seed)
    results["M1b_learned_weight_calibrated"] = score(
        dev_residual, np.stack([spd(learned_scale * m) for m in learned_moment]),
        dev_drives, loglik=learned_loglik)
    results["M1b_learned_weight_calibrated"]["calibration_scale"] = learned_scale
    results["M1b_learned_weight_calibrated"]["calibration"] = (
        "median of the scales each held-out fit drive required, with the mean model, the "
        "mixture and the regime classifier all rebuilt without that drive")

    from sklearn.metrics import brier_score_loss, roc_auc_score

    dev_responsibility = np.exp(
        np.log(np.clip(weights, 1e-9, 1)) + gaussian_logpdf(dev_residual, good) - loglik)
    dev_degraded = (dev_responsibility < 0.5).astype(int)
    results["M1b_regime_classifier"] = {
        "held_out_auc": float(roc_auc_score(dev_degraded, 1.0 - predicted_good)),
        "held_out_brier": float(brier_score_loss(dev_degraded, 1.0 - predicted_good)),
        "fit_degraded_rate": float(degraded.mean()),
        "development_degraded_rate": float(dev_degraded.mean()),
        "features": ["range_m", "inverse_range", "bbox_w", "bbox_h", "bbox_aspect",
                     "bottom_v_norm", "bottom_u_norm", "confidence", "camera one-hot"],
    }

    # gated view: refuse readings the mixture attributes to the degraded component
    responsibility = np.exp(np.log(np.clip(weights, 1e-9, 1))
                            + gaussian_logpdf(dev_residual, good) - loglik)
    gate_results = {}
    for threshold in (0.5, 0.7, 0.9):
        keep = responsibility >= threshold
        if keep.sum() < 50:
            continue
        entry = score(dev_residual[keep], good[keep], dev_drives[keep])
        entry["kept_fraction"] = float(keep.mean())
        gate_results[f"keep_p_good_ge_{threshold}"] = entry
    results["M_mixture_gated"] = gate_results
    results["M_mixture_gated_note"] = (
        "diagnostic only: this gate uses the residual, which needs ground truth and is "
        "not available at runtime. The deployable form is M1b_learned_weight.")

    ranking = {name: value["calibration_error"] for name, value in results.items()
               if isinstance(value, dict) and "calibration_error" in value}
    selected = min(ranking, key=lambda name: ranking[name]) if ranking else None

    report = {
        "schema": "commissioned_measurement_noise_model.v1",
        "capture_root": str(arguments.capture_root),
        "audit_analysis_permitted": False,
        "mean_model": "box_mlp",
        "conditioning": "camera x range tertile",
        "fit_drives": fit_drives,
        "development_drives": sorted(set(dev_drives.tolist())),
        "residual_diagnostics_fit": diagnostics,
        "route_generalisation_gap": gap,
        "models": results,
        "selected_by_calibration": selected,
        "mixture_parameters_pooled": mixture["pooled"],
        "strata_with_own_mixture": sorted(mixture["strata"]),
    }
    base.atomic_json(arguments.output / "measurement_noise_model.json", report)

    print(json.dumps({
        "residual_diagnostics": diagnostics,
        "held_out_drive_nll": {k: round(v["equal_drive_mean_nll"], 3)
                               for k, v in results.items() if "equal_drive_mean_nll" in v},
        "containment_95": {k: round(v["containment"]["95"], 3)
                           for k, v in results.items() if "containment" in v},
        "containment_50": {k: round(v["containment"]["50"], 3)
                           for k, v in results.items() if "containment" in v},
        "median_95_ellipse_cm2": {k: round(v["median_95_ellipse_area_m2"] * 1e4, 1)
                                  for k, v in results.items() if "median_95_ellipse_area_m2" in v},
        "gated": {k: {"kept": round(v["kept_fraction"], 3),
                      "cov95": round(v["containment"]["95"], 3)}
                  for k, v in gate_results.items()},
        "selected_by_calibration": selected,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
