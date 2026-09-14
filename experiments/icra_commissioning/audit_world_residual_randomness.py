#!/usr/bin/env python3
"""Audit whether a trained world-position correction leaves predictable residuals.

This is a static, single-camera observation audit. It tests conditional centring,
dependence on runtime observables, and repeatable camera/location structure. It does
not claim temporal whiteness; that requires ordered driving RGB and filter innovations.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train_world_position as training


def selected_prediction(results: dict, archive: np.lib.npyio.NpzFile, arm: str) -> tuple[np.ndarray, np.ndarray]:
    stack = np.asarray(archive[arm], dtype=float)
    method = results["arms"][arm]["aggregation"]["selected"]
    prediction = np.full(stack.shape[1:], np.nan, dtype=float)
    available = np.any(np.all(np.isfinite(stack), axis=2), axis=0)
    if method == "mean":
        prediction[available] = np.nanmean(stack[:, available], axis=0)
        return prediction, stack
    if method == "coordinate_median":
        prediction[available] = np.nanmedian(stack[:, available], axis=0)
        return prediction, stack
    raise ValueError(f"unsupported aggregation {method!r}")


def cluster_mean_ci(values: np.ndarray, clusters: np.ndarray, *, seed: int,
                    draws: int = 4000) -> dict[str, object]:
    unique = np.unique(clusters)
    observed = np.mean(values, axis=0)
    if len(unique) < 2:
        return {"clusters": int(len(unique)), "mean_cm": (100 * observed).tolist(),
                "ci95_cm": None}
    members = [np.flatnonzero(clusters == group) for group in unique]
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, values.shape[1]), dtype=float)
    for draw in range(draws):
        chosen = rng.integers(0, len(members), size=len(members))
        indices = np.concatenate([members[index] for index in chosen])
        samples[draw] = np.mean(values[indices], axis=0)
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
    return {
        "clusters": int(len(unique)),
        "mean_cm": (100 * observed).tolist(),
        "ci95_cm": np.column_stack((100 * lo, 100 * hi)).tolist(),
        "zero_in_ci_both_axes": bool(np.all((lo <= 0) & (hi >= 0))),
    }


def residual_summary(residual: np.ndarray, places: np.ndarray, mask: np.ndarray,
                     *, seed: int) -> dict[str, object]:
    values = residual[mask]
    if not len(values):
        return {"n": 0}
    norm = np.linalg.norm(values, axis=1)
    result = cluster_mean_ci(values, places[mask], seed=seed)
    result.update({
        "n": int(len(values)),
        "rms_cm": float(100 * math.sqrt(np.mean(norm ** 2))),
        "median_cm": float(100 * np.median(norm)),
        "covariance_ray_m2": np.cov(values.T, ddof=1).tolist() if len(values) > 1 else None,
    })
    return result


def benjamini_hochberg(rows: list[dict[str, object]]) -> None:
    order = np.argsort([float(row["p_value"]) for row in rows])
    count = len(rows)
    running = 1.0
    for rank in range(count - 1, -1, -1):
        index = int(order[rank])
        adjusted = min(running, float(rows[index]["p_value"]) * count / (rank + 1))
        rows[index]["q_value_bh"] = adjusted
        running = adjusted


def dependence_tests(data: dict[str, np.ndarray], residual: np.ndarray,
                     mask: np.ndarray) -> dict[str, object]:
    candidates = [(name, data["features"][:, index])
                  for index, name in enumerate(training.FEATURE_NAMES[:10])]
    candidates.append(("observed_height_ratio", data["height_ratio"]))
    indices = np.flatnonzero(mask)
    labels = np.asarray([f"{data['camera'][index]}|{data['place'][index]}" for index in indices])
    unique = np.unique(labels)
    rows: list[dict[str, object]] = []
    axis_names = ("along", "across")
    for feature_name, values in candidates:
        feature_group = np.asarray([np.mean(values[indices][labels == group]) for group in unique])
        residual_group = np.asarray([
            np.mean(residual[indices][labels == group], axis=0) for group in unique
        ])
        for axis, axis_name in enumerate(axis_names):
            rho, p_value = stats.spearmanr(feature_group, residual_group[:, axis])
            rows.append({"feature": feature_name, "axis": axis_name,
                         "spearman_rho": float(rho), "p_value": float(p_value)})
    benjamini_hochberg(rows)
    rows.sort(key=lambda row: abs(float(row["spearman_rho"])), reverse=True)
    return {
        "unit": "camera-by-spatial-place aggregate",
        "groups": int(len(unique)),
        "tests": rows,
        "significant_after_bh_0p05": int(sum(float(row["q_value_bh"]) < 0.05 for row in rows)),
        "largest_absolute_correlations": rows[:8],
    }


def cross_validated_predictability(data: dict[str, np.ndarray], residual: np.ndarray,
                                   mask: np.ndarray) -> dict[str, object]:
    indices = np.flatnonzero(mask)
    groups = data["place"][indices]
    unique = np.unique(groups)
    if len(unique) < 3:
        return {"n": int(len(indices)), "groups": int(len(unique)), "status": "insufficient_groups"}
    features = np.column_stack((data["features"][indices], data["height_ratio"][indices]))
    target = residual[indices]
    prediction = np.full_like(target, np.nan)
    splitter = GroupKFold(n_splits=min(5, len(unique)))
    for fit, test in splitter.split(features, target, groups):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(features[fit], target[fit])
        prediction[test] = model.predict(features[test])
    baseline_sse = float(np.sum(target ** 2))
    model_sse = float(np.sum((target - prediction) ** 2))
    return {
        "n": int(len(indices)), "groups": int(len(unique)),
        "features": list(training.FEATURE_NAMES) + ["observed_height_ratio"],
        "grouping": "spatial place; no place appears in both fit and test fold",
        "r2_along": float(r2_score(target[:, 0], prediction[:, 0])),
        "r2_across": float(r2_score(target[:, 1], prediction[:, 1])),
        "multivariate_r2_against_zero": float(1.0 - model_sse / baseline_sse),
        "interpretation": "positive held-place R2 is evidence that residual remains predictable; non-positive R2 does not prove randomness",
    }


def location_structure(residual: np.ndarray, camera: np.ndarray, place: np.ndarray,
                       mask: np.ndarray, *, seed: int, permutations: int = 2000) -> dict[str, object]:
    indices = np.flatnonzero(mask)
    labels = np.asarray([f"{camera[index]}|{place[index]}" for index in indices])
    values = residual[indices]
    counts = {label: int(np.sum(labels == label)) for label in np.unique(labels)}
    keep = np.asarray([counts[label] >= 2 for label in labels])
    labels, values = labels[keep], values[keep]
    if len(values) < 8:
        return {"n": int(len(values)), "status": "insufficient_repeats"}

    _, codes = np.unique(labels, return_inverse=True)
    group_counts = np.bincount(codes).astype(float)

    def eta2(candidate: np.ndarray) -> np.ndarray:
        global_mean = np.mean(candidate, axis=0)
        total = np.sum((candidate - global_mean) ** 2, axis=0)
        sums = np.column_stack([
            np.bincount(codes, weights=candidate[:, axis], minlength=len(group_counts))
            for axis in range(2)
        ])
        means = sums / group_counts[:, None]
        between = np.sum(group_counts[:, None] * (means - global_mean) ** 2, axis=0)
        return np.divide(between, total, out=np.zeros(2), where=total > 0)

    observed = eta2(values)
    rng = np.random.default_rng(seed)
    null = np.empty((permutations, 2))
    for index in range(permutations):
        null[index] = eta2(values[rng.permutation(len(values))])
    return {
        "n": int(len(values)), "camera_place_groups": int(len(np.unique(labels))),
        "minimum_repeats_per_group": 2,
        "eta_squared_along_across": observed.tolist(),
        "permutation_p_along_across": ((1 + np.sum(null >= observed, axis=0)) / (permutations + 1)).tolist(),
        "interpretation": "significant eta-squared means repeatable camera/location structure remains",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--augmentation-capture", type=Path, required=True)
    parser.add_argument("--transfer-capture", type=Path, required=True)
    parser.add_argument("--arm", default="rgb_balanced")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    base = training.prepare()
    base["split"] = np.where(base["split"] == "development", "historical_development", base["split"])
    base["images"] = np.empty((len(base["split"]), 0), dtype=np.uint8)
    data = training.combine_datasets([
        base,
        training.prepare_capture(args.augmentation_capture, load_images=False),
        training.prepare_capture(args.transfer_capture, role_override="development", load_images=False),
    ])
    results = json.loads((args.training_dir / "results.json").read_text())
    with np.load(args.training_dir / "predictions.npz", allow_pickle=False) as archive:
        if len(archive["target_ray"]) != len(data["target_ray"]):
            raise RuntimeError("training archive and reconstructed dataset lengths differ")
        prediction, stack = selected_prediction(results, archive, args.arm)

    residual = prediction - data["target_ray"]
    development = data["eligible"] & (data["split"] == "development")
    partial = development & (data["severity"] < 3)
    severe = development & (data["severity"] == 0)
    strata = {
        "all": development, "partial": partial, "severe": severe,
        **{f"{camera}_partial": partial & (data["camera"] == camera)
           for camera in training.CAMERAS},
    }
    summaries = {name: residual_summary(residual, data["place"], mask, seed=args.seed + index)
                 for index, (name, mask) in enumerate(strata.items())}

    disagreement = np.full(len(prediction), np.nan, dtype=float)
    available = np.all(np.isfinite(prediction), axis=1)
    disagreement[available] = np.sqrt(np.nanmean(
        np.sum((stack[:, available] - prediction[None, available]) ** 2, axis=2), axis=0))
    gate = results["arms"][args.arm].get("gates", {}).get("ensemble_disagreement")
    gated = None
    if gate:
        admitted = disagreement <= float(gate["threshold_m"])
        gated = {name: residual_summary(residual, data["place"], mask & admitted,
                                        seed=args.seed + 100 + index)
                 for index, (name, mask) in enumerate(strata.items())}

    report = {
        "schema": "world_residual_randomness_audit.v1",
        "status": "static_development_diagnostic_not_temporal_whiteness",
        "arm": args.arm,
        "coordinates": "camera-ray metres; residual = corrected world observation minus truth",
        "sample_unit": "eligible static single-camera detection",
        "resampling_unit": "spatial place",
        "summaries": summaries,
        "ensemble_disagreement_gate_summaries": gated,
        "dependence": {
            "all": dependence_tests(data, residual, development),
            "partial": dependence_tests(data, residual, partial),
        },
        "held_place_linear_predictability": {
            "all": cross_validated_predictability(data, residual, development),
            "partial": cross_validated_predictability(data, residual, partial),
        },
        "repeatable_camera_place_structure": {
            "all": location_structure(residual, data["camera"], data["place"], development,
                                      seed=args.seed),
            "partial": location_structure(residual, data["camera"], data["place"], partial,
                                          seed=args.seed + 1),
        },
        "decision_rule": {
            "mean": "cluster-bootstrap 95% CI must contain zero on both axes in every supported camera/occlusion stratum",
            "dependence": "no runtime-observable correlation may remain significant after BH correction",
            "repeatability": "camera/place eta-squared permutation test must not reject at 0.05",
            "scope": "passing these static tests is necessary but not sufficient; ordered driving innovations must also be white",
        },
    }
    supported = [value for value in summaries.values() if value.get("n", 0) >= 20]
    report["passes_supported_conditional_mean_rule"] = bool(
        supported and all(value.get("zero_in_ci_both_axes", False) for value in supported))
    report["passes_static_randomness_screen"] = bool(
        report["passes_supported_conditional_mean_rule"]
        and report["dependence"]["all"]["significant_after_bh_0p05"] == 0
        and report["dependence"]["partial"]["significant_after_bh_0p05"] == 0
        and all(min(value.get("permutation_p_along_across", [0])) >= 0.05
                for value in report["repeatable_camera_place_structure"].values()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
