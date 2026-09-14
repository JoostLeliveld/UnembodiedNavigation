#!/usr/bin/env python3
"""Test tail, temporal, and simultaneous-camera models on commissioned residuals.

This is a diagnostic for the perception-side predictive distribution.  It reads the
stored predictions of one fitted mean/covariance model, then compares residual models
with complete-drive leave-one-out fitting:

* conditionally independent Gaussian;
* conditionally independent covariance-matched Student-t;
* per-camera continuous-time OU dependence with Gaussian innovations;
* the same OU dependence with Student-t innovations.

Simultaneous-camera dependence is tested separately on exact capture timestamps.  A
2x2 cross-camera block is fitted for every camera pair after the temporal transform,
and held-out pair likelihood is compared with an independent block.  Separating the
two tests avoids pretending that a same-timestamp covariance accounts for dependence
between frames at different timestamps.

The input deployment model was fitted on all drives.  Results from this script are
therefore in-sample diagnostics of that fixed model, not thesis evidence or a model
selection.  The leave-one-drive-out boundary applies only to residual-model parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _moments(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    centred = values - np.mean(values)
    second = float(np.mean(centred**2))
    if not values.size or second <= 0.0:
        return {"mean": math.nan, "sd": math.nan, "skew": math.nan,
                "excess_kurtosis": math.nan}
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values)),
        "skew": float(np.mean(centred**3) / second**1.5),
        "excess_kurtosis": float(np.mean(centred**4) / second**2 - 3.0),
    }


def _acf(values: np.ndarray, lag: int) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) <= lag + 2:
        return math.nan
    first = values[:-lag] - np.mean(values[:-lag])
    second = values[lag:] - np.mean(values[lag:])
    denominator = float(np.sqrt(np.sum(first**2) * np.sum(second**2)))
    return float(np.sum(first * second) / denominator) if denominator > 0.0 else math.nan


def _sequence_indexes(drive: np.ndarray, camera: np.ndarray, stamp_ns: np.ndarray,
                      selected_drives: set[str] | None = None):
    for drive_id in sorted(set(drive.tolist())):
        if selected_drives is not None and drive_id not in selected_drives:
            continue
        for camera_id in sorted(set(camera[drive == drive_id].tolist())):
            indexes = np.flatnonzero((drive == drive_id) & (camera == camera_id))
            if indexes.size:
                yield drive_id, camera_id, indexes[np.argsort(stamp_ns[indexes], kind="stable")]


def _acf_summary(residual: np.ndarray, drive: np.ndarray, camera: np.ndarray,
                 stamp_ns: np.ndarray, lags=(1, 5)) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for lag in lags:
        values: list[float] = []
        for _, _, indexes in _sequence_indexes(drive, camera, stamp_ns):
            if len(indexes) <= lag + 2:
                continue
            for axis in range(2):
                value = _acf(residual[indexes, axis], lag)
                if math.isfinite(value):
                    values.append(value)
        array = np.asarray(values, dtype=float)
        output[str(lag)] = {
            "sequence_axis_count": int(len(array)),
            "median": float(np.median(array)),
            "q25": float(np.quantile(array, 0.25)),
            "q75": float(np.quantile(array, 0.75)),
        }
    return output


def _gaussian_nll(vectors: np.ndarray, covariance: np.ndarray | None = None) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    dimension = vectors.shape[1]
    if covariance is None:
        quadratic = np.sum(vectors**2, axis=1)
        logdet = 0.0
    else:
        covariance = np.asarray(covariance, dtype=float)
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ValueError("covariance must be positive definite")
        quadratic = np.einsum("ni,ij,nj->n", vectors, np.linalg.inv(covariance), vectors)
    return 0.5 * (dimension * math.log(2.0 * math.pi) + logdet + quadratic)


def _student_nll(vectors: np.ndarray, nu: float,
                 covariance: np.ndarray | None = None) -> np.ndarray:
    """Student-t NLL whose covariance, rather than scatter, is ``covariance``."""
    vectors = np.asarray(vectors, dtype=float)
    dimension = vectors.shape[1]
    covariance = np.eye(dimension) if covariance is None else np.asarray(covariance, dtype=float)
    scatter = covariance * ((nu - 2.0) / nu)
    sign, logdet = np.linalg.slogdet(scatter)
    if sign <= 0:
        raise ValueError("Student-t scatter must be positive definite")
    quadratic = np.einsum("ni,ij,nj->n", vectors, np.linalg.inv(scatter), vectors)
    log_normalizer = (
        gammaln((nu + dimension) / 2.0) - gammaln(nu / 2.0)
        - 0.5 * (dimension * math.log(nu * math.pi) + logdet)
    )
    return -log_normalizer + 0.5 * (nu + dimension) * np.log1p(quadratic / nu)


def _fit_nu(vectors: np.ndarray, covariances: list[np.ndarray] | None = None) -> float:
    vectors = np.asarray(vectors, dtype=float)
    if covariances is None:
        objective = lambda nu: float(np.mean(_student_nll(vectors, nu)))
    else:
        dimension = vectors.shape[1]
        logdet = np.empty(len(vectors), dtype=float)
        quadratic = np.empty(len(vectors), dtype=float)
        for index, (vector, covariance) in enumerate(
                zip(vectors, covariances, strict=True)):
            sign, value = np.linalg.slogdet(covariance)
            if sign <= 0:
                raise ValueError("covariance must be positive definite")
            logdet[index] = value
            quadratic[index] = vector @ np.linalg.solve(covariance, vector)

        def objective(nu: float) -> float:
            ratio = (nu - 2.0) / nu
            log_normalizer = (
                gammaln((nu + dimension) / 2.0) - gammaln(nu / 2.0)
                - 0.5 * (
                    dimension * math.log(nu * math.pi)
                    + logdet + dimension * math.log(ratio)
                )
            )
            values = -log_normalizer + 0.5 * (nu + dimension) * np.log1p(
                quadratic / (ratio * nu))
            return float(np.mean(values))
    result = minimize_scalar(objective, bounds=(2.05, 200.0), method="bounded",
                             options={"xatol": 1e-3})
    if not result.success:
        raise RuntimeError("Student-t degrees-of-freedom fit failed")
    return float(result.x)


def _varying_covariance_nll(vectors: np.ndarray, covariances: list[np.ndarray],
                            nu: float | None) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    dimension = vectors.shape[1]
    logdet = np.empty(len(vectors), dtype=float)
    quadratic = np.empty(len(vectors), dtype=float)
    for index, (vector, covariance) in enumerate(zip(vectors, covariances, strict=True)):
        sign, value = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ValueError("covariance must be positive definite")
        logdet[index] = value
        quadratic[index] = vector @ np.linalg.solve(covariance, vector)
    if nu is None:
        return 0.5 * (dimension * math.log(2.0 * math.pi) + logdet + quadratic)
    ratio = (nu - 2.0) / nu
    log_normalizer = (
        gammaln((nu + dimension) / 2.0) - gammaln(nu / 2.0)
        - 0.5 * (
            dimension * math.log(nu * math.pi)
            + logdet + dimension * math.log(ratio)
        )
    )
    return -log_normalizer + 0.5 * (nu + dimension) * np.log1p(
        quadratic / (ratio * nu))


def _fit_tau(sequences: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Fit an OU correlation time in seconds by Gaussian conditional likelihood."""
    def objective(log_tau: float) -> float:
        tau = math.exp(log_tau)
        total = 0.0
        count = 0
        for stamps, values in sequences:
            if len(values) < 2:
                continue
            dt = np.diff(stamps.astype(np.float64)) * 1e-9
            valid = dt > 0.0
            if not np.any(valid):
                continue
            rho = np.exp(-dt[valid] / tau)
            variance = np.maximum(1.0 - rho**2, 1e-8)
            innovation = values[1:][valid] - rho[:, None] * values[:-1][valid]
            total += float(np.sum(
                math.log(2.0 * math.pi) + np.log(variance)
                + 0.5 * np.sum(innovation**2, axis=1) / variance
            ))
            count += int(np.sum(valid))
        return total / max(count, 1)

    result = minimize_scalar(objective, bounds=(math.log(0.005), math.log(30.0)),
                             method="bounded", options={"xatol": 1e-4})
    if not result.success:
        raise RuntimeError("OU correlation-time fit failed")
    return float(math.exp(result.x))


def _sequences_for_camera(values: np.ndarray, drive: np.ndarray, camera: np.ndarray,
                          stamp_ns: np.ndarray, camera_id: str,
                          selected_drives: set[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    output = []
    for _, candidate, indexes in _sequence_indexes(drive, camera, stamp_ns, selected_drives):
        if candidate == camera_id:
            output.append((stamp_ns[indexes], values[indexes]))
    return output


def _temporal_innovations(values: np.ndarray, drive: np.ndarray, camera: np.ndarray,
                          stamp_ns: np.ndarray, taus: dict[str, float],
                          selected_drives: set[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformed, source_indexes, variances = [], [], []
    for _, camera_id, indexes in _sequence_indexes(drive, camera, stamp_ns, selected_drives):
        previous_index = None
        for index in indexes:
            if previous_index is None:
                transformed.append(values[index])
                variances.append(1.0)
            else:
                dt = (int(stamp_ns[index]) - int(stamp_ns[previous_index])) * 1e-9
                rho = math.exp(-dt / taus[camera_id]) if dt > 0.0 else 0.0
                variance = max(1.0 - rho * rho, 1e-8)
                transformed.append((values[index] - rho * values[previous_index]) / math.sqrt(variance))
                variances.append(variance)
            source_indexes.append(index)
            previous_index = index
    return (np.asarray(transformed), np.asarray(source_indexes, dtype=int),
            np.asarray(variances, dtype=float))


def _score_temporal(values: np.ndarray, drive: np.ndarray, camera: np.ndarray,
                    stamp_ns: np.ndarray, selected_drives: set[str],
                    taus: dict[str, float], nu: float | None) -> np.ndarray:
    scores = []
    for _, camera_id, indexes in _sequence_indexes(drive, camera, stamp_ns, selected_drives):
        previous_index = None
        for index in indexes:
            if previous_index is None:
                covariance = np.eye(2)
                innovation = values[index]
            else:
                dt = (int(stamp_ns[index]) - int(stamp_ns[previous_index])) * 1e-9
                rho = math.exp(-dt / taus[camera_id]) if dt > 0.0 else 0.0
                variance = max(1.0 - rho * rho, 1e-8)
                covariance = variance * np.eye(2)
                innovation = values[index] - rho * values[previous_index]
            score = (_gaussian_nll(innovation[None], covariance)[0] if nu is None
                     else _student_nll(innovation[None], nu, covariance)[0])
            scores.append(float(score))
            previous_index = index
    return np.asarray(scores)


def _paired(values: np.ndarray, drive: np.ndarray, camera: np.ndarray,
            stamp_ns: np.ndarray, camera_a: str, camera_b: str,
            selected_drives: set[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first, second, drive_ids = [], [], []
    for drive_id in sorted(selected_drives):
        indexes_a = np.flatnonzero((drive == drive_id) & (camera == camera_a))
        indexes_b = np.flatnonzero((drive == drive_id) & (camera == camera_b))
        by_stamp_a = {int(stamp_ns[index]): index for index in indexes_a}
        by_stamp_b = {int(stamp_ns[index]): index for index in indexes_b}
        for stamp in sorted(set(by_stamp_a) & set(by_stamp_b)):
            first.append(values[by_stamp_a[stamp]])
            second.append(values[by_stamp_b[stamp]])
            drive_ids.append(drive_id)
    return np.asarray(first), np.asarray(second), np.asarray(drive_ids)


def _cross_block(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    cross = first.T @ second / max(len(first), 1)
    left, singular, right = np.linalg.svd(cross, full_matrices=False)
    singular = np.minimum(singular, 0.95)
    return (left * singular) @ right


def _pair_covariance(cross: np.ndarray) -> np.ndarray:
    covariance = np.block([[np.eye(2), cross], [cross.T, np.eye(2)]])
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    return (eigenvectors * np.maximum(eigenvalues, 1e-4)) @ eigenvectors.T


def _canonical_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 8:
        return math.nan
    first = first - np.mean(first, axis=0)
    second = second - np.mean(second, axis=0)
    scale = float(len(first) - 1)
    covariance_first = first.T @ first / scale
    covariance_second = second.T @ second / scale
    cross = first.T @ second / scale

    def inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
        return (eigenvectors * (1.0 / np.sqrt(np.maximum(eigenvalues, 1e-10)))) @ eigenvectors.T

    standardized_cross = inverse_sqrt(covariance_first) @ cross @ inverse_sqrt(covariance_second)
    return float(min(np.linalg.svd(standardized_cross, compute_uv=False)[0], 1.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    predictions_path = args.model_dir / "training_predictions.npz"
    dataset_path = args.model_dir / "dataset_contract.npz"
    predictions = np.load(predictions_path, allow_pickle=False)
    dataset = np.load(dataset_path, allow_pickle=False)
    target = predictions["target_ray"].astype(float)
    mean = predictions["rgb_gaussian_mean"].astype(float)
    covariance = predictions["rgb_gaussian_covariance"].astype(float)
    basis = predictions["basis"].astype(float)
    drive = predictions["drive"]
    camera = predictions["camera"]
    stamp_ns = dataset["stamp_ns"].astype(np.int64)
    cameras = sorted(set(camera.tolist()))
    drives = sorted(set(drive.tolist()))

    residual_ray = target - mean
    raw_projection_residual_world = (
        predictions["truth_world"].astype(float)
        - predictions["raw_world"].astype(float)
    )
    raw_projection_residual_ray = np.einsum(
        "nji,nj->ni", basis, raw_projection_residual_world)
    factors = np.linalg.cholesky(covariance)
    whitened_ray = np.linalg.solve(factors, residual_ray[..., None])[..., 0]
    residual_world = np.einsum("nij,nj->ni", basis, residual_ray)
    covariance_world = np.einsum("nij,njk,nlk->nil", basis, covariance, basis)
    world_factors = np.linalg.cholesky(covariance_world)
    whitened_world = np.linalg.solve(world_factors, residual_world[..., None])[..., 0]

    d2 = np.sum(whitened_ray**2, axis=1)
    diagnostics = {
        "sample_count": int(len(target)),
        "drive_count": int(len(drives)),
        "camera_count": int(len(cameras)),
        "whitened_ray_axis_0": _moments(whitened_ray[:, 0]),
        "whitened_ray_axis_1": _moments(whitened_ray[:, 1]),
        "mahalanobis2": {
            "mean": float(np.mean(d2)),
            "median": float(np.median(d2)),
            "p95": float(np.quantile(d2, 0.95)),
            "containment_95": float(np.mean(d2 <= 5.991)),
        },
        "temporal_acf_raw_projection_residual": _acf_summary(
            raw_projection_residual_ray, drive, camera, stamp_ns),
        "temporal_acf_corrected_residual_before_whitening": _acf_summary(
            residual_ray, drive, camera, stamp_ns),
        "temporal_acf_whitened_residual": _acf_summary(
            whitened_ray, drive, camera, stamp_ns),
    }

    temporal_folds = []
    for held_out in drives:
        training_drives = set(drives) - {held_out}
        held_drives = {held_out}
        train_mask = np.asarray([value in training_drives for value in drive])
        held_mask = drive == held_out
        iid_nu = _fit_nu(whitened_ray[train_mask])
        taus = {
            camera_id: _fit_tau(_sequences_for_camera(
                whitened_ray, drive, camera, stamp_ns, camera_id, training_drives))
            for camera_id in cameras
        }
        train_innovations, _, _ = _temporal_innovations(
            whitened_ray, drive, camera, stamp_ns, taus, training_drives)
        ou_nu = _fit_nu(train_innovations)
        scores = {
            "iid_gaussian": float(np.mean(_gaussian_nll(whitened_ray[held_mask]))),
            "iid_student_t": float(np.mean(_student_nll(whitened_ray[held_mask], iid_nu))),
            "ou_gaussian": float(np.mean(_score_temporal(
                whitened_ray, drive, camera, stamp_ns, held_drives, taus, None))),
            "ou_student_t": float(np.mean(_score_temporal(
                whitened_ray, drive, camera, stamp_ns, held_drives, taus, ou_nu))),
        }
        temporal_folds.append({
            "held_out_drive": held_out,
            "readings": int(np.sum(held_mask)),
            "iid_student_nu": iid_nu,
            "ou_student_nu": ou_nu,
            "ou_tau_s_by_camera": taus,
            "mean_nll_per_2d_reading": scores,
        })

    temporal_summary = {}
    for model in ("iid_gaussian", "iid_student_t", "ou_gaussian", "ou_student_t"):
        values = np.asarray([fold["mean_nll_per_2d_reading"][model]
                             for fold in temporal_folds])
        weights = np.asarray([fold["readings"] for fold in temporal_folds], dtype=float)
        temporal_summary[model] = {
            "drive_mean_nll": float(np.mean(values)),
            "reading_weighted_mean_nll": float(np.average(values, weights=weights)),
            "drive_median_nll": float(np.median(values)),
            "drive_wins": int(sum(
                fold["mean_nll_per_2d_reading"][model]
                == min(fold["mean_nll_per_2d_reading"].values())
                for fold in temporal_folds
            )),
        }

    # Descriptive simultaneous-camera dependence before temporal modelling.
    cross_diagnostics = {}
    all_drives = set(drives)
    for camera_index, camera_a in enumerate(cameras):
        for camera_b in cameras[camera_index + 1:]:
            projection_a, projection_b, pair_drives = _paired(
                raw_projection_residual_world, drive, camera, stamp_ns,
                camera_a, camera_b, all_drives)
            corrected_a, corrected_b, _ = _paired(
                residual_world, drive, camera, stamp_ns, camera_a, camera_b, all_drives)
            white_a, white_b, _ = _paired(
                whitened_world, drive, camera, stamp_ns, camera_a, camera_b, all_drives)
            per_drive = []
            for drive_id in drives:
                selected = pair_drives == drive_id
                if np.sum(selected) >= 8:
                    per_drive.append(_canonical_correlation(
                        white_a[selected], white_b[selected]))
            cross_diagnostics[f"{camera_a}|{camera_b}"] = {
                "simultaneous_pairs": int(len(projection_a)),
                "canonical_correlation_raw_projection_world": _canonical_correlation(
                    projection_a, projection_b),
                "canonical_correlation_corrected_world_before_whitening": _canonical_correlation(
                    corrected_a, corrected_b),
                "canonical_correlation_whitened_world": _canonical_correlation(white_a, white_b),
                "median_within_drive_whitened_canonical_correlation": (
                    float(np.median(per_drive)) if per_drive else math.nan),
                "drives_with_at_least_8_pairs": int(len(per_drive)),
            }

    # LODO cross-camera likelihood after fitting the OU transform on training drives.
    cross_folds = []
    for held_out in drives:
        training_drives = set(drives) - {held_out}
        held_drives = {held_out}
        taus = {
            camera_id: _fit_tau(_sequences_for_camera(
                whitened_ray, drive, camera, stamp_ns, camera_id, training_drives))
            for camera_id in cameras
        }
        train_values, train_indexes, _ = _temporal_innovations(
            whitened_ray, drive, camera, stamp_ns, taus, training_drives)
        held_values, held_indexes, _ = _temporal_innovations(
            whitened_ray, drive, camera, stamp_ns, taus, held_drives)
        transformed = np.full_like(whitened_ray, np.nan)
        transformed[train_indexes] = train_values
        transformed[held_indexes] = held_values
        train_vectors: list[np.ndarray] = []
        train_covariances: list[np.ndarray] = []
        held_vectors: list[np.ndarray] = []
        held_covariances: list[np.ndarray] = []
        pair_reports = {}
        for camera_index, camera_a in enumerate(cameras):
            for camera_b in cameras[camera_index + 1:]:
                first, second, _ = _paired(
                    transformed, drive, camera, stamp_ns, camera_a, camera_b, training_drives)
                held_first, held_second, _ = _paired(
                    transformed, drive, camera, stamp_ns, camera_a, camera_b, held_drives)
                if len(first) < 20 or not len(held_first):
                    continue
                cross = _cross_block(first, second)
                pair_covariance = _pair_covariance(cross)
                train_vectors.extend(np.concatenate((first, second), axis=1))
                train_covariances.extend([pair_covariance] * len(first))
                held_vectors.extend(np.concatenate((held_first, held_second), axis=1))
                held_covariances.extend([pair_covariance] * len(held_first))
                pair_reports[f"{camera_a}|{camera_b}"] = {
                    "train_pairs": int(len(first)), "held_pairs": int(len(held_first)),
                    "largest_cross_singular_value": float(np.linalg.svd(cross, compute_uv=False)[0]),
                }
        if not held_vectors:
            continue
        train_array = np.asarray(train_vectors)
        held_array = np.asarray(held_vectors)
        joint_nu = _fit_nu(train_array, train_covariances)
        independent_gaussian = _gaussian_nll(held_array)
        independent_nu = _fit_nu(train_array)
        independent_student = _student_nll(held_array, independent_nu)
        correlated_gaussian = _varying_covariance_nll(
            held_array, held_covariances, None)
        correlated_student = _varying_covariance_nll(
            held_array, held_covariances, joint_nu)
        cross_folds.append({
            "held_out_drive": held_out,
            "pair_observations": int(len(held_array)),
            "independent_student_nu": independent_nu,
            "correlated_student_nu": joint_nu,
            "mean_nll_per_4d_pair": {
                "ou_independent_gaussian": float(np.mean(independent_gaussian)),
                "ou_independent_student_t": float(np.mean(independent_student)),
                "ou_correlated_gaussian": float(np.mean(correlated_gaussian)),
                "ou_correlated_student_t": float(np.mean(correlated_student)),
            },
            "pair_fits": pair_reports,
        })

    cross_summary = {}
    names = ("ou_independent_gaussian", "ou_independent_student_t",
             "ou_correlated_gaussian", "ou_correlated_student_t")
    for model in names:
        values = np.asarray([fold["mean_nll_per_4d_pair"][model] for fold in cross_folds])
        weights = np.asarray([fold["pair_observations"] for fold in cross_folds], dtype=float)
        cross_summary[model] = {
            "drive_mean_nll": float(np.mean(values)),
            "pair_weighted_mean_nll": float(np.average(values, weights=weights)),
            "drive_median_nll": float(np.median(values)),
            "drive_wins": int(sum(
                fold["mean_nll_per_4d_pair"][model]
                == min(fold["mean_nll_per_4d_pair"].values())
                for fold in cross_folds
            )),
        }

    result = {
        "schema": "perception_residual_dependence_models.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_only_fixed_base_model_was_fit_on_all_drives",
        "population": {
            "layer": "admitted per-camera corrected readings",
            "reference": "reference-controlled robot position at camera capture timestamp",
            "sample_unit": "physical camera frame",
            "residual_model_fold_unit": "complete drive",
            "base_model_fit_boundary": "same 18 drives; in-sample and not thesis evidence",
            "readings": int(len(target)), "drives": int(len(drives)),
            "cameras": cameras,
        },
        "inputs": {
            "model_directory": str(args.model_dir.resolve()),
            "training_predictions_sha256": sha256(predictions_path),
            "dataset_contract_sha256": sha256(dataset_path),
        },
        "diagnostics": diagnostics,
        "temporal_model_test": {
            "metric": "mean predictive negative log-likelihood per 2D reading",
            "design": "leave one complete drive out for residual-model parameters",
            "summary": temporal_summary,
            "folds": temporal_folds,
        },
        "simultaneous_camera_diagnostics": cross_diagnostics,
        "combined_temporal_cross_camera_test": {
            "metric": "mean pairwise predictive negative log-likelihood per synchronized 4D camera pair",
            "design": "leave one complete drive out; OU parameters and cross blocks fit on other drives",
            "warning": "pairs from timestamps with three or more cameras overlap; use for model comparison, not an independent-sample confidence interval",
            "summary": cross_summary,
            "folds": cross_folds,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "diagnostics": diagnostics,
        "temporal_summary": temporal_summary,
        "cross_camera_diagnostics": cross_diagnostics,
        "combined_summary": cross_summary,
        "output": str(args.out),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
