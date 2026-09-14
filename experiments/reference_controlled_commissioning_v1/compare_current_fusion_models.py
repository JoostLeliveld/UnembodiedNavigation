#!/usr/bin/env python3
"""Diagnostic fusion comparison for the current all-drive RGB Gaussian model.

The perception model and its marginal covariance were fitted on all commissioning
drives.  Cross-camera residual blocks are fitted leave-one-complete-drive-out here,
but this remains an in-sample diagnostic of the base model, not navigation evidence.
Only readings with exactly equal drive and capture timestamp are fused.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


CHI2_95_2D = 5.991464547107979
CHI2_99_2D = 9.21034037197618
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")


def spd(matrix: np.ndarray, floor: float = 1.0e-9) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    return vectors @ np.diag(np.maximum(values, floor)) @ vectors.T


def world_values(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = np.asarray(data["basis"], dtype=float)
    correction = np.asarray(data["rgb_gaussian_mean"], dtype=float)
    covariance_ray = np.asarray(data["rgb_gaussian_covariance"], dtype=float)
    estimate = np.asarray(data["raw_world"], dtype=float) + np.einsum(
        "nij,nj->ni", basis, correction
    )
    covariance = np.einsum("nij,njk,nlk->nil", basis, covariance_ray, basis)
    residual = estimate - np.asarray(data["truth_world"], dtype=float)
    return estimate, covariance, residual


def groups(data: dict[str, np.ndarray]) -> list[np.ndarray]:
    result: dict[tuple[str, int], list[int]] = {}
    for index, (drive, stamp) in enumerate(
        zip(data["drive"], data["stamp_ns"], strict=True)
    ):
        result.setdefault((str(drive), int(stamp)), []).append(index)
    return [np.asarray(value, dtype=int) for value in result.values()]


def standardized(residual: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    output = np.empty_like(residual)
    for index in range(len(residual)):
        output[index] = np.linalg.solve(np.linalg.cholesky(spd(covariance[index])), residual[index])
    return output


def fit_cross_blocks(
    data: dict[str, np.ndarray], residual: np.ndarray, covariance: np.ndarray, held_drive: str
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, int]]:
    z = standardized(residual, covariance)
    accum: dict[tuple[str, str], list[np.ndarray]] = {}
    for indices in groups(data):
        if str(data["drive"][indices[0]]) == held_drive:
            continue
        lookup = {str(data["camera"][index]): int(index) for index in indices}
        for ai, first in enumerate(CAMERAS):
            for second in CAMERAS[ai + 1 :]:
                if first in lookup and second in lookup:
                    accum.setdefault((first, second), []).append(
                        np.outer(z[lookup[first]], z[lookup[second]])
                    )
    blocks = {key: np.mean(value, axis=0) for key, value in accum.items()}
    return blocks, {"|".join(key): len(value) for key, value in accum.items()}


def independent_covariance(covariances: np.ndarray) -> np.ndarray:
    count = len(covariances)
    result = np.zeros((2 * count, 2 * count), dtype=float)
    for index, covariance in enumerate(covariances):
        result[2 * index : 2 * index + 2, 2 * index : 2 * index + 2] = covariance
    return result


def correlated_covariance(
    cameras: np.ndarray,
    covariances: np.ndarray,
    blocks: dict[tuple[str, str], np.ndarray],
) -> tuple[np.ndarray, float]:
    result = independent_covariance(covariances)
    cross: list[tuple[int, int, np.ndarray]] = []
    factors = [np.linalg.cholesky(spd(value)) for value in covariances]
    for first in range(len(cameras)):
        for second in range(first + 1, len(cameras)):
            a, b = str(cameras[first]), str(cameras[second])
            key = (a, b) if a < b else (b, a)
            block = blocks.get(key)
            if block is None:
                continue
            if a > b:
                block = block.T
            cross.append((first, second, factors[first] @ block @ factors[second].T))

    # Back off only if the empirical pair blocks do not form a joint SPD matrix.
    alpha = 1.0
    while alpha >= 1.0e-4:
        candidate = result.copy()
        for first, second, block in cross:
            candidate[2 * first : 2 * first + 2, 2 * second : 2 * second + 2] = alpha * block
            candidate[2 * second : 2 * second + 2, 2 * first : 2 * first + 2] = alpha * block.T
        if np.linalg.eigvalsh(candidate).min() > 1.0e-10:
            return candidate, alpha
        alpha *= 0.8
    return result, 0.0


def gls(observations: np.ndarray, joint_covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.tile(np.eye(2), (len(observations), 1))
    inverse = np.linalg.inv(spd(joint_covariance))
    information = h.T @ inverse @ h
    covariance = spd(np.linalg.inv(information))
    mean = covariance @ h.T @ inverse @ observations.reshape(-1)
    return mean, covariance


def score(records: list[dict[str, object]]) -> dict[str, float | int]:
    errors = np.asarray([value["error_m"] for value in records], dtype=float)
    mahalanobis = np.asarray([value["mahalanobis2"] for value in records], dtype=float)
    nll = np.asarray([value["nll"] for value in records], dtype=float)
    area = np.asarray([value["area95_cm2"] for value in records], dtype=float)
    return {
        "rounds": len(records),
        "mean_error_cm": float(100.0 * errors.mean()),
        "median_error_cm": float(100.0 * np.median(errors)),
        "rmse_cm": float(100.0 * math.sqrt(np.mean(errors**2))),
        "p95_error_cm": float(100.0 * np.quantile(errors, 0.95)),
        "mean_mahalanobis2": float(mahalanobis.mean()),
        "coverage95": float(np.mean(mahalanobis <= CHI2_95_2D)),
        "mean_nll": float(nll.mean()),
        "mean_ellipse_area95_cm2": float(area.mean()),
    }


def record(mean: np.ndarray, covariance: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = mean - truth
    inverse = np.linalg.inv(spd(covariance))
    maha = float(error @ inverse @ error)
    _, logdet = np.linalg.slogdet(spd(covariance))
    return {
        "error_m": float(np.linalg.norm(error)),
        "mahalanobis2": maha,
        "nll": float(0.5 * (2.0 * math.log(2.0 * math.pi) + logdet + maha)),
        "area95_cm2": float(math.pi * CHI2_95_2D * math.sqrt(np.linalg.det(covariance)) * 1.0e4),
    }


def joseph_update(
    mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray,
    measurement_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    innovation_covariance = spd(covariance + measurement_covariance)
    gain = covariance @ np.linalg.inv(innovation_covariance)
    identity = np.eye(2) - gain
    updated_mean = mean + gain @ (measurement - mean)
    updated_covariance = spd(
        identity @ covariance @ identity.T + gain @ measurement_covariance @ gain.T
    )
    return updated_mean, updated_covariance


def replay_drive(
    data: dict[str, np.ndarray], estimate: np.ndarray, covariance: np.ndarray,
    blocks: dict[tuple[str, str], np.ndarray], held_drive: str, fusion: str,
    operational: bool, process_noise: float = 0.02,
) -> dict[str, float | int]:
    drive_groups = [
        indices for indices in groups(data)
        if str(data["drive"][indices[0]]) == held_drive
    ]
    drive_groups.sort(key=lambda indices: int(data["stamp_ns"][indices[0]]))
    first_truth = np.asarray(data["truth_world"][drive_groups[0][0]], dtype=float)
    mean = first_truth.copy()
    belief_covariance = np.eye(2) * 0.25
    previous_truth = first_truth.copy()
    previous_stamp = int(data["stamp_ns"][drive_groups[0][0]])
    errors, nees = [], []
    accepted_stamps: list[int] = []
    nis_values: list[float] = []
    rejected = 0
    for indices in drive_groups:
        stamp = int(data["stamp_ns"][indices[0]])
        truth = np.asarray(data["truth_world"][indices[0]], dtype=float)
        dt = max((stamp - previous_stamp) * 1.0e-9, 0.0)
        mean += truth - previous_truth
        belief_covariance = spd(belief_covariance + np.eye(2) * process_noise**2 * dt)
        previous_truth, previous_stamp = truth, stamp

        covariances = covariance[indices]
        observations = estimate[indices]
        if fusion == "best_single":
            best = int(np.argmin(np.linalg.det(covariances)))
            measurement = observations[best]
            measurement_covariance = covariances[best]
        elif fusion == "independent_gls":
            measurement, measurement_covariance = gls(
                observations, independent_covariance(covariances)
            )
        elif fusion == "correlated_gls":
            joint, _ = correlated_covariance(data["camera"][indices], covariances, blocks)
            measurement, measurement_covariance = gls(observations, joint)
        else:
            raise ValueError(f"unknown fusion method {fusion}")
        innovation = measurement - mean
        innovation_covariance = spd(belief_covariance + measurement_covariance)
        nis = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
        nis_values.append(nis)
        if not operational or nis <= CHI2_99_2D:
            mean, belief_covariance = joseph_update(
                mean, belief_covariance, measurement, measurement_covariance
            )
            accepted_stamps.append(stamp)
        else:
            rejected += 1
        error = mean - truth
        errors.append(float(np.linalg.norm(error)))
        nees.append(float(error @ np.linalg.solve(belief_covariance, error)))

    stamps = [int(data["stamp_ns"][indices[0]]) for indices in drive_groups]
    anchors = sorted(set([stamps[0], *accepted_stamps, stamps[-1]]))
    longest_gap = max((b - a) * 1.0e-9 for a, b in zip(anchors[:-1], anchors[1:]))
    return {
        "rounds": len(drive_groups),
        "rmse_cm": float(100.0 * math.sqrt(np.mean(np.asarray(errors) ** 2))),
        "mean_nees": float(np.mean(nees)),
        "coverage95": float(np.mean(np.asarray(nees) <= CHI2_95_2D)),
        "mean_nis": float(np.mean(nis_values)),
        "rejected_rounds": rejected,
        "rejection_fraction": float(rejected / len(drive_groups)),
        "longest_without_accepted_correction_s": float(longest_gap),
    }


def aggregate_replays(per_drive: dict[str, dict[str, float | int]]) -> dict[str, float | int]:
    rows = list(per_drive.values())
    return {
        "complete_drives": len(rows),
        "equal_drive_rmse_cm": float(np.mean([row["rmse_cm"] for row in rows])),
        "equal_drive_mean_nees": float(np.mean([row["mean_nees"] for row in rows])),
        "equal_drive_coverage95": float(np.mean([row["coverage95"] for row in rows])),
        "equal_drive_mean_nis": float(np.mean([row["mean_nis"] for row in rows])),
        "total_rejected_rounds": int(sum(row["rejected_rounds"] for row in rows)),
        "equal_drive_rejection_fraction": float(np.mean([row["rejection_fraction"] for row in rows])),
        "equal_drive_longest_outage_s": float(np.mean([
            row["longest_without_accepted_correction_s"] for row in rows
        ])),
        "worst_drive_longest_outage_s": float(max([
            row["longest_without_accepted_correction_s"] for row in rows
        ])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract_path = args.model_root / "dataset_contract.npz"
    prediction_path = args.model_root / "training_predictions.npz"
    with np.load(contract_path, allow_pickle=False) as archive:
        contract = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(prediction_path, allow_pickle=False) as archive:
        prediction = {key: np.asarray(archive[key]) for key in archive.files}
    identity_fields = {
        "drive": "drive",
        "camera": "camera",
        "basis": "basis",
        "raw": "raw_world",
        "truth": "truth_world",
    }
    for contract_key, prediction_key in identity_fields.items():
        if not np.array_equal(contract[contract_key], prediction[prediction_key]):
            raise RuntimeError(f"prediction identity mismatch for {prediction_key}")
    data = dict(prediction)
    data["stamp_ns"] = contract["stamp_ns"]
    estimate, covariance, residual = world_values(data)

    methods = {name: [] for name in ("best_single", "independent_gls", "correlated_gls")}
    per_drive: dict[str, dict[str, dict[str, float | int]]] = {}
    estimator_per_drive = {
        f"{name}_{mode}": {} for name in methods for mode in ("forced", "nis99")
    }
    fold_metadata = []
    for held_drive in sorted(set(map(str, data["drive"]))):
        blocks, counts = fit_cross_blocks(data, residual, covariance, held_drive)
        local = {name: [] for name in methods}
        backoff = []
        for indices in groups(data):
            if str(data["drive"][indices[0]]) != held_drive:
                continue
            truth = np.asarray(data["truth_world"][indices[0]], dtype=float)
            if not np.allclose(data["truth_world"][indices], truth, atol=1.0e-7):
                raise RuntimeError("synchronized readings disagree on reference position")
            observations = estimate[indices]
            covariances = covariance[indices]
            best = int(np.argmin(np.linalg.det(covariances)))
            local["best_single"].append(record(observations[best], covariances[best], truth))
            independent = independent_covariance(covariances)
            mean, fused_covariance = gls(observations, independent)
            local["independent_gls"].append(record(mean, fused_covariance, truth))
            joint, alpha = correlated_covariance(data["camera"][indices], covariances, blocks)
            backoff.append(alpha)
            mean, fused_covariance = gls(observations, joint)
            local["correlated_gls"].append(record(mean, fused_covariance, truth))
        per_drive[held_drive] = {name: score(values) for name, values in local.items()}
        for name in methods:
            estimator_per_drive[f"{name}_forced"][held_drive] = replay_drive(
                data, estimate, covariance, blocks, held_drive, name, False
            )
            estimator_per_drive[f"{name}_nis99"][held_drive] = replay_drive(
                data, estimate, covariance, blocks, held_drive, name, True
            )
        for name in methods:
            methods[name].extend(local[name])
        fold_metadata.append({
            "held_out_drive": held_drive,
            "pair_training_counts": counts,
            "minimum_cross_block_scale": float(min(backoff)),
            "mean_cross_block_scale": float(np.mean(backoff)),
        })

    payload = {
        "schema": "current_fusion_diagnostic.v1",
        "status": "diagnostic_only",
        "reporting_boundary": (
            "The RGB mean and marginal covariance were trained on all 18 drives. "
            "Only cross-camera blocks are leave-one-drive-out. These are not independent "
            "navigation results and must not be registered as thesis evidence."
        ),
        "identity": {
            "model_root": str(args.model_root.resolve()),
            "dataset_contract": str(contract_path.resolve()),
            "training_predictions": str(prediction_path.resolve()),
        },
        "population": {
            "camera_readings": int(len(data["drive"])),
            "synchronized_rounds": len(groups(data)),
            "complete_drives": len(set(map(str, data["drive"]))),
        },
        "method": {
            "best_single": "reading with smallest predicted covariance determinant",
            "independent_gls": "Gaussian GLS with block-diagonal marginal covariance",
            "correlated_gls": "Gaussian GLS with leave-one-drive-out empirical standardized cross-camera blocks",
        },
        "aggregate": {name: score(values) for name, values in methods.items()},
        "per_drive": per_drive,
        "controlled_estimator_replay": {
            "boundary": (
                "Reference increments are used as the common motion mean with fixed "
                "Q=0.02 m/sqrt(s). This isolates measurement/fusion consistency and is "
                "not an end-to-end localization or navigation result."
            ),
            "process_noise_m_per_sqrt_s": 0.02,
            "nis_gate": {"dimensions": 2, "confidence": 0.99, "threshold": CHI2_99_2D},
            "aggregate": {
                name: aggregate_replays(values) for name, values in estimator_per_drive.items()
            },
            "per_drive": estimator_per_drive,
        },
        "fold_metadata": fold_metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
