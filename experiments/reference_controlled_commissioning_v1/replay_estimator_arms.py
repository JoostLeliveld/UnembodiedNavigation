#!/usr/bin/env python3
"""Replay a position belief along held-out drives under competing measurement models.

A calibrated per-reading covariance does not by itself make the belief honest.  What the
planner consumes is the belief, so the model has to be judged inside the filter: the same
recorded measurements and the same reference motion are pushed through every arm, and the
arms differ in exactly one thing, the covariance handed to the update.

  E0  one pooled Gaussian                      the incumbent baseline
  E1  Gaussian per camera and range tertile    conditioning, still one regime
  E2  mixture, learned weight, calibrated      the selected measurement model
  E3  E2 with a measured cross-camera block    simultaneous readings are not independent
  E4  E2 with conservative fusion              covariance intersection, dependence unknown

Reported per drive: position RMSE, NEES against the two-dimensional belief, 95 percent
coverage, accepted and rejected updates.  NEES is the number that decides honesty: it
should sit at 1.0, below means the belief is too wide and above means it is overconfident.
The replication unit is the drive, and the drives are the ones the models never saw.
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
import fit_measurement_noise_model as noise  # noqa: E402

CAMERAS = base.CAMERAS
CHI2_2 = base.CHI2_2


def replay(records: list[dict], rounds: list[dict], drive: str,
           covariance_of: dict[int, np.ndarray], cross: float,
           process_noise: float, nis_threshold: float) -> dict:
    """Propagate a 2-D position belief along one drive and fuse the admitted readings."""
    drive_rounds = sorted([r for r in rounds if r["drive"] == drive],
                          key=lambda item: item["stamp_ns"])
    by_stamp: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        if record["drive"] == drive:
            by_stamp.setdefault(record["stamp_ns"], []).append(index)
    if not drive_rounds:
        raise RuntimeError(f"no rounds for {drive}")

    mean = drive_rounds[0]["reference"].copy()
    covariance = np.eye(2) * 0.25
    previous = None
    errors, nees, nis_values = [], [], []
    accepted = rejected = 0

    for entry in drive_rounds:
        stamp = entry["stamp_ns"]
        if previous is not None:
            dt = (stamp - previous) / 1e9
            covariance = covariance + np.eye(2) * (process_noise ** 2) * dt
            heading = entry["reference_yaw"]
            mean = mean + np.asarray([math.cos(heading), math.sin(heading)]) \
                * entry["reference_speed_mps"] * dt
        previous = stamp

        members = [i for i in by_stamp.get(stamp, []) if i in covariance_of]
        if members:
            count = len(members)
            # corrected world position = raw projection + the predicted ray-frame
            # correction rotated into world coordinates
            innovation = np.concatenate([
                records[i]["raw"] + records[i]["basis"] @ records[i]["corrected_ray"] - mean
                for i in members])
            jacobian = np.vstack([np.eye(2)] * count)
            blocks = [records[i]["basis"] @ covariance_of[i] @ records[i]["basis"].T
                      for i in members]
            noise_matrix = np.zeros((2 * count, 2 * count))
            for position, block in enumerate(blocks):
                noise_matrix[2 * position:2 * position + 2,
                             2 * position:2 * position + 2] = block
            if cross > 0.0 and count > 1:
                if cross >= 1.0:
                    # Covariance intersection with equal weights: each reading's covariance
                    # is divided by its mixing coefficient, so simultaneous readings can
                    # never be counted as independent evidence.
                    for position in range(count):
                        noise_matrix[2 * position:2 * position + 2,
                                     2 * position:2 * position + 2] *= count
                else:
                    # Measured dependence: a shared component of the stated size.
                    for a in range(count):
                        for b in range(count):
                            if a == b:
                                continue
                            shared = cross * np.sqrt(np.outer(np.diag(blocks[a]),
                                                              np.diag(blocks[b])))
                            noise_matrix[2 * a:2 * a + 2, 2 * b:2 * b + 2] = shared
            noise_matrix = base.spd(noise_matrix)
            innovation_covariance = base.spd(jacobian @ covariance @ jacobian.T + noise_matrix)
            inverse = np.linalg.inv(innovation_covariance)
            distance = float(innovation @ inverse @ innovation)
            nis_values.append(distance / (2 * count))
            if distance <= nis_threshold * 2 * count:
                gain = covariance @ jacobian.T @ inverse
                mean = mean + gain @ innovation
                identity = np.eye(2) - gain @ jacobian
                covariance = base.spd(identity @ covariance @ identity.T
                                      + gain @ noise_matrix @ gain.T)
                accepted += 1
            else:
                rejected += 1

        error = mean - entry["reference"]
        errors.append(float(np.linalg.norm(error)))
        nees.append(float(error @ np.linalg.inv(base.spd(covariance)) @ error))

    errors, nees = np.asarray(errors), np.asarray(nees)
    return {
        "drive": drive,
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "median_error_m": float(np.median(errors)),
        "p95_error_m": float(np.percentile(errors, 95)),
        "mean_nees": float(np.mean(nees)),
        "median_nees": float(np.median(nees)),
        "coverage_95": float(np.mean(nees <= CHI2_2["95"])),
        "accepted_updates": accepted,
        "rejected_updates": rejected,
        "mean_normalised_nis": float(np.mean(nis_values)) if nis_values else math.nan,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--noise-model", required=True, type=Path,
                        help="measurement_noise_model.json from the fitting stage")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=260912)
    parser.add_argument("--process-noise", type=float, default=0.10)
    parser.add_argument("--nis-threshold", type=float, default=CHI2_2["99"] / 2.0)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)

    _, records, rounds = base.load_capture(arguments.capture_root.resolve())
    fitted = json.loads(arguments.noise_model.read_text(encoding="utf-8"))
    partition = np.asarray([r["partition"] for r in records])
    drive_of = np.asarray([r["drive"] for r in records])
    fit_index = np.flatnonzero(partition == "fit")
    dev_index = np.flatnonzero(partition == "development")
    target = np.stack([r["target_ray"] for r in records])

    # frozen mean correction, leave one drive out inside fit, all-fit model on development
    corrected = np.zeros((len(records), 2))
    for held in sorted(set(drive_of[fit_index].tolist())):
        inner = fit_index[drive_of[fit_index] != held]
        outer = fit_index[drive_of[fit_index] == held]
        corrected[outer] = base.predict_mean(
            records, outer, "box_mlp",
            base.fit_mean(records, inner, "box_mlp", arguments.seed))
    corrected[dev_index] = base.predict_mean(
        records, dev_index, "box_mlp",
        base.fit_mean(records, fit_index, "box_mlp", arguments.seed))
    for index, record in enumerate(records):
        record["corrected_ray"] = corrected[index]
    residual = target - corrected

    fit_labels, edges = noise.strata(records, fit_index, None)
    dev_labels, _ = noise.strata(records, dev_index, edges)

    gaussian = noise.fit_gaussian(residual[fit_index], fit_labels)
    pooled = np.asarray(gaussian["pooled"])
    mixture = noise.fit_mixture(residual[fit_index], fit_labels, arguments.seed)

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

    fit_weights, fit_good, fit_bad = noise.mixture_components(mixture, fit_labels)
    fit_responsibility = np.exp(
        np.log(np.clip(fit_weights, 1e-9, 1))
        + noise.gaussian_logpdf(residual[fit_index], fit_good)
        - noise.mixture_loglik(residual[fit_index], fit_weights, fit_good, fit_bad))
    classifier = GradientBoostingClassifier(random_state=arguments.seed)
    classifier.fit(regime_features(fit_index), (fit_responsibility < 0.5).astype(int))

    _, dev_good, dev_bad = noise.mixture_components(mixture, dev_labels)
    predicted_good = np.clip(
        1.0 - classifier.predict_proba(regime_features(dev_index))[:, 1], 0.02, 0.98)
    scale = float(fitted["models"]["M1b_learned_weight_calibrated"]["calibration_scale"])
    mixture_covariance = np.stack([
        base.spd(scale * m) for m in
        (predicted_good[:, None, None] * dev_good
         + (1.0 - predicted_good)[:, None, None] * dev_bad)])

    # Cross-camera dependence, measured on THIS capture rather than imported: simultaneous
    # residuals are whitened by the selected covariance and their pairwise correlation is
    # taken over rounds where two or more cameras reported at the same instant.
    by_round: dict[tuple[str, int], list[tuple[str, np.ndarray]]] = {}
    for position, i in enumerate(dev_index):
        factor = np.linalg.cholesky(mixture_covariance[position])
        by_round.setdefault((records[i]["drive"], records[i]["stamp_ns"]), []).append(
            (records[i]["camera"], np.linalg.solve(factor, residual[i])))
    pairs: dict[str, list[float]] = {}
    for members in by_round.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                key = "|".join(sorted([members[a][0], members[b][0]]))
                pairs.setdefault(key, []).append(
                    float(np.dot(members[a][1], members[b][1]) / 2.0))
    pairwise = {}
    for key, values in sorted(pairs.items()):
        if len(values) < 30:
            continue
        v = np.asarray(values)
        pairwise[key] = {"n": int(len(v)), "mean_correlation": float(v.mean()),
                         "se": float(v.std(ddof=1) / math.sqrt(len(v)))}
    significant = [v["mean_correlation"] for v in pairwise.values()
                   if abs(v["mean_correlation"]) > 2 * v["se"]]
    # Only positive shared error inflates a fused covariance; a negative correlation would
    # make fusion sharper, and assuming that is not conservative.
    measured_cross = float(np.mean([c for c in significant if c > 0])) if any(
        c > 0 for c in significant) else 0.0

    gaussian_pooled = np.repeat(pooled[None], len(dev_index), axis=0)
    gaussian_strata = noise.predict_gaussian(gaussian, dev_labels)

    ARMS = {
        "E0_pooled_gaussian": (gaussian_pooled, 0.0),
        "E1_stratified_gaussian": (gaussian_strata, 0.0),
        "E2_mixture_calibrated": (mixture_covariance, 0.0),
        "E3_mixture_cross_camera": (mixture_covariance, measured_cross),
        "E4_mixture_conservative": (mixture_covariance, 1.0),
    }

    dev_drives = sorted(set(drive_of[dev_index].tolist()))
    results: dict[str, Any] = {}
    for name, (covariances, cross) in ARMS.items():
        lookup = {int(i): covariances[position] for position, i in enumerate(dev_index)}
        per_drive = [replay(records, rounds, drive, lookup, cross,
                            arguments.process_noise, arguments.nis_threshold)
                     for drive in dev_drives]
        results[name] = {
            "cross_camera_term": cross,
            "equal_drive_rmse_m": float(np.mean([d["rmse_m"] for d in per_drive])),
            "equal_drive_p95_error_m": float(np.mean([d["p95_error_m"] for d in per_drive])),
            "equal_drive_mean_nees": float(np.mean([d["mean_nees"] for d in per_drive])),
            "nees_sd_over_drives": float(np.std([d["mean_nees"] for d in per_drive], ddof=1)),
            "equal_drive_coverage_95": float(np.mean([d["coverage_95"] for d in per_drive])),
            "total_accepted": int(sum(d["accepted_updates"] for d in per_drive)),
            "total_rejected": int(sum(d["rejected_updates"] for d in per_drive)),
            "per_drive": per_drive,
        }

    report = {
        "schema": "commissioned_estimator_replay.v1",
        "capture_root": str(arguments.capture_root),
        "audit_analysis_permitted": False,
        "noise_model": str(arguments.noise_model),
        "cross_camera_pairwise": pairwise,
        "cross_camera_term_used": measured_cross,
        "cross_camera_rule": ("mean of the significantly positive pairwise correlations; "
                              "negative correlations are not used because assuming them "
                              "would sharpen the fusion rather than guard it"),
        "development_drives": dev_drives,
        "process_noise_m_per_sqrt_s": arguments.process_noise,
        "nis_threshold_per_dof": arguments.nis_threshold,
        "arms": results,
    }
    base.atomic_json(arguments.output / "estimator_replay.json", report)
    print(json.dumps({
        "measured_cross_camera": round(measured_cross, 4),
        "cross_camera_pairs_significant": len(significant),
        "rmse_cm": {k: round(v["equal_drive_rmse_m"] * 100, 2) for k, v in results.items()},
        "mean_nees": {k: round(v["equal_drive_mean_nees"], 3) for k, v in results.items()},
        "nees_sd": {k: round(v["nees_sd_over_drives"], 3) for k, v in results.items()},
        "coverage_95": {k: round(v["equal_drive_coverage_95"], 3) for k, v in results.items()},
        "accepted": {k: v["total_accepted"] for k, v in results.items()},
        "rejected": {k: v["total_rejected"] for k, v in results.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
