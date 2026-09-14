#!/usr/bin/env python3
"""Matched belief replay for every commissioned covariance candidate.

All covariance models are fitted on the six fit drives.  Belief consistency is
reported on the six development drives.  The audit partition is never opened.

Two replays are produced for every R model:

* forced assimilation, which gives every R exactly the same update batches
* operational assimilation, which applies the same fixed 2-D NIS gate

The motion mean uses reference increments so this is a controlled sensor and
belief test, not an end-to-end navigation result.  Ground truth is used only to
construct those common offline increments and to score the resulting belief.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

import compare_r_methods_supervisor as covariance_analysis


CHI2 = covariance_analysis.CHI2
DISPLAY = covariance_analysis.DISPLAY
COLORS = covariance_analysis.COLORS
FLOOR_M2 = covariance_analysis.FLOOR_M2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def spd(matrix: np.ndarray, floor: float = FLOOR_M2) -> np.ndarray:
    return covariance_analysis.spd(matrix, floor=floor)


def load_rounds(campaign_root: Path) -> list[dict[str, Any]]:
    execution = json.loads((campaign_root / "campaign_execution.json").read_text(encoding="utf-8"))
    if execution.get("status") != "collection_complete_audit_sealed":
        raise RuntimeError("campaign is not complete with audit sealed")
    if execution.get("audit_analysis_permitted") is not False:
        raise RuntimeError("audit must remain sealed")
    sources = [entry for entry in execution["drive_sources"]
               if entry["partition"] in {"fit", "development"}]
    if any("audit" in entry["drive_id"] or "audit" in entry["source"] for entry in sources):
        raise RuntimeError("refusing an audit source")
    rounds: list[dict[str, Any]] = []
    for entry in sources:
        table = Path(entry["source"]) / "tables/network_rounds.csv"
        with table.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["reference_supported"] != "1":
                    continue
                if row["drive_id"] != entry["drive_id"] or row["partition"] != entry["partition"]:
                    raise RuntimeError("network round crossed its declared partition")
                rounds.append({
                    "drive": row["drive_id"],
                    "partition": row["partition"],
                    "stamp": int(row["capture_stamp_ns"]),
                    "reference": np.asarray([float(row["reference_x"]), float(row["reference_y"])]),
                })
    rounds.sort(key=lambda item: (item["drive"], item["stamp"]))
    identities = [(item["drive"], item["stamp"]) for item in rounds]
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate network round identity")
    return rounds


def joseph_update(mean: np.ndarray, covariance: np.ndarray,
                  measurement: np.ndarray, measurement_covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    innovation_covariance = spd(covariance + measurement_covariance)
    gain = covariance @ np.linalg.inv(innovation_covariance)
    innovation = measurement - mean
    updated_mean = mean + gain @ innovation
    identity = np.eye(2) - gain
    updated_covariance = spd(
        identity @ covariance @ identity.T + gain @ measurement_covariance @ gain.T
    )
    return updated_mean, updated_covariance


def joint_measurement(observations: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = len(observations)
    if covariance.shape != (2 * count, 2 * count):
        raise ValueError("joint covariance shape does not match observations")
    h = np.tile(np.eye(2), (count, 1))
    inverse = np.linalg.inv(spd(covariance))
    information = h.T @ inverse @ h
    fused_covariance = spd(np.linalg.inv(information))
    fused_mean = fused_covariance @ h.T @ inverse @ observations.reshape(-1)
    return fused_mean, fused_covariance


def independent_joint_covariance(covariances: np.ndarray) -> np.ndarray:
    count = len(covariances)
    result = np.zeros((2 * count, 2 * count))
    for index, matrix in enumerate(covariances):
        result[2 * index:2 * index + 2, 2 * index:2 * index + 2] = matrix
    return result


def shared_joint_covariance(covariance_ray: np.ndarray, bases: np.ndarray,
                            shared_block: np.ndarray) -> np.ndarray:
    count = len(covariance_ray)
    world = covariance_analysis.world_covariances(
        {"basis": bases}, covariance_ray
    )
    result = independent_joint_covariance(world)
    factors = [np.linalg.cholesky(spd(matrix)) for matrix in covariance_ray]
    for a in range(count):
        for b in range(a):
            cross_ray = factors[a] @ shared_block @ factors[b].T
            cross_world = bases[a] @ cross_ray @ bases[b].T
            result[2 * a:2 * a + 2, 2 * b:2 * b + 2] = cross_world
            result[2 * b:2 * b + 2, 2 * a:2 * a + 2] = cross_world.T
    return spd(result, floor=1.0e-5)


def observation_groups(data: dict[str, np.ndarray]) -> dict[tuple[str, int], np.ndarray]:
    grouped: dict[tuple[str, int], list[int]] = {}
    for index, (drive, stamp) in enumerate(zip(data["drive"], data["stamp"], strict=True)):
        grouped.setdefault((str(drive), int(stamp)), []).append(index)
    return {key: np.asarray(value, dtype=int) for key, value in grouped.items()}


def longest_gap(stamps: list[int], accepted_stamps: list[int]) -> float:
    if len(stamps) < 2:
        return 0.0
    anchors = [stamps[0], *accepted_stamps, stamps[-1]]
    anchors = sorted(set(anchors))
    if len(anchors) < 2:
        return float((stamps[-1] - stamps[0]) / 1.0e9)
    return float(max(b - a for a, b in zip(anchors[:-1], anchors[1:], strict=True)) / 1.0e9)


def summarize_drive(errors: np.ndarray, nees: np.ndarray, areas: np.ndarray,
                    nis: np.ndarray, accepted_rounds: int, rejected_rounds: int,
                    accepted_frames: int, rejected_frames: int, gap_s: float) -> dict[str, Any]:
    total_rounds = accepted_rounds + rejected_rounds
    total_frames = accepted_frames + rejected_frames
    return {
        "belief_samples": int(len(errors)),
        "rmse_m": float(math.sqrt(np.mean(errors ** 2))),
        "mean_error_m": float(np.mean(errors)),
        "median_error_m": float(np.median(errors)),
        "p95_error_m": float(np.quantile(errors, 0.95)),
        "mean_nees": float(np.mean(nees)),
        "median_nees": float(np.median(nees)),
        "belief_coverage": {str(level): float(np.mean(nees <= threshold))
                            for level, threshold in CHI2.items()},
        "mean_95_ellipse_area_cm2": float(np.mean(areas)),
        "median_95_ellipse_area_cm2": float(np.median(areas)),
        "attempted_update_rounds": int(total_rounds),
        "accepted_update_rounds": int(accepted_rounds),
        "rejected_update_rounds": int(rejected_rounds),
        "accepted_round_fraction": float(accepted_rounds / max(total_rounds, 1)),
        "attempted_frames": int(total_frames),
        "accepted_frames": int(accepted_frames),
        "rejected_frames": int(rejected_frames),
        "accepted_frame_fraction": float(accepted_frames / max(total_frames, 1)),
        "longest_without_accepted_correction_s": gap_s,
        "mean_nis": float(np.mean(nis)) if len(nis) else 0.0,
        "nis_coverage": {str(level): float(np.mean(nis <= threshold)) if len(nis) else 0.0
                         for level, threshold in CHI2.items()},
    }


def aggregate_drives(per_drive: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = list(per_drive.values())
    return {
        "complete_drives": len(values),
        "equal_drive_rmse_m": float(np.mean([value["rmse_m"] for value in values])),
        "equal_drive_mean_error_m": float(np.mean([value["mean_error_m"] for value in values])),
        "equal_drive_median_error_m": float(np.mean([value["median_error_m"] for value in values])),
        "equal_drive_p95_error_m": float(np.mean([value["p95_error_m"] for value in values])),
        "equal_drive_mean_nees": float(np.mean([value["mean_nees"] for value in values])),
        "equal_drive_median_nees": float(np.mean([value["median_nees"] for value in values])),
        "equal_drive_belief_coverage": {
            str(level): float(np.mean([value["belief_coverage"][str(level)] for value in values]))
            for level in CHI2
        },
        "equal_drive_mean_95_ellipse_area_cm2": float(np.mean([
            value["mean_95_ellipse_area_cm2"] for value in values
        ])),
        "equal_drive_accepted_round_fraction": float(np.mean([
            value["accepted_round_fraction"] for value in values
        ])),
        "equal_drive_accepted_frame_fraction": float(np.mean([
            value["accepted_frame_fraction"] for value in values
        ])),
        "equal_drive_longest_gap_s": float(np.mean([
            value["longest_without_accepted_correction_s"] for value in values
        ])),
        "worst_drive_longest_gap_s": float(np.max([
            value["longest_without_accepted_correction_s"] for value in values
        ])),
        "equal_drive_mean_nis": float(np.mean([value["mean_nis"] for value in values])),
        "equal_drive_nis_coverage": {
            str(level): float(np.mean([value["nis_coverage"][str(level)] for value in values]))
            for level in CHI2
        },
        "by_drive": per_drive,
    }


def replay_direct(data: dict[str, np.ndarray], rounds: list[dict[str, Any]],
                  covariance_ray: np.ndarray, mode: str, process_noise: float,
                  nis_threshold: float, shared_block: np.ndarray | None = None) -> dict[str, Any]:
    if mode not in {"forced", "operational"}:
        raise ValueError("unknown replay mode")
    world_covariance = covariance_analysis.world_covariances(data, covariance_ray)
    groups = observation_groups(data)
    drives = sorted(set(data["drive"].tolist()))
    per_drive: dict[str, dict[str, Any]] = {}
    consumed: set[tuple[str, str, int]] = set()
    for drive in drives:
        drive_rounds = [entry for entry in rounds
                        if entry["partition"] == "development" and entry["drive"] == drive]
        if not drive_rounds:
            raise RuntimeError(f"no development rounds for {drive}")
        mean = drive_rounds[0]["reference"].copy()
        belief_covariance = np.eye(2) * 0.25
        previous_reference = drive_rounds[0]["reference"].copy()
        previous_stamp = drive_rounds[0]["stamp"]
        errors, nees, areas, nis_values = [], [], [], []
        accepted_rounds = rejected_rounds = accepted_frames = rejected_frames = 0
        accepted_stamps: list[int] = []
        all_stamps = [entry["stamp"] for entry in drive_rounds]
        for entry in drive_rounds:
            stamp = entry["stamp"]
            dt = max((stamp - previous_stamp) / 1.0e9, 0.0)
            delta = entry["reference"] - previous_reference
            mean = mean + delta
            belief_covariance = spd(
                belief_covariance + np.eye(2) * process_noise ** 2 * dt
            )
            previous_reference = entry["reference"]
            previous_stamp = stamp
            selected = groups.get((drive, stamp), np.asarray([], dtype=int))
            if len(selected):
                identities = [(drive, str(data["camera"][index]), stamp) for index in selected]
                if any(identity in consumed for identity in identities):
                    raise RuntimeError("a physical camera frame was consumed twice")
                consumed.update(identities)
                if shared_block is None:
                    joint = independent_joint_covariance(world_covariance[selected])
                else:
                    joint = shared_joint_covariance(
                        covariance_ray[selected], data["basis"][selected], shared_block
                    )
                measurement, measurement_covariance = joint_measurement(
                    data["corrected"][selected], joint
                )
                innovation = measurement - mean
                innovation_covariance = spd(belief_covariance + measurement_covariance)
                nis = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
                nis_values.append(nis)
                accepted = mode == "forced" or nis <= nis_threshold
                if accepted:
                    mean, belief_covariance = joseph_update(
                        mean, belief_covariance, measurement, measurement_covariance
                    )
                    accepted_rounds += 1
                    accepted_frames += len(selected)
                    accepted_stamps.append(stamp)
                else:
                    rejected_rounds += 1
                    rejected_frames += len(selected)
            error = mean - entry["reference"]
            errors.append(float(np.linalg.norm(error)))
            nees.append(float(error @ np.linalg.solve(belief_covariance, error)))
            areas.append(float(
                math.pi * CHI2[0.95] * math.sqrt(np.linalg.det(belief_covariance)) * 1.0e4
            ))
        per_drive[drive] = summarize_drive(
            np.asarray(errors), np.asarray(nees), np.asarray(areas), np.asarray(nis_values),
            accepted_rounds, rejected_rounds, accepted_frames, rejected_frames,
            longest_gap(all_stamps, accepted_stamps),
        )
    expected_identities = {
        (str(drive), str(camera), int(stamp))
        for drive, camera, stamp in zip(data["drive"], data["camera"], data["stamp"], strict=True)
    }
    if consumed != expected_identities:
        raise RuntimeError("not every development camera frame was accounted exactly once")
    result = aggregate_drives(per_drive)
    result["mode"] = mode
    result["nis_threshold"] = nis_threshold if mode == "operational" else None
    result["physical_frames_accounted_once"] = len(consumed)
    return result


def replay_cascade_forced(data: dict[str, np.ndarray], rounds: list[dict[str, Any]],
                          covariance_ray: np.ndarray, process_noise: float) -> dict[str, Any]:
    world_covariance = covariance_analysis.world_covariances(data, covariance_ray)
    groups = observation_groups(data)
    per_drive: dict[str, dict[str, Any]] = {}
    for drive in sorted(set(data["drive"].tolist())):
        drive_rounds = [entry for entry in rounds
                        if entry["partition"] == "development" and entry["drive"] == drive]
        mean = drive_rounds[0]["reference"].copy()
        belief_covariance = np.eye(2) * 0.25
        tracks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        previous_reference = drive_rounds[0]["reference"].copy()
        previous_stamp = drive_rounds[0]["stamp"]
        errors, nees, areas, nis_values = [], [], [], []
        accepted_stamps: list[int] = []
        accepted_rounds = accepted_frames = 0
        for entry in drive_rounds:
            stamp = entry["stamp"]
            dt = max((stamp - previous_stamp) / 1.0e9, 0.0)
            delta = entry["reference"] - previous_reference
            process = np.eye(2) * process_noise ** 2 * dt
            mean = mean + delta
            belief_covariance = spd(belief_covariance + process)
            tracks = {
                camera: (track_mean + delta, spd(track_covariance + process))
                for camera, (track_mean, track_covariance) in tracks.items()
            }
            previous_reference = entry["reference"]
            previous_stamp = stamp
            selected = groups.get((drive, stamp), np.asarray([], dtype=int))
            if len(selected):
                pseudo_means, pseudo_covariances = [], []
                for index in selected:
                    camera = str(data["camera"][index])
                    if camera not in tracks:
                        tracks[camera] = (
                            data["corrected"][index].copy(), world_covariance[index].copy()
                        )
                    else:
                        tracks[camera] = joseph_update(
                            tracks[camera][0], tracks[camera][1],
                            data["corrected"][index], world_covariance[index]
                        )
                    pseudo_means.append(tracks[camera][0])
                    pseudo_covariances.append(tracks[camera][1])
                measurement, measurement_covariance = joint_measurement(
                    np.asarray(pseudo_means), independent_joint_covariance(np.asarray(pseudo_covariances))
                )
                innovation = measurement - mean
                innovation_covariance = spd(belief_covariance + measurement_covariance)
                nis_values.append(float(innovation @ np.linalg.solve(innovation_covariance, innovation)))
                mean, belief_covariance = joseph_update(
                    mean, belief_covariance, measurement, measurement_covariance
                )
                accepted_rounds += 1
                accepted_frames += len(selected)
                accepted_stamps.append(stamp)
            error = mean - entry["reference"]
            errors.append(float(np.linalg.norm(error)))
            nees.append(float(error @ np.linalg.solve(belief_covariance, error)))
            areas.append(float(
                math.pi * CHI2[0.95] * math.sqrt(np.linalg.det(belief_covariance)) * 1.0e4
            ))
        per_drive[drive] = summarize_drive(
            np.asarray(errors), np.asarray(nees), np.asarray(areas), np.asarray(nis_values),
            accepted_rounds, 0, accepted_frames, 0,
            longest_gap([entry["stamp"] for entry in drive_rounds], accepted_stamps),
        )
    result = aggregate_drives(per_drive)
    result["mode"] = "forced"
    result["architecture"] = "double_gaussian_cascade_negative_control"
    return result


def analytical_checks(data: dict[str, np.ndarray], predictions: dict[str, np.ndarray],
                      rounds: list[dict[str, Any]]) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    prior_mean = np.asarray([0.3, -0.4])
    prior_covariance = np.asarray([[0.20, 0.03], [0.03, 0.12]])
    measurement = np.asarray([0.7, -0.1])
    measurement_covariance = np.asarray([[0.08, -0.01], [-0.01, 0.05]])
    updated_mean, updated_covariance = joseph_update(
        prior_mean, prior_covariance, measurement, measurement_covariance
    )
    information_covariance = np.linalg.inv(
        np.linalg.inv(prior_covariance) + np.linalg.inv(measurement_covariance)
    )
    information_mean = information_covariance @ (
        np.linalg.solve(prior_covariance, prior_mean)
        + np.linalg.solve(measurement_covariance, measurement)
    )
    error = max(
        float(np.max(np.abs(updated_mean - information_mean))),
        float(np.max(np.abs(updated_covariance - information_covariance))),
    )
    checks["single_update_matches_information_form"] = {
        "passed": error < 1.0e-10, "maximum_absolute_error": error
    }

    observations = np.asarray([[0.5, -0.2], [0.8, -0.3], [0.6, 0.0]])
    matrices = np.asarray([
        [[0.08, 0.01], [0.01, 0.06]],
        [[0.12, -0.02], [-0.02, 0.09]],
        [[0.05, 0.00], [0.00, 0.10]],
    ])
    fused_mean, fused_covariance = joint_measurement(
        observations, independent_joint_covariance(matrices)
    )
    batch_mean, batch_covariance = joseph_update(
        prior_mean, prior_covariance, fused_mean, fused_covariance
    )
    sequential_mean, sequential_covariance = prior_mean.copy(), prior_covariance.copy()
    for local_mean, local_covariance in zip(observations, matrices, strict=True):
        sequential_mean, sequential_covariance = joseph_update(
            sequential_mean, sequential_covariance, local_mean, local_covariance
        )
    error = max(
        float(np.max(np.abs(batch_mean - sequential_mean))),
        float(np.max(np.abs(batch_covariance - sequential_covariance))),
    )
    checks["joint_update_matches_sequential_independent_updates"] = {
        "passed": error < 1.0e-10, "maximum_absolute_error": error
    }
    reordered_mean, reordered_covariance = joint_measurement(
        observations[::-1], independent_joint_covariance(matrices[::-1])
    )
    error = max(
        float(np.max(np.abs(fused_mean - reordered_mean))),
        float(np.max(np.abs(fused_covariance - reordered_covariance))),
    )
    checks["independent_fusion_is_order_invariant"] = {
        "passed": error < 1.0e-10, "maximum_absolute_error": error
    }

    angle = 0.73
    rotation = np.asarray([[math.cos(angle), -math.sin(angle)],
                           [math.sin(angle), math.cos(angle)]])
    source = np.asarray([[0.13, 0.04], [0.04, 0.07]])
    rotated = rotation @ source @ rotation.T
    restored = rotation.T @ rotated @ rotation
    error = float(np.max(np.abs(source - restored)))
    eigen_error = float(np.max(np.abs(np.linalg.eigvalsh(source) - np.linalg.eigvalsh(rotated))))
    checks["ray_to_world_rotation_preserves_covariance"] = {
        "passed": max(error, eigen_error) < 1.0e-10,
        "round_trip_error": error,
        "eigenvalue_error": eigen_error,
    }

    minimum_eigenvalue = min(
        float(np.min(np.linalg.eigvalsh(covariance)))
        for covariance in predictions.values()
    )
    checks["all_predicted_covariances_are_positive_definite"] = {
        "passed": minimum_eigenvalue > 0.0,
        "minimum_eigenvalue_m2": minimum_eigenvalue,
    }
    identities = [
        (str(drive), str(camera), int(stamp))
        for drive, camera, stamp in zip(data["drive"], data["camera"], data["stamp"], strict=True)
    ]
    round_keys = {(entry["drive"], entry["stamp"]) for entry in rounds
                  if entry["partition"] == "development"}
    supported = all((drive, stamp) in round_keys for drive, _camera, stamp in identities)
    checks["physical_frame_identity_is_unique_and_round_supported"] = {
        "passed": len(identities) == len(set(identities)) and supported,
        "frames": len(identities),
        "unique_frames": len(set(identities)),
        "all_frames_have_reference_round": supported,
    }
    if not all(value["passed"] for value in checks.values()):
        failed = [name for name, value in checks.items() if not value["passed"]]
        raise RuntimeError(f"analytical checks failed: {failed}")
    return checks


def write_summary_csv(path: Path, results: dict[str, dict[str, dict[str, Any]]]) -> None:
    fields = [
        "model", "mode", "equal_drive_rmse_cm", "equal_drive_p95_error_cm",
        "equal_drive_mean_nees", "belief_coverage_95", "mean_95_ellipse_area_cm2",
        "accepted_round_fraction", "accepted_frame_fraction", "mean_longest_gap_s",
        "worst_longest_gap_s", "mean_nis", "nis_coverage_99",
    ]
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model, modes in results.items():
            for mode, value in modes.items():
                writer.writerow({
                    "model": model,
                    "mode": mode,
                    "equal_drive_rmse_cm": 100.0 * value["equal_drive_rmse_m"],
                    "equal_drive_p95_error_cm": 100.0 * value["equal_drive_p95_error_m"],
                    "equal_drive_mean_nees": value["equal_drive_mean_nees"],
                    "belief_coverage_95": value["equal_drive_belief_coverage"]["0.95"],
                    "mean_95_ellipse_area_cm2": value["equal_drive_mean_95_ellipse_area_cm2"],
                    "accepted_round_fraction": value["equal_drive_accepted_round_fraction"],
                    "accepted_frame_fraction": value["equal_drive_accepted_frame_fraction"],
                    "mean_longest_gap_s": value["equal_drive_longest_gap_s"],
                    "worst_longest_gap_s": value["worst_drive_longest_gap_s"],
                    "mean_nis": value["equal_drive_mean_nis"],
                    "nis_coverage_99": value["equal_drive_nis_coverage"]["0.99"],
                })


def write_per_drive_csv(path: Path, results: dict[str, dict[str, dict[str, Any]]]) -> None:
    fields = [
        "model", "mode", "drive", "rmse_cm", "p95_error_cm", "mean_nees",
        "belief_coverage_95", "mean_95_ellipse_area_cm2", "accepted_round_fraction",
        "accepted_frame_fraction", "longest_gap_s", "mean_nis",
    ]
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model, modes in results.items():
            for mode, aggregate in modes.items():
                for drive, value in aggregate["by_drive"].items():
                    writer.writerow({
                        "model": model,
                        "mode": mode,
                        "drive": drive,
                        "rmse_cm": 100.0 * value["rmse_m"],
                        "p95_error_cm": 100.0 * value["p95_error_m"],
                        "mean_nees": value["mean_nees"],
                        "belief_coverage_95": value["belief_coverage"]["0.95"],
                        "mean_95_ellipse_area_cm2": value["mean_95_ellipse_area_cm2"],
                        "accepted_round_fraction": value["accepted_round_fraction"],
                        "accepted_frame_fraction": value["accepted_frame_fraction"],
                        "longest_gap_s": value["longest_without_accepted_correction_s"],
                        "mean_nis": value["mean_nis"],
                    })


def page_title(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.055, 0.955, title, fontsize=20, fontweight="bold", va="top")
    fig.text(0.055, 0.913, subtitle, fontsize=9.5, color="#555555", va="top")


def add_footer(fig: plt.Figure, page: int) -> None:
    fig.text(0.055, 0.025, "Matched belief replay | exploratory | audit sealed", fontsize=8, color="#777777")
    fig.text(0.945, 0.025, str(page), fontsize=8, color="#777777", ha="right")


def table(axis: plt.Axes, columns: list[str], rows: list[list[str]], widths: list[float]) -> None:
    axis.axis("off")
    item = axis.table(cellText=rows, colLabels=columns, cellLoc="left", colLoc="left",
                      colWidths=widths, loc="center")
    item.auto_set_font_size(False)
    item.set_fontsize(8.2)
    item.scale(1.0, 1.45)
    for (row, _column), cell in item.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor("#eaf1f8")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f7f7f7")


def report_table_rows(names: list[str], results: dict[str, dict[str, dict[str, Any]]],
                      mode: str) -> list[list[str]]:
    rows = []
    for name in names:
        value = results[name][mode]
        rows.append([
            DISPLAY[name],
            f"{100 * value['equal_drive_rmse_m']:.2f}",
            f"{100 * value['equal_drive_p95_error_m']:.2f}",
            f"{value['equal_drive_mean_nees']:.2f}",
            f"{100 * value['equal_drive_belief_coverage']['0.95']:.1f}%",
            f"{value['equal_drive_mean_95_ellipse_area_cm2']:.1f}",
        ])
    return rows


def generate_report(path: Path, payload: dict[str, Any]) -> None:
    results = payload["belief_replay"]
    names = list(covariance_analysis.BUILDERS)
    shortlist = list(dict.fromkeys([
        "global_constant", "geometric_range_angle", payload["best_measurement_model"]
    ]))
    page = 0
    with PdfPages(path) as pdf:
        page += 1
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        page_title(fig, "Does each R produce an honest belief?", "Matched replay on six development drives")
        fig.text(0.06, 0.80, "Controlled comparison", fontsize=12, fontweight="bold", color="#315b7d")
        fig.text(0.06, 0.745,
                 "Every R receives the same corrected observations, timestamps, initial belief, reference motion increments, and fixed Q.\n"
                 "The forced replay isolates uncertainty propagation. The operational replay adds the same 2-D NIS gate.",
                 fontsize=12, linespacing=1.55, va="top")
        fig.text(0.06, 0.57, "Interpretation", fontsize=12, fontweight="bold", color="#315b7d")
        fig.text(0.06, 0.515,
                 "Low RMSE is accuracy. Mean NEES near 2 and nominal ellipse containment are consistency.\n"
                 "Ellipse area is sharpness. A method must be judged on all three.",
                 fontsize=12, linespacing=1.55, va="top")
        checks = payload["analytical_checks"]
        fig.text(0.06, 0.36, "Implementation checks", fontsize=12, fontweight="bold", color="#315b7d")
        fig.text(0.06, 0.305,
                 f"{sum(value['passed'] for value in checks.values())}/{len(checks)} checks passed\n"
                 f"{payload['metadata']['development_rows']:,} physical frames accounted exactly once in every direct replay\n"
                 "Audit drives were not opened",
                 fontsize=12, linespacing=1.55, va="top")
        fig.text(0.06, 0.13,
                 "Boundary: reference increments make this a controlled sensor and belief test. It is not an end-to-end navigation result.",
                 fontsize=9.5, color="#a04431")
        add_footer(fig, page)
        pdf.savefig(fig)
        plt.close(fig)

        for mode, title_text, subtitle in (
            ("forced", "Forced assimilation", "Every admitted measurement batch is fused, so all R methods receive identical updates"),
            ("operational", "Operational NIS-gated replay", f"The same fixed NIS threshold of {payload['configuration']['nis_threshold']:.2f} is applied to every R"),
        ):
            page += 1
            fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
            page_title(fig, title_text, subtitle)
            axis = fig.add_axes([0.045, 0.14, 0.91, 0.68])
            table(axis, ["R model", "RMSE cm", "P95 cm", "Mean NEES", "95% belief", "Area cm2"],
                  report_table_rows(names, results, mode),
                  [0.30, 0.12, 0.12, 0.13, 0.14, 0.14])
            fig.text(0.055, 0.085,
                     "Reference values: mean planar NEES 2.0 and 95% belief containment 95%. Smaller area is sharper only when coverage remains honest.",
                     fontsize=9, color="#555555")
            add_footer(fig, page)
            pdf.savefig(fig)
            plt.close(fig)

        page += 1
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.subplots_adjust(top=0.82, bottom=0.14, left=0.08, right=0.96, wspace=0.30)
        page_title(fig, "Accuracy and consistency must agree", "Point size is mean 95% belief-ellipse area")
        for axis, mode in zip(axes, ("forced", "operational"), strict=True):
            areas = np.asarray([results[name][mode]["equal_drive_mean_95_ellipse_area_cm2"] for name in names])
            sizes = 45.0 + 160.0 * (areas - areas.min()) / max(float(np.ptp(areas)), 1.0e-9)
            for index, name in enumerate(names):
                value = results[name][mode]
                marker = "*" if name == payload["best_measurement_model"] else "o"
                axis.scatter(100 * value["equal_drive_rmse_m"], value["equal_drive_mean_nees"],
                             s=sizes[index] * (1.35 if marker == "*" else 1.0), marker=marker,
                             color=COLORS[name], label=DISPLAY[name])
            axis.axhline(2.0, color="#222222", linestyle="--", linewidth=1)
            axis.set_title(mode.title())
            axis.set_xlabel("Equal-drive belief RMSE [cm]")
            axis.set_ylabel("Equal-drive mean NEES")
            axis.set_yscale("log")
            axis.grid(alpha=0.2)
        axes[1].legend(fontsize=7, loc="best")
        add_footer(fig, page)
        pdf.savefig(fig)
        plt.close(fig)

        page += 1
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.subplots_adjust(top=0.82, bottom=0.14, left=0.08, right=0.96, wspace=0.30)
        page_title(fig, "Belief calibration", "Observed development containment for the paper shortlist")
        levels = list(CHI2)
        for axis, mode in zip(axes, ("forced", "operational"), strict=True):
            axis.plot(levels, levels, "--", color="#222222", label="Ideal")
            for name in shortlist:
                coverage = results[name][mode]["equal_drive_belief_coverage"]
                axis.plot(levels, [coverage[str(level)] for level in levels], marker="o",
                          color=COLORS[name], label=DISPLAY[name])
            axis.set_title(mode.title())
            axis.set(xlabel="Nominal probability", ylabel="Observed containment",
                     xlim=(0.47, 1.0), ylim=(0.47, 1.0))
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        add_footer(fig, page)
        pdf.savefig(fig)
        plt.close(fig)

        page += 1
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.subplots_adjust(top=0.82, bottom=0.18, left=0.20, right=0.96, wspace=0.18)
        page_title(fig, "What the NIS gate changes", "Operational replay only")
        accepted = [100 * results[name]["operational"]["equal_drive_accepted_frame_fraction"] for name in names]
        gaps = [results[name]["operational"]["worst_drive_longest_gap_s"] for name in names]
        axes[0].barh(range(len(names)), accepted, color=[COLORS[name] for name in names])
        axes[0].set_yticks(range(len(names)), [DISPLAY[name] for name in names], fontsize=8)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Accepted physical frames [%]")
        axes[0].set_title("Acceptance")
        axes[0].set_xlim(0, 100)
        axes[0].grid(axis="x", alpha=0.2)
        axes[1].barh(range(len(names)), gaps, color=[COLORS[name] for name in names])
        axes[1].set_yticks(range(len(names)), [""] * len(names))
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Worst drive gap without an accepted correction [s]")
        axes[1].set_title("Correction gap")
        axes[1].grid(axis="x", alpha=0.2)
        fig.text(0.09, 0.085,
                 "If forced replay is consistent but operational replay is not, the difference comes from NIS gating and the resulting correction gaps.",
                 fontsize=9.5)
        add_footer(fig, page)
        pdf.savefig(fig)
        plt.close(fig)

        page += 1
        fig, axes = plt.subplots(1, 3, figsize=(11.69, 8.27))
        fig.subplots_adjust(top=0.80, bottom=0.17, left=0.07, right=0.96, wspace=0.34)
        page_title(fig, "Architecture diagnostic with the selected R", "Forced assimilation keeps the measurement set fixed")
        architecture = payload["architecture_diagnostic"]
        labels = ["Direct\nindependent", "Direct\nshared block", "Double Gaussian\ncascade"]
        keys = ["direct_independent", "direct_shared", "double_cascade"]
        values = [architecture[key] for key in keys]
        axes[0].bar(labels, [100 * value["equal_drive_rmse_m"] for value in values],
                    color=["#4c78a8", "#b279a2", "#e45756"])
        axes[0].set_ylabel("Belief RMSE [cm]")
        axes[1].bar(labels, [value["equal_drive_mean_nees"] for value in values],
                    color=["#4c78a8", "#b279a2", "#e45756"])
        axes[1].axhline(2.0, color="#222222", linestyle="--")
        axes[1].set_ylabel("Mean NEES")
        axes[1].set_yscale("log")
        axes[2].bar(labels, [100 * value["equal_drive_belief_coverage"]["0.95"] for value in values],
                    color=["#4c78a8", "#b279a2", "#e45756"])
        axes[2].axhline(95.0, color="#222222", linestyle="--")
        axes[2].set_ylabel("95% belief containment [%]")
        for axis in axes:
            axis.grid(axis="y", alpha=0.2)
            axis.tick_params(axis="x", labelsize=7)
        fig.text(0.07, 0.075,
                 "The cascade is a negative control. Lower RMSE does not validate it if NEES and containment reveal repeated or overweighted information.",
                 fontsize=9.5)
        add_footer(fig, page)
        pdf.savefig(fig)
        plt.close(fig)

        page += 1
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        page_title(fig, "Decision for the methodology", "Use belief replay to validate R, not to select R after seeing downstream accuracy")
        best = payload["best_measurement_model"]
        forced = results[best]["forced"]
        operational = results[best]["operational"]
        fig.text(0.06, 0.79, "Selected measurement model", fontsize=12, fontweight="bold", color="#315b7d")
        fig.text(0.06, 0.735,
                 f"{DISPLAY[best]} was selected by measurement-level equal-drive development NLL.",
                 fontsize=13)
        fig.text(0.06, 0.63, "Belief validation", fontsize=12, fontweight="bold", color="#315b7d")
        fig.text(0.06, 0.575,
                 f"Forced: RMSE {100 * forced['equal_drive_rmse_m']:.2f} cm, mean NEES {forced['equal_drive_mean_nees']:.2f}, "
                 f"95% containment {100 * forced['equal_drive_belief_coverage']['0.95']:.1f}%\n"
                 f"Operational: RMSE {100 * operational['equal_drive_rmse_m']:.2f} cm, mean NEES {operational['equal_drive_mean_nees']:.2f}, "
                 f"95% containment {100 * operational['equal_drive_belief_coverage']['0.95']:.1f}%",
                 fontsize=12, linespacing=1.55, va="top")
        fig.text(0.06, 0.41, "How to read this", fontsize=12, fontweight="bold", color="#315b7d")
        fig.text(0.06, 0.355,
                 "If the forced result is inconsistent, changing the gate cannot repair the underlying uncertainty model.\n"
                 "If only the operational result changes materially, investigate NIS rejection and correction gaps.\n"
                 "If all R methods fail similarly, inspect Q, correlation, and the initial belief before changing R.",
                 fontsize=11.5, linespacing=1.55, va="top")
        fig.text(0.06, 0.16, "Evidence boundary", fontsize=12, fontweight="bold", color="#a04431")
        fig.text(0.06, 0.105,
                 "These are development-split diagnostics. Freeze the method before opening the audit. Do not report these values as final thesis evidence.",
                 fontsize=10.5)
        add_footer(fig, page)
        pdf.savefig(fig)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--process-noise", type=float, default=0.02,
                        help="fixed planar process noise in metres per square-root second")
    parser.add_argument("--nis-threshold", type=float, default=CHI2[0.99])
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    data, metadata = covariance_analysis.load_data(
        args.campaign_root.resolve(), args.prediction_root.resolve()
    )
    rounds = load_rounds(args.campaign_root.resolve())
    fit_index = np.flatnonzero(data["partition"] == "fit")
    development_index = np.flatnonzero(data["partition"] == "development")
    fit = covariance_analysis.subset(data, fit_index)
    development = covariance_analysis.subset(data, development_index)
    static, predictions, models = covariance_analysis.fit_static_models(fit, development)
    learned = [name for name in covariance_analysis.BUILDERS
               if name not in {"global_constant", "geometric_range_angle"}]
    best = min(learned, key=lambda name: static[name]["development"]["equal_drive_nll"])
    best_fit_covariance = models[best].predict(fit) * static[best]["fit_oof_calibration_scale"]
    correlation = covariance_analysis.fit_shared_correlation(fit, best_fit_covariance)
    shared_block = np.asarray(correlation["shared_block"])

    checks = analytical_checks(development, predictions, rounds)
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for index, name in enumerate(covariance_analysis.BUILDERS, start=1):
        print(f"[{index}/{len(covariance_analysis.BUILDERS)}] replaying {name}", flush=True)
        results[name] = {
            mode: replay_direct(
                development, rounds, predictions[name], mode,
                args.process_noise, args.nis_threshold,
            )
            for mode in ("forced", "operational")
        }

    architecture = {
        "direct_independent": results[best]["forced"],
        "direct_shared": replay_direct(
            development, rounds, predictions[best], "forced",
            args.process_noise, args.nis_threshold, shared_block=shared_block,
        ),
        "double_cascade": replay_cascade_forced(
            development, rounds, predictions[best], args.process_noise
        ),
    }
    payload = {
        "schema": "commissioned_belief_handling_supervisor.v1",
        "status": "complete_exploratory_development_split",
        "paper_facing_evidence": False,
        "audit_opened": False,
        "metadata": metadata,
        "configuration": {
            "process_noise_m_per_sqrt_s": args.process_noise,
            "nis_threshold": args.nis_threshold,
            "initial_position_covariance_m2": [[0.25, 0.0], [0.0, 0.25]],
            "motion_mean": "reference increments shared by every arm",
            "belief_layer": "2-D position at network-round timestamps",
            "aggregation": "within complete drive first, then equal weight across six drives",
        },
        "analytical_checks": checks,
        "measurement_level_results": static,
        "best_measurement_model": best,
        "belief_replay": results,
        "cross_camera_fit": correlation,
        "architecture_diagnostic": architecture,
    }
    write_summary_csv(output / "belief_replay_summary.csv", results)
    write_per_drive_csv(output / "belief_replay_by_drive.csv", results)
    atomic_json(output / "belief_handling_comparison.json", payload)
    report_path = output / "belief_handling_supervisor_report.pdf"
    generate_report(report_path, payload)
    completion = {
        "schema": "commissioned_belief_handling_completion.v1",
        "status": "complete_exploratory_development_split",
        "audit_opened": False,
        "best_measurement_model": best,
        "all_analytical_checks_passed": all(value["passed"] for value in checks.values()),
        "artifacts": {name: sha256(output / name) for name in (
            "belief_replay_summary.csv", "belief_replay_by_drive.csv",
            "belief_handling_comparison.json", "belief_handling_supervisor_report.pdf",
        )},
        "source_sha256": sha256(Path(__file__)),
    }
    atomic_json(output / "completion.json", completion)
    print(json.dumps({
        "best_measurement_model": best,
        "selected_forced": results[best]["forced"],
        "selected_operational": results[best]["operational"],
        "analytical_checks_passed": len(checks),
        "report": str(report_path),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
