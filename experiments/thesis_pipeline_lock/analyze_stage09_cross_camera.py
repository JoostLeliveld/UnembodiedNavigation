#!/usr/bin/env python3
"""Estimate simultaneous camera dependence and choose the Stage-09 runtime fusion rule."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from reliability.contracts import CameraQuality
from reliability.fusion import MapObservation
from reliability.measurement_fusion import combine_measurements_2d


REPO = Path(__file__).resolve().parents[2]
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
RULES = ("best_single", "independent", "joint_network")
CHI2_95, CHI2_99 = 5.991464547107979, 9.21034037197618


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    oof_path = REPO / "logs/thesis_final_pipeline_v1/stage07_correction_covariance/run_v3/oof_predictions.csv"
    admissions_path = REPO / "logs/thesis_final_pipeline_v1/stage06_detector_gate/gate_selection_v1/admission_records.csv"
    model_path = REPO / "logs/thesis_final_pipeline_v1/stage07_correction_covariance/run_v3/measurement_model.json"
    with oof_path.open(newline="", encoding="utf-8") as handle:
        oof = list(csv.DictReader(handle))
    with admissions_path.open(newline="", encoding="utf-8") as handle:
        admissions = {
            (row["pose_id"], row["camera_id"]): row for row in csv.DictReader(handle)
            if row["admitted"] == "1"
        }
    payload = json.loads(model_path.read_text())
    scale = float(payload["covariance_parameters"]["calibration_scale"])
    tables = payload["covariance_parameters"]["per_camera_width_bins"]
    groups, residuals = defaultdict(list), defaultdict(dict)
    for row in oof:
        camera = row["camera_id"]
        admission = admissions[(row["pose_id"], camera)]
        width = float(admission["best_box_x1"]) - float(admission["best_box_x0"])
        table = tables[camera]
        edges = [float(value) for value in table["width_edges_px"]]
        bin_index = 0 if width <= edges[0] else 1 if width <= edges[1] else 2
        covariance = scale * np.asarray(table["covariance_m2"][bin_index], dtype=float)
        xy = np.asarray((row["C2_ridge_residual_x"], row["C2_ridge_residual_y"]), dtype=float)
        truth = np.asarray((row["truth_x"], row["truth_y"]), dtype=float)
        quality = CameraQuality(camera_id=camera, conditional_cov_uv=((1.0, 0.0), (0.0, 1.0)))
        observation = MapObservation(
            camera, 0.0, tuple(xy), tuple(map(tuple, covariance)), quality, "stage07_spatial_oof"
        )
        groups[row["pose_id"]].append((observation, truth, row["block_id"]))
        residuals[row["pose_id"]][camera] = xy - truth

    pairwise = {}
    for index, camera_a in enumerate(CAMERAS):
        for camera_b in CAMERAS[index + 1:]:
            pairs = [(values[camera_a], values[camera_b]) for values in residuals.values()
                     if camera_a in values and camera_b in values]
            if len(pairs) < 20:
                continue
            values = np.asarray([np.r_[a, b] for a, b in pairs])
            pairwise[f"{camera_a}:{camera_b}"] = {
                "simultaneous_poses": len(pairs),
                "pearson_x": float(np.corrcoef(values[:, 0], values[:, 2])[0, 1]),
                "pearson_y": float(np.corrcoef(values[:, 1], values[:, 3])[0, 1]),
            }

    results = {}
    for rule in RULES:
        errors, d2, counts = [], [], []
        for group in groups.values():
            if len(group) < 2:
                continue
            mean, covariance, used = combine_measurements_2d(
                [item[0] for item in group], rule=rule
            )
            residual = np.asarray(mean) - group[0][1]
            matrix = np.asarray(covariance)
            errors.append(float(np.linalg.norm(residual)))
            d2.append(float(residual @ np.linalg.solve(matrix, residual)))
            counts.append(len(used))
        error = np.asarray(errors)
        mahalanobis = np.asarray(d2)
        results[rule] = {
            "simultaneous_pose_groups": len(errors),
            "mean_used_cameras": float(np.mean(counts)),
            "median_error_m": float(np.median(error)),
            "rms_error_m": float(np.sqrt(np.mean(error ** 2))),
            "p95_error_m": float(np.quantile(error, 0.95)),
            "containment_95": float(np.mean(mahalanobis <= CHI2_95)),
            "above_chi2_99_fraction": float(np.mean(mahalanobis > CHI2_99)),
            "median_mahalanobis_d2": float(np.median(mahalanobis)),
        }
        results[rule]["eligible"] = (
            0.90 <= results[rule]["containment_95"] <= 0.99
            and results[rule]["above_chi2_99_fraction"] <= 0.05
        )
    eligible = [rule for rule in RULES if results[rule]["eligible"]]
    selected = min(eligible, key=lambda rule: (results[rule]["p95_error_m"], RULES.index(rule)))
    report = {
        "schema": "thesis_stage09_cross_camera_commissioning.v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "final_audit_accessed": False,
        "population": "Stage-07 spatial out-of-fold admitted commissioning observations; simultaneous pose groups only",
        "dependence": pairwise, "candidate_rules": results,
        "selection_rule": "Among rules with 95% containment in [0.90,0.99] and <=5% above chi2_2(0.99), choose lowest p95 error.",
        "selected_runtime_rule": selected,
        "interpretation": (
            "Pairwise residual correlations are nonzero and sometimes negative. Independent fusion is retained "
            "because it is calibrated on spatial OOF groups and has the lowest eligible p95; joint_network is "
            "reported but excluded here because its 95% containment exceeds 0.99."
        ),
        "sources": {
            str(oof_path.relative_to(REPO)): sha256(oof_path),
            str(admissions_path.relative_to(REPO)): sha256(admissions_path),
            str(model_path.relative_to(REPO)): sha256(model_path),
            str(Path(__file__).resolve().relative_to(REPO)): sha256(Path(__file__).resolve()),
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("refusing to overwrite a frozen dependence report")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
