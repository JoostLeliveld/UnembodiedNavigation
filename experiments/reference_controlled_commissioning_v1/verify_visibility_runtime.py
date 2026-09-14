#!/usr/bin/env python3
"""Verify offline/runtime visibility-residual parity and CPU timing before audit access."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src/reliability"), str(REPO / "src/unav_common")]

from analyze_commissioned_model import load_capture  # noqa: E402
from reliability.commissioned_visibility import CommissionedVisibilitySensorModel  # noqa: E402
from visibility_patch_rgb import visibility_grid_from_saved_context as training_grid  # noqa: E402
from unav_common.visibility_patch import visibility_grid_from_saved_context as runtime_grid  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--visibility-root", required=True, type=Path)
    parser.add_argument("--runtime-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    execution, records, rounds = load_capture(args.capture_root.resolve())
    if execution.get("audit_analysis_permitted") is not False:
        raise RuntimeError("audit must remain sealed during runtime verification")
    prediction_path = args.visibility_root.resolve() / "candidate_predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as archive:
        partition = np.asarray(archive["partition"])
        expected = np.asarray(archive["candidate_prediction_ray_m"], dtype=float)
    if len(records) != len(expected):
        raise RuntimeError("prediction and capture row counts differ")
    heading = {(row["drive"], row["stamp_ns"]): row["reference_yaw"] for row in rounds}
    model = CommissionedVisibilitySensorModel(args.runtime_model.resolve())
    selected = np.flatnonzero(partition == "development")
    differences = []
    covariance_eigenvalues = []
    timings_ms = []
    grid_difference = []
    for count, index in enumerate(selected):
        record = records[index]
        with np.load(record["crop_path"], allow_pickle=False) as archive:
            crop = np.asarray(archive["crop"])
            image_shape = np.asarray(archive["image_shape"])
        grid_training = training_grid(crop, record["bbox_xyxy"], image_shape)
        grid_runtime = runtime_grid(crop, record["bbox_xyxy"], image_shape)
        grid_difference.append(float(np.max(np.abs(grid_training - grid_runtime))))
        start = time.perf_counter()
        correction = model.correction_ray(
            record["camera"], record["raw"], record["bbox_xyxy"],
            record["confidence"], grid_runtime,
        )
        if count < 1000:
            _, covariance = model.correct_and_covariance(
                record["camera"], record["raw"], record["bbox_xyxy"],
                record["confidence"], grid_runtime,
                heading[(record["drive"], record["stamp_ns"])],
            )
            timings_ms.append((time.perf_counter() - start) * 1000.0)
            covariance_eigenvalues.extend(np.linalg.eigvalsh(np.asarray(covariance)).tolist())
        differences.append(float(np.max(np.abs(correction - expected[index]))))
    report = {
        "schema": "commissioned_visibility_runtime_verification.v1",
        "status": "passed",
        "audit_accessed": False,
        "development_rows": int(len(selected)),
        "maximum_grid_absolute_difference": max(grid_difference),
        "maximum_correction_absolute_difference_m": max(differences),
        "minimum_runtime_covariance_eigenvalue_m2": min(covariance_eigenvalues),
        "single_measurement_cpu_timing_ms": {
            "n": len(timings_ms),
            "median": float(np.median(timings_ms)),
            "p95": float(np.percentile(timings_ms, 95)),
            "maximum": max(timings_ms),
        },
        "passed_checks": {
            "grid_exact": max(grid_difference) == 0.0,
            "correction_within_1e-5_m": max(differences) <= 1e-5,
            "covariance_positive_definite": min(covariance_eigenvalues) > 0.0,
        },
        "source_hashes": {
            "runtime_model": sha256(args.runtime_model.resolve()),
            "candidate_predictions": sha256(prediction_path),
            "verifier": sha256(Path(__file__)),
        },
    }
    if not all(report["passed_checks"].values()):
        report["status"] = "failed"
    atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
