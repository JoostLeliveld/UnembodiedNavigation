#!/usr/bin/env python3
"""Export in-sample residuals of the frozen 12-drive correction fit.

These residuals are for provisional matched-R fitting.  They are deliberately
labelled in-sample and must not be described as whole-drive out-of-fold evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


MODEL_ID = "box_mlp_visibility_residual"
PREDICTION_KEY = f"{MODEL_ID}_prediction_ray_m"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(residual: np.ndarray) -> dict:
    error = np.linalg.norm(residual, axis=1)
    return {
        "n": int(len(error)),
        "signed_bias_ray_m": residual.mean(axis=0).tolist(),
        "sample_covariance_ray_m2": np.cov(residual, rowvar=False, ddof=1).tolist(),
        "mean_error_m": float(error.mean()),
        "median_error_m": float(np.median(error)),
        "rmse_m": float(np.sqrt(np.mean(error ** 2))),
        "p90_error_m": float(np.quantile(error, 0.90)),
        "p95_error_m": float(np.quantile(error, 0.95)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.comparison_root.resolve()
    output = args.output.resolve()
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists():
        raise FileExistsError(output if output.exists() else staging)
    staging.mkdir(parents=True)

    report_path = root / "correction_comparison.json"
    prediction_path = root / "candidate_predictions.npz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "held_out_correction_comparison_complete":
        raise RuntimeError("correction comparison is not complete")
    if report.get("selected_correction") != MODEL_ID:
        raise RuntimeError("requested residual model is not the selected correction")
    expected = report["artifacts"]["predictions"]["sha256"]
    if sha256(prediction_path) != expected:
        raise RuntimeError("candidate-prediction hash mismatch")

    with np.load(prediction_path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    partition = data["historical_partition"]
    index = np.flatnonzero(np.isin(partition, ["fit", "development"]))
    drives = data["drive"][index]
    if len(index) != 13813 or len(set(drives.tolist())) != 12:
        raise RuntimeError("expected the frozen 13,813-row/12-drive training population")
    if np.any(partition[index] == "audit"):
        raise RuntimeError("audit row leaked into training-residual export")

    target = data["target_ray_m"][index].astype(np.float64)
    prediction = data[PREDICTION_KEY][index].astype(np.float64)
    residual = target - prediction
    if not np.isfinite(residual).all():
        raise RuntimeError("non-finite residual")
    cameras = data["camera"][index]

    residual_path = staging / "train12_in_sample_residuals.npz"
    np.savez_compressed(
        residual_path,
        model_id=np.asarray([MODEL_ID]),
        residual_kind=np.asarray(["in_sample_frozen_train12"]),
        drive=drives,
        historical_partition=partition[index],
        camera=cameras,
        source_frame_id=data["source_frame_id"][index],
        stamp_ns=data["stamp_ns"][index],
        target_ray_m=target,
        prediction_ray_m=prediction,
        residual_ray_m= residual,
    )
    csv_path = staging / "train12_in_sample_residuals.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["drive_id", "historical_partition", "camera_id",
                         "source_frame_id", "capture_stamp_ns",
                         "target_along_m", "target_cross_m",
                         "prediction_along_m", "prediction_cross_m",
                         "residual_along_m", "residual_cross_m"])
        for i in range(len(index)):
            writer.writerow([drives[i], partition[index][i], cameras[i],
                             data["source_frame_id"][index][i], int(data["stamp_ns"][index][i]),
                             *target[i].tolist(), *prediction[i].tolist(), *residual[i].tolist()])

    manifest = {
        "schema": "selected_correction_train_residuals.v1",
        "status": "complete_provisional_in_sample_residuals",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "residual_definition": "target_ray_m - prediction_ray_m",
        "population": {"rows": int(len(index)), "complete_drives": 12,
                       "historical_partitions": ["fit", "development"],
                       "audit_rows": 0},
        "evidence_warning": (
            "The selected correction was fitted on these same 12 drives. These are "
            "in-sample residuals, not whole-drive out-of-fold residuals; covariance "
            "calibration inferred from them may be optimistic."
        ),
        "source": {"comparison_report": str(report_path),
                   "comparison_report_sha256": sha256(report_path),
                   "candidate_predictions": str(prediction_path),
                   "candidate_predictions_sha256": sha256(prediction_path)},
        "summary": summary(residual),
        "by_camera": {camera: summary(residual[cameras == camera])
                      for camera in sorted(set(cameras.tolist()))},
        "by_drive": {drive: summary(residual[drives == drive])
                     for drive in sorted(set(drives.tolist()))},
        "artifacts": {
            "npz": {"path": residual_path.name, "sha256": sha256(residual_path)},
            "csv": {"path": csv_path.name, "sha256": sha256(csv_path)},
        },
        "implementation_sha256": sha256(Path(__file__)),
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(staging, output)
    print(json.dumps({"status": manifest["status"], "population": manifest["population"],
                      "summary": manifest["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
