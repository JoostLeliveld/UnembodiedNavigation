#!/usr/bin/env python3
"""Package the three provisional perception-uncertainty methods for runtime.

This packages, but does not turn, the current all-drive diagnostics into
independent evaluation evidence. Navigation drives remain the independent test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(target: Path, parent: Path) -> str:
    return str(target.resolve().relative_to(parent.resolve())) if target.resolve().is_relative_to(parent.resolve()) else str(target.resolve())


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.models.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    prediction_path = source / "training_predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as archive:
        target = np.asarray(archive["target_ray"], dtype=np.float32)
        box_mean = np.asarray(archive["box_spatial_mean"], dtype=np.float32)
        residual = target - box_mean
        residual_path = output / "box_mlp_residual_population.npz"
        np.savez_compressed(
            residual_path,
            raw_world=np.asarray(archive["raw_world"], dtype=np.float32),
            residual_ray=residual,
            camera=np.asarray(archive["camera"]),
            drive=np.asarray(archive["drive"]),
        )

    box_checkpoint = source / "box_spatial_mlp_deployment.pt"
    rgb_checkpoint = source / "rgb_gaussian_deployment.pt"
    common = {
        "schema": "commissioned_perception_runtime_model.v1",
        "status": "provisional_all_drive_fit_navigation_evaluation_pending",
        "runtime_output": "corrected world XY and full 2x2 observation covariance",
        "forbidden_inputs": [
            "belief", "robot EKF", "ground truth", "map visibility", "line of sight",
            "obstacle height", "robot hull",
        ],
    }
    packages = {}
    for method, checkpoint in (
        ("hierarchical_residual", box_checkpoint),
        ("spatial_residual", box_checkpoint),
        ("joint_rgb_gaussian", rgb_checkpoint),
    ):
        package = dict(common)
        package.update({
            "method_id": method,
            "mean_checkpoint": relative(checkpoint, output),
            "mean_checkpoint_sha256": sha256(checkpoint),
        })
        if method != "joint_rgb_gaussian":
            package.update({
                "residual_artifact": residual_path.name,
                "residual_artifact_sha256": sha256(residual_path),
                "residual_source": "all-drive box-MLP training residuals",
                "neighbors": 120,
                "bandwidth_m": 0.75,
                "drive_weighting": "each drive capped before local aggregation",
            })
        path = output / f"{method}.json"
        write_json(path, package)
        packages[path.name] = sha256(path)

    write_json(output / "manifest.json", {
        "schema": "artifact_manifest.v1",
        "status": "complete_provisional_navigation_pilot_ready",
        "source_predictions": str(prediction_path),
        "source_predictions_sha256": sha256(prediction_path),
        "artifacts": {
            residual_path.name: sha256(residual_path),
            **packages,
        },
        "scientific_boundary": (
            "The residual population and deployed means were fit on all completed "
            "commissioning drives. Only independent navigation drives may be used as "
            "evaluation evidence."
        ),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
