#!/usr/bin/env python3
"""Package the selected train-12 visibility correction with four matched R methods."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src/reliability"), str(REPO / "src/unav_common")]
from reliability.commissioned_perception import CAMERAS  # noqa: E402
from reliability.projection import camera_model_from_world  # noqa: E402


METHODS = {
    "global_residual": "R0_global_full",
    "per_camera_residual": "R1_per_camera_full",
    "spatial_residual": "R2_spatial_residual",
    "hierarchical_residual": "R3_hierarchical_predictive",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_entry(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--residual-root", type=Path, required=True)
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists():
        raise FileExistsError(output if output.exists() else staging)
    staging.mkdir(parents=True)

    comparison = args.comparison_root.resolve()
    report_path = comparison / "correction_comparison.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("selected_correction") != "box_mlp_visibility_residual":
        raise RuntimeError("visibility-residual correction is not selected")
    base = comparison / "box_mlp_train12.joblib"
    patch = comparison / "box_mlp_visibility_residual_train12.pt"

    residual_source = args.residual_root.resolve() / "train12_in_sample_residuals.npz"
    dataset_path = args.dataset_contract.resolve()
    with np.load(residual_source, allow_pickle=False) as residual_archive:
        residual = np.asarray(residual_archive["residual_ray_m"], dtype=np.float32)
        drive = np.asarray(residual_archive["drive"])
        camera = np.asarray(residual_archive["camera"])
        frames = np.asarray(residual_archive["source_frame_id"])
        stamps = np.asarray(residual_archive["stamp_ns"], dtype=np.int64)
    with np.load(dataset_path, allow_pickle=False) as dataset:
        lookup = {
            (str(d), str(c), str(f), int(t)): index
            for index, (d, c, f, t) in enumerate(zip(
                dataset["drive"], dataset["camera"], dataset["source_frame_id"],
                dataset["stamp_ns"],
            ))
        }
        index = np.asarray([
            lookup[(str(d), str(c), str(f), int(t))]
            for d, c, f, t in zip(drive, camera, frames, stamps)
        ])
        raw_world = np.asarray(dataset["raw"][index], dtype=np.float32)
    residual_path = staging / "selected_correction_train12_residual_population.npz"
    np.savez_compressed(
        residual_path,
        raw_world=raw_world,
        residual_ray=residual,
        camera=camera,
        drive=drive,
    )

    world = args.world.resolve()
    includes = tuple(f"external_camera{suffix}" for suffix in ("", "_b", "_c", "_d", "_e"))
    camera_xy = {
        camera_id: list(map(float, camera_model_from_world(
            world, include_name=include,
        ).cam_pos[:2]))
        for camera_id, include in zip(CAMERAS, includes)
    }
    artifacts = {residual_path.name: sha256(residual_path)}
    for method, covariance in METHODS.items():
        package = {
            "schema": "commissioned_visibility_sensor_model.v1",
            "status": "provisional_train12_navigation_evaluation_pending",
            "audit_accessed": False,
            "mean_model": "box_mlp_visibility_residual",
            "runtime_covariance_model": covariance,
            "planner_covariance_model": covariance,
            "camera_order": list(CAMERAS),
            "camera_xy_m": camera_xy,
            "image_shape_hw": [720, 1280],
            "visibility_grid": {"shape": [1, 16, 16], "definition": "soft_target_blue_in_bbox"},
            "correction_base": verified_entry(base),
            "correction_patch": verified_entry(patch),
            "residual_population": {
                "path": str((output / residual_path.name).resolve()),
                "sha256": sha256(residual_path),
                "kind": "selected-correction train-12 in-sample residuals",
            },
            "neighbors": 120,
            "bandwidth_m": 0.75,
            "runtime_covariance_multiplier": 5.0,
            "runtime_rate_contract": (
                "Five 5 Hz covariance-inflated updates contribute no more nominal "
                "information than one uninflated 1 Hz planner opportunity."
            ),
            "fixed_mean_across_r_arms": True,
            "scientific_boundary": (
                "Provisional navigation evidence only: correction and R were fitted on the "
                "same 12 complete commissioning drives."
            ),
        }
        path = staging / f"{method}.json"
        path.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifacts[path.name] = sha256(path)
    manifest = {
        "schema": "selected_correction_r_runtime_manifest.v1",
        "status": "complete_provisional_navigation_ready",
        "selected_correction": "box_mlp_visibility_residual",
        "r_methods": METHODS,
        "source_hashes": {
            str(report_path): sha256(report_path),
            str(residual_source): sha256(residual_source),
            str(dataset_path): sha256(dataset_path),
            str(world): sha256(world),
        },
        "artifacts": artifacts,
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(staging, output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
