#!/usr/bin/env python3
"""Export spatial planner marginals for the three runtime uncertainty methods."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src/reliability"), str(REPO / "src/unav_common")]
from reliability.commissioned_perception import (  # noqa: E402
    CAMERAS, CommissionedPerceptionSensorModel, _ray_basis, _spd,
)
from reliability.projection import camera_model_from_world  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    template_path = args.template.resolve()
    package_root = args.packages.resolve()
    prediction_path = args.predictions.resolve()
    world = args.world.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    with np.load(template_path, allow_pickle=False) as source:
        template = {name: np.asarray(source[name]) for name in source.files if name != "metadata_json"}
        template_metadata = json.loads(str(source["metadata_json"].item()))
    xs, ys, headings = template["xs"], template["ys"], template["headings"]
    points = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    include_names = (
        "external_camera", "external_camera_b", "external_camera_c",
        "external_camera_d", "external_camera_e",
    )
    camera_xy = {
        camera: np.asarray(camera_model_from_world(world, include_name=include).cam_pos[:2])
        for camera, include in zip(CAMERAS, include_names)
    }
    models = {
        method: CommissionedPerceptionSensorModel(str(package_root / f"{method}.json"))
        for method in ("hierarchical_residual", "spatial_residual")
    }
    with np.load(prediction_path, allow_pickle=False) as archive:
        train_xy = np.asarray(archive["raw_world"], dtype=float)
        train_camera = np.asarray(archive["camera"]).astype(str)
        train_cov_ray = np.asarray(archive["rgb_gaussian_covariance"], dtype=float)

    fields = {
        method: np.empty((len(CAMERAS), len(ys), len(xs), 2, 2), dtype=float)
        for method in ("hierarchical_residual", "spatial_residual", "joint_rgb_gaussian")
    }
    for camera_index, camera in enumerate(CAMERAS):
        print(f"planner marginal {camera}", flush=True)
        use = train_camera == camera
        tree = cKDTree(train_xy[use])
        _distance, neighbor_index = tree.query(points, k=min(120, int(use.sum())))
        joint_values = train_cov_ray[use]
        joint_ray = np.mean(joint_values[neighbor_index], axis=1)
        for point_index, point in enumerate(points):
            basis = _ray_basis(camera_xy[camera], point)
            _mean, hierarchical_ray = models["hierarchical_residual"]._hierarchical_predictive(camera, point)
            _zero, spatial_ray = models["spatial_residual"]._spatial_covariance(camera, point)
            y_index, x_index = divmod(point_index, len(xs))
            for method, covariance_ray in (
                ("hierarchical_residual", hierarchical_ray),
                ("spatial_residual", spatial_ray),
                ("joint_rgb_gaussian", joint_ray[point_index]),
            ):
                fields[method][camera_index, y_index, x_index] = _spd(
                    basis @ covariance_ray @ basis.T
                )

    manifest_artifacts = {}
    for arm, method in (("U1", "hierarchical_residual"), ("U2", "spatial_residual"),
                        ("U3", "joint_rgb_gaussian")):
        field = np.broadcast_to(
            fields[method][:, None],
            (len(CAMERAS), len(headings), len(ys), len(xs), 2, 2),
        ).copy()
        constant = np.mean(fields[method], axis=(1, 2))
        package = package_root / f"{method}.json"
        metadata = dict(template_metadata)
        metadata.update({
            "arm": arm,
            "covariance_arm": method,
            "future_heading_approximation": (
                "position-conditioned marginal over commissioned current-frame inputs; "
                "broadcast over heading"
            ),
            "runtime_sensor_model_path": repo_relative(package),
            "runtime_sensor_model_sha256": sha256(package),
            "runtime_equivalence": (
                "planner uses the spatial marginal of the same method used by the runtime estimator"
            ),
            "source_hashes": {
                **dict(template_metadata.get("source_hashes", {})),
                repo_relative(package): sha256(package),
                repo_relative(prediction_path): sha256(prediction_path),
                repo_relative(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            },
        })
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            xs=xs, ys=ys, headings=headings, camera_ids=template["camera_ids"],
            score=template["score"], availability=template["availability"],
            R_cond_m2=constant,
            R_miss_proxy_m2=constant + np.eye(2)[None],
            R_cond_field_m2=field,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        path = output / f"{arm.lower()}_planner_field.npz"
        path.write_bytes(buffer.getvalue())
        manifest_artifacts[arm] = {
            "method": method, "path": str(path), "sha256": sha256(path),
        }
    (output / "manifest.json").write_text(json.dumps({
        "schema": "three_uncertainty_planner_fields.v1",
        "status": "provisional_navigation_campaign_ready",
        "spatial_marginal": (
            "condition on camera and world position; marginalize unavailable future RGB/box inputs"
        ),
        "artifacts": manifest_artifacts,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
