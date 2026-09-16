#!/usr/bin/env python3
"""Export planner R fields matched to the selected train-12 correction.

The four R-only arms reuse one commissioned availability field.  They differ
only in the covariance forecast built from the frozen correction residuals.
An optional RGB field is copied as a separately labelled mean-and-R ablation.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src/reliability"), str(REPO / "src/unav_common")]
from reliability.commissioned_perception import CAMERAS, _ray_basis, _spd  # noqa: E402
from reliability.projection import camera_model_from_world  # noqa: E402


METHODS = (
    ("U0", "global_residual_R"),
    ("U1", "per_camera_residual_R"),
    ("U2", "spatial_residual_R"),
    ("U3", "hierarchical_predictive_R"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scatter(values: np.ndarray) -> np.ndarray:
    return _spd(values.T @ values / max(len(values), 1))


def residual_population(residual_path: Path, dataset_path: Path) -> dict[str, np.ndarray]:
    with np.load(residual_path, allow_pickle=False) as archive:
        residual = np.asarray(archive["residual_ray_m"], dtype=float)
        drive = np.asarray(archive["drive"]).astype(str)
        camera = np.asarray(archive["camera"]).astype(str)
        frames = np.asarray(archive["source_frame_id"]).astype(str)
        stamps = np.asarray(archive["stamp_ns"], dtype=np.int64)
    with np.load(dataset_path, allow_pickle=False) as archive:
        lookup = {
            (str(d), str(c), str(f), int(t)): index
            for index, (d, c, f, t) in enumerate(zip(
                archive["drive"], archive["camera"], archive["source_frame_id"],
                archive["stamp_ns"],
            ))
        }
        index = np.asarray([
            lookup[(d, c, f, int(t))]
            for d, c, f, t in zip(drive, camera, frames, stamps)
        ])
        xy = np.asarray(archive["raw"][index], dtype=float)
    return {"residual": residual, "drive": drive, "camera": camera, "xy": xy}


def local_ray_covariances(
    points: np.ndarray,
    values: np.ndarray,
    drives: np.ndarray,
    queries: np.ndarray,
    *,
    neighbors: int = 120,
    bandwidth_m: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    global_covariance = scatter(values)
    tree = cKDTree(points)
    distance, neighbor = tree.query(queries, k=min(neighbors, len(points)))
    if neighbor.ndim == 1:
        neighbor = neighbor[:, None]
        distance = distance[:, None]
    spatial = np.empty((len(queries), 2, 2), dtype=float)
    hierarchical = np.empty_like(spatial)
    for row, (dist, local) in enumerate(zip(distance, neighbor)):
        weight = np.exp(-0.5 * (dist / bandwidth_m) ** 2)
        if float(weight.sum()) <= 1.0e-12:
            weight[np.argmin(dist)] = 1.0
        local_values = values[local]
        local_drives = drives[local]
        capped = weight.copy()
        for drive in np.unique(local_drives):
            use = local_drives == drive
            total = float(capped[use].sum())
            if total > 0.0:
                capped[use] /= total
        local_scatter = sum(
            w * np.outer(value, value) for w, value in zip(capped, local_values)
        )
        spatial[row] = _spd(
            (local_scatter + 4.0 * global_covariance) / (float(capped.sum()) + 4.0)
        )

        means, covariances, drive_weights = [], [], []
        for drive in np.unique(local_drives):
            use = local_drives == drive
            local_weight = weight[use]
            if float(local_weight.sum()) <= 1.0e-12:
                continue
            local_drive_values = local_values[use]
            mean = np.average(local_drive_values, axis=0, weights=local_weight)
            covariance = sum(
                w * np.outer(value - mean, value - mean)
                for w, value in zip(local_weight, local_drive_values)
            ) / float(local_weight.sum())
            means.append(mean)
            covariances.append(_spd(covariance))
            drive_weights.append(min(1.0, float(local_weight.sum()) / 5.0))
        wd = np.asarray(drive_weights, dtype=float)
        n = float(wd.sum())
        if n <= 1.0e-12:
            hierarchical[row] = global_covariance
            continue
        means_array = np.asarray(means, dtype=float)
        mean = np.average(means_array, axis=0, weights=wd)
        within = np.average(np.asarray(covariances), axis=0, weights=wd)
        between = sum(
            w * np.outer(value - mean, value - mean)
            for w, value in zip(wd, means_array)
        ) / n
        kappa, nu = 1.0 + n, 5.0 + n
        psi = 2.0 * global_covariance + n * (within + between) + (n / kappa) * np.outer(mean, mean)
        degrees = max(nu - 1.0, 3.01)
        scale = _spd(((kappa + 1.0) / (kappa * degrees)) * psi)
        hierarchical[row] = _spd(scale * degrees / (degrees - 2.0))
    return spatial, hierarchical


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--residuals", type=Path, required=True)
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--rgb-field", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists():
        raise FileExistsError(output if output.exists() else staging)
    staging.mkdir(parents=True)

    template_path = args.template.resolve()
    residual_path = args.residuals.resolve()
    dataset_path = args.dataset_contract.resolve()
    world_path = args.world.resolve()
    with np.load(template_path, allow_pickle=False) as archive:
        template = {
            name: np.asarray(archive[name])
            for name in archive.files if name != "metadata_json"
        }
        template_metadata = json.loads(str(archive["metadata_json"].item()))
    population = residual_population(residual_path, dataset_path)
    xs, ys, headings = template["xs"], template["ys"], template["headings"]
    queries = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    includes = tuple(f"external_camera{suffix}" for suffix in ("", "_b", "_c", "_d", "_e"))
    camera_xy = {
        camera: np.asarray(
            camera_model_from_world(world_path, include_name=include).cam_pos[:2], dtype=float
        )
        for camera, include in zip(CAMERAS, includes)
    }

    field = {
        method: np.empty((len(CAMERAS), len(ys), len(xs), 2, 2), dtype=float)
        for _arm, method in METHODS
    }
    pooled = scatter(population["residual"])
    for camera_index, camera in enumerate(CAMERAS):
        print(f"building selected-correction R fields for {camera}", flush=True)
        use = population["camera"] == camera
        values = population["residual"][use]
        per_camera = scatter(values)
        spatial, hierarchical = local_ray_covariances(
            population["xy"][use], values, population["drive"][use], queries,
        )
        for query_index, query in enumerate(queries):
            basis = _ray_basis(camera_xy[camera], query)
            y_index, x_index = divmod(query_index, len(xs))
            for method, covariance_ray in (
                ("global_residual_R", pooled),
                ("per_camera_residual_R", per_camera),
                ("spatial_residual_R", spatial[query_index]),
                ("hierarchical_predictive_R", hierarchical[query_index]),
            ):
                field[method][camera_index, y_index, x_index] = _spd(
                    basis @ covariance_ray @ basis.T
                )

    artifacts = {}
    for arm, method in METHODS:
        full = np.broadcast_to(
            field[method][:, None],
            (len(CAMERAS), len(headings), len(ys), len(xs), 2, 2),
        ).copy()
        constant = np.mean(field[method], axis=(1, 2))
        metadata = dict(template_metadata)
        metadata.update({
            "arm": arm,
            "comparison_role": "R_only_selected_correction_fixed",
            "mean_model": "box_mlp_visibility_residual",
            "covariance_arm": method,
            "residual_fit_population": "12 complete drives; correction-fit in-sample",
            "future_heading_approximation": "position-conditioned covariance broadcast over heading",
            "runtime_equivalence": "pending current selected-correction runtime packaging",
            "source_hashes": {
                str(residual_path.relative_to(REPO)): sha256(residual_path),
                str(dataset_path.relative_to(REPO)): sha256(dataset_path),
                str(Path(__file__).resolve().relative_to(REPO)): sha256(Path(__file__).resolve()),
            },
        })
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            xs=xs,
            ys=ys,
            headings=headings,
            camera_ids=template["camera_ids"],
            score=template["score"],
            availability=template["availability"],
            R_cond_m2=constant,
            R_miss_proxy_m2=constant + np.eye(2)[None],
            R_cond_field_m2=full,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        path = staging / f"{arm.lower()}_planner_field.npz"
        path.write_bytes(buffer.getvalue())
        artifacts[arm] = {"method": method, "path": path.name, "sha256": sha256(path)}

    if args.rgb_field:
        rgb_source = args.rgb_field.resolve()
        rgb_path = staging / "u4_planner_field.npz"
        rgb_path.write_bytes(rgb_source.read_bytes())
        artifacts["U4"] = {
            "method": "joint_RGB_Gaussian_mean_and_R_ablation",
            "path": rgb_path.name,
            "sha256": sha256(rgb_path),
            "source": str(rgb_source.relative_to(REPO)),
            "scientific_role": "end-to-end perception ablation; not an R-only arm",
        }

    manifest = {
        "schema": "selected_correction_r_planner_fields.v1",
        "status": "complete_route_selection_ready_runtime_packaging_pending",
        "mean_model": "box_mlp_visibility_residual",
        "availability": "identical commissioned position GP copied from template",
        "r_only_arms": [arm for arm, _method in METHODS],
        "separate_ablation_arm": "U4" if args.rgb_field else None,
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
