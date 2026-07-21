#!/usr/bin/env python3
"""Build the day-zero planner/research prior for warehouse_full_4cam.

No retraining is performed here. The artifact is built from known camera
calibration and the world-profile grid only. The current runtime planner still
uses a single detector stream on camera A, so the planner-facing
P_conservative_plan_map is camera_A-compatible. The same artifact also stores
four-camera union/best maps for offline fusion and the future fused-observation
runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "reliability"))

from reliability import (  # noqa: E402
    CalibratedPriorConfig,
    CameraCalibration,
    PriorGridSpec,
    build_multicamera_geometry_prior,
)


WORLD = "warehouse_full_4cam.world.sdf"
PROFILE = REPO / "src" / "experiments" / "config" / "world_profiles.yaml"
OUT = REPO / "paper_artifacts" / "gp" / "warehouse_full_4cam_dayzero_v1"
ARTIFACT = OUT / "camera_a_planner_with_four_camera_maps.npz"
MANIFEST = OUT / "prior_manifest.json"
ARTIFACT_NX = 240
ARTIFACT_NY = 184

CAMERA_POSES = {
    "camera_A": (-6.0, -10.0, 6.10, 0.0, 0.92, 1.5708),
    "camera_B": (-6.0, 10.0, 6.10, 0.0, 0.92, -1.5708),
    "camera_C": (6.0, -10.0, 6.10, 0.0, 0.92, 1.5708),
    "camera_D": (6.0, 10.0, 6.10, 0.0, 0.92, -1.5708),
}


def _load_profile() -> tuple[dict, dict]:
    payload = yaml.safe_load(PROFILE.read_text(encoding="utf-8")) or {}
    return payload["worlds"][WORLD], payload["camera_intrinsics"]


def main() -> None:
    profile, intrinsics = _load_profile()
    visibility = profile["visibility_defaults"]
    grid = PriorGridSpec.from_bounds(
        x_min=float(visibility["visibility_map_min_x"]),
        x_max=float(visibility["visibility_map_max_x"]),
        y_min=float(visibility["visibility_map_min_y"]),
        y_max=float(visibility["visibility_map_max_y"]),
        nx=ARTIFACT_NX,
        ny=ARTIFACT_NY,
    )
    cfg = CalibratedPriorConfig(
        target_z_m=float(visibility.get("visibility_target_height_m", 0.35)),
        base_reliability=0.98,
        good_range_m=8.0,
        range_decay_m=8.0,
        min_border_margin_px=20.0,
        min_pixel_scale_px_per_m=12.0,
        min_ground_incidence_sin=0.05,
        max_probability=0.995,
    )
    calibrations = [
        CameraCalibration.from_gazebo_pose(
            camera_id,
            pose,
            img_width_px=int(intrinsics["img_width"]),
            img_height_px=int(intrinsics["img_height"]),
            fov_h_rad=float(intrinsics["fov_h_rad"]),
        )
        for camera_id, pose in CAMERA_POSES.items()
    ]
    prior = build_multicamera_geometry_prior(calibrations, grid, cfg)

    planner = prior.camera_maps["camera_A"].probability
    per_camera = {
        camera_id: camera_map.probability
        for camera_id, camera_map in sorted(prior.camera_maps.items())
    }

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ARTIFACT,
        xs=np.asarray(prior.xs, dtype=float),
        ys=np.asarray(prior.ys, dtype=float),
        P_mean_map=planner,
        P_conservative_plan_map=planner,
        P_camera_A_map=per_camera["camera_A"],
        P_camera_B_map=per_camera["camera_B"],
        P_camera_C_map=per_camera["camera_C"],
        P_camera_D_map=per_camera["camera_D"],
        P_union_4cam_map=prior.union_probability,
        P_best_4cam_map=prior.best_probability,
        coverage_count=prior.coverage_count,
        best_camera_id=prior.best_camera_id.astype(str),
        camera_ids=np.asarray(sorted(per_camera.keys()), dtype=str),
        target_height=np.asarray([cfg.target_z_m], dtype=float),
    )

    manifest = {
        "world": WORLD,
        "artifact": str(ARTIFACT.relative_to(REPO)),
        "planner_map": "P_conservative_plan_map",
        "planner_map_semantics": "camera_A day-zero calibrated prior; compatible with current single-camera planner measurement stream",
        "multicamera_maps": ["P_union_4cam_map", "P_best_4cam_map", "coverage_count", "best_camera_id"],
        "training_data_used": False,
        "ground_truth_used": False,
        "camera_poses_xyz_rpy": {key: list(value) for key, value in CAMERA_POSES.items()},
        "grid": {
            "nx": len(prior.xs),
            "ny": len(prior.ys),
            "source_profile_nx": int(visibility["visibility_map_nx"]),
            "source_profile_ny": int(visibility["visibility_map_ny"]),
            "x_min": prior.xs[0],
            "x_max": prior.xs[-1],
            "y_min": prior.ys[0],
            "y_max": prior.ys[-1],
        },
        "summary": prior.to_dict(include_maps=False),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {ARTIFACT.relative_to(REPO)}")
    print(f"wrote {MANIFEST.relative_to(REPO)}")
    print(
        "planner camera_A mean="
        f"{float(np.mean(planner)):.3f}, four-camera union mean={float(np.mean(prior.union_probability)):.3f}"
    )


if __name__ == "__main__":
    main()
