#!/usr/bin/env python3
"""Compare the Stage-08 runtime query with the frozen deployment formula."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from reliability.commissioned_availability import CommissionedAvailabilityModel
from reliability.projection import camera_model_from_world
from unav_common.occlusion_geometry import parse_occlusion_scene_from_world, segment_occluded


REPO = Path(__file__).resolve().parents[2]
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--opportunities", type=Path, required=True)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact_path, world = args.artifact.resolve(), args.world.resolve()
    payload = json.loads(artifact_path.read_text())
    runtime = CommissionedAvailabilityModel(
        artifact_path, world, expected_sha256=sha256(artifact_path)
    )
    include_names = dict(zip(CAMERAS, (
        "external_camera", "external_camera_b", "external_camera_c",
        "external_camera_d", "external_camera_e",
    )))
    cameras = {
        camera: camera_model_from_world(world, include_name=include)
        for camera, include in include_names.items()
    }
    scene = parse_occlusion_scene_from_world(
        str(world), model_name="warehouse_v2_occluders", geometry_tags=("collision",)
    )
    maximum = 0.0
    checked = 0
    with args.opportunities.resolve().open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            camera_id = row["camera_id"]
            camera = cameras[camera_id]
            x, y, yaw = float(row["robot_x"]), float(row["robot_y"]), float(row["robot_yaw"])
            runtime_probability = runtime.probability(camera_id, x, y, yaw, camera)
            box = tuple(float(row[key]) for key in (
                "expected_box_x0", "expected_box_y0", "expected_box_x1", "expected_box_y1"
            ))
            width, height = box[2] - box[0], box[3] - box[1]
            inclusion = float(
                box[0] >= 0.0 and box[1] >= 0.0
                and box[2] < camera.img_width and box[3] < camera.img_height
            )
            line_of_sight = float(not segment_occluded(
                scene.prisms, camera.cam_pos, np.asarray((x, y, 0.2))
            ))
            distance = math.hypot(x - camera.cam_pos[0], y - camera.cam_pos[1])
            parameters = payload["parameters"][camera_id]
            continuous = (distance, width, height)
            features = tuple(
                (continuous[index] - parameters["continuous_mean"][index])
                / parameters["continuous_scale"][index]
                for index in range(3)
            ) + (inclusion, line_of_sight)
            logit = parameters["intercept"] + sum(
                weight * value for weight, value in zip(parameters["coefficient"], features)
            )
            expected = 1.0 / (1.0 + math.exp(-max(min(logit, 60.0), -60.0)))
            expected = min(max(expected, 0.001), 0.999)
            maximum = max(maximum, abs(runtime_probability - expected))
            checked += 1
    report = {
        "schema": "thesis_stage08_runtime_verification.v1",
        "status": "pass" if maximum <= 1e-12 and checked == 9600 else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "final_audit_accessed": False,
        "rows_checked": checked,
        "maximum_probability_absolute_difference": maximum,
        "artifact_sha256": sha256(artifact_path),
        "opportunities_sha256": sha256(args.opportunities.resolve()),
        "world_sha256": sha256(world),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
