from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
for relative in ("src/reliability", "src/unav_common"):
    location = str(ROOT / relative)
    if location not in sys.path:
        sys.path.insert(0, location)

from reliability.contracts import CameraObservation  # noqa: E402
from reliability.projection import (  # noqa: E402
    camera_model_from_world,
    project_observation_to_world,
)

WORLD_SDF = ROOT / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
CAMERA_INCLUDES = {
    "camera_A": "external_camera",
    "camera_B": "external_camera_b",
    "camera_C": "external_camera_c",
    "camera_D": "external_camera_d",
}


def _observation(camera_id: str, *, u: float, v: float) -> CameraObservation:
    return CameraObservation(
        camera_id=camera_id,
        timestamp_s=10.0,
        pixel_uv=(u, v),
        detection_valid=True,
        detector_score=0.9,
        measurement_age_s=0.02,
        calibration_id=f"warehouse_full_4cam_{camera_id}",
        image_frame_id=camera_id,
    )


def test_recorder_delegates_projection_to_library():
    """The recorder must reuse the library projection, not carry a fork."""

    tools = ROOT / "experiments/multicamera_commissioning_bigwarehouse/tools"
    sys.path.insert(0, str(tools))
    try:
        import record_operational_logs as recorder
    finally:
        sys.path.remove(str(tools))

    assert recorder.project_observation_to_world is project_observation_to_world
    assert recorder.camera_model_from_world is camera_model_from_world


def test_along_bearing_offset_moves_point_away_from_camera():
    import math

    model = camera_model_from_world(WORLD_SDF, include_name="external_camera_c")
    observation = _observation("camera_C", u=640.0, v=500.0)
    base = project_observation_to_world(observation, model, contact_z_m=0.05)
    shifted = project_observation_to_world(
        observation, model, contact_z_m=0.05, along_bearing_offset_m=0.20
    )
    assert base is not None and shifted is not None
    moved = math.hypot(shifted[0] - base[0], shifted[1] - base[1])
    assert moved == pytest.approx(0.20, abs=1e-9)
    base_dist = math.hypot(base[0] - model.cam_pos[0], base[1] - model.cam_pos[1])
    shifted_dist = math.hypot(shifted[0] - model.cam_pos[0], shifted[1] - model.cam_pos[1])
    assert shifted_dist == pytest.approx(base_dist + 0.20, abs=1e-9)


def test_load_projection_calibration_round_trip(tmp_path):
    import json

    from reliability.projection import load_projection_calibration

    payload = {
        "kind": "projection_along_bearing_offsets",
        "cameras": {
            "camera_B": {"intercept_m": -0.045, "slope_per_m": 0.0175, "samples": 280},
            # Legacy constant-offset entries stay readable as intercept-only.
            "camera_C": {"along_bearing_offset_m": 0.166, "samples": 320},
            "camera_D": 0.105,
        },
    }
    path = tmp_path / "projection_calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    calibrations = load_projection_calibration(path)
    assert calibrations == {
        "camera_B": {"intercept_m": -0.045, "slope_per_m": 0.0175},
        "camera_C": {"intercept_m": 0.166, "slope_per_m": 0.0},
        "camera_D": {"intercept_m": 0.105, "slope_per_m": 0.0},
    }


def test_distance_slope_scales_correction_with_range():
    import math

    model = camera_model_from_world(WORLD_SDF, include_name="external_camera_c")
    observation = _observation("camera_C", u=640.0, v=500.0)
    base = project_observation_to_world(observation, model, contact_z_m=0.05)
    corrected = project_observation_to_world(
        observation, model, contact_z_m=0.05,
        along_bearing_offset_m=-0.021, along_bearing_slope_per_m=0.0153,
    )
    distance = math.hypot(base[0] - model.cam_pos[0], base[1] - model.cam_pos[1])
    expected = -0.021 + 0.0153 * distance
    moved = math.hypot(corrected[0] - base[0], corrected[1] - base[1])
    assert moved == pytest.approx(abs(expected), abs=1e-9)


def test_projection_rejects_invalid_detection():
    model = camera_model_from_world(WORLD_SDF, include_name="external_camera")
    invalid = CameraObservation(camera_id="camera_A", timestamp_s=1.0, detection_valid=False)
    assert project_observation_to_world(invalid, model, contact_z_m=0.05) is None


def test_all_four_cameras_project_into_world_bounds():
    for camera_id, include in CAMERA_INCLUDES.items():
        model = camera_model_from_world(WORLD_SDF, include_name=include)
        world_xy = project_observation_to_world(
            _observation(camera_id, u=640.0, v=520.0), model, contact_z_m=0.05
        )
        assert world_xy is not None
        x, y = world_xy
        assert -12.5 <= x <= 12.5
        assert -10.5 <= y <= 10.5


def test_node_module_imports_without_ros():
    """The node module must stay importable in non-ROS tooling contexts."""

    from reliability.nodes import camera_manager_node

    assert hasattr(camera_manager_node, "CameraManagerNode")
    assert hasattr(camera_manager_node, "main")
    quality = camera_manager_node._contract_quality(
        _observation("camera_B", u=100.0, v=200.0)
    )
    assert quality.camera_id == "camera_B"
    assert quality.p_available == 1.0
    assert quality.source_model == "live_contract"
