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


def test_library_projection_matches_recorder_copy():
    """The library helper and the recorder's original must agree exactly."""

    tools = ROOT / "experiments/multicamera_commissioning_bigwarehouse/tools"
    sys.path.insert(0, str(tools))
    try:
        import record_operational_logs as recorder
    finally:
        sys.path.remove(str(tools))

    observation = _observation("camera_C", u=640.0, v=500.0)
    for include in CAMERA_INCLUDES.values():
        library_model = camera_model_from_world(WORLD_SDF, include_name=include)
        recorder_model = recorder.camera_model_from_world(WORLD_SDF, include_name=include)
        library_xy = project_observation_to_world(observation, library_model, contact_z_m=0.05)
        recorder_xy = recorder.project_observation_to_world(
            observation, recorder_model, contact_z_m=0.05
        )
        assert library_xy == pytest.approx(recorder_xy, abs=1e-12)


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
