from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
for relative in ("src/reliability", "src/unav_common"):
    location = str(ROOT / relative)
    if location not in sys.path:
        sys.path.insert(0, location)

from reliability.contracts import CameraObservation, CameraQuality  # noqa: E402
from reliability.projection import (  # noqa: E402
    camera_model_from_world,
    project_observation_to_world,
    project_observation_to_world_with_covariance,
)

WORLD_SDF = ROOT / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
CAMERA_INCLUDES = {
    "camera_A": "external_camera",
    "camera_B": "external_camera_b",
    "camera_C": "external_camera_c",
    "camera_D": "external_camera_d",
}


def _observation(
    camera_id: str,
    *,
    u: float,
    v: float,
    conditional_cov_uv=((2.5**2, 0.0), (0.0, 2.5**2)),
) -> CameraObservation:
    return CameraObservation(
        camera_id=camera_id,
        timestamp_s=10.0,
        pixel_uv=(u, v),
        detection_valid=True,
        detector_score=0.9,
        measurement_age_s=0.02,
        calibration_id=f"warehouse_full_4cam_{camera_id}",
        image_frame_id=camera_id,
        conditional_cov_uv=conditional_cov_uv,
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
    # Every entry now also carries the cross-bearing degree of freedom, which an
    # along-only file leaves at zero so the correction is unchanged. See
    # tests/reliability/test_projection_cross_bearing.py.
    assert calibrations == {
        "camera_B": {"intercept_m": -0.045, "slope_per_m": 0.0175,
                     "cross_intercept_m": 0.0, "cross_slope_per_m": 0.0},
        "camera_C": {"intercept_m": 0.166, "slope_per_m": 0.0,
                     "cross_intercept_m": 0.0, "cross_slope_per_m": 0.0},
        "camera_D": {"intercept_m": 0.105, "slope_per_m": 0.0,
                     "cross_intercept_m": 0.0, "cross_slope_per_m": 0.0},
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


def test_paper1_pixel_covariance_is_propagated_as_full_map_ellipse():
    model = camera_model_from_world(WORLD_SDF, include_name="external_camera_c")
    observation = _observation(
        "camera_C",
        u=710.0,
        v=470.0,
        conditional_cov_uv=((6.25, 1.5), (1.5, 25.0)),
    )
    projected = project_observation_to_world_with_covariance(
        observation,
        model,
        contact_z_m=0.05,
        along_bearing_offset_m=-0.021,
        along_bearing_slope_per_m=0.0153,
    )
    assert projected is not None
    world_xy, covariance = projected
    assert world_xy == pytest.approx(
        project_observation_to_world(
            observation,
            model,
            contact_z_m=0.05,
            along_bearing_offset_m=-0.021,
            along_bearing_slope_per_m=0.0153,
        )
    )
    assert covariance[0][1] == pytest.approx(covariance[1][0])
    assert abs(covariance[0][1]) > 1.0e-8
    assert covariance[0][0] > 0.0
    assert covariance[1][1] > 0.0
    assert covariance[0][0] * covariance[1][1] - covariance[0][1] ** 2 > 0.0


def test_projected_covariance_scales_linearly_with_pixel_covariance():
    model = camera_model_from_world(WORLD_SDF, include_name="external_camera")
    base = project_observation_to_world_with_covariance(
        _observation(
            "camera_A", u=640.0, v=500.0,
            conditional_cov_uv=((6.25, 0.0), (0.0, 6.25)),
        ),
        model,
        contact_z_m=0.05,
    )
    scaled = project_observation_to_world_with_covariance(
        _observation(
            "camera_A", u=640.0, v=500.0,
            conditional_cov_uv=((25.0, 0.0), (0.0, 25.0)),
        ),
        model,
        contact_z_m=0.05,
    )
    assert base is not None and scaled is not None
    for row in range(2):
        for column in range(2):
            assert scaled[1][row][column] == pytest.approx(
                4.0 * base[1][row][column], rel=1.0e-9, abs=1.0e-12
            )


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


def test_paper1_profile_preserves_fused_cross_covariance_without_report_floor():
    from reliability.nodes.camera_manager_node import (
        PAPER1_HISTORICAL_COVARIANCE,
        _fusion_report_covariance,
    )

    covariance = ((0.0025, -0.001), (-0.001, 0.010))
    reported = _fusion_report_covariance(
        covariance,
        covariance_profile=PAPER1_HISTORICAL_COVARIANCE,
        report_std_m=0.30,
    )
    assert reported == covariance


def test_legacy_profile_retains_old_diagonal_report_floor():
    from reliability.nodes.camera_manager_node import (
        LEGACY_MULTICAM_COVARIANCE,
        _fusion_report_covariance,
    )

    reported = _fusion_report_covariance(
        ((0.0025, -0.001), (-0.001, 0.010)),
        covariance_profile=LEGACY_MULTICAM_COVARIANCE,
        report_std_m=0.30,
    )
    assert reported[0] == pytest.approx((0.09, 0.0))
    assert reported[1] == pytest.approx((0.0, 0.09))


def test_paper1_profile_does_not_apply_multicam_handover_inflation():
    from reliability.fusion import MapObservation
    from reliability.nodes.camera_manager_node import (
        PAPER1_HISTORICAL_COVARIANCE,
        _handover_profile_observation,
    )

    quality = CameraQuality(
        camera_id="camera_B",
        p_available=0.9,
        conditional_cov_uv=((6.25, 0.0), (0.0, 6.25)),
        association_confidence=0.9,
    )
    selected = MapObservation(
        camera_id="camera_B",
        timestamp_s=1.0,
        xy_m=(1.0, 2.0),
        covariance_m2=((0.02, 0.005), (0.005, 0.04)),
        quality=quality,
    )
    inflated = MapObservation(
        camera_id="camera_B",
        timestamp_s=1.0,
        xy_m=(1.0, 2.0),
        covariance_m2=((0.18, 0.045), (0.045, 0.36)),
        quality=quality,
    )
    assert _handover_profile_observation(
        selected,
        inflated,
        covariance_profile=PAPER1_HISTORICAL_COVARIANCE,
    ) is selected


def test_paper1_fusion_is_prior_free_and_rejects_a_metric_outlier():
    from reliability.fusion import MapObservation
    from reliability.nodes.camera_manager_node import (
        _paper1_precision_fusion_with_disagreement_gate,
    )

    def observation(camera_id, x):
        return MapObservation(
            camera_id=camera_id,
            timestamp_s=1.0,
            xy_m=(x, 0.0),
            covariance_m2=((0.04, 0.0), (0.0, 0.04)),
            quality=CameraQuality(
                camera_id=camera_id,
                p_available=0.9,
                conditional_cov_uv=((6.25, 0.0), (0.0, 6.25)),
                association_confidence=0.9,
            ),
        )

    result = _paper1_precision_fusion_with_disagreement_gate(
        [
            observation("camera_A", 0.0),
            observation("camera_B", 0.2),
            observation("camera_C", 5.0),
        ],
        disagreement_gate_m=0.6,
    )
    assert result.accepted_camera_ids == ("camera_A", "camera_B")
    assert result.rejected_camera_ids == ("camera_C",)
    assert result.mean_xy == pytest.approx((0.1, 0.0))
    assert result.covariance_m2[0] == pytest.approx((0.02, 0.0))
    assert result.covariance_m2[1] == pytest.approx((0.0, 0.02))


def test_fusion_candidates_share_manager_gates_and_drop_cached_skewed_views():
    from reliability.camera_manager import CameraManager, CameraManagerConfig
    from reliability.fusion import MapObservation
    from reliability.nodes.camera_manager_node import _synchronous_fusion_candidates

    def map_obs(camera_id, timestamp_s, *, association=0.9):
        return MapObservation(
            camera_id=camera_id,
            timestamp_s=timestamp_s,
            xy_m=(0.0, 0.0),
            covariance_m2=((0.1, 0.0), (0.0, 0.1)),
            quality=CameraQuality(
                camera_id=camera_id,
                p_available=0.9,
                conditional_cov_uv=((1.0, 0.0), (0.0, 1.0)),
                association_confidence=association,
                epistemic_score=0.0,
                source_model="test",
            ),
        )

    manager = CameraManager(CameraManagerConfig(
        min_spatial_trust=0.1,
        min_association_confidence=0.5,
        max_measurement_age_s=1.0,
        age_decay_s=1.0,
    ))
    fresh, scores, rejected, skewed = _synchronous_fusion_candidates(
        manager,
        now_s=10.0,
        observations=[
            map_obs("camera_A", 10.0),
            map_obs("camera_B", 9.6),
            map_obs("camera_C", 10.0, association=0.2),
        ],
        max_timestamp_spread_s=0.25,
    )

    assert [obs.camera_id for obs in fresh] == ["camera_A"]
    assert scores["camera_A"] > 0.0
    assert skewed == ["camera_B"]
    assert "association_below_minimum" in rejected["camera_C"]
