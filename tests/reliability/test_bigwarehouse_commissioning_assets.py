from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_fullwarehouse_study_is_dedicated_to_the_four_camera_world() -> None:
    study = ROOT / "experiments" / "multicamera_commissioning_bigwarehouse"
    config = yaml.safe_load((study / "config" / "study.yaml").read_text(encoding="utf-8"))

    assert (study / "README.md").is_file()
    assert (study / "TODO.md").is_file()
    assert config["world"] == "warehouse_full_4cam.world.sdf"
    assert config["world_name"] == "warehouse_full_4cam"
    assert config["camera_ids"] == ["camera_A", "camera_B", "camera_C", "camera_D"]
    assert config["cameras"]["camera_A"]["image_topic"] == "/external_camera/image_raw"
    assert config["cameras"]["camera_B"]["image_topic"] == "/external_camera_b/image_raw"
    assert config["cameras"]["camera_C"]["image_topic"] == "/external_camera_c/image_raw"
    assert config["cameras"]["camera_D"]["image_topic"] == "/external_camera_d/image_raw"
    assert config["collection"]["passive_only"] is True
    routes = config["collection"]["routes"]
    assert {route["name"] for route in routes} == {
        "camera_A_south_west_pass",
        "camera_B_north_west_pass",
        "camera_C_south_east_pass",
        "camera_D_north_east_pass",
        "south_to_north_handover",
        "north_to_south_handover",
        "south_pair_overlap",
        "north_pair_overlap",
        "central_overlap_sweep",
    }
    # All initial centre lines stay inside the 4.5 m central aisle; the first
    # live pass must complete before lateral offsets are admitted.
    for route in routes:
        assert abs(float(route["start"]["x"])) <= 1.8
        assert abs(float(route["goal"]["x"])) <= 1.8


def test_fullwarehouse_launch_targets_the_four_camera_world() -> None:
    launch = (
        ROOT / "src" / "experiments" / "launch" / "warehouse_full4cam_commissioning.launch.py"
    ).read_text(encoding="utf-8")

    assert "warehouse_full_4cam.world.sdf" in launch
    assert '"world_name", default_value="warehouse_full_4cam"' in launch
    assert '"bridge_camera_b": PythonExpression([' in launch
    assert "direct_gz' else 'true'" in launch
    assert '"bridge_contacts": LaunchConfiguration("bridge_contacts")' in launch
    assert '"bridge_contacts", default_value="true"' in launch
    assert '"use_nvidia_prime_offload", default_value="false"' in launch
    assert '"__NV_PRIME_RENDER_OFFLOAD", "1"' in launch
    assert '"__EGL_VENDOR_LIBRARY_FILENAMES"' in launch
    assert "external_camera_2" not in launch
    assert '_detector_node("camera_A", "/external_camera/image_raw")' in launch
    assert '_detector_node("camera_B", "/external_camera_b/image_raw")' in launch
    assert '_detector_node("camera_C", "/external_camera_c/image_raw")' in launch
    assert '_detector_node("camera_D", "/external_camera_d/image_raw")' in launch
    assert 'f"/perception/{camera_id}/pixel_pose"' in launch
    assert 'f"/perception/camera_observation/{camera_id}"' in launch
    assert "ParameterValue(LaunchConfiguration(device_arg), value_type=str)" in launch
    assert 'executable="encoder_noise_node"' in launch
    assert '"output_topic": "/odom_noisy"' in launch
    for frozen_default in (
        '"manager_min_spatial_trust", default_value="0.45"',
        '"manager_min_association_confidence", default_value="0.70"',
        '"manager_max_measurement_age_s", default_value="0.15"',
        '"manager_candidate_score_margin", default_value="0.08"',
        '"manager_required_consecutive_better_frames", default_value="3"',
        '"manager_max_cross_camera_disagreement_m", default_value="0.30"',
        '"manager_max_overlap_time_delta_s", default_value="0.05"',
    ):
        assert frozen_default in launch
    assert '"manager_require_projection_calibration", default_value="true"' in launch
    assert '"manager_require_gp_artifacts", default_value="true"' in launch
    assert '"yolo_cpu_num_threads_camera_a", default_value="2"' in launch


def test_every_central_sweep_offset_clears_the_support_pillar() -> None:
    study = yaml.safe_load(
        (ROOT / "experiments" / "multicamera_commissioning_bigwarehouse" / "config" / "study.yaml").read_text(
            encoding="utf-8"
        )
    )
    profile = yaml.safe_load((ROOT / "src" / "experiments" / "config" / "world_profiles.yaml").read_text(
        encoding="utf-8"
    ))["worlds"]["warehouse_full_4cam.world.sdf"]
    pillar = next(region for region in profile["known_2d_regions"] if region["name"] == "central_support_pillar")
    sweep = next(route for route in study["collection"]["routes"] if route["name"] == "central_overlap_sweep")

    # Eastbound route offsets shift along +y.  Require 0.25 m centre clearance,
    # which exceeds the TurtleBot half-width plus the commissioning margin.
    for offset_m in study["collection"]["lateral_offsets_m"]:
        route_y = float(sweep["start"]["y"]) + float(offset_m)
        assert not (
            float(pillar["ymin"]) - 0.25 <= route_y <= float(pillar["ymax"]) + 0.25
        )


def test_bigwarehouse_study_separates_control_and_operational_odometry() -> None:
    config = yaml.safe_load(
        (ROOT / "experiments" / "multicamera_commissioning_bigwarehouse" / "config" / "study.yaml").read_text(
            encoding="utf-8"
        )
    )
    collection = config["collection"]

    assert collection["control_odom_topic"] == "/odom"
    assert collection["operational_odom_topic"] == "/odom_noisy"
    assert collection["duration_clock"] == "simulation"
    assert collection["command_topic"] == "/cmd_vel"
