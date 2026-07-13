from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]


def test_extension_world_contains_two_isolated_camera_models() -> None:
    world_path = ROOT / "src" / "sim" / "gazebo_worlds" / "worlds" / "warehouse_multicamera_extension.world.sdf"
    root = ET.parse(world_path).getroot()
    includes = root.findall(".//include")
    include_by_name = {item.findtext("name"): item for item in includes}

    assert "external_camera" in include_by_name
    assert "external_camera_b" in include_by_name
    assert include_by_name["external_camera"].findtext("uri") == "model://external_camera"
    assert include_by_name["external_camera_b"].findtext("uri") == "model://external_camera_b"
    assert root.find(".//world").get("name") == "warehouse_multicamera_extension"


def test_external_camera_b_model_uses_b_topics_only() -> None:
    model_path = ROOT / "src" / "sim" / "models" / "external_camera_b" / "model.sdf"
    text = model_path.read_text(encoding="utf-8")

    assert "external_camera_b/image_raw" in text
    assert "external_camera_b/depth" in text
    assert "external_camera_b/segmentation" in text
    assert "external_camera/image_raw" not in text


def test_bringup_camera_b_bridge_is_opt_in() -> None:
    bringup = (ROOT / "src" / "sim" / "launch" / "bringup_sim.launch.py").read_text(encoding="utf-8")

    assert '"bridge_camera_b"' in bringup
    assert 'default_value="false"' in bringup
    assert "/external_camera_b/image_raw@sensor_msgs/msg/Image[gz.msgs.Image" in bringup
    assert "condition=IfCondition(bridge_camera_b)" in bringup


def test_multicamera_detector_launch_uses_isolated_topics() -> None:
    launch = (ROOT / "src" / "experiments" / "launch" / "warehouse_multicamera_extension.launch.py").read_text(
        encoding="utf-8"
    )

    assert "warehouse_multicamera_extension.world.sdf" in launch
    assert '"bridge_camera_b": "true"' in launch
    assert "yolo_robot_detector_camera_a" in launch
    assert "yolo_robot_detector_camera_b" in launch
    assert "/perception/camera_A/pixel_pose" in launch
    assert "/perception/camera_B/pixel_pose" in launch
    assert "/perception/camera_A/detection_diagnostics" in launch
    assert "/perception/camera_B/detection_diagnostics" in launch
    assert "/perception/camera_observation/camera_A" in launch
    assert "/perception/camera_observation/camera_B" in launch
    assert '"use_sim_time": LaunchConfiguration("use_sim_time")' in launch
    assert 'DeclareLaunchArgument("use_sim_time", default_value="true")' in launch


def test_active_warehouse_campaign_does_not_use_extension_world() -> None:
    config = (ROOT / "scripts" / "visibility_comparison" / "warehouse_visibility_campaign.yaml").read_text(
        encoding="utf-8"
    )

    assert "warehouse_multicamera_extension" not in config
    assert "external_camera_b" not in config
