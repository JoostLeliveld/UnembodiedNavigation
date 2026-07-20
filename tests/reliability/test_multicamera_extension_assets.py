from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]


def _parse_world(filename: str) -> ET.Element:
    path = ROOT / "src" / "sim" / "gazebo_worlds" / "worlds" / filename
    return ET.parse(path).getroot()


def _include_by_name(root: ET.Element) -> dict[str, ET.Element]:
    return {item.findtext("name"): item for item in root.findall(".//include")}


def _pose_values(include: ET.Element) -> list[float]:
    return [float(value) for value in include.findtext("pose", "").split()]


def _ground_size(root: ET.Element) -> tuple[float, float]:
    size = root.find(".//model[@name='ground_plane']//plane/size")
    assert size is not None
    sx, sy = [float(value) for value in size.text.split()]
    return sx, sy


def test_full_warehouse_four_camera_world_is_canonical_aws_mesh_layout() -> None:
    path = ROOT / "src" / "sim" / "gazebo_worlds" / "worlds" / "warehouse_full_4cam.world.sdf"
    root = ET.parse(path).getroot()
    text = path.read_text(encoding="utf-8")
    include_by_name = _include_by_name(root)

    assert root.find(".//world").get("name") == "warehouse_full_4cam"
    assert _ground_size(root) == (24.5, 20.5)
    assert include_by_name["external_camera"].findtext("uri") == "model://external_camera"
    assert include_by_name["external_camera_b"].findtext("uri") == "model://external_camera_b"
    assert include_by_name["external_camera_c"].findtext("uri") == "model://external_camera_c"
    assert include_by_name["external_camera_d"].findtext("uri") == "model://external_camera_d"
    assert "external_camera_2" not in text

    expected_camera_poses = {
        "external_camera": (-6.0, -10.0, 1.5708),
        "external_camera_b": (-6.0, 10.0, -1.5708),
        "external_camera_c": (6.0, -10.0, 1.5708),
        "external_camera_d": (6.0, 10.0, -1.5708),
    }
    for camera_name, (x, y, yaw) in expected_camera_poses.items():
        pose = _pose_values(include_by_name[camera_name])
        assert pose[0] == x
        assert pose[1] == y
        assert pose[2] == 6.10
        assert pose[4] == 0.92
        assert pose[5] == yaw

    rack_collisions = [
        link
        for link in root.findall(".//model[@name='warehouse_rack_occluders']/link")
        if link.get("name", "").startswith("rack_") and link.find("collision") is not None
    ]
    mesh_visuals = root.findall(".//model[@name='aws_shelf_mesh_visuals']/link")
    assert len(rack_collisions) == 27
    assert len(mesh_visuals) == 27
    rack_xs = sorted({
        round(float(link.findtext("pose", "").split()[0]), 3)
        for link in rack_collisions
    })
    assert rack_xs == [-11.635, -8.825, -6.725, -4.625, -2.525, 2.525, 4.625, 6.725, 8.825]
    inner_clearance_m = (2.525 - 0.55 / 2.0) - (-2.525 + 0.55 / 2.0)
    assert 4.45 <= inner_clearance_m <= 4.55
    assert "rack_W0_south" in text
    assert "model://aws_robomaker_warehouse_ShelfD_01" in text
    assert "model://aws_robomaker_warehouse_ShelfE_01" in text
    assert include_by_name["desk_qc_ne"].findtext("uri") == "model://aws_robomaker_warehouse_DeskC_01"
    assert include_by_name["palletjack_apron"].findtext("uri") == "model://aws_robomaker_warehouse_PalletJackB_01"
    assert include_by_name["clutterA_sw"].findtext("uri") == "model://aws_robomaker_warehouse_ClutteringA_01"


def test_full_warehouse_layout_map_documents_the_chosen_structure() -> None:
    doc = (ROOT / "docs" / "warehouse_full_4cam_layout.md").read_text(encoding="utf-8")
    svg = (ROOT / "docs" / "assets" / "warehouse_full_4cam_map.svg").read_text(encoding="utf-8")

    assert "warehouse_full_4cam.world.sdf" in doc
    assert "Camera A" in doc
    assert "Camera B" in doc
    assert "Camera C" in doc
    assert "Camera D" in doc
    assert "(-6.0, -10)" in doc
    assert "(6.0, 10)" in doc
    assert "south wall, west dock column" in doc
    assert "north wall, east dock column" in doc
    assert "DeskC_01" in doc
    assert "W2_north" in doc
    assert "E3_south" in doc
    assert "4.50 m` central aisle" in doc
    assert "wall-backed" in doc
    assert "Green No-Go Outlines" in doc
    assert "crossing a green line" in doc
    assert "warehouse_full_4cam: 24.5 x 20.5 m AWS-style warehouse" in svg
    assert "4.5 m central aisle" in svg
    assert "green no-go/collision outline" in svg


def test_extension_camera_models_use_isolated_topics_only() -> None:
    for suffix in ["b", "c", "d"]:
        model_path = ROOT / "src" / "sim" / "models" / f"external_camera_{suffix}" / "model.sdf"
        text = model_path.read_text(encoding="utf-8")

        assert f"external_camera_{suffix}/image_raw" in text
        assert f"external_camera_{suffix}/depth" in text
        assert f"external_camera_{suffix}/segmentation" in text
        assert "external_camera/image_raw" not in text


def test_bringup_extension_camera_bridges_are_opt_in() -> None:
    bringup = (ROOT / "src" / "sim" / "launch" / "bringup_sim.launch.py").read_text(encoding="utf-8")

    for suffix in ["b", "c", "d"]:
        assert f'"bridge_camera_{suffix}"' in bringup
        assert f"/external_camera_{suffix}/image_raw@sensor_msgs/msg/Image[gz.msgs.Image" in bringup
        assert f"/external_camera_{suffix}/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo" in bringup
        assert f"condition=IfCondition(bridge_camera_{suffix})" in bringup


def test_bringup_extension_segmentation_bridges_are_independently_opt_in() -> None:
    bringup = (ROOT / "src" / "sim" / "launch" / "bringup_sim.launch.py").read_text(encoding="utf-8")

    for suffix in ["b", "c", "d"]:
        assert f'"bridge_segmentation_{suffix}"' in bringup
        assert (
            f"/external_camera_{suffix}/segmentation/labels_map"
            "@sensor_msgs/msg/Image[gz.msgs.Image"
        ) in bringup
        assert f"condition=IfCondition(bridge_segmentation_{suffix})" in bringup


def test_multicamera_detector_launch_uses_isolated_topics() -> None:
    launch = (ROOT / "src" / "experiments" / "launch" / "warehouse_multicamera_extension.launch.py").read_text(
        encoding="utf-8"
    )

    assert "warehouse_full_4cam.world.sdf" in launch
    assert "warehouse_full_4cam" in launch
    assert '"bridge_camera_b": "true"' in launch
    assert '"bridge_camera_c": "true"' in launch
    assert '"bridge_camera_d": "true"' in launch
    for camera in ["A", "B", "C", "D"]:
        assert f'_detector_node("{camera}",' in launch
    assert 'f"/perception/{camera_id}/pixel_pose"' in launch
    assert 'f"/perception/{camera_id}/detection_diagnostics"' in launch
    assert 'f"/perception/camera_observation/{camera_id}"' in launch
    assert '"use_sim_time": LaunchConfiguration("use_sim_time")' in launch
    assert 'DeclareLaunchArgument("use_sim_time", default_value="true")' in launch


def test_primary_comparison_defaults_to_forward_four_camera_world() -> None:
    launch = (ROOT / "src" / "experiments" / "launch" / "warehouse_primary_comparison.launch.py").read_text(
        encoding="utf-8"
    )
    shared = (ROOT / "src" / "experiments" / "experiments" / "core" / "visibility_launch_common.py").read_text(
        encoding="utf-8"
    )

    assert "warehouse_full_4cam.world.sdf" in launch
    assert "warehouse_full_4cam.world.sdf" in shared
    for suffix in ["b", "c", "d"]:
        assert f"DeclareLaunchArgument('bridge_camera_{suffix}', default_value='true'" in launch
        assert f"'bridge_camera_{suffix}': _as_bool(" in shared
        assert f"'bridge_camera_{suffix}': 'true' if cfg.get('bridge_camera_{suffix}', False) else 'false'" in shared


def test_forward_four_camera_campaign_uses_new_world_and_dayzero_artifact() -> None:
    config_path = ROOT / "scripts" / "visibility_comparison" / "warehouse_full_4cam_campaign.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["world"] == "warehouse_full_4cam.world.sdf"
    assert config["launch_file"] == "warehouse_primary_comparison.launch.py"
    assert config["bridge_camera_b"] is True
    assert config["bridge_camera_c"] is True
    assert config["bridge_camera_d"] is True
    assert "driveable_geometry_json" not in config
    assert config["gp_artifact"] == (
        "paper_artifacts/gp/warehouse_full_4cam_dayzero_v1/"
        "camera_a_planner_with_four_camera_maps.npz"
    )
    assert set(config["tasks"]) == {"full_traverse_handover", "route_tall_shadow_west"}
    for task in config["tasks"].values():
        assert task["conditions"] == ["C1", "C2"]


def test_full_four_camera_dayzero_artifact_keeps_planner_field_single_camera_compatible() -> None:
    artifact = ROOT / "paper_artifacts" / "gp" / "warehouse_full_4cam_dayzero_v1" / (
        "camera_a_planner_with_four_camera_maps.npz"
    )
    manifest = ROOT / "paper_artifacts" / "gp" / "warehouse_full_4cam_dayzero_v1" / "prior_manifest.json"

    with np.load(artifact, allow_pickle=False) as data:
        planner = data["P_conservative_plan_map"]
        camera_a = data["P_camera_A_map"]
        union = data["P_union_4cam_map"]
        best = data["P_best_4cam_map"]
        coverage = data["coverage_count"]

    assert planner.shape == camera_a.shape == union.shape == best.shape == coverage.shape
    assert planner.shape == (184, 240)
    assert np.allclose(planner, camera_a)
    assert np.all(union + 1.0e-12 >= best)
    assert np.nanmean(union) > np.nanmean(planner)
    assert np.max(coverage) >= 2.0
    manifest_text = manifest.read_text(encoding="utf-8")
    assert "camera_A day-zero calibrated prior" in manifest_text
    assert "training_data_used" in manifest_text


def test_active_warehouse_campaign_does_not_use_extension_world() -> None:
    config = (ROOT / "scripts" / "visibility_comparison" / "warehouse_visibility_campaign.yaml").read_text(
        encoding="utf-8"
    )

    assert "warehouse_multicamera_extension" not in config
    assert "external_camera_b" not in config
