from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "src/sim/models"
WORLDS = ROOT / "src/sim/gazebo_worlds/worlds"


def test_laptop_four_camera_models_are_distinct_and_16_by_9() -> None:
    expected = {
        "external_camera_laptop": "external_camera",
        "external_camera_b_laptop": "external_camera_b",
        "external_camera_c_laptop": "external_camera_c",
        "external_camera_d_laptop": "external_camera_d",
    }
    for model_name, runtime_topic_prefix in expected.items():
        root = ET.parse(MODELS / model_name / "model.sdf").getroot()
        assert root.find("model").attrib["name"] == model_name
        rgb = root.find(".//sensor[@name='camera']/camera/image")
        assert rgb.findtext("width") == "640"
        assert rgb.findtext("height") == "360"
        assert root.findtext(".//sensor[@name='camera']/topic") == (
            f"{runtime_topic_prefix}/image_raw"
        )


def test_laptop_world_uses_laptop_camera_assets_without_changing_instance_contract() -> None:
    root = ET.parse(WORLDS / "warehouse_full_4cam_laptop_640x360.world.sdf").getroot()
    world = root.find("world")
    assert world.attrib["name"] == "warehouse_full_4cam_laptop_640x360"
    includes = {
        include.findtext("name"): include.findtext("uri")
        for include in world.findall("include")
    }
    assert {key: includes[key] for key in (
        "external_camera", "external_camera_b", "external_camera_c", "external_camera_d"
    )} == {
        "external_camera": "model://external_camera_laptop",
        "external_camera_b": "model://external_camera_b_laptop",
        "external_camera_c": "model://external_camera_c_laptop",
        "external_camera_d": "model://external_camera_d_laptop",
    }


def test_laptop_3hz_successor_has_isolated_models_and_preserves_topics() -> None:
    expected = {
        "external_camera_laptop_3hz": "external_camera",
        "external_camera_b_laptop_3hz": "external_camera_b",
        "external_camera_c_laptop_3hz": "external_camera_c",
        "external_camera_d_laptop_3hz": "external_camera_d",
    }
    for model_name, runtime_topic_prefix in expected.items():
        root = ET.parse(MODELS / model_name / "model.sdf").getroot()
        assert root.find("model").attrib["name"] == model_name
        sensor = root.find(".//sensor[@name='camera']")
        assert sensor.findtext("update_rate") == "3"
        assert sensor.findtext("topic") == f"{runtime_topic_prefix}/image_raw"

    root = ET.parse(
        WORLDS / "warehouse_full_4cam_laptop_640x360_3hz.world.sdf"
    ).getroot()
    world = root.find("world")
    assert world.attrib["name"] == "warehouse_full_4cam_laptop_640x360_3hz"
    includes = {
        include.findtext("name"): include.findtext("uri")
        for include in world.findall("include")
    }
    assert {key: includes[key] for key in expected.values()} == {
        instance_name: f"model://{model_name}"
        for model_name, instance_name in expected.items()
    }
