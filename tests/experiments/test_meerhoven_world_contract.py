from pathlib import Path
import xml.etree.ElementTree as ET
import importlib.util
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORLD_NAME = "warehouse_meerhoven.world.sdf"


def test_meerhoven_profile_camera_registry_matches_the_generated_world():
    profiles = yaml.safe_load(
        (ROOT / "src/experiments/config/world_profiles.yaml").read_text(encoding="utf-8")
    )
    profile = profiles["worlds"][WORLD_NAME]
    ids = profile["camera_ids"]
    includes = profile["camera_model_includes"]
    roles = profile["camera_roles"]

    assert ids == [f"camera_{letter}" for letter in "ABCDEFGHIJKL"]
    assert len(ids) == len(includes) == len(roles) == 12
    assert len(set(includes)) == 12

    root = ET.parse(ROOT / "src/sim/gazebo_worlds/worlds" / WORLD_NAME).getroot()
    world_includes = {
        (node.findtext("name") or "").strip(): (node.findtext("uri") or "").strip()
        for node in root.findall(".//world/include")
    }
    assert list(world_includes)[-12:] == includes
    for include in includes:
        assert world_includes[include] == f"model://{include}"


def test_meerhoven_tasks_are_direct_goal_only():
    tasks = yaml.safe_load(
        (ROOT / "src/experiments/config/tasks.yaml").read_text(encoding="utf-8")
    )["tasks"][WORLD_NAME]

    assert {task["name"] for task in tasks} == {
        "meerhoven_haul_lane_sanity",
        "meerhoven_west_to_returns_direct",
    }
    assert all("waypoints" not in task for task in tasks)
    assert all("start" in task and "goal" in task for task in tasks)


def test_meerhoven_dataset_merger_contract_covers_a_through_l():
    path = ROOT / "experiments/warehouse_layout_sketches/merge_meerhoven_yolo.py"
    spec = importlib.util.spec_from_file_location("test_meerhoven_merger", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    contracts = module.camera_contracts()

    assert list(contracts) == [f"camera_{letter}" for letter in "ABCDEFGHIJKL"]
    assert contracts["camera_A"]["camera_model"] == "external_camera"
    assert contracts["camera_L"]["labels_topic"] == "/external_camera_l/segmentation/labels_map"
