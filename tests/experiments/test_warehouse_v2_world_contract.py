from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import yaml


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/warehouse_v2_sketches"
WORLD_DIR = ROOT / "src/sim/gazebo_worlds/worlds"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(HERE))
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(HERE))
    return module


def test_peak_and_shipout_profiles_match_internal_world_names_and_tasks():
    profiles = yaml.safe_load(
        (ROOT / "src/experiments/config/world_profiles.yaml").read_text()
    )["worlds"]
    tasks = yaml.safe_load(
        (ROOT / "src/experiments/config/tasks.yaml").read_text()
    )["tasks"]
    expected_names = {
        "warehouse_v2.world.sdf": "warehouse_v2",
        "warehouse_v2_shipout.world.sdf": "warehouse_v2_shipout",
    }
    for filename, world_name in expected_names.items():
        parsed_name = ET.parse(WORLD_DIR / filename).getroot().find("world").get("name")
        assert parsed_name == world_name
        assert profiles[filename]["world_name"] == world_name
        assert profiles[filename]["recommended_task"] in {
            task["name"] for task in tasks[filename]
        }


def test_both_worlds_and_profiles_declare_the_same_five_camera_registry():
    profiles = yaml.safe_load(
        (ROOT / "src/experiments/config/world_profiles.yaml").read_text()
    )["worlds"]
    expected_ids = [f"camera_{letter}" for letter in "ABCDE"]
    expected_models = [
        "external_camera",
        "external_camera_b",
        "external_camera_c",
        "external_camera_d",
        "external_camera_e",
    ]
    for filename in ("warehouse_v2.world.sdf", "warehouse_v2_shipout.world.sdf"):
        profile = profiles[filename]
        assert profile["camera_ids"] == expected_ids
        assert profile["camera_model_includes"] == expected_models
        world = ET.parse(WORLD_DIR / filename).getroot().find("world")
        included = {
            include.findtext("name"): include.findtext("uri")
            for include in world.findall("include")
        }
        assert {name: included[name] for name in expected_models} == {
            name: f"model://{name}" for name in expected_models
        }


def test_generated_worlds_and_freeze_manifest_are_exact():
    module = _load_module("warehouse_v2_world_freeze", HERE / "world_freeze.py")
    expected = json.loads((HERE / "world_freeze_manifest.json").read_text())
    # refreeze_history records WHY the freeze was replaced -- provenance about the
    # manifest rather than a property of the world -- so world_freeze.py excludes it
    # from its own comparison and so does this test.
    expected.pop("refreeze_history", None)
    assert module.build_manifest() == expected
    assert all(record["generator_matches_file"] for record in expected["worlds"].values())
    assert all(record["identical"] for record in expected["paired_invariants"].values())


def test_batch_mode_camera_set_comes_from_the_runtime_contract_not_a_literal():
    """No camera may be dropped without saying so, and no literal may go stale.

    This guard used to require the launch path to refuse anything but cameras A-D, from
    when the batched detector was four-camera and five-camera fusion was uncommissioned.
    The batch runtime is now contract v2 and carries warehouse_v2's five wall cameras, so a
    repeated literal here would refuse every warehouse_v2 arm instead of protecting
    anything. The property that mattered is kept: the admissible camera set is the
    perception layer's own contract, and a mismatch is refused rather than trimmed.
    """

    source = (
        ROOT / "src/experiments/experiments/core/visibility_launch_common.py"
    ).read_text()
    assert "BATCHED_CAMERA_ORDER" in source
    assert "Refusing to silently omit cameras" in source
    assert "frozen_batch_ids" not in source

    spec = importlib.util.spec_from_file_location(
        "four_camera_runtime_contract",
        ROOT / "src/perception/perception/core/four_camera_runtime_contract.py",
    )
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    profiles = yaml.safe_load(
        (ROOT / "src/experiments/config/world_profiles.yaml").read_text()
    )["worlds"]
    assert tuple(profiles["warehouse_v2.world.sdf"]["camera_ids"]) == tuple(
        contract.BATCHED_CAMERA_ORDER
    )

    bringup = (ROOT / "src/sim/launch/bringup_sim.launch.py").read_text()
    assert 'world_name = LaunchConfiguration("world_name").perform(context)' in bringup


def test_fusion_campaign_cannot_merge_adjacent_capture_rounds():
    campaign = yaml.safe_load((
        ROOT / "scripts/visibility_comparison/fusion_on_fixed_routes_campaign.yaml"
    ).read_text())
    camera_period_s = 0.20
    assert 0.0 <= campaign["yolo_max_batch_stamp_skew_s"] < camera_period_s
    assert campaign["manager_fusion_max_timestamp_spread_s"] < camera_period_s
    assert campaign["require_state_correction_envelope"] is True
    assert campaign["stale_belief_inflate_m2_per_s"] == 0.0
    assert campaign["stale_belief_inflate_cap_m2"] == 0.0
