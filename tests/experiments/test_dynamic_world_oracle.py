"""Guards for the dynamic world and its visibility oracle.

Fast checks only — none of these start Gazebo. The end-to-end acceptance run
lives in ``experiments/dynamic_world_oracle/verify_acceptance.py``; what is
enforced here is the part that must never regress silently: the generated
assets stay consistent, the scenario loader refuses impossible timelines, the
fast ray cast still agrees with the repo's scalar primitive, and no runtime
package has started reading ground truth.
"""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / "experiments" / "dynamic_world_oracle"
WORLDS = REPO / "src" / "sim" / "gazebo_worlds" / "worlds"
STATIC_WORLD = WORLDS / "warehouse_full_4cam.world.sdf"
DYNAMIC_WORLD = WORLDS / "warehouse_full_4cam_dynamic.world.sdf"
STAGE_JSON = WORLDS / "warehouse_full_4cam_dynamic.stage.json"

sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(REPO / "scripts" / "shared"))

import oracle as ora  # noqa: E402
from scenario import ScenarioError, load_scenario  # noqa: E402

from unav_common.occlusion_geometry import segment_occluded  # noqa: E402


@pytest.fixture(scope="module")
def stage() -> dict:
    import json
    return json.loads(STAGE_JSON.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- generated assets
def test_dynamic_world_is_the_flagship_world_under_a_different_name():
    """Only the world name and one header comment may differ, so runs stay comparable.

    If the two worlds ever diverge structurally, a dynamic-scenario result stops
    being comparable with a static four-camera result, which is the whole reason
    the variant is a rename rather than a redesign.
    """
    static = STATIC_WORLD.read_text(encoding="utf-8")
    dynamic = DYNAMIC_WORLD.read_text(encoding="utf-8")

    lines = dynamic.splitlines()
    starts = [i for i, line in enumerate(lines)
              if line.lstrip().startswith("<!-- warehouse_full_4cam_dynamic:")]
    assert len(starts) == 1, "expected exactly one injected header comment"
    start = starts[0]
    end = next(i for i in range(start, len(lines)) if lines[i].rstrip().endswith("-->"))
    without_header = "\n".join(lines[:start] + lines[end + 1:]) + "\n"

    assert '<world name="warehouse_full_4cam_dynamic">' in dynamic
    assert without_header.replace("warehouse_full_4cam_dynamic", "warehouse_full_4cam") == static


def test_generated_assets_are_well_formed():
    for path in (DYNAMIC_WORLD,
                 REPO / "src/sim/models/dyn_pallet_box/model.sdf",
                 REPO / "src/sim/models/dyn_pallet_box/model.config",
                 REPO / "src/sim/models/dyn_forklift/model.sdf",
                 REPO / "src/sim/models/dyn_forklift/model.config"):
        assert path.exists(), f"{path} is missing; run make_warehouse_full.py --variant dynamic"
        ET.parse(path)  # raises on malformed XML


def test_obstacles_hold_a_commanded_pose():
    """static=false so gz accepts pose commands; gravity=false so it never drifts."""
    for name in ("dyn_pallet_box", "dyn_forklift"):
        root = ET.parse(REPO / f"src/sim/models/{name}/model.sdf").getroot()
        model = root.find("model")
        assert model is not None
        assert (model.findtext("static") or "").strip() == "false", name
        link = model.find("link")
        assert link is not None
        assert (link.findtext("gravity") or "").strip() == "false", name


def test_stage_cameras_match_the_world(stage):
    """The contract's extrinsics must be the poses the world actually places."""
    root = ET.parse(DYNAMIC_WORLD).getroot()
    poses = {}
    for include in root.findall(".//include"):
        name = (include.findtext("name") or "").strip()
        pose = include.findtext("pose")
        if name and pose:
            poses[name] = [float(v) for v in pose.split()]
    assert len(stage["cameras"]) == 4
    for camera in stage["cameras"]:
        assert camera["camera_id"] in poses, camera["camera_id"]
        assert poses[camera["camera_id"]] == pytest.approx(
            camera["extrinsics"]["sdf_pose_xyz_rpy"])


def test_scenario_aisle_is_clear_of_static_geometry(stage):
    aisle = stage["scenario_stage_aisle"]
    scene = ora.OracleScene.from_world(DYNAMIC_WORLD, [c["camera_id"] for c in stage["cameras"]])
    intruding = [
        prism.name for prism in scene.static_prisms
        if prism.xmin < aisle["xmax"] and prism.xmax > aisle["xmin"]
        and prism.ymin < aisle["ymax"] and prism.ymax > aisle["ymin"]
    ]
    assert not intruding, f"scenario aisle is not clear: {intruding}"


# ------------------------------------------------------------------ scenario spec
def test_shipped_scenarios_load():
    for path in sorted((STUDY / "scenarios").glob("*.yaml")):
        scenario = load_scenario(path)
        assert scenario.capture_times
        assert scenario.events
        for t in scenario.capture_times:
            steps = t / scenario.tick_s
            assert abs(steps - round(steps)) < 1e-9, f"{path}: capture {t} is off the sensor tick"


def test_scenarios_bracket_every_event():
    """Without a capture either side of an event, its timing cannot be checked."""
    for path in sorted((STUDY / "scenarios").glob("*.yaml")):
        scenario = load_scenario(path)
        for event in scenario.events:
            assert any(t < event.t for t in scenario.capture_times), f"{path}: {event.kind}"
            assert any(t > event.t for t in scenario.capture_times), f"{path}: {event.kind}"


def test_loader_rejects_a_move_before_its_spawn(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "scenario_id: bad\nworld_file: w.sdf\nworld_name: w\nduration_s: 2.0\n"
        "events:\n  - {t: 0.4, kind: move, entity: ghost, pose: {x: 0, y: 0}}\n"
        "captures: {start_t: 0.2, period_s: 0.2, end_t: 1.0}\n",
        encoding="utf-8")
    with pytest.raises(ScenarioError, match="not spawned"):
        load_scenario(path)


def test_loader_rejects_a_tick_that_is_not_whole_physics_steps(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "scenario_id: bad\nworld_file: w.sdf\nworld_name: w\ntick_s: 0.00015\n"
        "events: []\ncaptures: {start_t: 0.0, period_s: 0.00015, end_t: 0.0003}\n",
        encoding="utf-8")
    with pytest.raises(ScenarioError, match="physics steps"):
        load_scenario(path)


# ------------------------------------------------------------------------- oracle
def test_placing_an_obstacle_is_exact_on_quarter_turns_and_conservative_otherwise():
    parts = ora.parts_from_extent((2.0, 1.0, 1.5))

    at_zero = ora.place_obstacle("e", "m", parts, {"x": 0.0, "y": 0.0, "yaw": 0.0}).aabb
    assert at_zero.xmax - at_zero.xmin == pytest.approx(2.0)
    assert at_zero.ymax - at_zero.ymin == pytest.approx(1.0)
    assert at_zero.zmin == pytest.approx(0.0)
    assert at_zero.zmax == pytest.approx(1.5)

    quarter = ora.place_obstacle("e", "m", parts, {"x": 0.0, "y": 0.0, "yaw": np.pi / 2})
    assert quarter.aabb_is_exact
    assert quarter.aabb.xmax - quarter.aabb.xmin == pytest.approx(1.0)
    assert quarter.aabb.ymax - quarter.aabb.ymin == pytest.approx(2.0)

    oblique = ora.place_obstacle("e", "m", parts, {"x": 0.0, "y": 0.0, "yaw": 0.5})
    assert not oblique.aabb_is_exact
    assert oblique.aabb.xmax - oblique.aabb.xmin > 1.0


def test_obstacle_geometry_comes_from_its_collision_parts_not_one_box():
    """A single bounding box over-occludes: the load is narrower than the pallet base."""
    parts = ora.parts_from_model_sdf(REPO / "src/sim/models/dyn_pallet_box/model.sdf",
                                     "dyn_pallet_box")
    assert len(parts) == 2, "pallet should be a base plus a narrower load"
    base = max(parts, key=lambda p: p.xmax - p.xmin)
    load = min(parts, key=lambda p: p.xmax - p.xmin)
    assert (load.xmax - load.xmin) < (base.xmax - base.xmin)
    assert load.zmin == pytest.approx(base.zmax), "the load must sit on the base"

    forklift = ora.parts_from_model_sdf(REPO / "src/sim/models/dyn_forklift/model.sdf",
                                        "dyn_forklift")
    assert len(forklift) >= 5, "forklift should be several parts, not one slab"
    bound = ora.place_obstacle("f", "dyn_forklift", forklift, {"x": 0, "y": 0, "yaw": 0}).aabb
    solid = sum((p.xmax - p.xmin) * (p.ymax - p.ymin) * (p.zmax - p.zmin) for p in forklift)
    envelope = ((bound.xmax - bound.xmin) * (bound.ymax - bound.ymin)
                * (bound.zmax - bound.zmin))
    assert solid < 0.5 * envelope, "a forklift is mostly air; one box would claim it is solid"


def test_vectorised_ray_cast_agrees_with_the_repo_primitive(stage):
    """oracle.segments_hit_any_prism is a speed twin of segment_occluded, not a fork."""
    camera_ids = [c["camera_id"] for c in stage["cameras"]]
    scene = ora.OracleScene.from_world(DYNAMIC_WORLD, camera_ids)
    bounds = stage["site_bounds"]
    grid = ora.FloorGrid(bounds["xmin"], bounds["xmax"], bounds["ymin"], bounds["ymax"], 0.5)
    points = grid.points_at_height(0.35)
    rng = np.random.default_rng(20260811)
    index = rng.choice(points.shape[0], 250, replace=False)
    camera = scene.cameras[camera_ids[0]]
    fast = ora.segments_hit_any_prism(camera.cam_pos, points[index], scene.static_prisms)
    slow = np.array([segment_occluded(scene.static_prisms, camera.cam_pos, p)
                     for p in points[index]])
    assert (fast == slow).all()


def test_an_obstacle_hides_some_cells_and_leaves_others(stage):
    """The s01 spawn pose must cost camera A cells and cost the others none."""
    camera_ids = [c["camera_id"] for c in stage["cameras"]]
    scene = ora.OracleScene.from_world(DYNAMIC_WORLD, camera_ids)
    bounds = stage["site_bounds"]
    grid = ora.FloorGrid(bounds["xmin"], bounds["xmax"], bounds["ymin"], bounds["ymax"], 0.25)
    baseline = ora.visibility_grids(scene.cameras, grid, scene.static_prisms)

    scenario = load_scenario(STUDY / "scenarios" / "s01_box_in_aisle.yaml")
    spawn = next(e for e in scenario.events if e.kind == "spawn")
    model_sdf = next(o["model_sdf"] for o in stage["obstacle_catalogue"]
                     if o["model_name"] == spawn.model)
    parts = ora.parts_from_model_sdf(REPO / model_sdf, spawn.model)
    box = ora.place_obstacle(spawn.entity, spawn.model, parts, spawn.pose)
    occluded = ora.visibility_grids(scene.cameras, grid, scene.static_prisms, box.prisms)

    lost = {c: int((baseline[c] == ora.VISIBLE).sum()) - int((occluded[c] == ora.VISIBLE).sum())
            for c in camera_ids}
    selected = [c for c, n in lost.items() if n > 0]
    assert len(selected) == 1, f"expected exactly one affected camera, got {lost}"
    assert lost[selected[0]] > 0
    assert int((occluded[selected[0]] == ora.VISIBLE).sum()) > lost[selected[0]], \
        "the obstacle should hide a minority of the camera's floor, not all of it"


def test_removing_the_obstacle_restores_the_map_exactly(stage):
    camera_ids = [c["camera_id"] for c in stage["cameras"]]
    scene = ora.OracleScene.from_world(DYNAMIC_WORLD, camera_ids)
    bounds = stage["site_bounds"]
    grid = ora.FloorGrid(bounds["xmin"], bounds["xmax"], bounds["ymin"], bounds["ymax"], 0.5)
    before = ora.visibility_grids(scene.cameras, grid, scene.static_prisms)
    parts = ora.parts_from_model_sdf(REPO / "src/sim/models/dyn_pallet_box/model.sdf",
                                     "dyn_pallet_box")
    box = ora.place_obstacle("b", "dyn_pallet_box", parts,
                             {"x": -5.675, "y": -4.5, "yaw": np.pi / 2})
    during = ora.visibility_grids(scene.cameras, grid, scene.static_prisms, box.prisms)
    after = ora.visibility_grids(scene.cameras, grid, scene.static_prisms, [])
    assert any(not (before[c] == during[c]).all() for c in camera_ids)
    for c in camera_ids:
        assert (before[c] == after[c]).all()


# ----------------------------------------------------------------------- boundary
@pytest.mark.parametrize("needle", ["dynamic_world_oracle", "oracle_visibility", "oracle_depth"])
def test_no_runtime_package_reads_the_oracle(needle):
    """Ground truth is evaluation-only; src/ must never reference it."""
    result = subprocess.run(
        ["grep", "-rln", "--include=*.py", "--include=*.yaml", "--include=*.launch.py",
         needle, str(REPO / "src")],
        capture_output=True, text=True,
    )
    hits = [line for line in result.stdout.splitlines() if line.strip()]
    assert not hits, f"runtime files reference {needle!r}: {hits}"
