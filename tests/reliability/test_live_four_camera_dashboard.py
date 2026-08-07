from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import time

import numpy as np
import pytest


pytest.importorskip("cv_bridge")

ROOT = Path(__file__).resolve().parents[2]
RELIABILITY = str(ROOT / "src/reliability")
if RELIABILITY not in sys.path:
    sys.path.insert(0, RELIABILITY)


def _dashboard_module():
    path = ROOT / "scripts/perception/live_four_camera_dashboard.py"
    spec = importlib.util.spec_from_file_location("live_four_camera_dashboard", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _renderer(module):
    return module.DashboardRenderer(
        ROOT / "logs/visibility_comparison/spawn_grid_20260727/fused_planner_four_camera.npz",
        ROOT / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf",
        (-10.0, -6.0),
        (-3.57, 6.0),
        -math.pi / 2.0,
        ROOT / "logs/visibility_comparison/spawn_grid_20260727/gp",
    )


def test_nearest_timestamped_requires_a_sync_match():
    module = _dashboard_module()
    rows = [(9.8, "old"), (10.03, "aligned"), (10.6, "future")]
    assert module._nearest_timestamped(rows, 10.0, 0.1)[1] == "aligned"
    assert module._nearest_timestamped(rows, 11.0, 0.1) is None


def test_gp_query_fails_closed_outside_fitted_domain():
    module = _dashboard_module()
    renderer = _renderer(module)
    assert math.isnan(renderer._gp_value("camera_A", (999.0, 999.0)))


def test_dashboard_map_uses_effective_driveable_union_and_sdf_obstacles():
    module = _dashboard_module()
    renderer = _renderer(module)

    def at(mask, x, y):
        ix = int(np.argmin(np.abs(renderer.xs - x)))
        iy = len(renderer.ys) - 1 - int(np.argmin(np.abs(renderer.ys - y)))
        return bool(mask[iy, ix])

    assert at(renderer.site_mask, 11.4, 0.0)
    assert not at(renderer.site_mask, -11.4, 0.0)
    assert at(renderer.driveable_mask, 1.0, 0.0)

    # The pillar comes from current collision geometry. Its exact footprint is
    # a dark obstacle and its larger safety envelope is subtracted from the
    # driveable union.
    assert at(renderer.obstacle_mask, 0.0, -0.9)
    assert at(renderer.nogo_mask, 0.5, -0.9)
    assert not at(renderer.driveable_mask, 0.5, -0.9)
    # The old shelf-end crates occupied this through-aisle point and are gone.
    assert not at(renderer.obstacle_mask, -8.825, -6.85)
    assert at(renderer.driveable_mask, -8.825, -6.85)


def test_dashboard_error_is_fused_belief_against_synchronized_ground_truth():
    module = _dashboard_module()
    renderer = _renderer(module)
    now = time.monotonic()
    covariance = np.asarray(((0.04, 0.0), (0.0, 0.09)), dtype=float)
    belief = (10.0, 1.0, 2.0, 0.0, covariance)
    truth = (10.0, 1.3, 2.4, 0.0)
    snapshot = {
        "frames": {cam: None for cam in module.CAMERAS},
        "frame_wall": {cam: 0.0 for cam in module.CAMERAS},
        "frame_sync_delta": {cam: math.inf for cam in module.CAMERAS},
        "obs": {cam: {} for cam in module.CAMERAS},
        "obs_wall": {cam: 0.0 for cam in module.CAMERAS},
        "histories": {cam: [] for cam in module.CAMERAS},
        "odom": (0.0, 0.0, 0.0),
        "odom_noisy": None,
        "odom_trail": [],
        "ground_truth": truth,
        "ground_truth_wall": now,
        "ground_truth_history": [truth],
        "belief": belief,
        "belief_wall": now,
        "belief_history": [belief],
        "state_update": None,
        "mission_start": (-10.0, -6.0, -math.pi / 2.0),
        "goal": (-3.57, 6.0),
        "plan": [(-10.0, -6.0), (-10.0, -7.5), (-3.57, -7.5)],
    }

    frame = renderer.render(snapshot)

    assert frame.shape == (1080, 1920, 3)
    assert renderer.belief_error_history[-1][1] == pytest.approx(0.5)
    assert renderer.belief_error_history[-1][2] == pytest.approx(0.3)


def test_dashboard_error_pairs_fresh_pose_arrivals_when_header_clocks_differ():
    module = _dashboard_module()
    renderer = _renderer(module)
    now = time.monotonic()
    covariance = np.asarray(((0.01, 0.0), (0.0, 0.04)), dtype=float)
    belief = (10.0, 1.0, 2.0, 0.0, covariance)
    truth = (1000.0, 1.3, 2.4, 0.0)
    snapshot = {
        "frames": {cam: None for cam in module.CAMERAS},
        "frame_wall": {cam: 0.0 for cam in module.CAMERAS},
        "frame_sync_delta": {cam: math.inf for cam in module.CAMERAS},
        "obs": {cam: {} for cam in module.CAMERAS},
        "obs_wall": {cam: 0.0 for cam in module.CAMERAS},
        "histories": {cam: [] for cam in module.CAMERAS},
        "odom": None,
        "odom_noisy": None,
        "odom_trail": [],
        "ground_truth": truth,
        "ground_truth_wall": now,
        "ground_truth_history": [truth],
        "belief": belief,
        "belief_wall": now + 0.05,
        "belief_history": [belief],
        "state_update": None,
        "mission_start": (-10.0, -6.0, -math.pi / 2.0),
        "goal": (-3.57, 6.0),
        "plan": [],
    }

    renderer.render(snapshot)

    assert renderer.belief_error_history[-1][1] == pytest.approx(0.5)
    assert renderer.belief_error_history[-1][2] == pytest.approx(0.2)
