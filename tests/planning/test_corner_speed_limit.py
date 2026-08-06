from __future__ import annotations

import numpy as np
import pytest

from planning.nodes.efe_agent_node import (
    _ff_fb_forward_speed,
    _geometric_route_time_cost,
    _preview_corner_speed_limit,
    _route_length_from,
    _route_states,
    _tracking_waypoints,
)


def test_straight_path_keeps_one_metre_per_second_ceiling():
    path = np.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)))
    assert _preview_corner_speed_limit(
        path, 0, np.asarray((0.0, 0.0)), v_max=1.0
    ) == pytest.approx(1.0)


def test_ninety_degree_corner_triggers_braking_before_turn():
    path = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0)))
    limit = _preview_corner_speed_limit(
        path, 1, np.asarray((0.80, 0.0)), v_max=1.0
    )
    assert 0.30 < limit < 1.0


def test_corner_outside_preview_does_not_slow_early_straight():
    path = np.asarray(((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)))
    assert _preview_corner_speed_limit(
        path, 0, np.asarray((0.0, 0.0)), v_max=1.0, preview_m=0.9
    ) == pytest.approx(1.0)


def test_tracker_retains_previous_corner_as_cross_track_anchor():
    route = [(-10.0, 7.75), (-3.57, 7.75), (-3.57, 6.0)]
    path = _tracking_waypoints(route, 1, np.asarray((-8.0, 7.75)))
    assert path.tolist() == [
        [-10.0, 7.75],
        [-3.57, 7.75],
        [-3.57, 6.0],
    ]


def test_tracker_uses_current_state_before_first_route_waypoint():
    route = [(-10.0, -2.0), (-3.57, -2.0), (-3.57, 6.0)]
    path = _tracking_waypoints(route, 0, np.asarray((-10.0, -6.0)))
    assert path.tolist() == [
        [-10.0, -6.0],
        [-10.0, -2.0],
        [-3.57, -2.0],
        [-3.57, 6.0],
    ]


def test_geometric_route_length_includes_start_to_first_waypoint():
    assert _route_length_from(
        np.asarray((-10.0, -6.0)),
        [(-10.0, 2.0), (-3.5, 2.0), (-3.5, 6.0)],
    ) == pytest.approx(18.5)


def test_geometric_route_states_include_start_and_corner_headings():
    states = _route_states(
        np.asarray((0.0, 0.0, -1.0)), [(0.0, 2.0), (3.0, 2.0)]
    )
    assert states[:, :2].tolist() == [[0.0, 0.0], [0.0, 2.0], [3.0, 2.0]]
    assert states[0, 2] == pytest.approx(np.pi / 2.0)
    assert states[1, 2] == pytest.approx(0.0)


def test_ff_fb_pivots_in_place_when_badly_misaligned():
    assert _ff_fb_forward_speed(
        0.25, 0.30, -1.5, -1.49, v_max=1.0, yaw_gate_rad=0.60
    ) == 0.0


def test_ff_fb_keeps_one_metre_per_second_on_aligned_straight():
    assert _ff_fb_forward_speed(
        1.0, 1.0, 0.0, 0.0, v_max=1.0, yaw_gate_rad=0.60
    ) == pytest.approx(1.0)


def test_ff_fb_respects_arrival_or_corner_speed_cap():
    assert _ff_fb_forward_speed(
        1.0, 0.30, 0.0, 0.0, v_max=1.0, yaw_gate_rad=0.60
    ) == pytest.approx(0.30)


def test_geometric_time_cost_prefers_heading_aligned_route_over_180_pivot():
    start = np.asarray((0.0, 0.0, -np.pi / 2.0))
    short_with_uturn = [(0.0, 4.0), (4.0, 4.0), (4.0, 8.0)]
    wider_aligned_route = [(0.0, -2.0), (4.0, -2.0), (4.0, 8.0)]
    assert _route_length_from(start[:2], short_with_uturn) < _route_length_from(
        start[:2], wider_aligned_route
    )
    assert _geometric_route_time_cost(
        start, wider_aligned_route, v_max=1.0
    ) < _geometric_route_time_cost(start, short_with_uturn, v_max=1.0)
