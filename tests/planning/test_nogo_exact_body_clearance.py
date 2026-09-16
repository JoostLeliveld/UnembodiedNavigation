"""Regression tests for the no-go cost's oriented-body clearance."""

import json
import math

import pytest

from planning.core.nogo_cost import NogoCostConfig, NogoZoneCostModel


def lane_scene():
    return json.dumps({
        "prisms": [{
            "xmin": -3.0,
            "xmax": 3.0,
            "ymin": -0.35,
            "ymax": 0.35,
            "zmin": 0.0,
            "zmax": 1.0,
        }],
    })


def test_aligned_body_uses_lateral_extent_not_longitudinal_extent():
    model = NogoZoneCostModel(NogoCostConfig(
        geometry_json=lane_scene(),
        mode="keep_in",
        weight=1.0,
        safe_distance=0.325,
        robot_half_length=0.4,
        robot_half_width=0.275,
        body_margin=0.05,
    ))

    # The exact body has 7.5 cm to the lane edge; the separate margin leaves
    # 2.5 cm.  The superseded max-axis surrogate returned -10 cm here.
    assert model.clearance_state_np([0.0, 0.0, 0.0]) == pytest.approx(0.025)


def test_sideways_body_still_fails_the_same_narrow_lane():
    model = NogoZoneCostModel(NogoCostConfig(
        geometry_json=lane_scene(),
        mode="keep_in",
        weight=1.0,
        safe_distance=0.325,
        robot_half_length=0.4,
        robot_half_width=0.275,
        body_margin=0.05,
    ))

    assert model.clearance_state_np([0.0, 0.0, math.pi / 2.0]) < 0.0


def test_point_surrogate_is_unchanged_when_no_body_is_configured():
    model = NogoZoneCostModel(NogoCostConfig(
        geometry_json=lane_scene(),
        mode="keep_in",
        weight=1.0,
        safe_distance=0.325,
    ))

    assert model.clearance_state_np([0.0, 0.0, 0.0]) == pytest.approx(0.025)
