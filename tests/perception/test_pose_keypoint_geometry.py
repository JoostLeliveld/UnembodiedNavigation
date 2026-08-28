"""Sanity-check the marker geometry against the URDF source-of-truth."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PERCEPTION_PKG = REPO_ROOT / 'src' / 'perception'
URDF_DIR = REPO_ROOT / 'src' / 'sim' / 'robot_description' / 'urdf'
URDF_PATH = URDF_DIR / 'warehouse_amr.urdf.xacro'
BURGER_URDF_PATH = URDF_DIR / 'turtlebot3_burger.urdf.xacro'

if str(PERCEPTION_PKG) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_PKG))

from perception.core.pose_keypoints import (
    BURGER_MARKER_GEOMETRY,
    DEFAULT_MARKER_GEOMETRY,
    KEYPOINT_FRONT,
    KEYPOINT_REAR,
    NUM_KEYPOINTS,
)


def _read_xacro_property(name: str, path=URDF_PATH) -> float:
    text = path.read_text(encoding='utf-8')
    pattern = re.compile(
        rf'<xacro:property\s+name="{re.escape(name)}"\s+value="(?P<value>[-+0-9.eE]+)"',
    )
    match = pattern.search(text)
    if match is None:
        raise AssertionError(f'xacro property "{name}" not found in {path}')
    return float(match.group('value'))


def test_python_marker_offsets_match_urdf() -> None:
    g = DEFAULT_MARKER_GEOMETRY
    assert g.front_x == _read_xacro_property('pm_front_x')
    assert g.rear_x == _read_xacro_property('pm_rear_x')
    assert g.marker_y == _read_xacro_property('pm_marker_y')
    assert g.marker_z_in_base_link == _read_xacro_property('pm_marker_z')


def test_burger_offsets_match_its_own_urdf() -> None:
    """Pre-2026-08-20 captures and models used the Burger. Its geometry must stay
    exactly reproducible, so it is pinned against its own URDF, not the AMR's."""
    g = BURGER_MARKER_GEOMETRY
    assert g.front_x == _read_xacro_property('pm_front_x', BURGER_URDF_PATH)
    assert g.rear_x == _read_xacro_property('pm_rear_x', BURGER_URDF_PATH)
    assert g.marker_y == _read_xacro_property('pm_marker_y', BURGER_URDF_PATH)
    assert g.marker_z_in_base_link == _read_xacro_property('pm_marker_z', BURGER_URDF_PATH)
    assert abs(g.world_z(0.0) - 0.210) < 1e-9


def test_amr_marker_plane_and_baseline() -> None:
    """The warehouse AMR reads at a 0.350 m plane on a 0.600 m baseline. Both
    numbers are quoted in analyses, so pin them here."""
    g = DEFAULT_MARKER_GEOMETRY
    assert abs(g.world_z(0.0) - 0.350) < 1e-9
    assert abs((g.front_x - g.rear_x) - 0.600) < 1e-9


def test_keypoints_lie_within_the_chassis() -> None:
    """A virtual keypoint anchored off the body would be unlearnable. Both sit
    inside the 0.800 x 0.550 m deck, on its centreline."""
    g = DEFAULT_MARKER_GEOMETRY
    body_l = _read_xacro_property('body_l')
    assert abs(g.front_x) <= body_l / 2
    assert abs(g.rear_x) <= body_l / 2
    assert g.marker_y == 0.0


def test_keypoint_separation_at_least_8cm() -> None:
    """Separation must be enough to give usable yaw resolution (1 px keypoint
    error -> a few degrees of yaw error)."""
    g = DEFAULT_MARKER_GEOMETRY
    assert (g.front_x - g.rear_x) >= 0.08


def test_keypoint_index_constants() -> None:
    assert KEYPOINT_FRONT == 0
    assert KEYPOINT_REAR == 1
    assert NUM_KEYPOINTS == 2


def test_world_z_combines_offsets() -> None:
    g = DEFAULT_MARKER_GEOMETRY
    z = g.world_z(robot_footprint_z=0.05)
    assert abs(z - (0.05 + g.base_joint_z + g.marker_z_in_base_link)) < 1e-12


def test_front_offset_strictly_ahead_of_rear() -> None:
    g = DEFAULT_MARKER_GEOMETRY
    assert g.front_x > g.rear_x
    assert (g.front_x - g.rear_x) >= 0.05  # at least 5 cm separation
