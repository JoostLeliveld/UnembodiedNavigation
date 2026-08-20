"""Sanity-check the marker geometry against the URDF source-of-truth."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PERCEPTION_PKG = REPO_ROOT / 'src' / 'perception'
URDF_PATH = REPO_ROOT / 'src' / 'sim' / 'robot_description' / 'urdf' / 'turtlebot3_burger.urdf.xacro'

if str(PERCEPTION_PKG) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_PKG))

from perception.core.pose_keypoints import (
    DEFAULT_MARKER_GEOMETRY,
    KEYPOINT_FRONT,
    KEYPOINT_REAR,
    NUM_KEYPOINTS,
)


def _read_xacro_property(name: str) -> float:
    text = URDF_PATH.read_text(encoding='utf-8')
    pattern = re.compile(
        rf'<xacro:property\s+name="{re.escape(name)}"\s+value="(?P<value>[-+0-9.eE]+)"',
    )
    match = pattern.search(text)
    if match is None:
        raise AssertionError(f'xacro property "{name}" not found in {URDF_PATH}')
    return float(match.group('value'))


def test_python_marker_offsets_match_urdf() -> None:
    g = DEFAULT_MARKER_GEOMETRY
    assert g.front_x == _read_xacro_property('pm_front_x')
    assert g.rear_x == _read_xacro_property('pm_rear_x')
    assert g.marker_y == _read_xacro_property('pm_marker_y')
    assert g.marker_z_in_base_link == _read_xacro_property('pm_marker_z')


def test_v2_marker_disk_geometry() -> None:
    """The v2 pose-marker anchors are explicit disks, not the old lidar-centre
    virtual feature. Keep this aligned with the xacro pm_* source of truth."""
    g = DEFAULT_MARKER_GEOMETRY
    assert abs(g.front_x - 0.040) < 1e-9
    assert abs(g.rear_x - (-0.100)) < 1e-9
    assert abs(g.marker_z_in_base_link - 0.200) < 1e-9


def test_keypoint_separation_at_least_8cm() -> None:
    """With existing geometry separation must be enough to give usable yaw
    resolution (1 px localisation -> a few degrees of yaw error)."""
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
