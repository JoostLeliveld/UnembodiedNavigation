"""Shared geometry and label conventions for the front/rear pose-keypoint pipeline.

These offsets must match the `pm_*` properties in the URDF of whichever robot
is being driven -- `warehouse_amr.urdf.xacro` (current) or
`turtlebot3_burger.urdf.xacro` (pre-2026-08-20 campaigns). The
`test_pose_keypoint_geometry.py` test asserts both.

Two keypoint indices are used by both the dataset capture script, the YOLO
training YAML, the perception node, and the state node:

  KEYPOINT_FRONT = 0   # +x in robot frame (deck front, above the wheelbase)
  KEYPOINT_REAR  = 1   # -x in robot frame (deck rear, above the wheelbase)
"""

from __future__ import annotations

from dataclasses import dataclass


KEYPOINT_FRONT = 0
KEYPOINT_REAR = 1
NUM_KEYPOINTS = 2

# Visibility flag values in the YOLO pose label format.
KPT_VIS_NOT_LABELED = 0.0
KPT_VIS_OCCLUDED = 1.0
KPT_VIS_VISIBLE = 2.0


@dataclass(frozen=True)
class MarkerGeometry:
    """Keypoint positions in the base_link frame (metres).

    Each keypoint is the centroid of the visible TOP surface of a coloured
    panel mounted on the robot's top plate. The panel geometry (size, plate
    dimensions) is documented in the URDF only — projection and back-projection
    only need the centroid (x, y, z) and the base_joint Z offset.
    """

    front_x: float
    rear_x: float
    marker_y: float
    marker_z_in_base_link: float
    base_joint_z: float

    def front_in_base_link(self) -> tuple[float, float, float]:
        return (self.front_x, self.marker_y, self.marker_z_in_base_link)

    def rear_in_base_link(self) -> tuple[float, float, float]:
        return (self.rear_x, self.marker_y, self.marker_z_in_base_link)

    def world_z(self, robot_footprint_z: float) -> float:
        return float(robot_footprint_z) + self.base_joint_z + self.marker_z_in_base_link


# Default values must match the driven robot's URDF pm_* properties.
#
# warehouse_amr (current, 2026-08-20): low-deck 0.80 x 0.55 m AMR. Keypoints sit
# at the deck edges above the wheelbase, a 0.600 m baseline against the Burger's
# 0.140 m. keypoint_marker_world_z = base_joint_z(0.010) + 0.340 = 0.350 with the
# robot resting on the floor. No marker disks: the cues are the chassis itself.
DEFAULT_MARKER_GEOMETRY = MarkerGeometry(
    front_x=0.300,
    rear_x=-0.300,
    marker_y=0.0,
    marker_z_in_base_link=0.340,
    base_joint_z=0.010,
)

# turtlebot3_burger (pre-2026-08-20): explicit cyan(front)/magenta(rear) disks at
# z=0.200 in base_link, world plane 0.210. Kept so captures, trained models and
# analyses from before the robot swap stay reproducible -- read
# `keypoint_marker_world_z` from a capture manifest rather than assuming either.
BURGER_MARKER_GEOMETRY = MarkerGeometry(
    front_x=0.040,
    rear_x=-0.100,
    marker_y=0.0,
    marker_z_in_base_link=0.200,
    base_joint_z=0.010,
)
