"""Shared geometry and label conventions for the front/rear pose-keypoint pipeline.

These offsets must match the `pose_markers` block in
`src/sim/robot_description/urdf/turtlebot3_burger.urdf.xacro`. The
`test_pose_keypoint_geometry.py` test asserts that.

Two keypoint indices are used by both the dataset capture script, the YOLO
training YAML, the perception node, and the state node:

  KEYPOINT_FRONT = 0   # red disk, +x in robot frame
  KEYPOINT_REAR  = 1   # blue disk, -x in robot frame
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


# Default values must match turtlebot3_burger.urdf.xacro pm_* properties.
# Keypoints are anchored to the existing red base + blue lidar geometry — no
# extra markers are added to the URDF. Both keypoints sit at the lidar top
# (z=0.188 in base_link) so the state node can back-project to a single
# z-plane without bias.
#   rear_x  = lidar centre x (-0.032 in base_link)
#   front_x = virtual point ~0.082 m forward of lidar centre
DEFAULT_MARKER_GEOMETRY = MarkerGeometry(
    front_x=0.050,
    rear_x=-0.032,
    marker_y=0.0,
    marker_z_in_base_link=0.188,
    base_joint_z=0.010,
)
