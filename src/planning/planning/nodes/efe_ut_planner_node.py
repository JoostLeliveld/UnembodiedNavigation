"""ROS 2 node for EFE-UT planner."""

import rclpy

from planning.nodes.unicycle_planner_node import UnicyclePlannerNode
from planning.planners.efe_ut_planner import EfeUtPlanner


class EfeUtPlannerNode(UnicyclePlannerNode):
    NODE_NAME = 'efe_ut_planner'
    PLANNER_CLASS = EfeUtPlanner


def main(args=None):
    rclpy.init(args=args)
    node = EfeUtPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
