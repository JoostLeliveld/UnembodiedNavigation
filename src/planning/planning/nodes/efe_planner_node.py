"""ROS 2 node for configurable EFE planner (ET1/ET2)."""

import rclpy

from planning.nodes.unicycle_planner_node import UnicyclePlannerNode
from planning.planners.base_planner import UnicyclePlannerBase


class EfePlannerNode(UnicyclePlannerNode):
    NODE_NAME = 'efe_planner'
    PLANNER_CLASS = UnicyclePlannerBase


def main(args=None):
    rclpy.init(args=args)
    node = EfePlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
