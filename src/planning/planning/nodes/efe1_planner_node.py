"""ROS 2 node for EFE1 planner."""

import rclpy

from planning.nodes.unicycle_planner_node import UnicyclePlannerNode
from planning.planners.efe1_planner import Efe1Planner


class Efe1PlannerNode(UnicyclePlannerNode):
    NODE_NAME = 'efe1_planner'
    PLANNER_CLASS = Efe1Planner


def main(args=None):
    rclpy.init(args=args)
    node = Efe1PlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
