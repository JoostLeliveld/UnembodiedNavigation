"""ROS 2 node for EFE2 planner."""

import rclpy

from planning.nodes.unicycle_planner_node import UnicyclePlannerNode
from planning.planners.efe2_planner import Efe2Planner


class Efe2PlannerNode(UnicyclePlannerNode):
    NODE_NAME = 'efe2_planner'
    PLANNER_CLASS = Efe2Planner


def main(args=None):
    rclpy.init(args=args)
    node = Efe2PlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
