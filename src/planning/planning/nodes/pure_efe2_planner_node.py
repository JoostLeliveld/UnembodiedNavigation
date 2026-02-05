"""ROS 2 node for pure reference EFE2 planner."""

import rclpy

from planning.nodes.pure_efe_planner_node import PureEfePlannerNodeBase
from planning.planners.pure_efe_planner import PureEfe2Planner


class PureEfe2PlannerNode(PureEfePlannerNodeBase):
    NODE_NAME = 'pure_efe2_planner'
    PLANNER_CLASS = PureEfe2Planner


def main(args=None):
    rclpy.init(args=args)
    node = PureEfe2PlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
