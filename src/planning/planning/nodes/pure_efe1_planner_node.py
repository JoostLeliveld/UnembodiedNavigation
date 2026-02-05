"""ROS 2 node for pure reference EFE1 planner."""

import rclpy

from planning.nodes.pure_efe_planner_node import PureEfePlannerNodeBase
from planning.planners.pure_efe_planner import PureEfe1Planner


class PureEfe1PlannerNode(PureEfePlannerNodeBase):
    NODE_NAME = 'pure_efe1_planner'
    PLANNER_CLASS = PureEfe1Planner


def main(args=None):
    rclpy.init(args=args)
    node = PureEfe1PlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
