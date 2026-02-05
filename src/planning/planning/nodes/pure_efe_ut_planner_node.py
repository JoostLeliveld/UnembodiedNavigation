"""ROS 2 node for pure reference EFE-UT planner."""

import rclpy

from planning.nodes.pure_efe_planner_node import PureEfePlannerNodeBase
from planning.planners.pure_efe_planner import PureEfeUtPlanner


class PureEfeUtPlannerNode(PureEfePlannerNodeBase):
    NODE_NAME = 'pure_efe_ut_planner'
    PLANNER_CLASS = PureEfeUtPlanner


def main(args=None):
    rclpy.init(args=args)
    node = PureEfeUtPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
