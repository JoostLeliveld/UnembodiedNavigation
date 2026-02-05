"""ROS 2 node for pure reference MPC-style planner."""

import rclpy

from planning.nodes.pure_efe_planner_node import PureEfePlannerNodeBase
from planning.planners.pure_efe_planner import PureMpcPlanner


class PureMpcPlannerNode(PureEfePlannerNodeBase):
    NODE_NAME = 'pure_mpc_planner'
    PLANNER_CLASS = PureMpcPlanner


def main(args=None):
    rclpy.init(args=args)
    node = PureMpcPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
