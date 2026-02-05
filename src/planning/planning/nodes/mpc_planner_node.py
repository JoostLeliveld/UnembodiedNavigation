"""ROS 2 node for MPC-style planner."""

import rclpy

from planning.nodes.unicycle_planner_node import UnicyclePlannerNode
from planning.planners.mpc_planner import MpcPlanner


class MpcPlannerNode(UnicyclePlannerNode):
    NODE_NAME = 'mpc_planner'
    PLANNER_CLASS = MpcPlanner


def main(args=None):
    rclpy.init(args=args)
    node = MpcPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
