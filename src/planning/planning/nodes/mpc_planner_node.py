"""ROS 2 node for MPC-style risk-only planning (ET1, no ambiguity)."""

import rclpy

from planning.nodes.unicycle_planner_node import UnicyclePlannerNode
from planning.planners.base_planner import UnicyclePlannerBase


class MpcPlannerNode(UnicyclePlannerNode):
    NODE_NAME = 'mpc_planner'
    PLANNER_CLASS = UnicyclePlannerBase
    PARAM_DEFAULT_OVERRIDES = {
        # Notebook/Kouws-aligned MPC baseline:
        # first-order approximation, risk terms on, ambiguity off.
        'approx_method': 'ET1',
        'add_ambiguity': False,
        'use_ambiguity': False,
        'use_obs_risk': True,
    }


def main(args=None):
    rclpy.init(args=args)
    node = MpcPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
