"""ROS 2 node for EFER risk-only planning (ET2, no ambiguity)."""

import rclpy

from planning.nodes.unicycle_planner_node import UnicyclePlannerNode
from planning.planners.base_planner import UnicyclePlannerBase


class EferPlannerNode(UnicyclePlannerNode):
    NODE_NAME = 'efer_planner'
    PLANNER_CLASS = UnicyclePlannerBase
    PARAM_DEFAULT_OVERRIDES = {
        # Notebook/Kouws-aligned EFER:
        # second-order transform, risk terms on, ambiguity off.
        'approx_method': 'ET2',
        'add_ambiguity': False,
        'use_ambiguity': False,
        'use_obs_risk': True,
    }


def main(args=None):
    rclpy.init(args=args)
    node = EferPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
