"""ROS 2 node for configurable EFE planner (ET1/ET2)."""

import rclpy

from planning.nodes.unicycle_planner_node import UnicyclePlannerNode
from planning.planners.base_planner import UnicyclePlannerBase


class EfePlannerNode(UnicyclePlannerNode):
    NODE_NAME = 'efe_planner'
    PLANNER_CLASS = UnicyclePlannerBase
    PARAM_DEFAULT_OVERRIDES = {
        # Canonical EFE defaults; launch files can override for EFE1/ablation runs.
        'approx_method': 'ET2',
        'add_ambiguity': True,
        'use_ambiguity': True,
        'use_obs_risk': True,
    }


def main(args=None):
    rclpy.init(args=args)
    node = EfePlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
