"""ROS 2 node for EFE agent that publishes cmd_vel directly (unicycle dynamics)."""

import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist

from planning.nodes.unicycle_planner_node import UnicyclePlannerNode
from planning.planners.base_planner import UnicyclePlannerBase


class EfeAgentNode(UnicyclePlannerNode):
    NODE_NAME = 'efe_agent'
    PLANNER_CLASS = UnicyclePlannerBase

    def __init__(self):
        super().__init__()

        if not self.has_parameter('cmd_topic'):
            self.declare_parameter('cmd_topic', '/cmd_vel')
        self.cmd_topic = self.get_parameter('cmd_topic').value
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)

    def _after_plan_result(self, result):
        # Agent mode extends the shared planner behavior with direct command output.
        cmd = Twist()
        cmd.linear.x = float(result.controls[0, 0])
        cmd.angular.z = float(result.controls[0, 1])
        self.cmd_pub.publish(cmd)
        self.last_cmd = np.array([cmd.linear.x, cmd.angular.z], dtype=float)

    def _publish_safe_stop_command(self):
        if not hasattr(self, 'cmd_pub'):
            return
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)
        self.last_cmd = np.array([0.0, 0.0], dtype=float)


def main(args=None):
    rclpy.init(args=args)
    node = EfeAgentNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
