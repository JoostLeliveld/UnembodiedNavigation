"""ROS 2 node for EFE agent that publishes cmd_vel directly (unicycle dynamics)."""

import numpy as np
import time

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
        self._active_controls = None
        self._active_plan_started_at = None
        self._cmd_timer_period_s = max(0.02, min(self.dt, 0.1))
        self._cmd_timer = self.create_timer(
            self._cmd_timer_period_s,
            self._publish_active_plan_command,
            callback_group=self._io_group,
        )

    def _publish_command(self, v_cmd: float, w_cmd: float):
        cmd = Twist()
        cmd.linear.x = float(v_cmd)
        cmd.angular.z = float(w_cmd)
        self.cmd_pub.publish(cmd)
        self.last_cmd = np.array([cmd.linear.x, cmd.angular.z], dtype=float)

    def _publish_active_plan_command(self):
        with self._data_lock:
            controls = None if self._active_controls is None else self._active_controls.copy()
            started_at = self._active_plan_started_at
        if controls is None or controls.size == 0 or started_at is None:
            return

        elapsed_s = max(time.monotonic() - started_at, 0.0)
        step_dt = max(float(self.dt), 1e-3)
        step_idx = min(int(elapsed_s / step_dt), controls.shape[0] - 1)
        u = controls[step_idx]
        self._publish_command(u[0], u[1])

    def _after_plan_result(self, result):
        # Keep following the current planned control sequence until replanning replaces it.
        controls = np.asarray(result.controls, dtype=float)
        with self._data_lock:
            self._active_controls = controls.copy()
            self._active_plan_started_at = time.monotonic()
        if controls.size == 0:
            return
        self._publish_command(controls[0, 0], controls[0, 1])

    def _publish_safe_stop_command(self):
        if not hasattr(self, 'cmd_pub'):
            return
        with self._data_lock:
            self._active_controls = None
            self._active_plan_started_at = None
        self._publish_command(0.0, 0.0)


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
