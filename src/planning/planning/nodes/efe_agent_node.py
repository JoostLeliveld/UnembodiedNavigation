"""ROS 2 node for EFE agent that publishes cmd_vel directly (unicycle dynamics)."""

import math
import time
import numpy as np

import rclpy
from rclpy.time import Time
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Float64MultiArray

from planning.nodes.unicycle_planner_node import UnicyclePlannerNode
from planning.planners.base_planner import UnicyclePlannerBase, CostmapData
from planning.core.efe_utils import wrap_angle


class EfeAgentNode(UnicyclePlannerNode):
    NODE_NAME = 'efe_agent'
    PLANNER_CLASS = UnicyclePlannerBase

    def __init__(self):
        super().__init__()

        if not self.has_parameter('cmd_topic'):
            self.declare_parameter('cmd_topic', '/cmd_vel')
        self.cmd_topic = self.get_parameter('cmd_topic').value
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)

    def _plan_once(self):
        if self.goal_msg is None or self.costmap is None:
            return

        if self.use_pixel_correction:
            if self.belief_m is None or self.belief_S is None:
                if not self._init_belief_from_state():
                    return
            m0 = self.belief_m.copy()
            S0 = self.belief_S.copy()
            if self.belief_stamp is not None:
                try:
                    now = self.get_clock().now()
                    stamp = Time.from_msg(self.belief_stamp)
                    age = (now - stamp).nanoseconds * 1e-9
                except Exception:
                    age = 0.0
                if age > self.pixel_timeout_s:
                    now_wall = time.monotonic()
                    if now_wall - self._last_stale_log > 2.0:
                        self.get_logger().warn(f"Pixel belief stale (age {age:.2f}s)")
                        self._last_stale_log = now_wall
        else:
            if self.state_msg is None:
                return
            q = self.state_msg.pose.pose.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            theta = math.atan2(siny_cosp, cosy_cosp)
            m0 = np.array([
                self.state_msg.pose.pose.position.x,
                self.state_msg.pose.pose.position.y,
                theta,
            ], dtype=float)

            cov = self.state_msg.pose.covariance
            S0 = np.diag([
                cov[0] if len(cov) > 0 else 1e-6,
                cov[7] if len(cov) > 7 else 1e-6,
                cov[35] if len(cov) > 35 else 1e-6,
            ]).astype(float)
            if self.min_state_cov > 0.0:
                for i in range(min(3, S0.shape[0])):
                    if S0[i, i] < self.min_state_cov:
                        S0[i, i] = self.min_state_cov

        goal_xy = (
            float(self.goal_msg.pose.position.x),
            float(self.goal_msg.pose.position.y),
        )

        result = self.planner.plan(m0, S0, goal_xy, self.costmap)
        if result is None:
            return

        path = Path()
        frame_id = self.costmap.frame_id or (self.state_msg.header.frame_id if self.state_msg else '') or 'map_bev'
        path.header.frame_id = frame_id
        path.header.stamp = self.get_clock().now().to_msg()

        for state in result.states:
            p = PoseStamped()
            p.header = path.header
            p.pose.position.x = float(state[0])
            p.pose.position.y = float(state[1])
            path.poses.append(p)

        goal_pose = PoseStamped()
        goal_pose.header = path.header
        goal_pose.pose.position.x = float(goal_xy[0])
        goal_pose.pose.position.y = float(goal_xy[1])
        path.poses.append(goal_pose)

        self.path_pub.publish(path)

        msg = Float64MultiArray()
        msg.data = [
            float(result.total_cost),
            float(result.risk_cost),
            float(result.ambiguity_cost),
            float(result.control_cost),
            float(result.boundary_cost),
        ]
        self.metrics_pub.publish(msg)

        # Publish first control as cmd_vel
        cmd = Twist()
        cmd.linear.x = float(result.controls[0, 0])
        cmd.angular.z = float(result.controls[0, 1])
        self.cmd_pub.publish(cmd)
        self.last_cmd = np.array([cmd.linear.x, cmd.angular.z], dtype=float)


def main(args=None):
    rclpy.init(args=args)
    node = EfeAgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
