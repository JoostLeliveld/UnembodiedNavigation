"""Base ROS 2 node for pure reference EFE planners."""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.time import Time

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from std_msgs.msg import Float64MultiArray


class PureEfePlannerNodeBase(Node):
    NODE_NAME = 'pure_efe_planner'
    PLANNER_CLASS = None

    def __init__(self):
        super().__init__(self.NODE_NAME)

        if self.PLANNER_CLASS is None:
            raise RuntimeError('PLANNER_CLASS is not set.')

        self.declare_parameter('plan_rate', 1.0)
        self.declare_parameter('horizon', 8)
        self.declare_parameter('num_samples', 256)
        self.declare_parameter('dt', 0.2)
        self.declare_parameter('eta', 0.1)
        self.declare_parameter('rho', 0.05)
        self.declare_parameter('obs_sigma', 2.0)
        self.declare_parameter('seed', 0)

        self.declare_parameter('cam_pos', [-3.0, -3.0, 6.0])
        self.declare_parameter('look_at', [1.5, 1.5, 0.0])
        self.declare_parameter('img_width', 1280)
        self.declare_parameter('img_height', 720)
        self.declare_parameter('fov_h_rad', 1.5708)

        self.plan_rate = float(self.get_parameter('plan_rate').value)
        self.H = int(self.get_parameter('horizon').value)
        self.N = int(self.get_parameter('num_samples').value)
        self.dt = float(self.get_parameter('dt').value)
        self.eta = float(self.get_parameter('eta').value)
        self.rho = self.get_parameter('rho').value
        self.obs_sigma = self.get_parameter('obs_sigma').value
        self.seed = int(self.get_parameter('seed').value)

        camera_params = {
            'cam_pos': self.get_parameter('cam_pos').value,
            'look_at': self.get_parameter('look_at').value,
            'img_width': int(self.get_parameter('img_width').value),
            'img_height': int(self.get_parameter('img_height').value),
            'fov_h_rad': float(self.get_parameter('fov_h_rad').value),
        }

        self.planner = self.PLANNER_CLASS(
            horizon=self.H,
            num_samples=self.N,
            dt=self.dt,
            eta=self.eta,
            rho=self.rho,
            obs_sigma=self.obs_sigma,
            seed=self.seed,
            camera_params=camera_params,
        )

        state_qos = QoSProfile(depth=1)
        state_qos.durability = DurabilityPolicy.VOLATILE
        self.state_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/state/bev', self._state_cb, qos_profile=state_qos
        )
        goal_qos = QoSProfile(depth=1)
        goal_qos.durability = DurabilityPolicy.VOLATILE
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_bev', self._goal_cb, qos_profile=goal_qos
        )

        self.u_pub = self.create_publisher(Float64MultiArray, '/efe_ref/argmin_u', 10)
        self.value_pub = self.create_publisher(Float64MultiArray, '/efe_ref/value', 10)

        self.mu0 = None
        self.S0 = None
        self.goal_obs = None
        self.prev_state_xy = None
        self.prev_state_time = None

        self.create_timer(1.0 / max(self.plan_rate, 0.1), self._plan_once)
        self.get_logger().info(f"{self.NODE_NAME} started")

    def _state_cb(self, msg: PoseWithCovarianceStamped):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)

        vx = 0.0
        vy = 0.0
        try:
            now = Time.from_msg(msg.header.stamp)
            if self.prev_state_xy is not None and self.prev_state_time is not None:
                dt_s = (now - self.prev_state_time).nanoseconds * 1e-9
                if dt_s > 1e-6:
                    vx = (x - self.prev_state_xy[0]) / dt_s
                    vy = (y - self.prev_state_xy[1]) / dt_s
            self.prev_state_time = now
            self.prev_state_xy = (x, y)
        except Exception:
            pass

        self.mu0 = np.array([x, y, vx, vy], dtype=float)

        cov = msg.pose.covariance
        cov_x = float(cov[0])
        cov_y = float(cov[7])
        vel_cov_x = max(float(self.planner.rho[0]), 1e-6)
        vel_cov_y = max(float(self.planner.rho[1]), 1e-6)
        self.S0 = np.diag([cov_x, cov_y, vel_cov_x, vel_cov_y])

    def _goal_cb(self, msg: PoseStamped):
        goal_state = np.array([
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            0.0,
            0.0,
        ])
        self.goal_obs = self.planner.g(goal_state)

    def _plan_once(self):
        if self.mu0 is None or self.S0 is None or self.goal_obs is None:
            return

        best_U, best_cost = self.planner.plan(self.mu0, self.S0, self.goal_obs)
        if best_U is None:
            return

        msg = Float64MultiArray()
        msg.data = best_U.flatten().tolist()
        self.u_pub.publish(msg)

        val = Float64MultiArray()
        val.data = [float(best_cost)]
        self.value_pub.publish(val)
