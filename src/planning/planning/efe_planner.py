#!/usr/bin/env python3
"""
Bayesian planner using Expected Free Energy (EFE).

Implements EFE1 (ET1) and EFE2 (UT) approximations over BEV state.
Publishes a BEV Path for the controller to follow.
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Float64MultiArray

from planning import search_based_path_planning
from planning.efe_utils import ET1, UT, ambiguity, risk, wrap_angle
from planning.camera_model import ObliqueCameraModel


class EFEPlanner(Node):
    def __init__(self):
        super().__init__('efe_planner', allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)

        def _declare_if_not(name, default_value):
            if not self.has_parameter(name):
                self.declare_parameter(name, default_value)

        # Planner params
        _declare_if_not('planner_mode', 'efe1')  # efe1 | efe2
        _declare_if_not('plan_rate', 1.0)
        _declare_if_not('horizon', 10)
        _declare_if_not('num_samples', 64)
        _declare_if_not('dt', 0.2)
        _declare_if_not('v_min', 0.0)
        _declare_if_not('v_max', 0.22)
        _declare_if_not('w_min', -1.0)
        _declare_if_not('w_max', 1.0)
        _declare_if_not('control_weight', 0.1)
        _declare_if_not('boundary_weight', 1.0)
        _declare_if_not('max_cost', 100.0)
        _declare_if_not('lethal_cost_threshold', 99.0)
        _declare_if_not('seed', 0)

        # Process/observation noise
        _declare_if_not('process_noise_xy', 0.01)
        _declare_if_not('process_noise_theta', 0.02)
        _declare_if_not('obs_noise_uv', 2.0)
        _declare_if_not('obs_noise_yaw', 0.05)

        # Goal covariance
        _declare_if_not('goal_sigma_xy', 0.25)
        _declare_if_not('goal_sigma_theta', 0.5)

        # Camera model params (must match sim)
        _declare_if_not('cam_pos', [-3.0, -3.0, 6.0])
        _declare_if_not('look_at', [1.5, 1.5, 0.0])
        _declare_if_not('img_width', 1920)
        _declare_if_not('img_height', 1080)
        _declare_if_not('fov_h_rad', 1.5708)

        self.planner_mode = self.get_parameter('planner_mode').value
        self.plan_rate = float(self.get_parameter('plan_rate').value)
        self.horizon = int(self.get_parameter('horizon').value)
        self.num_samples = int(self.get_parameter('num_samples').value)
        self.dt = float(self.get_parameter('dt').value)
        self.v_min = float(self.get_parameter('v_min').value)
        self.v_max = float(self.get_parameter('v_max').value)
        self.w_min = float(self.get_parameter('w_min').value)
        self.w_max = float(self.get_parameter('w_max').value)
        self.control_weight = float(self.get_parameter('control_weight').value)
        self.boundary_weight = float(self.get_parameter('boundary_weight').value)
        self.max_cost = float(self.get_parameter('max_cost').value)
        self.lethal_cost_threshold = float(self.get_parameter('lethal_cost_threshold').value)
        self.seed = int(self.get_parameter('seed').value)

        self.process_noise_xy = float(self.get_parameter('process_noise_xy').value)
        self.process_noise_theta = float(self.get_parameter('process_noise_theta').value)
        self.obs_noise_uv = float(self.get_parameter('obs_noise_uv').value)
        self.obs_noise_yaw = float(self.get_parameter('obs_noise_yaw').value)

        self.goal_sigma_xy = float(self.get_parameter('goal_sigma_xy').value)
        self.goal_sigma_theta = float(self.get_parameter('goal_sigma_theta').value)

        self.rng = np.random.default_rng(self.seed)

        # Camera model
        self.camera = ObliqueCameraModel(
            cam_pos=self.get_parameter('cam_pos').value,
            look_at=self.get_parameter('look_at').value,
            img_width=self.get_parameter('img_width').value,
            img_height=self.get_parameter('img_height').value,
            fov_h_rad=self.get_parameter('fov_h_rad').value,
        )

        # Subscriptions
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
        costmap_qos = QoSProfile(depth=1)
        costmap_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, '/costmap', self._costmap_cb, qos_profile=costmap_qos
        )

        # Publisher
        path_qos = QoSProfile(depth=1)
        path_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.path_pub = self.create_publisher(Path, '/plan', qos_profile=path_qos)
        self.metrics_pub = self.create_publisher(Float64MultiArray, '/efe/metrics', 10)

        # State
        self.state_msg = None
        self.goal_msg = None
        self.costmap_msg = None

        self.create_timer(1.0 / max(self.plan_rate, 0.1), self._plan_once)
        self.get_logger().info(f"EFE planner started in mode {self.planner_mode}")

    def _state_cb(self, msg: PoseWithCovarianceStamped):
        self.state_msg = msg

    def _goal_cb(self, msg: PoseStamped):
        self.goal_msg = msg

    def _costmap_cb(self, msg: OccupancyGrid):
        self.costmap_msg = msg

    def _unicycle_step(self, state, control):
        x, y, theta = state
        v, w = control
        x = x + v * self.dt * math.cos(theta)
        y = y + v * self.dt * math.sin(theta)
        theta = wrap_angle(theta + w * self.dt)
        return np.array([x, y, theta], dtype=float)

    def _unicycle_jacobian(self, state, control):
        _, _, theta = state
        v, _ = control
        F = np.eye(3)
        F[0, 2] = -v * self.dt * math.sin(theta)
        F[1, 2] = v * self.dt * math.cos(theta)
        return F

    def _cost_at(self, x, y):
        if self.costmap_msg is None:
            return 0.0
        origin = np.asarray([
            self.costmap_msg.info.origin.position.x,
            self.costmap_msg.info.origin.position.y
        ])
        res = self.costmap_msg.info.resolution
        grid = search_based_path_planning.world_to_grid([x, y], origin=origin, resolution=res)[0]
        i, j = int(grid[0]), int(grid[1])
        if i < 0 or j < 0 or i >= self.costmap_msg.info.height or j >= self.costmap_msg.info.width:
            return self.max_cost
        costmap_matrix = np.array(self.costmap_msg.data).reshape(
            self.costmap_msg.info.height, self.costmap_msg.info.width
        )
        cost = costmap_matrix[i, j]
        if cost < 0:
            return self.max_cost
        return float(cost)

    def _cost_at_raw(self, x, y):
        if self.costmap_msg is None:
            return 0.0, True
        origin = np.asarray([
            self.costmap_msg.info.origin.position.x,
            self.costmap_msg.info.origin.position.y
        ])
        res = self.costmap_msg.info.resolution
        grid = search_based_path_planning.world_to_grid([x, y], origin=origin, resolution=res)[0]
        i, j = int(grid[0]), int(grid[1])
        if i < 0 or j < 0 or i >= self.costmap_msg.info.height or j >= self.costmap_msg.info.width:
            return self.max_cost, False
        costmap_matrix = np.array(self.costmap_msg.data).reshape(
            self.costmap_msg.info.height, self.costmap_msg.info.width
        )
        cost = float(costmap_matrix[i, j])
        return cost, True

    def _plan_once(self):
        if self.state_msg is None or self.goal_msg is None or self.costmap_msg is None:
            return

        # Current state mean and covariance
        q = self.state_msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        theta = math.atan2(siny_cosp, cosy_cosp)
        m0 = np.array([
            self.state_msg.pose.pose.position.x,
            self.state_msg.pose.pose.position.y,
            theta
        ], dtype=float)

        cov = self.state_msg.pose.covariance
        S0 = np.diag([
            cov[0] if len(cov) > 0 else 1e-6,
            cov[7] if len(cov) > 7 else 1e-6,
            cov[35] if len(cov) > 35 else 1e-6
        ]).astype(float)

        # Goal mean/cov in state space
        goal_theta = theta
        goal_mean = np.array([
            self.goal_msg.pose.position.x,
            self.goal_msg.pose.position.y,
            goal_theta
        ], dtype=float)
        goal_cov = np.diag([
            self.goal_sigma_xy ** 2,
            self.goal_sigma_xy ** 2,
            self.goal_sigma_theta ** 2
        ])

        # Process noise
        Q = np.diag([
            self.process_noise_xy ** 2,
            self.process_noise_xy ** 2,
            self.process_noise_theta ** 2
        ])

        # Observation noise (pixel space)
        R = np.diag([
            self.obs_noise_uv ** 2,
            self.obs_noise_uv ** 2,
            self.obs_noise_yaw ** 2
        ])

        # Choose approximation
        use_ut = (self.planner_mode.lower() == 'efe2')

        best_cost = float('inf')
        best_controls = None
        best_metrics = None

        for _ in range(self.num_samples):
            # Sample control sequence
            vs = self.rng.uniform(self.v_min, self.v_max, size=self.horizon)
            ws = self.rng.uniform(self.w_min, self.w_max, size=self.horizon)
            controls = np.column_stack([vs, ws])

            m = m0.copy()
            S = S0.copy()
            total = 0.0
            total_risk = 0.0
            total_amb = 0.0
            total_control = 0.0
            total_boundary = 0.0

            for t in range(self.horizon):
                u = controls[t]
                # Predict
                m = self._unicycle_step(m, u)
                F = self._unicycle_jacobian(m, u)
                S = F @ S @ F.T + Q

                # Observation model
                def g_fn(x):
                    return self.camera.g(x)

                if use_ut:
                    mu_y, Sigma_y, Gamma = UT(m, S, g_fn, addmatrix=R, forceHermitian=True)
                else:
                    mu_y, Sigma_y, Gamma = ET1(m, S, g_fn, addmatrix=R, forceHermitian=True)

                # EFE terms (risk in state space + ambiguity in observation space)
                r = risk(m, S, (goal_mean, goal_cov))
                a = ambiguity(Sigma_y, Gamma, S)
                c = self.control_weight * float(u[0] ** 2 + u[1] ** 2)
                total_risk += r
                total_amb += a
                total_control += c
                total += r + a + c

                # Boundary cost
                if self.boundary_weight > 0.0:
                    cell_cost, in_bounds = self._cost_at_raw(m[0], m[1])
                    if (not in_bounds) or (cell_cost < 0.0) or (cell_cost >= self.lethal_cost_threshold):
                        total = float('inf')
                        total_boundary += self.boundary_weight
                        break
                    b = self.boundary_weight * (cell_cost / max(self.max_cost, 1.0))
                    total_boundary += b
                    total += b

            if total < best_cost:
                best_cost = total
                best_controls = controls
                best_metrics = (total, total_risk, total_amb, total_control, total_boundary)

        if best_controls is None:
            return

        # Build a path from best controls
        path = Path()
        path.header.frame_id = self.costmap_msg.header.frame_id or self.state_msg.header.frame_id or 'map_bev'
        path.header.stamp = self.get_clock().now().to_msg()

        m = m0.copy()
        start_pose = PoseStamped()
        start_pose.header = path.header
        start_pose.pose.position.x = float(m[0])
        start_pose.pose.position.y = float(m[1])
        path.poses.append(start_pose)

        for t in range(self.horizon):
            m = self._unicycle_step(m, best_controls[t])
            p = PoseStamped()
            p.header = path.header
            p.pose.position.x = float(m[0])
            p.pose.position.y = float(m[1])
            path.poses.append(p)

        # Append goal for controller convenience
        goal_pose = PoseStamped()
        goal_pose.header = path.header
        goal_pose.pose.position.x = float(goal_mean[0])
        goal_pose.pose.position.y = float(goal_mean[1])
        path.poses.append(goal_pose)

        self.path_pub.publish(path)

        if best_metrics is not None:
            msg = Float64MultiArray()
            msg.data = [
                float(best_metrics[0]),
                float(best_metrics[1]),
                float(best_metrics[2]),
                float(best_metrics[3]),
                float(best_metrics[4]),
            ]
            self.metrics_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = EFEPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
