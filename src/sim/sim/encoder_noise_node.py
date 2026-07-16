"""Simulated wheel-encoder odometry with realistic slip and drift.

Gazebo's diff-drive plugin publishes a perfect integral of whatever velocity
it was commanded.  Real cheap encoders accumulate independent slip errors on
top of that.  This node corrupts the reported velocity with a correlated
multiplicative slip model (same AR(1) structure as actuation_noise_node) and
republishes an integrated pose that drifts away from the simulator truth.

The planner subscribes to /odom_noisy instead of /odom and uses the noisy
velocity for its EKF predict step, making the belief diverge from truth
whenever camera observations stop arriving.

Topic wiring:
  /odom          <- Gazebo (truth of actual motion, already includes actuation slip)
  /odom_noisy    -> consumed by the planner's predict step and heading correction
"""

import math
import random

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time


def _wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _quaternion_from_yaw(yaw: float):
    from geometry_msgs.msg import Quaternion
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


class EncoderNoiseNode(Node):
    """Republishes /odom with independent encoder slip added to the velocity.

    Pose is integrated from the first received odom message; it drifts from
    the Gazebo ground truth at a rate determined by the slip parameters.
    """

    def __init__(self):
        super().__init__('encoder_noise_node')

        self.declare_parameter('enabled', True)
        self.declare_parameter('input_topic', '/odom')
        self.declare_parameter('output_topic', '/odom_noisy')
        self.declare_parameter('seed', 0)

        # Multiplicative slip on the true velocity (independent of actuation slip).
        # mean > 0 models systematic under-reading (e.g. wheel compression).
        self.declare_parameter('linear_slip_mean', 0.02)
        self.declare_parameter('linear_slip_std', 0.05)
        self.declare_parameter('angular_slip_mean', 0.00)
        self.declare_parameter('angular_slip_std', 0.03)

        # Additive white noise (vibration, quantisation).
        self.declare_parameter('linear_additive_std', 0.004)
        self.declare_parameter('angular_additive_std', 0.020)

        # AR(1) temporal correlation of the slip state.
        self.declare_parameter('correlation_alpha', 0.80)

        # Deadband: do not inject drift on a hard stop command.
        self.declare_parameter('stop_linear_deadband', 1e-4)
        self.declare_parameter('stop_angular_deadband', 1e-4)

        # Maximum plausible dt between consecutive odom messages (seconds).
        # Messages with larger gaps are skipped to avoid huge integration jumps.
        self.declare_parameter('max_dt_s', 0.5)

        # A propagated covariance accompanies the corrupted encoder pose.  It
        # is operational metadata: it depends only on the encoder-noise model
        # and commanded odometry, never on Gazebo truth.  Commissioning logs
        # use it as the uncertain-input GP covariance and later check its
        # calibration against evaluation-only truth.
        self.declare_parameter('initial_position_std_m', 0.01)
        self.declare_parameter('initial_yaw_std_rad', 0.01)
        self.declare_parameter('covariance_floor_m2', 1.0e-8)
        self.declare_parameter('covariance_floor_yaw_rad2', 1.0e-8)

        self.enabled = bool(self.get_parameter('enabled').value)
        input_topic = str(self.get_parameter('input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)

        self.linear_slip_mean = float(self.get_parameter('linear_slip_mean').value)
        self.linear_slip_std = max(0.0, float(self.get_parameter('linear_slip_std').value))
        self.angular_slip_mean = float(self.get_parameter('angular_slip_mean').value)
        self.angular_slip_std = max(0.0, float(self.get_parameter('angular_slip_std').value))
        self.linear_additive_std = max(0.0, float(self.get_parameter('linear_additive_std').value))
        self.angular_additive_std = max(0.0, float(self.get_parameter('angular_additive_std').value))
        self.correlation_alpha = min(max(float(self.get_parameter('correlation_alpha').value), 0.0), 0.999)
        self.stop_linear_deadband = max(0.0, float(self.get_parameter('stop_linear_deadband').value))
        self.stop_angular_deadband = max(0.0, float(self.get_parameter('stop_angular_deadband').value))
        self.max_dt_s = max(0.01, float(self.get_parameter('max_dt_s').value))
        self.initial_position_std_m = max(0.0, float(self.get_parameter('initial_position_std_m').value))
        self.initial_yaw_std_rad = max(0.0, float(self.get_parameter('initial_yaw_std_rad').value))
        self.covariance_floor_m2 = max(0.0, float(self.get_parameter('covariance_floor_m2').value))
        self.covariance_floor_yaw_rad2 = max(
            0.0, float(self.get_parameter('covariance_floor_yaw_rad2').value)
        )

        seed = int(self.get_parameter('seed').value)
        # Use a different seed offset from actuation_noise_node so the two
        # slip processes are independent even with the same base seed.
        self._rng = random.Random(seed + 31337)
        self._linear_slip_state = 0.0
        self._angular_slip_state = 0.0

        # Encoder-integrated pose (world frame).  Initialised from first odom.
        self._pose_x: float | None = None
        self._pose_y: float | None = None
        self._pose_theta: float | None = None
        self._last_stamp = None
        self._pose_cov = self._initial_pose_covariance()

        self._pub = self.create_publisher(Odometry, output_topic, 10)
        self.create_subscription(Odometry, input_topic, self._odom_cb, 10)

        self.get_logger().info(
            f'Encoder noise node: {input_topic} -> {output_topic}, enabled={self.enabled}, '
            f'seed={seed}, lin_slip_mean={self.linear_slip_mean:.3f}, '
            f'lin_slip_std={self.linear_slip_std:.3f}, '
            f'ang_slip_std={self.angular_slip_std:.3f}, '
            f'alpha={self.correlation_alpha:.2f}'
        )

    def _update_correlated_state(self, old_value: float, std: float) -> float:
        if std <= 0.0:
            return 0.0
        innov_scale = math.sqrt(max(1.0 - self.correlation_alpha ** 2, 0.0))
        return self.correlation_alpha * old_value + innov_scale * self._rng.gauss(0.0, std)

    def _initial_pose_covariance(self):
        pos_var = max(self.initial_position_std_m ** 2, self.covariance_floor_m2)
        yaw_var = max(self.initial_yaw_std_rad ** 2, self.covariance_floor_yaw_rad2)
        return [[pos_var, 0.0, 0.0], [0.0, pos_var, 0.0], [0.0, 0.0, yaw_var]]

    def _propagate_pose_covariance(self, *, theta: float, v_true: float, w_true: float, dt: float) -> tuple[float, float]:
        """Propagate planar encoder uncertainty for one noisy integration step.

        The AR(1) state has stationary standard deviation ``*_slip_std``.
        Treating that variance as the one-step velocity uncertainty is
        conservative when the correlation is high, which is appropriate for a
        commissioning gate.  A later evaluation-only calibration report tests
        whether the resulting covariance has correct coverage.
        """

        c = math.cos(theta)
        s = math.sin(theta)
        f = (
            (1.0, 0.0, -v_true * dt * s),
            (0.0, 1.0, v_true * dt * c),
            (0.0, 0.0, 1.0),
        )
        p = self._pose_cov
        fp = [[sum(f[i][k] * p[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        propagated = [[sum(fp[i][k] * f[j][k] for k in range(3)) for j in range(3)] for i in range(3)]

        var_v = (v_true * self.linear_slip_std) ** 2 + self.linear_additive_std ** 2
        var_w = (w_true * self.angular_slip_std) ** 2 + self.angular_additive_std ** 2
        g_v = (dt * c, dt * s, 0.0)
        g_w = (0.0, 0.0, dt)
        for i in range(3):
            for j in range(3):
                propagated[i][j] += g_v[i] * var_v * g_v[j] + g_w[i] * var_w * g_w[j]
        for i in range(3):
            propagated[i][i] = max(
                propagated[i][i],
                self.covariance_floor_yaw_rad2 if i == 2 else self.covariance_floor_m2,
            )
        self._pose_cov = propagated
        return var_v, var_w

    def _write_covariances(self, message: Odometry, *, var_v: float, var_w: float) -> None:
        """Write planar covariance into ROS's 6x6 pose/twist conventions."""

        p = self._pose_cov
        pose_cov = [0.0] * 36
        pose_cov[0] = p[0][0]
        pose_cov[1] = pose_cov[6] = p[0][1]
        pose_cov[5] = pose_cov[30] = p[0][2]
        pose_cov[7] = p[1][1]
        pose_cov[11] = pose_cov[31] = p[1][2]
        pose_cov[35] = p[2][2]
        message.pose.covariance = pose_cov

        twist_cov = [0.0] * 36
        twist_cov[0] = max(var_v, self.covariance_floor_m2)
        twist_cov[35] = max(var_w, self.covariance_floor_yaw_rad2)
        message.twist.covariance = twist_cov

    def _odom_cb(self, msg: Odometry) -> None:
        stamp = msg.header.stamp

        # First message: initialise encoder pose from Gazebo truth.
        if self._last_stamp is None:
            self._pose_x = msg.pose.pose.position.x
            self._pose_y = msg.pose.pose.position.y
            self._pose_theta = _yaw_from_quaternion(msg.pose.pose.orientation)
            self._last_stamp = stamp
            return

        # Compute dt.
        try:
            dt = (Time.from_msg(stamp) - Time.from_msg(self._last_stamp)).nanoseconds * 1e-9
        except Exception:
            dt = 0.0
        self._last_stamp = stamp

        if dt <= 0.0 or dt > self.max_dt_s:
            return

        # True velocity from Gazebo (reflects actuation noise already applied).
        v_true = float(msg.twist.twist.linear.x)
        w_true = float(msg.twist.twist.angular.z)

        stop = (abs(v_true) <= self.stop_linear_deadband
                and abs(w_true) <= self.stop_angular_deadband)

        # Evolve correlated slip states regardless of motion to keep continuity.
        self._linear_slip_state = self._update_correlated_state(
            self._linear_slip_state, self.linear_slip_std)
        self._angular_slip_state = self._update_correlated_state(
            self._angular_slip_state, self.angular_slip_std)

        if (not self.enabled) or stop:
            v_enc = 0.0 if stop else v_true
            w_enc = 0.0 if stop else w_true
        else:
            v_mult = max(0.0, 1.0 - self.linear_slip_mean + self._linear_slip_state)
            w_mult = 1.0 - self.angular_slip_mean + self._angular_slip_state
            v_enc = v_true * v_mult + self._rng.gauss(0.0, self.linear_additive_std)
            # Angular encoder error also fires when the robot is turning with v=0.
            moving = (abs(v_true) > self.stop_linear_deadband
                      or abs(w_true) > self.stop_angular_deadband)
            w_enc = (w_true * w_mult + self._rng.gauss(0.0, self.angular_additive_std)
                     if moving else w_true)

        # Propagate covariance before applying the mean motion so its Jacobian
        # uses the previous heading, matching the Euler integration below.
        var_v, var_w = self._propagate_pose_covariance(
            theta=self._pose_theta,
            v_true=v_true,
            w_true=w_true,
            dt=dt,
        )

        # Integrate noisy velocity.
        self._pose_x += v_enc * dt * math.cos(self._pose_theta)
        self._pose_y += v_enc * dt * math.sin(self._pose_theta)
        self._pose_theta = _wrap_angle(self._pose_theta + w_enc * dt)

        # Publish noisy odometry.
        out = Odometry()
        out.header.stamp = stamp
        out.header.frame_id = msg.header.frame_id or 'odom'
        out.child_frame_id = msg.child_frame_id or 'base_footprint'
        out.pose.pose.position.x = self._pose_x
        out.pose.pose.position.y = self._pose_y
        out.pose.pose.position.z = 0.0
        out.pose.pose.orientation = _quaternion_from_yaw(self._pose_theta)
        out.twist.twist.linear.x = v_enc
        out.twist.twist.angular.z = w_enc
        self._write_covariances(out, var_v=var_v, var_w=var_w)
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = EncoderNoiseNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except RuntimeError:
            pass


if __name__ == '__main__':
    main()
