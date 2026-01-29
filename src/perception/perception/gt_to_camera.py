import math
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D
from tf_transformations import euler_from_quaternion


class GroundTruthToCamera(Node):
    def __init__(self):
        super().__init__('gt_to_camera')

        # ===============================
        # Camera intrinsics (from SDF)
        # ===============================
        self.fx = 554.256  # pixels
        self.fy = 554.256
        self.cx = 320.0
        self.cy = 240.0

        # ===============================
        # Camera pose in world frame
        # (must match world.sdf)
        # ===============================
        self.cam_x = 0.0
        self.cam_y = 0.0
        self.cam_z = 2.5

        # Roll, pitch, yaw of camera
        self.cam_roll = 0.0
        self.cam_pitch = -math.pi / 2.0
        self.cam_yaw = 0.0

        # Precompute rotation matrix
        self.R_wc = self.rotation_matrix(
            self.cam_roll,
            self.cam_pitch,
            self.cam_yaw
        )

        # ===============================
        # ROS interfaces
        # ===============================
        self.sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.pub = self.create_publisher(
            Pose2D,
            '/state/s_cam',
            10
        )

        self.get_logger().info('GT → camera projection node started')

    def rotation_matrix(self, roll, pitch, yaw):
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll),  math.cos(roll)]
        ])
        Ry = np.array([
            [ math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])
        Rz = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw),  math.cos(yaw), 0],
            [0, 0, 1]
        ])
        return Rz @ Ry @ Rx

    def odom_callback(self, msg: Odometry):
        # Robot position in world
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = 0.0

        # Robot orientation
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # World → camera coordinates
        p_world = np.array([
            px - self.cam_x,
            py - self.cam_y,
            pz - self.cam_z
        ])

        p_cam = self.R_wc @ p_world

        # Reject points behind camera
        if p_cam[2] <= 0.0:
            return

        # Perspective projection
        u = self.fx * (p_cam[0] / p_cam[2]) + self.cx
        v = self.fy * (p_cam[1] / p_cam[2]) + self.cy

        # Heading in camera frame
        theta_cam = yaw - self.cam_yaw

        msg_out = Pose2D()
        msg_out.x = float(u)
        msg_out.y = float(v)
        msg_out.theta = float(theta_cam)

        self.pub.publish(msg_out)


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthToCamera()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
