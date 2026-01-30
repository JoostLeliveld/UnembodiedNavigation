#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion
import numpy as np
import math

# Local import
from control import path_follow_tools

class ControlNode(Node):
    
    def __init__(self):
        super().__init__('control_node', allow_undeclared_parameters=True, 
                         automatically_declare_parameters_from_overrides=True)
        
        # Parameters
        self.lookahead_distance = 0.5 # meters
        self.max_speed = 0.22 # m/s (Turtlebot3 limit)
        self.kp_angular = 2.0
        self.goal_tolerance = 0.1 # meters
        
        # State
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_yaw = 0.0
        self.current_path = None # numpy array of shape (N, 2)
        
        # Subscribers
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        path_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Path, '/plan', self.path_callback, qos_profile=path_qos)
        
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Control Loop
        self.create_timer(0.1, self.control_loop) # 10 Hz
        
        self.get_logger().info("Control Node Started (Pure Pursuit)")

    def odom_callback(self, msg):
        self.pose_x = msg.pose.pose.position.x
        self.pose_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.pose_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

    def path_callback(self, msg):
        path_points = []
        for pose in msg.poses:
            path_points.append([pose.pose.position.x, pose.pose.position.y])
        
        if len(path_points) > 0:
            self.current_path = np.array(path_points)
            self.get_logger().info(f"Received new path with {len(path_points)} points")
        else:
            self.current_path = None

    def scan_callback(self, msg):
        # Basic safety check placeholder
        # In a real "SafePathFollower", we would stop if min(ranges) < threshold
        pass

    def control_loop(self):
        if self.current_path is None:
            return
            
        robot_pos = np.array([self.pose_x, self.pose_y])
        
        # Check if we reached the end
        dist_to_end = np.linalg.norm(robot_pos - self.current_path[-1])
        if dist_to_end < self.goal_tolerance:
            self.stop_robot()
            self.get_logger().info("Goal Reached")
            self.current_path = None # Clear path
            return

        # Pure Pursuit: Find lookahead point
        # path_goal_sphere returns the intersection point further along the path
        lookahead_point = path_follow_tools.path_goal_sphere(self.current_path, robot_pos, self.lookahead_distance)
        
        if lookahead_point is None:
            # If no intersection (e.g., path is entirely inside radius, or far away), 
            # target the last point if close, or re-plan/stop.
            # For robustness, target the closest point or end point.
            lookahead_point = self.current_path[-1] 
            
        # Log status every 10 iterations (1s)
        timestamp = self.get_clock().now().seconds_nanoseconds()[0]
        if timestamp % 2 == 0: 
             self.get_logger().info(f"Pose: ({self.pose_x:.2f}, {self.pose_y:.2f}) | Target: ({lookahead_point[0]:.2f}, {lookahead_point[1]:.2f}) | Dist: {dist_to_end:.2f}")

        # Calculate control commands
        target_x, target_y = lookahead_point
        dx = target_x - self.pose_x
        dy = target_y - self.pose_y
        
        # Desired heading
        desired_yaw = math.atan2(dy, dx)
        
        # Heading error
        yaw_error = desired_yaw - self.pose_yaw
        
        # Normalize error to [-pi, pi]
        while yaw_error > math.pi: yaw_error -= 2*math.pi
        while yaw_error < -math.pi: yaw_error += 2*math.pi
            
        # P-Control for angular velocity
        angular_vel = self.kp_angular * yaw_error
        
        # Limit speed based on sharpness of turn (simple heuristic)
        linear_vel = self.max_speed
        if abs(yaw_error) > 0.5:
            linear_vel = 0.1 # Slow down for sharp turns
            
        # Publish
        cmd = Twist()
        cmd.linear.x = float(linear_vel)
        cmd.angular.z = float(angular_vel)
        self.cmd_vel_pub.publish(cmd)

    def stop_robot(self):
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
