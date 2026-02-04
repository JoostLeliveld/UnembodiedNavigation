#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.clock import Clock, ClockType
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from tf_transformations import euler_from_quaternion
import numpy as np
import math
from control import path_follow_tools

class ControlNode(Node):
    
    def __init__(self):
        super().__init__('control_node')
        
        # Parameters
        self.declare_parameter('lookahead_distance', 0.5) # meters
        self.declare_parameter('max_speed', 0.22) # m/s (Turtlebot3 limit)
        self.declare_parameter('kp_angular', 2.0)
        self.declare_parameter('max_angular', 1.0) # rad/s cap to avoid tight circles
        self.declare_parameter('turn_in_place_threshold', 0.7) # rad; above this, rotate in place
        self.declare_parameter('slowdown_distance', 0.5) # meters; taper speed near goal
        self.declare_parameter('goal_tolerance', 0.1) # meters
        self.declare_parameter('state_timeout_s', 0.5) # seconds; stale-state watchdog

        self.lookahead_distance = float(self.get_parameter('lookahead_distance').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.kp_angular = float(self.get_parameter('kp_angular').value)
        self.max_angular = float(self.get_parameter('max_angular').value)
        self.turn_in_place_threshold = float(self.get_parameter('turn_in_place_threshold').value)
        self.slowdown_distance = float(self.get_parameter('slowdown_distance').value)
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        self.state_timeout_s = float(self.get_parameter('state_timeout_s').value)
        

        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_yaw = 0.0
        self.current_path = None # numpy array of shape (N, 2)
        self.last_state_time = None
        self.last_stale_warn_time = None
        

        self.pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/state/bev',
            self.pose_callback,
            10
        )
        
        path_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Path, '/plan', self.path_callback, qos_profile=path_qos)
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Control Loop (use steady time so publishing doesn't stall if /clock is late)
        self.control_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.1, self.control_loop, clock=self.control_clock) # 10 Hz
        
        self.get_logger().info("Control Node Started (Pure Pursuit)")

    def pose_callback(self, msg):
        self.pose_x = msg.pose.pose.position.x
        self.pose_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.pose_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.last_state_time = self.control_clock.now()

    def path_callback(self, msg):
        path_points = []
        for pose in msg.poses:
            path_points.append([pose.pose.position.x, pose.pose.position.y])
        
        if len(path_points) > 0:
            self.current_path = np.array(path_points)
            self.get_logger().info(f"Received new path with {len(path_points)} points")
        else:
            self.current_path = None

    def control_loop(self):
        if self.current_path is None:
            return

        now = self.control_clock.now()
        if self.last_state_time is None:
            return
        if (now - self.last_state_time).nanoseconds > self.state_timeout_s * 1e9:
            # Stale state: stop and warn at most every 2 seconds
            if (self.last_stale_warn_time is None or
                    (now - self.last_stale_warn_time).nanoseconds > 2e9):
                self.get_logger().warn("State is stale; stopping until /state/bev updates")
                self.last_stale_warn_time = now
            self.stop_robot()
            return
            
        robot_pos = np.array([self.pose_x, self.pose_y])
        

        dist_to_end = np.linalg.norm(robot_pos - self.current_path[-1])
        if dist_to_end < self.goal_tolerance:
            self.stop_robot()
            self.get_logger().info("Goal Reached")
            self.current_path = None # Clear path
            return

        lookahead_point = path_follow_tools.path_goal_sphere(self.current_path, robot_pos, self.lookahead_distance)
        
        if lookahead_point is None:
            lookahead_point = self.current_path[-1] 
            

        timestamp = self.get_clock().now().seconds_nanoseconds()[0]
        if timestamp % 2 == 0: 
             self.get_logger().info(f"Pose: ({self.pose_x:.2f}, {self.pose_y:.2f}) | Target: ({lookahead_point[0]:.2f}, {lookahead_point[1]:.2f}) | Dist: {dist_to_end:.2f}")

        target_x, target_y = lookahead_point
        dx = target_x - self.pose_x
        dy = target_y - self.pose_y
        

        desired_yaw = math.atan2(dy, dx)
        yaw_error = desired_yaw - self.pose_yaw
        
        while yaw_error > math.pi: yaw_error -= 2*math.pi
        while yaw_error < -math.pi: yaw_error += 2*math.pi
            
  
        angular_vel = self.kp_angular * yaw_error
        angular_vel = max(-self.max_angular, min(self.max_angular, angular_vel))

        if abs(yaw_error) > self.turn_in_place_threshold:
            linear_vel = 0.0
        else:
            heading_scale = max(0.1, math.cos(yaw_error))
            linear_vel = self.max_speed * heading_scale
            if dist_to_end < self.slowdown_distance:
                linear_vel *= max(0.1, dist_to_end / self.slowdown_distance)

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
