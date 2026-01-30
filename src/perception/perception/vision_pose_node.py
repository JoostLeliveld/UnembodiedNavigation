#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import numpy as np

class VisionPoseNode(Node):
    def __init__(self):
        super().__init__('vision_pose_node')
        
        # Parameters
        self.declare_parameter('pose_noise_std', 0.0)
        self.pose_noise_std = self.get_parameter('pose_noise_std').value
        
        # Subscribe to Odom (Ground Truth Proxy)
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
            
        # Publish Vision Pose
        self.publisher = self.create_publisher(
            PoseStamped,
            '/perception/robot_pose_cam',
            10)
            
        self.get_logger().info('Vision Pose Node Started (Frame: map_cam)')

    def odom_callback(self, msg):
        # Create output message
        pose_out = PoseStamped()
        
        # Set Header
        pose_out.header.stamp = self.get_clock().now().to_msg()
        pose_out.header.frame_id = 'map_cam' # The new global frame
        
        # Copy Position
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        
        # Add Noise (Optional)
        if self.pose_noise_std > 0.0:
            x += np.random.normal(0, self.pose_noise_std)
            y += np.random.normal(0, self.pose_noise_std)
        
        pose_out.pose.position.x = x
        pose_out.pose.position.y = y
        pose_out.pose.position.z = z
        
        # Copy Orientation
        pose_out.pose.orientation = msg.pose.pose.orientation
        
        # Publish
        self.publisher.publish(pose_out)

def main(args=None):
    rclpy.init(args=args)
    node = VisionPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
