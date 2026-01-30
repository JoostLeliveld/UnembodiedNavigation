#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import time

class MissionNode(Node):
    def __init__(self):
        super().__init__('mission_node')
        
        # Mission Parameters
        # Note: Odom frame always starts at (0,0) regardless of Gazebo spawn position
        self.declare_parameter('start_x', 0.0)
        self.declare_parameter('start_y', 0.0)
        self.declare_parameter('goal_x', 3.0)
        self.declare_parameter('goal_y', 3.0)
        self.declare_parameter('delay_seconds', 8.0)
        
        self.start_x = self.get_parameter('start_x').value
        self.start_y = self.get_parameter('start_y').value
        self.goal_x = self.get_parameter('goal_x').value
        self.goal_y = self.get_parameter('goal_y').value
        self.delay = self.get_parameter('delay_seconds').value
        
        # Publishers
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, 
            '/initialpose', 
            10
        )
        self.goal_pub = self.create_publisher(
            PoseStamped, 
            '/goal_pose', 
            10
        )
        
        # Timer to send mission after delay
        self.create_timer(self.delay, self.send_mission)
        
        self.mission_sent = False
        self.get_logger().info(f"Mission Node Ready. Will send goal in {self.delay}s")
        self.get_logger().info(f"  Start: ({self.start_x}, {self.start_y})")
        self.get_logger().info(f"  Goal:  ({self.goal_x}, {self.goal_y})")

    def send_mission(self):
        if self.mission_sent:
            return
            
        # Send Initial Pose
        initial_pose = PoseWithCovarianceStamped()
        initial_pose.header.stamp = self.get_clock().now().to_msg()
        initial_pose.header.frame_id = 'map_cam'
        initial_pose.pose.pose.position.x = self.start_x
        initial_pose.pose.pose.position.y = self.start_y
        initial_pose.pose.pose.orientation.w = 1.0
        
        self.initial_pose_pub.publish(initial_pose)
        self.get_logger().info("✓ Initial Pose Published")
        
        # Small delay between messages
        time.sleep(0.5)
        
        # Send Goal
        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = 'map_cam'
        goal_pose.pose.position.x = self.goal_x
        goal_pose.pose.position.y = self.goal_y
        goal_pose.pose.orientation.w = 1.0
        
        self.goal_pub.publish(goal_pose)
        self.get_logger().info("✓ Goal Published")
        self.get_logger().info("🚀 Mission Active!")
        
        self.mission_sent = True

def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
