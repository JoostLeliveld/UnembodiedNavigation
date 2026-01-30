#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        self.get_logger().info("Control node started")
        
        # Subscriber to path planning
        self.plan_subscription = self.create_subscription(
            Path,
            '/plan',
            self.plan_callback,
            10
        )
        
        # Publisher for robot velocity
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
    def plan_callback(self, msg):
        self.get_logger().info(f"Received path with {len(msg.poses)} poses")
        # Placeholder: Stop for now, later implement path following
        # For now, just logging.
        
def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
