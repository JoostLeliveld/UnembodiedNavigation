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
        
        # Declare parameter for test mode
        self.declare_parameter('test_mode', False)
        
        if self.get_parameter('test_mode').value:
            self.get_logger().info("Test mode enabled: Robot will drive in a circle")
            self.timer = self.create_timer(0.5, self.test_drive_callback)

    def test_drive_callback(self):
        msg = Twist()
        msg.linear.x = 0.2
        msg.angular.z = 0.2
        self.cmd_vel_publisher.publish(msg)
        
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
