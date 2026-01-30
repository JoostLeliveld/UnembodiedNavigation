#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        self.get_logger().info("Perception node started")
        
        # Subscribe to raw scan
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        # Publish processed/filtered scan
        self.scan_publisher = self.create_publisher(
            LaserScan,
            '/scan_processed',
            10
        )
        
    def scan_callback(self, msg):
        # Placeholder for processing logic (e.g. filtering)
        # For now, just republishing to verify flow
        self.scan_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
