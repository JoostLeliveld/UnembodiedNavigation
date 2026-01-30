#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry, Path

class AStarPlanner(Node):
    def __init__(self):
        super().__init__('astar_planner')
        self.get_logger().info("A* planner node started")
        
        # Subscribers
        self.costmap_subscription = self.create_subscription(
            OccupancyGrid,
            '/costmap',
            self.costmap_callback,
            10
        )
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        # Publisher
        self.plan_publisher = self.create_publisher(
            Path,
            '/plan',
            10
        )

    def costmap_callback(self, msg):
        pass

    def odom_callback(self, msg):
        pass

def main():
    rclpy.init()
    node = AStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
