#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

class AStarPlanner(Node):
    def __init__(self):
        super().__init__('astar_planner')
        self.get_logger().info("A* planner node started")

def main():
    rclpy.init()
    node = AStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
