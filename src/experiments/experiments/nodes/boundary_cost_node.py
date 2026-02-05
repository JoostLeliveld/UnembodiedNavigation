#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid


class BoundaryCostNode(Node):
    def __init__(self):
        super().__init__('boundary_cost_node')

        self.declare_parameter('min_x', -5.0)
        self.declare_parameter('max_x', 5.0)
        self.declare_parameter('min_y', -5.0)
        self.declare_parameter('max_y', 5.0)
        self.declare_parameter('wall_margin', 0.2)
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('frame_id', 'map_bev')
        # Simple static obstacle (defaults place it on the diagonal start->goal path)
        self.declare_parameter('obstacle_enabled', True)
        self.declare_parameter('obstacle_center_x', 1.5)
        self.declare_parameter('obstacle_center_y', 1.5)
        self.declare_parameter('obstacle_radius', 0.4)
        self.declare_parameter('obstacle_value', 100)

        self.min_x = float(self.get_parameter('min_x').value)
        self.max_x = float(self.get_parameter('max_x').value)
        self.min_y = float(self.get_parameter('min_y').value)
        self.max_y = float(self.get_parameter('max_y').value)
        self.wall_margin = float(self.get_parameter('wall_margin').value)
        self.resolution = float(self.get_parameter('resolution').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.obstacle_enabled = bool(self.get_parameter('obstacle_enabled').value)
        self.obstacle_center_x = float(self.get_parameter('obstacle_center_x').value)
        self.obstacle_center_y = float(self.get_parameter('obstacle_center_y').value)
        self.obstacle_radius = float(self.get_parameter('obstacle_radius').value)
        self.obstacle_value = int(self.get_parameter('obstacle_value').value)

        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(OccupancyGrid, '/costmap', qos)

        self._publish_costmap()
        self.get_logger().info('Boundary costmap published (static)')

    def _publish_costmap(self):
        width = int((self.max_x - self.min_x) / self.resolution)
        height = int((self.max_y - self.min_y) / self.resolution)
        if width <= 0 or height <= 0:
            self.get_logger().error('Invalid boundary dimensions for costmap')
            return

        data = [0] * (width * height)

        for y in range(height):
            for x in range(width):
                wx = self.min_x + (x + 0.5) * self.resolution
                wy = self.min_y + (y + 0.5) * self.resolution
                if (wx <= self.min_x + self.wall_margin or
                        wx >= self.max_x - self.wall_margin or
                        wy <= self.min_y + self.wall_margin or
                        wy >= self.max_y - self.wall_margin):
                    data[y * width + x] = 100
                elif self.obstacle_enabled:
                    dx = wx - self.obstacle_center_x
                    dy = wy - self.obstacle_center_y
                    if (dx * dx + dy * dy) <= (self.obstacle_radius * self.obstacle_radius):
                        data[y * width + x] = self.obstacle_value

        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = self.frame_id
        grid.info.resolution = self.resolution
        grid.info.width = width
        grid.info.height = height
        grid.info.origin.position.x = self.min_x
        grid.info.origin.position.y = self.min_y
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0
        grid.data = data

        self.publisher.publish(grid)
        if self.obstacle_enabled:
            self.get_logger().info(
                f"Obstacle enabled at ({self.obstacle_center_x:.2f}, {self.obstacle_center_y:.2f}) "
                f"r={self.obstacle_radius:.2f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = BoundaryCostNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
