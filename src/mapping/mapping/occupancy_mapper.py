#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped
import numpy as np
import math
from rclpy.qos import QoSProfile, DurabilityPolicy

class OccupancyMapper(Node):
    def __init__(self):
        super().__init__('occupancy_mapper',
                         allow_undeclared_parameters=True, 
                         automatically_declare_parameters_from_overrides=True)
        
        # Parameters
        self.resolution = 0.05 # m/cell
        self.width_m = 20.0 # meters
        self.height_m = 20.0 # meters
        self.origin_x = -10.0
        self.origin_y = -10.0
        
        self.width = int(self.width_m / self.resolution)
        self.height = int(self.height_m / self.resolution)
        
        # Grid: -1 unknown, 0 free, 100 occupied
        # Using simple counting model: +1 for hit, -1 for free (not implemented for raytrace speed)
        # For this simple version, we just mark hits.
        self.map_data = np.zeros((self.height, self.width), dtype=np.int8)
        
        # Subscribe
        self.scan_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.pose_subscription = self.create_subscription(PoseStamped, '/perception/robot_pose_cam', self.pose_callback, 10)
        
        # Publish
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_publisher = self.create_publisher(OccupancyGrid, '/map', map_qos)
        
        self.current_pose = None
        self.get_logger().info("Occupancy Mapper started")

    def pose_callback(self, msg):
        self.current_pose = msg.pose

    def scan_callback(self, msg):
        if self.current_pose is None:
            return
            
        # Orientation quaternion to yaw
        q = self.current_pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y
        
        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                # Calculate hit point in world frame
                hit_x = robot_x + r * math.cos(yaw + angle)
                hit_y = robot_y + r * math.sin(yaw + angle)
                
                # Convert to grid coordinates
                grid_x = int((hit_x - self.origin_x) / self.resolution)
                grid_y = int((hit_y - self.origin_y) / self.resolution)
                
                if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                    self.map_data[grid_y, grid_x] = 100
                    
            angle += msg.angle_increment

        self.publish_map()

    def publish_map(self):
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = 'map_bev'
        
        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.width
        grid_msg.info.height = self.height
        grid_msg.info.origin.position.x = self.origin_x
        grid_msg.info.origin.position.y = self.origin_y
        grid_msg.info.origin.orientation.w = 1.0
        
        grid_msg.data = self.map_data.flatten().tolist()
        
        self.map_publisher.publish(grid_msg)

def main(args=None):
    rclpy.init(args=args)
    node = OccupancyMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
