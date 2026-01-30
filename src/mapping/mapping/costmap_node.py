#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid
import numpy as np
import scipy.ndimage
import skimage.morphology

# Local import from the same package
from mapping import occupancy_grid_costmap

class UniformCostmap(Node):

    def __init__(self):
        super().__init__('costmap_node', allow_undeclared_parameters=True, 
                         automatically_declare_parameters_from_overrides=True)
        
        # Node Parameters
        self.min_cost = 1.0 # Minimum cost in the range of [0, 100]
        self.max_cost = 100.0 # Maximum (unsafe) cost in the range of [0, 100]
        self.safety_margin = 0.3 # Default safety margin 0.3m
        self.occupancy_threshold = 0.5 

        # Create a subscriber to the map topic
        map_qos_profile = QoSProfile(depth=1)
        map_qos_profile.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.map_subscriber = self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos_profile=map_qos_profile)
        
        # Create a publisher for the costmap
        costmap_qos_profile = QoSProfile(depth=1)
        costmap_qos_profile.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.costmap_publisher = self.create_publisher(OccupancyGrid, '/costmap', qos_profile=costmap_qos_profile)
        
        self.get_logger().info('Costmap Node Started')

    def map_callback(self, occgrid_msg):
        self.get_logger().info('Occupancy grid map received! (Generating Costmap)')

        occupancy_matrix = np.array(occgrid_msg.data).reshape(occgrid_msg.info.height, occgrid_msg.info.width)
        binary_occupancy_matrix = occupancy_matrix > 100*self.occupancy_threshold

        safety_margin_in_cells = self.safety_margin/occgrid_msg.info.resolution
        
        # Using the utility function we copied
        dilated_occupancy_matrix = occupancy_grid_costmap.dilate(binary_occupancy_matrix, radius = safety_margin_in_cells)
        
        cost_matrix = (self.max_cost - self.min_cost) * dilated_occupancy_matrix + self.min_cost 
        cost_matrix = np.clip(cost_matrix, 0, 100)
        cost_matrix = np.int8(cost_matrix)

        # Publish the costmap
        costmap_msg = occgrid_msg
        costmap_msg.data = cost_matrix.flatten().tolist()
        self.costmap_publisher.publish(costmap_msg)
        self.get_logger().info('Costmap published!')

def main(args=None):
    rclpy.init(args=args)
    node = UniformCostmap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
