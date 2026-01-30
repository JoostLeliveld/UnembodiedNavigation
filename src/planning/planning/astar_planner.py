#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import numpy as np

# Local import
from planning import search_based_path_planning

class AStarPlanner(Node):

    def __init__(self):
        super().__init__('astar_planner', allow_undeclared_parameters=True, 
                         automatically_declare_parameters_from_overrides=True)
        
        # Node Parameters
        self.max_cost = 100.0 # Maximum (unsafe) cost
        
        # Subscribe to Initial Pose (standard RViz topic)
        # Original used 'start', but standard is /initialpose from RViz
        start_qos_profile = QoSProfile(depth=1)
        start_qos_profile.durability = DurabilityPolicy.VOLATILE
        self.start_subscriber = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.start_callback, qos_profile=start_qos_profile)
        self.start_msg = None

        # Subscribe to Goal Pose (standard RViz topic)
        # Original used 'goal', standard is /goal_pose
        goal_qos_profile = QoSProfile(depth=1)
        goal_qos_profile.durability = DurabilityPolicy.VOLATILE
        self.goal_subscriber = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, qos_profile=goal_qos_profile)
        self.goal_msg = None

        # Subscribe to Costmap
        costmap_qos_profile = QoSProfile(depth=1)
        costmap_qos_profile.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.costmap_subscriber = self.create_subscription(OccupancyGrid, '/costmap', self.costmap_callback, qos_profile=costmap_qos_profile)
        self.costmap_msg = None

        # Publish Path
        path_qos_profile = QoSProfile(depth=1)
        path_qos_profile.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.path_publisher = self.create_publisher(Path, '/plan', qos_profile=path_qos_profile) # Standard topic /plan

        self.get_logger().info('A* Planner Started. Waiting for /costmap, /initialpose, and /goal_pose...')

    def start_callback(self, msg):
        # Convert PoseWithCovarianceStamped to PoseStamped for internal use
        self.start_msg = PoseStamped()
        self.start_msg.header = msg.header
        self.start_msg.pose = msg.pose.pose
        self.get_logger().info('Start pose received')
        self.publish_path()

    def goal_callback(self, msg):
        self.goal_msg = msg
        self.get_logger().info('Goal pose received')
        self.publish_path()

    def costmap_callback(self, msg):
        self.get_logger().info('Costmap received!')
        self.costmap_msg = msg
        self.publish_path()

    def publish_path(self):

        if (self.start_msg is None) or (self.goal_msg is None) or (self.costmap_msg is None):
            return
        
        self.get_logger().info('Planning path...')

        start_position = np.asarray([self.start_msg.pose.position.x, self.start_msg.pose.position.y])
        goal_position = np.asarray([self.goal_msg.pose.position.x, self.goal_msg.pose.position.y])

        costmap_origin = np.asarray([self.costmap_msg.info.origin.position.x, self.costmap_msg.info.origin.position.y])
        costmap_resolution = self.costmap_msg.info.resolution 
        costmap_matrix = np.array(self.costmap_msg.data).reshape(self.costmap_msg.info.height, self.costmap_msg.info.width)
        costmap_matrix = np.float64(costmap_matrix)
        
        # Mark obstacles as -1
        costmap_matrix[costmap_matrix>=self.max_cost] = -1

        start_cell = search_based_path_planning.world_to_grid(start_position, origin=costmap_origin, resolution=costmap_resolution)[0]
        goal_cell = search_based_path_planning.world_to_grid(goal_position, origin=costmap_origin, resolution=costmap_resolution)[0]

        # Use the copied library function
        path_grid = search_based_path_planning.shortest_path_networkx(costmap_matrix, start_cell, goal_cell, diagonal_connectivity=True)
        path_world = search_based_path_planning.grid_to_world(path_grid, costmap_origin, costmap_resolution)

        path_msg = Path()
        path_msg.header.frame_id = 'odom' # The costmap is in 'odom' frame usually for this simple setup (or map)
        if self.costmap_msg.header.frame_id:
             path_msg.header.frame_id = self.costmap_msg.header.frame_id
        
        path_msg.header.stamp = self.get_clock().now().to_msg()

        if path_world.size > 0:
            path_msg.poses.append(self.start_msg)
            for waypoint in path_world:
                pose_msg = PoseStamped()
                pose_msg.header = path_msg.header
                pose_msg.pose.position.x = waypoint[0]
                pose_msg.pose.position.y = waypoint[1]
                path_msg.poses.append(pose_msg)
            path_msg.poses.append(self.goal_msg)
            self.get_logger().info(f'Path found with {len(path_msg.poses)} waypoints.')
        else:
            self.get_logger().warn('No path found!')

        self.path_publisher.publish(path_msg)    

def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
