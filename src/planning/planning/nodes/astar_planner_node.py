"""ROS 2 node for A* planner."""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

from planning.planners.astar_planner import AStarPlannerCore


class AStarPlannerNode(Node):
    def __init__(self):
        super().__init__('astar_planner', allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)

        if not self.has_parameter('max_cost'):
            self.declare_parameter('max_cost', 100.0)

        self.max_cost = float(self.get_parameter('max_cost').value)
        self.planner = AStarPlannerCore(max_cost=self.max_cost, diagonal_connectivity=True)

        # Subscribe to BEV state (authoritative state estimate)
        state_qos_profile = QoSProfile(depth=1)
        state_qos_profile.durability = DurabilityPolicy.VOLATILE
        self.state_subscriber = self.create_subscription(
            PoseWithCovarianceStamped,
            '/state/bev',
            self.state_callback,
            qos_profile=state_qos_profile,
        )
        self.state_msg = None

        # Subscribe to BEV Goal
        goal_qos_profile = QoSProfile(depth=1)
        goal_qos_profile.durability = DurabilityPolicy.VOLATILE
        self.goal_subscriber = self.create_subscription(
            PoseStamped,
            '/goal_bev',
            self.goal_callback,
            qos_profile=goal_qos_profile,
        )
        self.goal_msg = None

        # Subscribe to Costmap
        costmap_qos_profile = QoSProfile(depth=1)
        costmap_qos_profile.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.costmap_subscriber = self.create_subscription(
            OccupancyGrid,
            '/costmap',
            self.costmap_callback,
            qos_profile=costmap_qos_profile,
        )
        self.costmap_msg = None

        # Publish Path
        path_qos_profile = QoSProfile(depth=1)
        path_qos_profile.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.path_publisher = self.create_publisher(Path, '/plan', qos_profile=path_qos_profile)

        self.get_logger().info('A* Planner Started. Waiting for /costmap, /state/bev, and /goal_bev...')

    def state_callback(self, msg):
        # Convert PoseWithCovarianceStamped to PoseStamped for internal use
        self.state_msg = PoseStamped()
        self.state_msg.header = msg.header
        self.state_msg.pose = msg.pose.pose
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
        if (self.state_msg is None) or (self.goal_msg is None) or (self.costmap_msg is None):
            return

        self.get_logger().info('Planning path...')

        start_position = np.asarray([
            self.state_msg.pose.position.x,
            self.state_msg.pose.position.y,
        ])
        goal_position = np.asarray([
            self.goal_msg.pose.position.x,
            self.goal_msg.pose.position.y,
        ])

        costmap_origin = np.asarray([
            self.costmap_msg.info.origin.position.x,
            self.costmap_msg.info.origin.position.y,
        ])
        costmap_resolution = self.costmap_msg.info.resolution
        costmap_matrix = np.array(self.costmap_msg.data).reshape(
            self.costmap_msg.info.height,
            self.costmap_msg.info.width,
        )
        costmap_matrix = np.float64(costmap_matrix)

        result = self.planner.plan(
            costmap_matrix,
            costmap_origin,
            costmap_resolution,
            start_position,
            goal_position,
        )

        path_msg = Path()
        path_msg.header.frame_id = self.state_msg.header.frame_id or 'map_bev'
        if self.costmap_msg.header.frame_id:
            path_msg.header.frame_id = self.costmap_msg.header.frame_id

        path_msg.header.stamp = self.get_clock().now().to_msg()

        if result.path_world.size > 0:
            path_msg.poses.append(self.state_msg)
            for waypoint in result.path_world:
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
    node = AStarPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
