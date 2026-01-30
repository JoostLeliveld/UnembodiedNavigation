#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import numpy as np
import cv2

class HomographySimNode(Node):
    def __init__(self):
        super().__init__('homography_sim_node')
        
        # Virtual Camera Parameters
        self.cam_height = 3.0 # meters
        self.img_width = 1920
        self.img_height = 1080
        self.fov_deg = 90.0
        
        # 1. Compute Intrinsics (K)
        # f = (W/2) / tan(FOV/2)
        f = (self.img_width / 2.0) / np.tan(np.deg2rad(self.fov_deg / 2.0))
        cx = self.img_width / 2.0
        cy = self.img_height / 2.0
        
        self.K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])
        
        # 2. Compute Extrinsics (RT) World -> Camera
        # World: X-Forward, Y-Left, Z-Up
        # Camera: X-Right, Y-Down, Z-Forward
        # Looking Down: CamZ aligns with World -Z.
        # Orientation: CamX aligns with World -Y (Right vs Left). CamY aligns with World -X (Down vs Forward).
        
        # Rotation Matrix (World Basis Vectors expressed in Camera Frame)
        # World X (1,0,0) -> Cam Y (-1) -> [0, -1, 0]
        # World Y (0,1,0) -> Cam X (-1) -> [-1, 0, 0]
        # World Z (0,0,1) -> Cam Z (-1) -> [0, 0, -1] (Wait, -Z world is +Z cam). 
        # Verified: Cross([0,-1,0], [-1,0,0]) = [0,0,-1]. Correct.
        
        R = np.array([
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, -1]
        ])
        
        # Translation
        # Camera Position in World C = [0, 0, H]
        # T = -R @ C
        C = np.array([0, 0, self.cam_height])
        T = -R @ C
        
        self.RT = np.zeros((3, 4))
        self.RT[:3, :3] = R
        self.RT[:3, 3] = T
        
        # 3. Compute Homography (H)
        # P_pixel = K @ RT @ P_world
        # On Ground Plane, Z=0. So we drop the 3rd column of RT (which multiplies Z).
        # H = K @ RT_planar
        
        RT_planar = self.RT[:, [0, 1, 3]] # Columns 0(X), 1(Y), 3(T)
        self.H = self.K @ RT_planar
        self.H_inv = np.linalg.inv(self.H)
        
        # Verify Math on startup
        self.get_logger().info(f"Homography Matrix:\n{self.H}")
        test_pt = np.array([0, 0, 1]) # Center of world
        uv = self.H @ test_pt
        uv = uv / uv[2]
        self.get_logger().info(f"World (0,0) -> Pixel {uv[:2]}")
        
        # Subscribers
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.pub = self.create_publisher(PoseStamped, '/perception/robot_pose_cam', 10)
        
        self.logger_timer = 0

    def odom_callback(self, msg):
        # 1. Get True Pose
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # 2. Simulate Projection (World -> Pixel)
        # This is what the Camera "Sees"
        world_pt = np.array([x, y, 1.0])
        pixel_pt_homo = self.H @ world_pt
        u = pixel_pt_homo[0] / pixel_pt_homo[2]
        v = pixel_pt_homo[1] / pixel_pt_homo[2]
        
        # Check if in view
        if not (0 <= u < self.img_width and 0 <= v < self.img_height):
            # Robot out of camera view!
            # For simplicity, we keep tracking or warn
            # self.get_logger().warn(f"Robot OOB: ({u:.1f}, {v:.1f})")
            pass

        # 3. Simulate Estimation (Pixel -> World)
        # This is what our Algorithms will do
        
        # Add Noise (Optional - clean for now)
        u_noise = u
        v_noise = v
        
        pixel_pt_est = np.array([u_noise, v_noise, 1.0])
        world_pt_est_homo = self.H_inv @ pixel_pt_est
        
        x_est = world_pt_est_homo[0] / world_pt_est_homo[2]
        y_est = world_pt_est_homo[1] / world_pt_est_homo[2]
        
        # 4. Publish
        out_msg = PoseStamped()
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = 'map_cam'
        out_msg.pose.position.x = x_est
        out_msg.pose.position.y = y_est
        out_msg.pose.position.z = 0.0
        out_msg.pose.orientation = msg.pose.pose.orientation # Pass-through orientation for now (Homography is XYZ)
        
        self.pub.publish(out_msg)
        
        # Debug Log
        self.logger_timer += 1
        if self.logger_timer % 20 == 0: # 2 Hz
            err = np.linalg.norm([x - x_est, y - y_est])
            self.get_logger().info(f"True:({x:.2f},{y:.2f}) -> Pix:({u:.0f},{v:.0f}) -> Est:({x_est:.2f},{y_est:.2f}) | Err: {err:.4f}")

def main(args=None):
    rclpy.init(args=args)
    node = HomographySimNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
