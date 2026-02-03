#!/usr/bin/env python3
"""
Homography Simulation Node for 45° Oblique Camera

This node simulates camera-based perception by:
1. Taking ground truth pose from /odom
2. Projecting to image coordinates using the camera model
3. Publishing a pixel-space pose on /perception/pixel_pose

The camera model accounts for:
- 45° oblique viewing angle (creates projective nonlinearity)
- Proper perspective projection
- Ground plane assumption (Z=0)

This provides the pixel observation source for boundary-only pipelines.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import numpy as np


class HomographySimNode(Node):
    def __init__(self):
        super().__init__('homography_sim_node', 
                         allow_undeclared_parameters=True, 
                         automatically_declare_parameters_from_overrides=True)
        
        # ===========================================
        # Camera Parameters (must match camera.sdf)
        # ===========================================
        
        # Camera position in odom frame
        self.cam_pos = np.array([-3.0, -3.0, 6.0])
        
        # Camera look-at point (midpoint of robot path)
        self.look_at = np.array([1.5, 1.5, 0.0])
        
        # Image parameters
        self.img_width = 1920
        self.img_height = 1080
        self.fov_h_rad = 1.5708  # 90 degrees horizontal FOV
        
        # Noise parameters (set to 0 for Pipeline A2 ablation)
        self.pixel_noise_std = 0.0  # Standard deviation in pixels
        
        # ===========================================
        # Compute Camera Matrices
        # ===========================================
        
        # 1. Intrinsic Matrix K
        # f = (width/2) / tan(fov/2)
        f = (self.img_width / 2.0) / np.tan(self.fov_h_rad / 2.0)
        cx = self.img_width / 2.0
        cy = self.img_height / 2.0
        
        self.K = np.array([
            [f,  0, cx],
            [0,  f, cy],
            [0,  0,  1]
        ])
        
        # 2. Rotation Matrix R (world to camera) using look-at formulation
        self.R = self._compute_lookat_rotation(self.cam_pos, self.look_at)
        
        # 3. Translation vector t = -R @ C
        self.t = -self.R @ self.cam_pos
        
        # 4. Homography for ground plane (Z=0)
        # H = K @ [r1, r2, t] where r1, r2 are first two columns of R
        RT_planar = np.column_stack([self.R[:, 0], self.R[:, 1], self.t])
        self.H = self.K @ RT_planar
        
        # 5. Inverse homography (for pixel -> world)
        self.H_inv = np.linalg.inv(self.H)
        
        # ===========================================
        # Verify Setup
        # ===========================================
        self._verify_camera_setup()
        
        # ===========================================
        # ROS Interface
        # ===========================================
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.pub = self.create_publisher(PoseStamped, '/perception/pixel_pose', 10)
        
        self.log_counter = 0
        self.get_logger().info("Homography Sim Node started (45° oblique camera)")
    
    def _compute_lookat_rotation(self, cam_pos, look_at, up_hint=None):
        """
        Compute rotation matrix using look-at formulation.
        This transforms points from world frame to camera frame.
        
        Camera frame convention (OpenCV/standard CV):
        - X: right in image
        - Y: down in image  
        - Z: forward (optical axis)
        
        Args:
            cam_pos: Camera position in world coordinates
            look_at: Point the camera is looking at
            up_hint: World up direction (default: [0, 0, 1])
            
        Returns:
            R: 3x3 rotation matrix (world to camera)
        """
        if up_hint is None:
            up_hint = np.array([0.0, 0.0, 1.0])
        
        # Camera Z axis = forward direction (toward look_at point)
        z_cam = look_at - cam_pos
        z_cam = z_cam / np.linalg.norm(z_cam)
        
        # Camera X axis = right direction (perpendicular to forward and up)
        x_cam = np.cross(z_cam, up_hint)
        x_cam = x_cam / np.linalg.norm(x_cam)
        
        # Camera Y axis = down direction (perpendicular to forward and right)
        y_cam = np.cross(z_cam, x_cam)
        y_cam = y_cam / np.linalg.norm(y_cam)
        
        # Rotation matrix: rows are camera axes expressed in world coordinates
        # This transforms world coordinates to camera coordinates
        R = np.array([x_cam, y_cam, z_cam])
        
        return R
    
    def _verify_camera_setup(self):
        """Log camera parameters and verify the setup is correct."""
        # Compute effective pitch angle
        direction = self.look_at - self.cam_pos
        horiz_dist = np.sqrt(direction[0]**2 + direction[1]**2)
        pitch_deg = np.degrees(np.arctan2(-direction[2], horiz_dist))
        yaw_deg = np.degrees(np.arctan2(direction[1], direction[0]))
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("Camera Configuration (45° Oblique View):")
        self.get_logger().info(f"  Position (odom): ({self.cam_pos[0]}, {self.cam_pos[1]}, {self.cam_pos[2]})")
        self.get_logger().info(f"  Look-at point: ({self.look_at[0]}, {self.look_at[1]}, {self.look_at[2]})")
        self.get_logger().info(f"  Effective angles: Yaw={yaw_deg:.1f}°, Pitch={pitch_deg:.1f}° down")
        self.get_logger().info(f"  FOV: {np.degrees(self.fov_h_rad):.1f}° horizontal")
        self.get_logger().info(f"  Resolution: {self.img_width}x{self.img_height}")
        
        # Test projection of key points
        test_points = [
            (0.0, 0.0, "Start (0,0)"),
            (3.0, 3.0, "Goal (3,3)"),
            (1.5, 1.5, "Midpoint (1.5,1.5)"),
        ]
        
        self.get_logger().info("Ground point projections:")
        for x, y, name in test_points:
            u, v, visible = self.world_to_pixel(x, y)
            status = "IN VIEW" if visible else "OUT OF VIEW"
            self.get_logger().info(f"  {name} -> pixel ({u:.0f}, {v:.0f}) [{status}]")
        
        # Test round-trip accuracy
        self.get_logger().info("Round-trip accuracy test:")
        for x, y, name in test_points:
            u, v, visible = self.world_to_pixel(x, y)
            if visible:
                x_est, y_est = self.pixel_to_world(u, v)
                err = np.sqrt((x - x_est)**2 + (y - y_est)**2)
                self.get_logger().info(f"  {name}: error = {err:.6f}m")
        
        self.get_logger().info("=" * 50)
    
    def world_to_pixel(self, x, y, z=0.0):
        """
        Project a world point to pixel coordinates.
        
        Args:
            x, y, z: World coordinates (z=0 for ground plane)
            
        Returns:
            u, v: Pixel coordinates
            visible: Whether the point is in the image
        """
        # Homogeneous world point
        world_pt = np.array([x, y, 1.0])  # Using homography (assumes z=0)
        
        # Project using homography
        pixel_homo = self.H @ world_pt
        
        # Dehomogenize
        if abs(pixel_homo[2]) < 1e-10:
            return 0, 0, False
            
        u = pixel_homo[0] / pixel_homo[2]
        v = pixel_homo[1] / pixel_homo[2]
        
        # Check if in image bounds
        visible = (0 <= u < self.img_width) and (0 <= v < self.img_height)
        
        # Also check if point is in front of camera
        # Transform world point to camera coordinates
        world_3d = np.array([x, y, z])
        cam_pt = self.R @ (world_3d - self.cam_pos)
        if cam_pt[2] <= 0:  # Behind camera
            visible = False
        
        return u, v, visible
    
    def pixel_to_world(self, u, v):
        """
        Back-project a pixel to world coordinates (assuming ground plane z=0).
        
        Args:
            u, v: Pixel coordinates
            
        Returns:
            x, y: World coordinates on ground plane
        """
        # Homogeneous pixel point
        pixel_pt = np.array([u, v, 1.0])
        
        # Apply inverse homography
        world_homo = self.H_inv @ pixel_pt
        
        # Dehomogenize
        x = world_homo[0] / world_homo[2]
        y = world_homo[1] / world_homo[2]
        
        return x, y
    
    def odom_callback(self, msg):
        """Process odometry and publish estimated pose."""
        # 1. Get ground truth pose
        x_true = msg.pose.pose.position.x
        y_true = msg.pose.pose.position.y
        
        # 2. Project to pixel coordinates
        u, v, visible = self.world_to_pixel(x_true, y_true)
        
        if not visible:
            # Robot out of camera view - could publish last known or skip
            self.log_counter += 1
            if self.log_counter % 50 == 0:
                self.get_logger().warn(f"Robot at ({x_true:.2f}, {y_true:.2f}) is OUT OF CAMERA VIEW")
            return
        
        # 3. Publish pixel pose (u,v) with yaw in orientation
        out_msg = PoseStamped()
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = 'image'
        out_msg.pose.position.x = float(u)
        out_msg.pose.position.y = float(v)
        out_msg.pose.position.z = 0.0
        out_msg.pose.orientation = msg.pose.pose.orientation  # Pass through orientation
        
        self.pub.publish(out_msg)
        
        # 4. Debug logging (every ~2 seconds at 10Hz odom)
        self.log_counter += 1
        if self.log_counter % 20 == 0:
            self.get_logger().info(
                f"True:({x_true:.2f},{y_true:.2f}) -> "
                f"Pix:({u:.0f},{v:.0f})"
            )


def main(args=None):
    rclpy.init(args=args)
    node = HomographySimNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
