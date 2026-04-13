import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

from perception.core.detection_diagnostics import (
    DETECTION_DIAGNOSTICS_TOPIC,
    diagnostics_message,
)
from perception.core.ros_image import image_msg_to_bgr8
from unav_common.camera_model import ObliqueCameraModel


class ImageMarkerDetectorNode(Node):
    """Detect a ground-contact proxy from the red robot body in the rendered image."""

    def __init__(self):
        super().__init__('image_marker_detector_node')

        def _declare_if_missing(name, default):
            if not self.has_parameter(name):
                self.declare_parameter(name, default)

        # Shared launch parameters are declared explicitly so typos fail fast even
        # though this detector only uses the camera/image subset directly.
        _declare_if_missing('cam_pos', [-3.0, -3.0, 6.0])
        _declare_if_missing('look_at', [1.5, 1.5, 0.0])
        _declare_if_missing('img_width', 1280)
        _declare_if_missing('img_height', 720)
        _declare_if_missing('fov_h_rad', 1.5708)
        _declare_if_missing('pixel_noise_sigma', 0.0)
        _declare_if_missing('seed', 0)
        _declare_if_missing('world_frame', 'map_bev')
        _declare_if_missing('use_geometry_occlusion', True)
        _declare_if_missing('visibility_geometry_json', '')
        _declare_if_missing('visibility_target_height_m', 0.0)
        _declare_if_missing('log_noisy_pixels', False)
        _declare_if_missing('min_blob_area_px', 8.0)
        _declare_if_missing('morph_kernel_px', 3)
        _declare_if_missing('red_h1_low', 0)
        _declare_if_missing('red_h1_high', 12)
        _declare_if_missing('red_h2_low', 168)
        _declare_if_missing('red_h2_high', 180)
        _declare_if_missing('red_s_min', 70)
        _declare_if_missing('red_v_min', 50)

        self.cam_pos = np.array(self.get_parameter('cam_pos').value, dtype=float)
        self.look_at = np.array(self.get_parameter('look_at').value, dtype=float)
        self.img_width = int(self.get_parameter('img_width').value)
        self.img_height = int(self.get_parameter('img_height').value)
        self.fov_h_rad = float(self.get_parameter('fov_h_rad').value)
        self.model = ObliqueCameraModel(
            cam_pos=self.cam_pos,
            look_at=self.look_at,
            img_width=self.img_width,
            img_height=self.img_height,
            fov_h_rad=self.fov_h_rad,
        )

        self.pixel_noise_std = float(self.get_parameter('pixel_noise_sigma').value)
        self.seed = int(self.get_parameter('seed').value)
        self.log_noisy_pixels = str(self.get_parameter('log_noisy_pixels').value).lower() == 'true'
        self.min_blob_area_px = float(self.get_parameter('min_blob_area_px').value)
        self.morph_kernel_px = max(1, int(self.get_parameter('morph_kernel_px').value))
        self.red_h1_low = int(self.get_parameter('red_h1_low').value)
        self.red_h1_high = int(self.get_parameter('red_h1_high').value)
        self.red_h2_low = int(self.get_parameter('red_h2_low').value)
        self.red_h2_high = int(self.get_parameter('red_h2_high').value)
        self.red_s_min = int(self.get_parameter('red_s_min').value)
        self.red_v_min = int(self.get_parameter('red_v_min').value)

        self.rng = np.random.default_rng(self.seed)
        self.log_counter = 0

        self.pub = self.create_publisher(PoseStamped, '/perception/pixel_pose', 10)
        self.diag_pub = self.create_publisher(Float64MultiArray, DETECTION_DIAGNOSTICS_TOPIC, 10)
        self.create_subscription(Image, '/external_camera/image_raw', self._image_cb, 10)

        self.get_logger().info(
            f'Image marker detector started (camera xy only; no visual heading, '
            f'pixel_noise_sigma={self.pixel_noise_std:.3f}, '
            f'min_blob_area_px={self.min_blob_area_px:.1f}, morph_kernel_px={self.morph_kernel_px})'
        )

    @staticmethod
    def _publish_detection_failure(stamp_msg, reason: str, diag_pub, logger, log_counter: int):
        diag_pub.publish(
            diagnostics_message(
                stamp=stamp_msg.sec + stamp_msg.nanosec * 1e-9,
                detected=False,
                u_mid=np.nan,
                v_mid=np.nan,
                yaw_est=np.nan,
                u_red=np.nan,
                v_red=np.nan,
                red_area_px=np.nan,
                u_blue=np.nan,
                v_blue=np.nan,
                blue_area_px=np.nan,
                separation_px=np.nan,
                border_margin_px=np.nan,
            )
        )
        log_counter += 1
        if log_counter % 50 == 0:
            logger.warn(f'Image detector missed red body blob: {reason}')
        return log_counter

    def _publish_failure(self, stamp_msg, reason: str):
        self.log_counter = self._publish_detection_failure(
            stamp_msg,
            reason,
            self.diag_pub,
            self.get_logger(),
            self.log_counter,
        )

    def _largest_blob(self, mask: np.ndarray):
        n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels <= 1:
            return None
        best_idx = None
        best_area = -1.0
        for idx in range(1, n_labels):
            area = float(stats[idx, cv2.CC_STAT_AREA])
            if area < self.min_blob_area_px:
                continue
            if area > best_area:
                best_idx = idx
                best_area = area
        if best_idx is None:
            return None
        centroid = centroids[best_idx]
        x0 = int(stats[best_idx, cv2.CC_STAT_LEFT])
        y0 = int(stats[best_idx, cv2.CC_STAT_TOP])
        width = int(stats[best_idx, cv2.CC_STAT_WIDTH])
        height = int(stats[best_idx, cv2.CC_STAT_HEIGHT])
        return {
            'centroid_u': float(centroid[0]),
            'centroid_v': float(centroid[1]),
            'area_px': float(best_area),
            'bbox_left': x0,
            'bbox_top': y0,
            'bbox_width': width,
            'bbox_height': height,
        }

    @staticmethod
    def _blob_ground_reference(blob):
        # For a ground robot seen obliquely, the bottom-center of the detected body blob
        # is a better proxy for contact position on the ground plane than the blob centroid.
        u_ref = float(blob['bbox_left'] + 0.5 * blob['bbox_width'])
        v_ref = float(blob['bbox_top'] + blob['bbox_height'] - 1.0)
        return u_ref, v_ref

    def _red_mask(self, hsv: np.ndarray):
        kernel = None
        if self.morph_kernel_px > 1:
            kernel = np.ones((self.morph_kernel_px, self.morph_kernel_px), dtype=np.uint8)

        lower1 = np.array([self.red_h1_low, self.red_s_min, self.red_v_min], dtype=np.uint8)
        upper1 = np.array([self.red_h1_high, 255, 255], dtype=np.uint8)
        lower2 = np.array([self.red_h2_low, self.red_s_min, self.red_v_min], dtype=np.uint8)
        upper2 = np.array([self.red_h2_high, 255, 255], dtype=np.uint8)
        mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

        if kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _image_cb(self, msg: Image):
        try:
            image_bgr = image_msg_to_bgr8(msg)
        except ValueError as exc:
            self.get_logger().warn(f'Failed to convert image: {exc}')
            return

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        red = self._largest_blob(self._red_mask(hsv))
        if red is None:
            self._publish_failure(msg.header.stamp, 'red body blob not found')
            return

        u_ref, v_ref = self._blob_ground_reference(red)
        if self.pixel_noise_std > 0.0:
            u_ref += float(self.rng.normal(0.0, self.pixel_noise_std))
            v_ref += float(self.rng.normal(0.0, self.pixel_noise_std))

        world = self.model.pixel_to_world(u_ref, v_ref)
        if world is None:
            self._publish_failure(msg.header.stamp, 'red blob reference outside homography footprint')
            return

        img_h, img_w = image_bgr.shape[:2]
        border_margin_px = float(
            min(
                u_ref,
                v_ref,
                img_w - 1.0 - u_ref,
                img_h - 1.0 - v_ref,
            )
        )

        self.diag_pub.publish(
            diagnostics_message(
                stamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                detected=True,
                u_mid=u_ref,
                v_mid=v_ref,
                yaw_est=np.nan,
                u_red=red['centroid_u'],
                v_red=red['centroid_v'],
                red_area_px=red['area_px'],
                u_blue=np.nan,
                v_blue=np.nan,
                blue_area_px=np.nan,
                separation_px=np.nan,
                border_margin_px=border_margin_px,
            )
        )

        out_msg = PoseStamped()
        out_msg.header = msg.header
        out_msg.header.frame_id = 'image'
        out_msg.pose.position.x = float(u_ref)
        out_msg.pose.position.y = float(v_ref)
        out_msg.pose.position.z = 0.0
        out_msg.pose.orientation.w = 1.0
        self.pub.publish(out_msg)

        self.log_counter += 1
        if self.log_counter % 20 == 0:
            self.get_logger().info(
                f'Detected red centroid=({red["centroid_u"]:.0f},{red["centroid_v"]:.0f}) '
                f'ground_ref=({u_ref:.0f},{v_ref:.0f}) area={red["area_px"]:.1f}px'
            )


def main(args=None):
    rclpy.init(args=args)
    node = ImageMarkerDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
