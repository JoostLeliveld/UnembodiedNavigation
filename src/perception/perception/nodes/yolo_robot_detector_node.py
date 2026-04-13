import math
from pathlib import Path

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


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on')


def _resolve_model_ref(model_ref: str, hf_filename: str) -> str:
    """Resolve local/Ultralytics/Hugging Face model references to a YOLO-loadable path."""
    ref = str(model_ref or '').strip()
    if not ref:
        return 'yolo11n.pt'
    if ref.startswith('hf://'):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "yolo_model uses hf:// but huggingface_hub is not installed. "
                "Install it with `pip install huggingface_hub` or pass a local .pt path."
            ) from exc

        raw = ref[len('hf://'):].strip('/')
        parts = [p for p in raw.split('/') if p]
        if len(parts) < 2:
            raise RuntimeError(
                "Invalid hf:// YOLO model reference. Use hf://owner/repo or "
                "hf://owner/repo/path/to/model.pt"
            )
        repo_id = '/'.join(parts[:2])
        filename = '/'.join(parts[2:]) if len(parts) > 2 else str(hf_filename or 'best.pt')
        return hf_hub_download(repo_id=repo_id, filename=filename)
    return ref


def _target_class_ids(names, target_class: str):
    target = str(target_class or '').strip().lower()
    if target in ('', '*', 'any', 'all'):
        return None
    if target.lstrip('-').isdigit():
        return {int(target)}
    if isinstance(names, dict):
        return {int(idx) for idx, name in names.items() if str(name).strip().lower() == target}
    return set()


def _confidence_logit(score: float, eps: float = 1e-6) -> float:
    p = float(np.clip(score, eps, 1.0 - eps))
    return float(math.log(p / (1.0 - p)))


class YoloRobotDetectorNode(Node):
    """Detect the robot with Ultralytics YOLO and publish the existing pixel observation contract."""

    def __init__(self):
        super().__init__('yolo_robot_detector_node')

        def _declare_if_missing(name, default):
            if not self.has_parameter(name):
                self.declare_parameter(name, default)

        _declare_if_missing('cam_pos', [-3.0, -3.0, 6.0])
        _declare_if_missing('look_at', [1.5, 1.5, 0.0])
        _declare_if_missing('img_width', 1280)
        _declare_if_missing('img_height', 720)
        _declare_if_missing('fov_h_rad', 1.5708)
        _declare_if_missing('pixel_noise_sigma', 0.0)
        _declare_if_missing('seed', 0)
        _declare_if_missing('yolo_model', 'yolo11n.pt')
        _declare_if_missing('yolo_hf_filename', 'best.pt')
        _declare_if_missing('yolo_device', '')
        _declare_if_missing('yolo_imgsz', 640)
        _declare_if_missing('yolo_conf_threshold', 0.25)
        _declare_if_missing('yolo_iou_threshold', 0.45)
        _declare_if_missing('yolo_target_class', 'robot')
        _declare_if_missing('yolo_use_masks', True)
        _declare_if_missing('yolo_min_mask_area_px', 12.0)
        _declare_if_missing('yolo_mask_bottom_band_px', 3.0)
        _declare_if_missing('yolo_verbose', False)

        self.cam_pos = np.array(self.get_parameter('cam_pos').value, dtype=float)
        self.look_at = np.array(self.get_parameter('look_at').value, dtype=float)
        self.img_width = int(self.get_parameter('img_width').value)
        self.img_height = int(self.get_parameter('img_height').value)
        self.fov_h_rad = float(self.get_parameter('fov_h_rad').value)
        self.camera_model = ObliqueCameraModel(
            cam_pos=self.cam_pos,
            look_at=self.look_at,
            img_width=self.img_width,
            img_height=self.img_height,
            fov_h_rad=self.fov_h_rad,
        )

        self.pixel_noise_std = float(self.get_parameter('pixel_noise_sigma').value)
        self.seed = int(self.get_parameter('seed').value)
        self.rng = np.random.default_rng(self.seed)
        self.yolo_model_ref = str(self.get_parameter('yolo_model').value)
        self.yolo_hf_filename = str(self.get_parameter('yolo_hf_filename').value)
        self.yolo_device = str(self.get_parameter('yolo_device').value).strip()
        self.yolo_imgsz = int(self.get_parameter('yolo_imgsz').value)
        self.yolo_conf_threshold = float(self.get_parameter('yolo_conf_threshold').value)
        self.yolo_iou_threshold = float(self.get_parameter('yolo_iou_threshold').value)
        self.yolo_target_class = str(self.get_parameter('yolo_target_class').value)
        self.yolo_use_masks = _as_bool(self.get_parameter('yolo_use_masks').value)
        self.yolo_min_mask_area_px = float(self.get_parameter('yolo_min_mask_area_px').value)
        self.yolo_mask_bottom_band_px = float(self.get_parameter('yolo_mask_bottom_band_px').value)
        self.yolo_verbose = _as_bool(self.get_parameter('yolo_verbose').value)

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required for perception_backend:=yolo. "
                "Install it with `pip install ultralytics` in the active ROS environment."
            ) from exc

        resolved_model = _resolve_model_ref(self.yolo_model_ref, self.yolo_hf_filename)
        self.yolo = YOLO(resolved_model)
        self.target_ids = _target_class_ids(getattr(self.yolo, 'names', {}), self.yolo_target_class)
        if self.target_ids == set():
            names = getattr(self.yolo, 'names', {})
            self.get_logger().warn(
                f"Target class {self.yolo_target_class!r} not found in YOLO names {names!r}; "
                "no detections will pass the target-class filter."
            )

        self.log_counter = 0
        self.pub = self.create_publisher(PoseStamped, '/perception/pixel_pose', 10)
        self.diag_pub = self.create_publisher(Float64MultiArray, DETECTION_DIAGNOSTICS_TOPIC, 10)
        self.create_subscription(Image, '/external_camera/image_raw', self._image_cb, 10)

        device_text = self.yolo_device if self.yolo_device else 'auto'
        target_text = 'any' if self.target_ids is None else sorted(self.target_ids)
        self.get_logger().info(
            f'YOLO robot detector started (model={self.yolo_model_ref!r}, resolved={Path(str(resolved_model)).name!r}, '
            f'target={target_text}, imgsz={self.yolo_imgsz}, conf={self.yolo_conf_threshold:.2f}, '
            f'iou={self.yolo_iou_threshold:.2f}, use_masks={self.yolo_use_masks}, device={device_text})'
        )

    def _publish_failure(self, stamp_msg, reason: str, *, yolo_score: float = 0.0):
        stamp = stamp_msg.sec + stamp_msg.nanosec * 1e-9
        self.diag_pub.publish(
            diagnostics_message(
                stamp=stamp,
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
                yolo_score=float(yolo_score),
                bbox_area_px=np.nan,
                bbox_xmin=np.nan,
                bbox_ymin=np.nan,
                bbox_xmax=np.nan,
                bbox_ymax=np.nan,
                class_id=np.nan,
                logit_margin=np.nan,
                class_entropy=np.nan,
                mask_area_px=np.nan,
                mask_bottom_u=np.nan,
                mask_bottom_v=np.nan,
                mask_used=0.0,
                mask_polygon_points=np.nan,
                confidence_logit=_confidence_logit(yolo_score) if float(yolo_score) > 0.0 else np.nan,
            )
        )
        self.log_counter += 1
        if self.log_counter % 50 == 0:
            self.get_logger().warn(f'YOLO detector missed robot: {reason}')

    def _predict(self, image_bgr: np.ndarray):
        kwargs = {
            'source': image_bgr,
            'imgsz': self.yolo_imgsz,
            'conf': self.yolo_conf_threshold,
            'iou': self.yolo_iou_threshold,
            'verbose': self.yolo_verbose,
        }
        if self.yolo_device:
            kwargs['device'] = self.yolo_device
        return self.yolo.predict(**kwargs)

    def _select_detection(self, result):
        boxes = getattr(result, 'boxes', None)
        if boxes is None or len(boxes) == 0:
            return None
        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int)
        if xyxy.size == 0:
            return None

        mask = np.isfinite(conf) & (conf >= self.yolo_conf_threshold)
        if self.target_ids is not None:
            mask &= np.asarray([int(c) in self.target_ids for c in cls], dtype=bool)
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            return None
        best = int(valid[np.argmax(conf[valid])])
        x_min, y_min, x_max, y_max = [float(v) for v in xyxy[best]]
        return {
            'index': best,
            'bbox': (x_min, y_min, x_max, y_max),
            'score': float(conf[best]),
            'class_id': int(cls[best]),
        }

    @staticmethod
    def _polygon_area(pts: np.ndarray) -> float:
        if pts.shape[0] < 3:
            return 0.0
        x = pts[:, 0]
        y = pts[:, 1]
        return float(abs(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))))

    def _mask_ground_reference(self, result, detection):
        if not self.yolo_use_masks:
            return None
        masks = getattr(result, 'masks', None)
        if masks is None:
            return None
        mask_polygons = getattr(masks, 'xy', None)
        if mask_polygons is None:
            return None
        idx = int(detection.get('index', -1))
        if idx < 0 or idx >= len(mask_polygons):
            return None
        pts = np.asarray(mask_polygons[idx], dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
            return None
        pts = pts[np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]), :2]
        if pts.shape[0] < 3:
            return None

        area_px = self._polygon_area(pts)
        if area_px < self.yolo_min_mask_area_px:
            return None

        v_bottom = float(np.max(pts[:, 1]))
        band = pts[pts[:, 1] >= v_bottom - max(self.yolo_mask_bottom_band_px, 0.0)]
        if band.size == 0:
            u_bottom = float(np.mean(pts[:, 0]))
        else:
            u_bottom = float(np.mean(band[:, 0]))
        return {
            'u': u_bottom,
            'v': v_bottom,
            'area_px': area_px,
            'n_points': int(pts.shape[0]),
        }

    def _image_cb(self, msg: Image):
        try:
            image_bgr = image_msg_to_bgr8(msg)
        except ValueError as exc:
            self.get_logger().warn(f'Failed to convert image: {exc}')
            return

        try:
            results = self._predict(image_bgr)
        except Exception as exc:
            self.get_logger().error(f'YOLO inference failed: {exc}')
            self._publish_failure(msg.header.stamp, 'inference failed')
            return

        if not results:
            self._publish_failure(msg.header.stamp, 'no YOLO result returned')
            return
        detection = self._select_detection(results[0])
        if detection is None:
            self._publish_failure(msg.header.stamp, 'no target-class box above threshold')
            return

        x_min, y_min, x_max, y_max = detection['bbox']
        mask_ref = self._mask_ground_reference(results[0], detection)
        if mask_ref is None:
            u_ref = 0.5 * (x_min + x_max)
            v_ref = y_max
            mask_used = 0.0
            mask_area_px = math.nan
            mask_bottom_u = math.nan
            mask_bottom_v = math.nan
            mask_polygon_points = math.nan
        else:
            u_ref = float(mask_ref['u'])
            v_ref = float(mask_ref['v'])
            mask_used = 1.0
            mask_area_px = float(mask_ref['area_px'])
            mask_bottom_u = u_ref
            mask_bottom_v = v_ref
            mask_polygon_points = float(mask_ref['n_points'])
        if self.pixel_noise_std > 0.0:
            u_ref += float(self.rng.normal(0.0, self.pixel_noise_std))
            v_ref += float(self.rng.normal(0.0, self.pixel_noise_std))

        world = self.camera_model.pixel_to_world(u_ref, v_ref)
        if world is None:
            self._publish_failure(
                msg.header.stamp,
                'YOLO box bottom-center outside homography footprint',
                yolo_score=detection['score'],
            )
            return

        img_h, img_w = image_bgr.shape[:2]
        border_margin_px = float(min(u_ref, v_ref, img_w - 1.0 - u_ref, img_h - 1.0 - v_ref))
        bbox_area_px = max(x_max - x_min, 0.0) * max(y_max - y_min, 0.0)

        self.diag_pub.publish(
            diagnostics_message(
                stamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                detected=True,
                u_mid=u_ref,
                v_mid=v_ref,
                yaw_est=np.nan,
                u_red=np.nan,
                v_red=np.nan,
                red_area_px=np.nan,
                u_blue=np.nan,
                v_blue=np.nan,
                blue_area_px=np.nan,
                separation_px=np.nan,
                border_margin_px=border_margin_px,
                yolo_score=detection['score'],
                bbox_area_px=bbox_area_px,
                bbox_xmin=x_min,
                bbox_ymin=y_min,
                bbox_xmax=x_max,
                bbox_ymax=y_max,
                class_id=detection['class_id'],
                logit_margin=np.nan,
                class_entropy=np.nan,
                mask_area_px=mask_area_px,
                mask_bottom_u=mask_bottom_u,
                mask_bottom_v=mask_bottom_v,
                mask_used=mask_used,
                mask_polygon_points=mask_polygon_points,
                confidence_logit=_confidence_logit(detection['score']),
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
                f'YOLO robot box=({x_min:.0f},{y_min:.0f},{x_max:.0f},{y_max:.0f}) '
                f'ground_ref=({u_ref:.0f},{v_ref:.0f}) score={detection["score"]:.3f} mask_used={bool(mask_used)}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = YoloRobotDetectorNode()
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
