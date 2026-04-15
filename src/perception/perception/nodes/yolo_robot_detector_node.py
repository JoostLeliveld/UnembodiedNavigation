import math
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from ultralytics import YOLO

from perception.core.detection_diagnostics import (
    DETECTION_DIAGNOSTICS_TOPIC,
    diagnostics_message,
)
from perception.core.ros_image import image_msg_to_bgr8


SELECTED_PIXEL_SOURCE_NONE = 0.0
SELECTED_PIXEL_SOURCE_BBOX_BOTTOM = 1.0
SELECTED_PIXEL_SOURCE_MASK_BOTTOM = 2.0


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on')


def _target_class_ids(names, class_name: str, class_id: int):
    if int(class_id) >= 0:
        return {int(class_id)}
    target = str(class_name or '').strip().lower()
    if target in ('', '*', 'any', 'all'):
        return None
    if isinstance(names, dict):
        return {int(idx) for idx, name in names.items() if str(name).strip().lower() == target}
    return set()


def _confidence_logit(score: float, eps: float = 1e-6) -> float:
    p = float(np.clip(score, eps, 1.0 - eps))
    return float(math.log(p / (1.0 - p)))


def _mask_bottom_pixel(points: np.ndarray, band_px: float) -> tuple[float, float]:
    v_bottom = float(np.max(points[:, 1]))
    band = points[points[:, 1] >= v_bottom - max(float(band_px), 0.0)]
    if band.size == 0:
        return float(np.mean(points[:, 0])), v_bottom
    return float(np.mean(band[:, 0])), v_bottom


class YoloRobotDetectorNode(Node):
    """Run one local YOLO model and publish the selected image-space robot observation."""

    def __init__(self):
        super().__init__('yolo_robot_detector_node')

        self.declare_parameter('model_path', '')
        self.declare_parameter('device', '')
        self.declare_parameter('image_size', 640)
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('class_name', 'robot')
        self.declare_parameter('class_id', -1)
        self.declare_parameter('use_masks', True)
        self.declare_parameter('mask_min_area', 12.0)
        self.declare_parameter('mask_bottom_band_px', 3.0)
        self.declare_parameter('pixel_noise_sigma', 0.0)
        self.declare_parameter('seed', 0)

        model_path = Path(str(self.get_parameter('model_path').value).strip()).expanduser()
        if not str(model_path):
            raise RuntimeError('model_path must be set for yolo_robot_detector_node')
        if not model_path.is_file():
            raise RuntimeError(f'model_path does not exist: {model_path}')

        self.model_path = model_path.resolve()
        self.device = str(self.get_parameter('device').value).strip()
        self.image_size = int(self.get_parameter('image_size').value)
        self.confidence_threshold = float(self.get_parameter('confidence_threshold').value)
        self.iou_threshold = float(self.get_parameter('iou_threshold').value)
        self.class_name = str(self.get_parameter('class_name').value)
        self.class_id = int(self.get_parameter('class_id').value)
        self.use_masks = bool(self.get_parameter('use_masks').value) if isinstance(self.get_parameter('use_masks').value, bool) else _as_bool(self.get_parameter('use_masks').value)
        self.mask_min_area = float(self.get_parameter('mask_min_area').value)
        self.mask_bottom_band_px = float(self.get_parameter('mask_bottom_band_px').value)
        self.pixel_noise_sigma = float(self.get_parameter('pixel_noise_sigma').value)
        self.rng = np.random.default_rng(int(self.get_parameter('seed').value))

        self.model = YOLO(str(self.model_path))
        self.target_ids = _target_class_ids(getattr(self.model, 'names', {}), self.class_name, self.class_id)
        if self.target_ids == set():
            names = getattr(self.model, 'names', {})
            raise RuntimeError(
                f'class filter does not match any YOLO class names: class_name={self.class_name!r}, '
                f'class_id={self.class_id}, names={names!r}'
            )

        self.pixel_pub = self.create_publisher(PoseStamped, '/perception/pixel_pose', 10)
        self.diag_pub = self.create_publisher(Float64MultiArray, DETECTION_DIAGNOSTICS_TOPIC, 10)
        self.create_subscription(Image, '/external_camera/image_raw', self._image_cb, 10)

        self.get_logger().info(
            f'YOLO runtime detector started (model={self.model_path}, conf={self.confidence_threshold:.2f}, '
            f'iou={self.iou_threshold:.2f}, use_masks={self.use_masks}, device={self.device or "auto"})'
        )

    def _predict(self, image_bgr: np.ndarray):
        kwargs = {
            'source': image_bgr,
            'imgsz': self.image_size,
            'conf': self.confidence_threshold,
            'iou': self.iou_threshold,
            'verbose': False,
        }
        if self.device:
            kwargs['device'] = self.device
        return self.model.predict(**kwargs)

    def _select_detection(self, result):
        boxes = getattr(result, 'boxes', None)
        if boxes is None or len(boxes) == 0:
            return None
        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int)
        if xyxy.ndim != 2 or xyxy.shape[1] < 4:
            raise RuntimeError('YOLO output boxes are malformed')

        valid = np.isfinite(conf) & (conf >= self.confidence_threshold)
        if self.target_ids is not None:
            valid &= np.asarray([int(c) in self.target_ids for c in cls], dtype=bool)
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            return None
        best_idx = int(valid_idx[np.argmax(conf[valid_idx])])
        x0, y0, x1, y1 = [float(v) for v in xyxy[best_idx, :4]]
        return {
            'index': best_idx,
            'bbox': (x0, y0, x1, y1),
            'confidence': float(conf[best_idx]),
            'class_id': int(cls[best_idx]),
        }

    def _mask_reference(self, result, detection):
        if not self.use_masks:
            return None
        masks = getattr(result, 'masks', None)
        polygons = getattr(masks, 'xy', None) if masks is not None else None
        if polygons is None:
            return None
        idx = int(detection['index'])
        if idx < 0 or idx >= len(polygons):
            return None
        points = np.asarray(polygons[idx], dtype=float)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
            return None
        points = points[np.isfinite(points[:, 0]) & np.isfinite(points[:, 1]), :2]
        if points.shape[0] < 3:
            return None
        area = float(abs(0.5 * (np.dot(points[:, 0], np.roll(points[:, 1], -1)) - np.dot(points[:, 1], np.roll(points[:, 0], -1)))))
        if area < self.mask_min_area:
            return None
        u, v = _mask_bottom_pixel(points, self.mask_bottom_band_px)
        return {
            'u': u,
            'v': v,
            'area': area,
            'n_points': int(points.shape[0]),
        }

    def _publish_failure(self, stamp_msg: Image, *, confidence: float = 0.0) -> None:
        stamp = float(stamp_msg.sec) + float(stamp_msg.nanosec) * 1e-9
        self.diag_pub.publish(
            diagnostics_message(
                stamp=stamp,
                detected=False,
                u_mid=math.nan,
                v_mid=math.nan,
                yaw_est=math.nan,
                u_red=math.nan,
                v_red=math.nan,
                red_area_px=math.nan,
                u_blue=math.nan,
                v_blue=math.nan,
                blue_area_px=math.nan,
                separation_px=math.nan,
                border_margin_px=math.nan,
                yolo_score=float(confidence),
                bbox_area_px=math.nan,
                bbox_xmin=math.nan,
                bbox_ymin=math.nan,
                bbox_xmax=math.nan,
                bbox_ymax=math.nan,
                class_id=math.nan,
                logit_margin=math.nan,
                class_entropy=math.nan,
                mask_area_px=math.nan,
                mask_bottom_u=math.nan,
                mask_bottom_v=math.nan,
                mask_used=0.0,
                mask_polygon_points=math.nan,
                confidence_logit=_confidence_logit(confidence) if confidence > 0.0 else math.nan,
                mask_compactness=math.nan,
                mask_border_frac=math.nan,
                mask_score=math.nan,
                selected_pixel_source_code=SELECTED_PIXEL_SOURCE_NONE,
            )
        )

    def _image_cb(self, msg: Image):
        image_bgr = image_msg_to_bgr8(msg)
        results = self._predict(image_bgr)
        if not results:
            self._publish_failure(msg.header.stamp)
            return

        detection = self._select_detection(results[0])
        if detection is None:
            self._publish_failure(msg.header.stamp)
            return

        x0, y0, x1, y1 = detection['bbox']
        mask_ref = self._mask_reference(results[0], detection)
        if mask_ref is not None:
            selected_u = float(mask_ref['u'])
            selected_v = float(mask_ref['v'])
            selected_pixel_source_code = SELECTED_PIXEL_SOURCE_MASK_BOTTOM
            mask_used = 1.0
            mask_area = float(mask_ref['area'])
            mask_points = float(mask_ref['n_points'])
            mask_bottom_u = selected_u
            mask_bottom_v = selected_v
        else:
            selected_u = float(0.5 * (x0 + x1))
            selected_v = float(y1)
            selected_pixel_source_code = SELECTED_PIXEL_SOURCE_BBOX_BOTTOM
            mask_used = 0.0
            mask_area = math.nan
            mask_points = math.nan
            mask_bottom_u = math.nan
            mask_bottom_v = math.nan

        if self.pixel_noise_sigma > 0.0:
            selected_u += float(self.rng.normal(0.0, self.pixel_noise_sigma))
            selected_v += float(self.rng.normal(0.0, self.pixel_noise_sigma))

        h, w = image_bgr.shape[:2]
        border_margin = float(min(selected_u, selected_v, w - 1.0 - selected_u, h - 1.0 - selected_v))
        bbox_area = float(max(x1 - x0, 0.0) * max(y1 - y0, 0.0))
        confidence = float(detection['confidence'])

        self.diag_pub.publish(
            diagnostics_message(
                stamp=float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9,
                detected=True,
                u_mid=selected_u,
                v_mid=selected_v,
                yaw_est=math.nan,
                u_red=math.nan,
                v_red=math.nan,
                red_area_px=math.nan,
                u_blue=math.nan,
                v_blue=math.nan,
                blue_area_px=math.nan,
                separation_px=math.nan,
                border_margin_px=border_margin,
                yolo_score=confidence,
                bbox_area_px=bbox_area,
                bbox_xmin=x0,
                bbox_ymin=y0,
                bbox_xmax=x1,
                bbox_ymax=y1,
                class_id=float(detection['class_id']),
                logit_margin=math.nan,
                class_entropy=math.nan,
                mask_area_px=mask_area,
                mask_bottom_u=mask_bottom_u,
                mask_bottom_v=mask_bottom_v,
                mask_used=mask_used,
                mask_polygon_points=mask_points,
                confidence_logit=_confidence_logit(confidence),
                mask_compactness=math.nan,
                mask_border_frac=math.nan,
                mask_score=confidence,
                selected_pixel_source_code=selected_pixel_source_code,
            )
        )

        out = PoseStamped()
        out.header = msg.header
        out.header.frame_id = 'image'
        out.pose.position.x = float(selected_u)
        out.pose.position.y = float(selected_v)
        out.pose.position.z = 0.0
        out.pose.orientation.w = 1.0
        self.pixel_pub.publish(out)


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
