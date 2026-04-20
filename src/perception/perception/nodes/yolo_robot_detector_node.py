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
from perception.core.yolo_selection import select_best_detection, target_class_ids


SELECTED_PIXEL_SOURCE_NONE = 0.0
SELECTED_PIXEL_SOURCE_BBOX_BOTTOM = 1.0
SELECTED_PIXEL_SOURCE_MASK_BOTTOM = 2.0


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on')


def _confidence_logit(score: float, eps: float = 1e-6) -> float:
    p = float(np.clip(score, eps, 1.0 - eps))
    return float(math.log(p / (1.0 - p)))


def _selected_pixel_source_code(source: str) -> float:
    source = str(source or '').strip().lower()
    if source == 'mask_bottom':
        return SELECTED_PIXEL_SOURCE_MASK_BOTTOM
    if source == 'bbox_bottom':
        return SELECTED_PIXEL_SOURCE_BBOX_BOTTOM
    return SELECTED_PIXEL_SOURCE_NONE


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
        self.target_ids = target_class_ids(getattr(self.model, 'names', {}), self.class_name, self.class_id)
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
            'conf': 0.0,
            'iou': self.iou_threshold,
            'verbose': False,
        }
        if self.device:
            kwargs['device'] = self.device
        return self.model.predict(**kwargs)

    def _publish_diagnostics(self, stamp_msg, selection: dict | None) -> None:
        selection = dict(selection or {})
        stamp = float(stamp_msg.sec) + float(stamp_msg.nanosec) * 1e-9
        detected_after_threshold = bool(selection.get('detected_after_threshold', False))
        bbox = selection.get('bbox_xyxy')
        if bbox is None:
            x0 = y0 = x1 = y1 = math.nan
        else:
            x0, y0, x1, y1 = [float(v) for v in np.asarray(bbox, dtype=float).reshape(4)]
        selected_u = float(selection.get('selected_u', math.nan))
        selected_v = float(selection.get('selected_v', math.nan))
        raw_score = float(selection.get('raw_best_score', 0.0) or 0.0)
        selected_score = float(selection.get('selected_score', raw_score) or raw_score)
        mask_area = float(selection.get('mask_area', math.nan))
        mask_available = int(selection.get('mask_available', 0) or 0)
        mask_bottom_u = selected_u if mask_available else math.nan
        mask_bottom_v = selected_v if mask_available else math.nan
        mask_points = float(
            selection.get('detection', {}).get('polygon').shape[0]
        ) if mask_available and selection.get('detection', {}).get('polygon') is not None else math.nan
        if math.isfinite(selected_u) and math.isfinite(selected_v):
            h = float(getattr(self, '_latest_image_shape', (0, 0))[0] or 0.0)
            w = float(getattr(self, '_latest_image_shape', (0, 0))[1] or 0.0)
            border_margin = float(min(selected_u, selected_v, w - 1.0 - selected_u, h - 1.0 - selected_v))
        else:
            border_margin = math.nan
        bbox_area = float(max(x1 - x0, 0.0) * max(y1 - y0, 0.0)) if math.isfinite(x0) else math.nan
        self.diag_pub.publish(
            diagnostics_message(
                stamp=stamp,
                detected=detected_after_threshold,
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
                yolo_score_raw=raw_score,
                yolo_score_selected=selected_score,
                yolo_detected_after_threshold=1.0 if detected_after_threshold else 0.0,
                yolo_best_class_id=float(selection.get('best_class_id', math.nan)),
                yolo_target_candidate_count=float(selection.get('n_candidates', 0)),
                bbox_area_px=bbox_area,
                bbox_xmin=x0,
                bbox_ymin=y0,
                bbox_xmax=x1,
                bbox_ymax=y1,
                logit_margin=math.nan,
                class_entropy=math.nan,
                mask_area_px=mask_area,
                mask_bottom_u=mask_bottom_u,
                mask_bottom_v=mask_bottom_v,
                mask_used=1.0 if mask_available else 0.0,
                mask_polygon_points=mask_points,
                confidence_logit=_confidence_logit(raw_score) if raw_score > 0.0 else math.nan,
                mask_compactness=math.nan,
                mask_border_frac=math.nan,
                mask_score=selected_score if mask_available else math.nan,
                selected_pixel_source_code=_selected_pixel_source_code(selection.get('selected_pixel_source', 'none')),
            )
        )

    def _image_cb(self, msg: Image):
        image_bgr = image_msg_to_bgr8(msg)
        self._latest_image_shape = image_bgr.shape[:2]
        results = self._predict(image_bgr)
        if not results:
            self._publish_diagnostics(msg.header.stamp, None)
            return

        selection = select_best_detection(
            results[0],
            target_ids=self.target_ids,
            confidence_threshold=float(self.confidence_threshold),
            use_masks=self.use_masks,
            mask_min_area=float(self.mask_min_area),
            mask_bottom_band_px=float(self.mask_bottom_band_px),
        )
        if selection.get('bbox_xyxy') is None:
            self._publish_diagnostics(msg.header.stamp, selection)
            return

        selected_u = float(selection.get('selected_u', math.nan))
        selected_v = float(selection.get('selected_v', math.nan))

        if self.pixel_noise_sigma > 0.0:
            selected_u += float(self.rng.normal(0.0, self.pixel_noise_sigma))
            selected_v += float(self.rng.normal(0.0, self.pixel_noise_sigma))
        selection['selected_u'] = selected_u
        selection['selected_v'] = selected_v
        self._publish_diagnostics(msg.header.stamp, selection)
        if not bool(selection.get('detected_after_threshold', False)):
            return

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
