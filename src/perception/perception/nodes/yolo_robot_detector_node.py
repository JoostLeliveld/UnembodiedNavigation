import math
import threading
import time
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
from perception.core.pose_extraction import (
    model_exposes_keypoints,
    select_pose_detection,
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
        self.declare_parameter('min_keypoint_conf', 0.5)

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
        self.min_keypoint_conf = float(self.get_parameter('min_keypoint_conf').value)
        self._last_inference_ms = math.nan
        self._last_callback_ms = math.nan
        self._last_receive_stamp_s = math.nan
        self._last_predict_start_stamp_s = math.nan
        self._last_predict_finish_stamp_s = math.nan
        self._latest_image_shape = (0, 0)
        self._pending_image = None
        self._pending_receive_stamp_s = math.nan
        self._image_lock = threading.Lock()
        self._image_event = threading.Event()
        self._worker_stop = threading.Event()

        self.model = YOLO(str(self.model_path))
        self.target_ids = target_class_ids(getattr(self.model, 'names', {}), self.class_name, self.class_id)
        if self.target_ids == set():
            names = getattr(self.model, 'names', {})
            raise RuntimeError(
                f'class filter does not match any YOLO class names: class_name={self.class_name!r}, '
                f'class_id={self.class_id}, names={names!r}'
            )
        self.is_pose_model = model_exposes_keypoints(self.model)

        self.pixel_pub = self.create_publisher(PoseStamped, '/perception/pixel_pose', 10)
        self.diag_pub = self.create_publisher(Float64MultiArray, DETECTION_DIAGNOSTICS_TOPIC, 10)
        self.create_subscription(Image, '/external_camera/image_raw', self._image_cb, 1)
        self._worker = threading.Thread(target=self._inference_worker, daemon=True)
        self._worker.start()

        self.get_logger().info(
            f'YOLO runtime detector started (model={self.model_path}, conf={self.confidence_threshold:.2f}, '
            f'iou={self.iou_threshold:.2f}, use_masks={self.use_masks}, '
            f'task={"pose" if self.is_pose_model else "seg/det"}, device={self.device or "auto"}, '
            'input_policy=latest-frame-only)'
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

    def _publish_diagnostics(
        self,
        stamp_msg,
        selection: dict | None,
        pose_extra: dict | None = None,
    ) -> None:
        selection = dict(selection or {})
        pose_extra = dict(pose_extra or {})
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
        mask_bottom_u = float(selection.get('mask_bottom_u', math.nan)) if mask_available else math.nan
        mask_bottom_v = float(selection.get('mask_bottom_v', math.nan)) if mask_available else math.nan
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
        yaw_est = float(pose_extra.get('yaw_est', math.nan))
        u_front = float(pose_extra.get('u_front', math.nan))
        v_front = float(pose_extra.get('v_front', math.nan))
        u_rear = float(pose_extra.get('u_rear', math.nan))
        v_rear = float(pose_extra.get('v_rear', math.nan))
        if (
            math.isfinite(u_front) and math.isfinite(v_front)
            and math.isfinite(u_rear) and math.isfinite(v_rear)
        ):
            separation_px = float(math.hypot(u_front - u_rear, v_front - v_rear))
        else:
            separation_px = math.nan
        publish_stamp_s = float(self.get_clock().now().nanoseconds) * 1e-9
        if math.isfinite(self._last_predict_start_stamp_s) and math.isfinite(self._last_predict_finish_stamp_s):
            yolo_latency_s = max(self._last_predict_finish_stamp_s - self._last_predict_start_stamp_s, 0.0)
        elif math.isfinite(self._last_inference_ms):
            yolo_latency_s = max(self._last_inference_ms * 1e-3, 0.0)
        else:
            yolo_latency_s = math.nan
        frame_age_at_publish_s = max(publish_stamp_s - stamp, 0.0) if math.isfinite(stamp) else math.nan
        self.diag_pub.publish(
            diagnostics_message(
                stamp=stamp,
                detected=detected_after_threshold,
                u_mid=selected_u,
                v_mid=selected_v,
                yaw_est=yaw_est,
                u_red=u_front,
                v_red=v_front,
                red_area_px=math.nan,
                u_blue=u_rear,
                v_blue=v_rear,
                blue_area_px=math.nan,
                separation_px=separation_px,
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
                yolo_inference_ms=self._last_inference_ms,
                detector_callback_ms=self._last_callback_ms,
                yolo_receive_stamp=self._last_receive_stamp_s,
                yolo_start_stamp=self._last_predict_start_stamp_s,
                yolo_finish_stamp=self._last_predict_finish_stamp_s,
                yolo_publish_stamp=publish_stamp_s,
                yolo_latency_s=yolo_latency_s,
                frame_age_at_publish_s=frame_age_at_publish_s,
            )
        )

    def _image_cb(self, msg: Image):
        receive_stamp_s = float(self.get_clock().now().nanoseconds) * 1e-9
        with self._image_lock:
            self._pending_image = msg
            self._pending_receive_stamp_s = receive_stamp_s
        self._image_event.set()

    def _take_latest_image(self):
        with self._image_lock:
            msg = self._pending_image
            receive_stamp_s = self._pending_receive_stamp_s
            self._pending_image = None
            self._pending_receive_stamp_s = math.nan
        return msg, receive_stamp_s

    def _inference_worker(self):
        while not self._worker_stop.is_set():
            self._image_event.wait(timeout=0.1)
            if self._worker_stop.is_set():
                break
            self._image_event.clear()
            msg, receive_stamp_s = self._take_latest_image()
            if msg is None:
                continue
            self._process_image(msg, receive_stamp_s)
            with self._image_lock:
                has_newer_image = self._pending_image is not None
            if has_newer_image:
                self._image_event.set()

    def _process_image(self, msg: Image, receive_stamp_s: float):
        callback_start = time.perf_counter()
        self._last_receive_stamp_s = receive_stamp_s
        image_bgr = image_msg_to_bgr8(msg)
        self._latest_image_shape = image_bgr.shape[:2]
        predict_start = time.perf_counter()
        self._last_predict_start_stamp_s = float(self.get_clock().now().nanoseconds) * 1e-9
        results = self._predict(image_bgr)
        self._last_predict_finish_stamp_s = float(self.get_clock().now().nanoseconds) * 1e-9
        self._last_inference_ms = max((time.perf_counter() - predict_start) * 1000.0, 0.0)
        self._last_callback_ms = max((time.perf_counter() - callback_start) * 1000.0, 0.0)
        if not results:
            self._publish_diagnostics(msg.header.stamp, None)
            return

        if self.is_pose_model:
            self._handle_pose_result(msg, results[0])
        else:
            self._handle_seg_result(msg, results[0])

    def destroy_node(self):
        self._worker_stop.set()
        self._image_event.set()
        worker = getattr(self, '_worker', None)
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)
        return super().destroy_node()

    def _handle_seg_result(self, msg: Image, result) -> None:
        selection = select_best_detection(
            result,
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

    def _handle_pose_result(self, msg: Image, result) -> None:
        boxes = getattr(result, 'boxes', None)
        keypoints = getattr(result, 'keypoints', None)
        if boxes is None or keypoints is None or len(boxes) == 0:
            self._publish_diagnostics(msg.header.stamp, None)
            return

        boxes_xyxy = boxes.xyxy.detach().cpu().numpy()
        boxes_conf = boxes.conf.detach().cpu().numpy()
        boxes_cls = boxes.cls.detach().cpu().numpy().astype(int)
        kpt_data = keypoints.data.detach().cpu().numpy()

        sel = select_pose_detection(
            boxes_xyxy=boxes_xyxy,
            boxes_conf=boxes_conf,
            boxes_cls=boxes_cls,
            keypoints_data=kpt_data,
            target_ids=self.target_ids,
            confidence_threshold=float(self.confidence_threshold),
            min_keypoint_conf=float(self.min_keypoint_conf),
        )

        if sel.bbox_xyxy is None:
            self._publish_diagnostics(msg.header.stamp, None)
            return

        x0, y0, x1, y1 = sel.bbox_xyxy
        # Apply pixel noise consistently to the centre AND each keypoint so the
        # reported yaw stays internally consistent with what state node sees.
        if self.pixel_noise_sigma > 0.0:
            noise = self.rng.normal(0.0, self.pixel_noise_sigma, size=6)
            front_u = float(sel.front_u + noise[0])
            front_v = float(sel.front_v + noise[1])
            rear_u = float(sel.rear_u + noise[2])
            rear_v = float(sel.rear_v + noise[3])
        else:
            front_u, front_v = float(sel.front_u), float(sel.front_v)
            rear_u, rear_v = float(sel.rear_u), float(sel.rear_v)
        if math.isfinite(front_u) and math.isfinite(rear_u):
            selected_u = 0.5 * (front_u + rear_u)
            selected_v = 0.5 * (front_v + rear_v)
        else:
            selected_u = float(sel.selected_u)
            selected_v = float(sel.selected_v)

        detected_after_threshold = bool(sel.valid)
        selection = {
            'detected_after_threshold': detected_after_threshold,
            'bbox_xyxy': sel.bbox_xyxy,
            'selected_u': selected_u,
            'selected_v': selected_v,
            'raw_best_score': float(sel.box_score),
            'selected_score': float(sel.box_score),
            'mask_available': 0,
            'best_class_id': math.nan,
            'n_candidates': int(boxes_conf.size),
            'selected_pixel_source': 'pose_keypoints',
        }
        if detected_after_threshold and math.isfinite(front_u) and math.isfinite(rear_u):
            yaw_image = math.atan2(front_v - rear_v, front_u - rear_u)
        else:
            yaw_image = math.nan
        pose_extra = {
            'yaw_est': yaw_image,
            'u_front': front_u, 'v_front': front_v,
            'u_rear': rear_u, 'v_rear': rear_v,
        }
        self._publish_diagnostics(msg.header.stamp, selection, pose_extra=pose_extra)
        if not detected_after_threshold:
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
