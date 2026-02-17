#!/usr/bin/env python3
import math
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image

from perception.core.camera_config import load_camera_params
from perception.core.homography import HomographyModel


def _build_dict_name_to_id():
    if not hasattr(cv2, "aruco"):
        return {}
    return {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
    }


class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__("aruco_detector_node")

        self.declare_parameter("cam_pos", [-3.0, -3.0, 6.0])
        self.declare_parameter("look_at", [1.5, 1.5, 0.0])
        self.declare_parameter("img_width", 1280)
        self.declare_parameter("img_height", 720)
        self.declare_parameter("fov_h_rad", 1.5708)
        self.declare_parameter("aruco_dict", "DICT_4X4_50")
        self.declare_parameter("target_marker_id", 0)
        self.declare_parameter("publish_yaw_from_marker", True)
        self.declare_parameter("marker_size_m", 0.5)
        self.declare_parameter("marker_yaw_offset_rad", 0.0)
        self.declare_parameter("enable_multiscale_detection", False)
        self.declare_parameter("multiscale_every_n_misses", 15)
        self.declare_parameter("enable_corner_tracking", True)
        self.declare_parameter("tracking_max_gap_frames", 20)
        self.declare_parameter("tracking_min_quad_area_px2", 80.0)
        self.declare_parameter("tracking_max_edge_ratio", 3.0)
        self.declare_parameter("enable_template_fallback", True)
        self.declare_parameter("template_fallback_every_n_misses", 12)
        self.declare_parameter("template_fallback_max_candidates", 60)
        self.declare_parameter("template_fallback_roi_margin_px", 220)
        self.declare_parameter("template_fallback_downscale", 0.5)
        self.declare_parameter("marker_template_path", "")
        self.declare_parameter("template_match_threshold", 0.55)

        (
            self.cam_pos,
            self.look_at,
            self.img_width,
            self.img_height,
            self.fov_h_rad,
        ) = load_camera_params(self)
        self.model = HomographyModel(
            cam_pos=self.cam_pos,
            look_at=self.look_at,
            img_width=self.img_width,
            img_height=self.img_height,
            fov_h_rad=self.fov_h_rad,
        )

        dict_name_to_id = _build_dict_name_to_id()
        if not dict_name_to_id:
            raise RuntimeError(
                "OpenCV ArUco module is unavailable. Install python3-opencv with aruco support."
            )

        dict_name = str(self.get_parameter("aruco_dict").value)
        if dict_name not in dict_name_to_id:
            known = ", ".join(sorted(dict_name_to_id.keys()))
            raise RuntimeError(f"Unknown aruco_dict '{dict_name}'. Known: {known}")
        dict_id = dict_name_to_id[dict_name]

        self.target_marker_id = int(self.get_parameter("target_marker_id").value)
        self.publish_yaw_from_marker = bool(
            self.get_parameter("publish_yaw_from_marker").value
        )
        self.marker_size_m = float(self.get_parameter("marker_size_m").value)
        self.marker_yaw_offset_rad = float(
            self.get_parameter("marker_yaw_offset_rad").value
        )
        self.enable_multiscale_detection = bool(
            self.get_parameter("enable_multiscale_detection").value
        )
        self.multiscale_every_n_misses = max(
            1, int(self.get_parameter("multiscale_every_n_misses").value)
        )
        self.enable_corner_tracking = bool(
            self.get_parameter("enable_corner_tracking").value
        )
        self.tracking_max_gap_frames = max(
            0, int(self.get_parameter("tracking_max_gap_frames").value)
        )
        self.tracking_min_quad_area_px2 = float(
            self.get_parameter("tracking_min_quad_area_px2").value
        )
        self.tracking_max_edge_ratio = float(
            self.get_parameter("tracking_max_edge_ratio").value
        )
        self.enable_template_fallback = bool(
            self.get_parameter("enable_template_fallback").value
        )
        self.template_fallback_every_n_misses = max(
            1, int(self.get_parameter("template_fallback_every_n_misses").value)
        )
        self.template_fallback_max_candidates = max(
            1, int(self.get_parameter("template_fallback_max_candidates").value)
        )
        self.template_fallback_roi_margin_px = max(
            0, int(self.get_parameter("template_fallback_roi_margin_px").value)
        )
        self.template_fallback_downscale = float(
            self.get_parameter("template_fallback_downscale").value
        )
        self.template_match_threshold = float(
            self.get_parameter("template_match_threshold").value
        )

        self.multiscale_scales = (1.5, 2.0)
        self._template_patch_size = 200
        self._camera_matrix = self.model.camera.K.astype(np.float64)
        self._camera_rotation_world_to_cam = self.model.camera.R.astype(np.float64)
        self._camera_rotation_cam_to_world = self._camera_rotation_world_to_cam.T
        half_size = max(1e-4, 0.5 * self.marker_size_m)
        self._marker_object_points = np.array(
            [
                [-half_size, half_size, 0.0],
                [half_size, half_size, 0.0],
                [half_size, -half_size, 0.0],
                [-half_size, -half_size, 0.0],
            ],
            dtype=np.float32,
        )
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        self.bridge = CvBridge()
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        if hasattr(cv2.aruco, "DetectorParameters"):
            self.detector_params = cv2.aruco.DetectorParameters()
        elif hasattr(cv2.aruco, "DetectorParameters_create"):
            self.detector_params = cv2.aruco.DetectorParameters_create()
        else:
            raise RuntimeError(
                "OpenCV aruco module does not expose DetectorParameters API."
            )
        self._configure_detector_params()
        self.detector = (
            cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)
            if hasattr(cv2.aruco, "ArucoDetector")
            else None
        )
        self.template_variants = self._load_template_variants()
        if self.enable_template_fallback and not self.template_variants:
            self.get_logger().warn(
                "Template fallback enabled but marker template could not be loaded; "
                "fallback will be skipped."
            )

        self.pub = self.create_publisher(PoseStamped, "/perception/pixel_pose", 10)
        self.create_subscription(Image, "/external_camera/image_raw", self._image_cb, 10)

        self._prev_gray: Optional[np.ndarray] = None
        self._tracked_pts: Optional[np.ndarray] = None
        self._tracked_marker_id: Optional[int] = None
        self._tracking_gap = 0
        self._missed_frames = 0
        self._last_center: Optional[Tuple[float, float]] = None
        self._log_counter = 0

        self.get_logger().info(
            "aruco_detector_node started "
            f"dict={dict_name} target_marker_id={self.target_marker_id} "
            f"marker_size_m={self.marker_size_m:.3f} "
            f"multiscale={self.enable_multiscale_detection} "
            f"corner_tracking={self.enable_corner_tracking} "
            f"template_fallback={self.enable_template_fallback}"
        )

    def _configure_detector_params(self):
        # Tuned for oblique, moderate-size fiducials from the external camera.
        self.detector_params.minMarkerPerimeterRate = 0.005
        self.detector_params.maxMarkerPerimeterRate = 4.0
        self.detector_params.polygonalApproxAccuracyRate = 0.06
        self.detector_params.minCornerDistanceRate = 0.01
        self.detector_params.minDistanceToBorder = 2
        if hasattr(self.detector_params, "adaptiveThreshWinSizeMin"):
            self.detector_params.adaptiveThreshWinSizeMin = 3
            self.detector_params.adaptiveThreshWinSizeMax = 53
            self.detector_params.adaptiveThreshWinSizeStep = 4
        if hasattr(self.detector_params, "cornerRefinementMethod"):
            if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
                self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.detector_params.cornerRefinementWinSize = 5
            self.detector_params.cornerRefinementMaxIterations = 50
            self.detector_params.cornerRefinementMinAccuracy = 0.01
        if hasattr(self.detector_params, "detectInvertedMarker"):
            self.detector_params.detectInvertedMarker = True

    def _resolve_template_path(self) -> Optional[str]:
        user_path = str(self.get_parameter("marker_template_path").value).strip()
        if user_path:
            return user_path
        try:
            sim_share = get_package_share_directory("sim")
        except Exception:
            return None
        return os.path.join(
            sim_share, "robot_description", "meshes", "aruco", "aruco_4x4_50_id0.png"
        )

    def _load_template_variants(self):
        path = self._resolve_template_path()
        if not path:
            return []
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.get_logger().warn(f"Failed to load marker template from: {path}")
            return []

        # Normalize to a binary template for robust matching under illumination changes.
        _, base = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        base = cv2.resize(
            base,
            (self._template_patch_size, self._template_patch_size),
            interpolation=cv2.INTER_NEAREST,
        )
        variants = []
        for rot in range(4):
            r = np.ascontiguousarray(np.rot90(base, rot))
            variants.append(r)
            variants.append(cv2.flip(r, 1))
        self.get_logger().info(f"Loaded marker template from: {path}")
        return variants

    @staticmethod
    def _order_quad(pts: np.ndarray) -> np.ndarray:
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).reshape(-1)
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(s)]  # top-left
        ordered[2] = pts[np.argmax(s)]  # bottom-right
        ordered[1] = pts[np.argmin(d)]  # top-right
        ordered[3] = pts[np.argmax(d)]  # bottom-left
        return ordered

    @staticmethod
    def _binary_match_score(a: np.ndarray, b: np.ndarray) -> float:
        # Score in [0,1]; 1 means identical bitmaps.
        diff = cv2.absdiff(a, b)
        return 1.0 - float(np.mean(diff) / 255.0)

    def _is_valid_quad(self, pts: np.ndarray) -> bool:
        if pts.shape != (4, 2):
            return False
        q = pts.astype(np.float32)
        if not cv2.isContourConvex(q):
            return False
        area = abs(float(cv2.contourArea(q)))
        if area < self.tracking_min_quad_area_px2:
            return False
        edge_lengths = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
        if np.min(edge_lengths) < 3.0:
            return False
        ratio = float(np.max(edge_lengths) / max(np.min(edge_lengths), 1e-6))
        if ratio > self.tracking_max_edge_ratio:
            return False
        return True

    def _quad_roi(
        self,
        pts: np.ndarray,
        shape: Tuple[int, int],
        margin: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        height, width = shape
        x0 = max(0, int(np.floor(float(np.min(pts[:, 0])))) - margin)
        y0 = max(0, int(np.floor(float(np.min(pts[:, 1])))) - margin)
        x1 = min(width, int(np.ceil(float(np.max(pts[:, 0])))) + margin)
        y1 = min(height, int(np.ceil(float(np.max(pts[:, 1])))) + margin)
        if x1 - x0 < 20 or y1 - y0 < 20:
            return None
        return x0, y0, x1, y1

    def _center_roi(
        self,
        center: Tuple[float, float],
        shape: Tuple[int, int],
        margin: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        height, width = shape
        cx, cy = center
        x0 = max(0, int(round(cx)) - margin)
        y0 = max(0, int(round(cy)) - margin)
        x1 = min(width, int(round(cx)) + margin)
        y1 = min(height, int(round(cy)) + margin)
        if x1 - x0 < 20 or y1 - y0 < 20:
            return None
        return x0, y0, x1, y1

    def _quad_template_score(self, gray: np.ndarray, quad: np.ndarray) -> float:
        if not self.template_variants:
            return -1.0
        dst = np.array(
            [
                [0, 0],
                [self._template_patch_size - 1, 0],
                [self._template_patch_size - 1, self._template_patch_size - 1],
                [0, self._template_patch_size - 1],
            ],
            dtype=np.float32,
        )
        H = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        patch = cv2.warpPerspective(gray, H, (self._template_patch_size, self._template_patch_size))
        _, patch_bin = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return max(
            self._binary_match_score(patch_bin, tmpl) for tmpl in self.template_variants
        )

    def _template_fallback_detect(
        self,
        gray: np.ndarray,
        roi_hint: Optional[Tuple[int, int, int, int]],
    ):
        if not self.template_variants:
            return None

        work_img = gray
        offset_x = 0
        offset_y = 0
        if roi_hint is not None:
            x0, y0, x1, y1 = roi_hint
            roi = gray[y0:y1, x0:x1]
            if roi.size > 0:
                work_img = roi
                offset_x = x0
                offset_y = y0

        scale = self.template_fallback_downscale
        if scale <= 0.0:
            scale = 1.0
        if abs(scale - 1.0) > 1e-6:
            proc = cv2.resize(
                work_img,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            proc = work_img

        bw = cv2.adaptiveThreshold(
            proc,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            7,
        )
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contours = contours[: self.template_fallback_max_candidates]

        img_h, img_w = proc.shape[:2]
        min_area = 0.00008 * float(img_h * img_w)
        max_area = 0.08 * float(img_h * img_w)

        best = None
        best_score = -1.0
        inv_scale = 1.0 / scale
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            peri = cv2.arcLength(contour, True)
            if peri <= 1e-6:
                continue
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            quad = approx.reshape(4, 2).astype(np.float32)
            quad *= inv_scale
            quad[:, 0] += float(offset_x)
            quad[:, 1] += float(offset_y)
            quad = self._order_quad(quad)
            if not self._is_valid_quad(quad):
                continue

            local_best = self._quad_template_score(gray, quad)
            if local_best > best_score:
                best_score = local_best
                best = quad

        if best is None or best_score < self.template_match_threshold:
            return None
        return best, best_score

    def _detect_once(self, gray: np.ndarray):
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
            return corners, ids
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.detector_params
        )
        return corners, ids

    def _detect(self, gray: np.ndarray, allow_multiscale: bool = False):
        corners, ids = self._detect_once(gray)
        if ids is not None:
            return corners, ids
        if not (allow_multiscale and self.enable_multiscale_detection):
            return corners, ids

        for scale in self.multiscale_scales:
            up = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LINEAR,
            )
            sc_corners, sc_ids = self._detect_once(up)
            if sc_ids is None:
                continue
            scaled_back = [c.astype(np.float32) / scale for c in sc_corners]
            return scaled_back, sc_ids
        return corners, ids

    def _select_marker(
        self, corners: list, ids: Optional[np.ndarray]
    ) -> Optional[Tuple[np.ndarray, int]]:
        if ids is None or len(corners) == 0:
            return None
        ids_flat = ids.reshape(-1)

        if self.target_marker_id >= 0:
            for i, marker_id in enumerate(ids_flat):
                if int(marker_id) == self.target_marker_id:
                    return corners[i].reshape(4, 2).astype(np.float32), int(marker_id)
            return None

        # If target_marker_id < 0: pick largest marker by perimeter.
        best_i = 0
        best_perim = -1.0
        for i, marker_corners in enumerate(corners):
            pts = marker_corners.reshape(4, 2)
            perim = cv2.arcLength(pts.astype(np.float32), True)
            if perim > best_perim:
                best_perim = perim
                best_i = i
        return corners[best_i].reshape(4, 2).astype(np.float32), int(ids_flat[best_i])

    def _detect_in_roi(
        self,
        gray: np.ndarray,
        roi: Tuple[int, int, int, int],
    ) -> Optional[Tuple[np.ndarray, int]]:
        x0, y0, x1, y1 = roi
        patch = gray[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        corners, ids = self._detect(patch, allow_multiscale=True)
        selected = self._select_marker(corners, ids)
        if selected is None:
            return None
        pts, marker_id = selected
        pts = pts.copy()
        pts[:, 0] += float(x0)
        pts[:, 1] += float(y0)
        return pts, marker_id

    def _track_quad(self, gray: np.ndarray) -> Optional[np.ndarray]:
        if not self.enable_corner_tracking:
            return None
        if self._prev_gray is None or self._tracked_pts is None:
            return None
        if self._tracking_gap >= self.tracking_max_gap_frames:
            return None
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            gray,
            self._tracked_pts.astype(np.float32),
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if next_pts is None or status is None:
            return None
        status_flat = status.reshape(-1)
        if int(np.count_nonzero(status_flat)) < 4:
            return None
        tracked = next_pts.reshape(4, 2).astype(np.float32)
        tracked = self._order_quad(tracked)
        if not self._is_valid_quad(tracked):
            return None
        return tracked

    def _yaw_from_corners(self, pts: np.ndarray) -> Optional[float]:
        image_points = pts.astype(np.float32).reshape((4, 1, 2))
        pnp_flags = (
            cv2.SOLVEPNP_IPPE_SQUARE
            if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
            else cv2.SOLVEPNP_ITERATIVE
        )
        success, rvec, _ = cv2.solvePnP(
            self._marker_object_points,
            image_points,
            self._camera_matrix,
            self._dist_coeffs,
            flags=pnp_flags,
        )
        if not success:
            return None

        rot_cam_marker, _ = cv2.Rodrigues(rvec)
        rot_world_marker = self._camera_rotation_cam_to_world @ rot_cam_marker
        forward_world = rot_world_marker[:, 0]
        yaw = math.atan2(float(forward_world[1]), float(forward_world[0]))
        yaw += self.marker_yaw_offset_rad
        return math.atan2(math.sin(yaw), math.cos(yaw))

    def _image_cb(self, msg: Image):
        try:
            gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as exc:
            self.get_logger().warn(f"Failed to decode image: {exc}")
            return

        pts = None
        marker_id = self.target_marker_id if self.target_marker_id >= 0 else 0
        source = ""
        fallback_score = None

        tracked_guess = self._track_quad(gray)
        roi_hint = None
        if tracked_guess is not None:
            roi_hint = self._quad_roi(
                tracked_guess,
                gray.shape[:2],
                margin=max(80, self.template_fallback_roi_margin_px // 2),
            )
            if roi_hint is not None:
                selected_roi = self._detect_in_roi(gray, roi_hint)
                if selected_roi is not None:
                    pts, marker_id = selected_roi
                    source = "aruco_roi"

        if pts is None:
            allow_multiscale = (
                self.enable_multiscale_detection
                and self._missed_frames > 0
                and (self._missed_frames % self.multiscale_every_n_misses == 0)
            )
            corners, ids = self._detect(gray, allow_multiscale=allow_multiscale)
            selected = self._select_marker(corners, ids)
            if selected is not None:
                pts, marker_id = selected
                source = "aruco"

        if pts is None and tracked_guess is not None:
            pts = tracked_guess
            marker_id = (
                self._tracked_marker_id
                if self._tracked_marker_id is not None
                else marker_id
            )
            source = "track"
            self._tracking_gap += 1

        if pts is None and self.enable_template_fallback:
            if self._missed_frames % self.template_fallback_every_n_misses == 0:
                fallback_roi = roi_hint
                if fallback_roi is None and self._last_center is not None:
                    fallback_roi = self._center_roi(
                        self._last_center,
                        gray.shape[:2],
                        margin=self.template_fallback_roi_margin_px,
                    )
                fallback = self._template_fallback_detect(gray, fallback_roi)
                if fallback is not None:
                    pts, fallback_score = fallback
                    marker_id = self.target_marker_id if self.target_marker_id >= 0 else 0
                    source = "template"

        if pts is None:
            self._missed_frames += 1
            self._prev_gray = gray
            if self._missed_frames > self.tracking_max_gap_frames:
                self._tracked_pts = None
            return

        pts = self._order_quad(pts.astype(np.float32))
        center = np.mean(pts, axis=0)
        u = float(center[0])
        v = float(center[1])

        yaw = 0.0
        if self.publish_yaw_from_marker:
            yaw_est = self._yaw_from_corners(pts)
            if yaw_est is not None:
                yaw = yaw_est

        out = PoseStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = "image"
        out.pose.position.x = u
        out.pose.position.y = v
        out.pose.position.z = 0.0
        out.pose.orientation.x = 0.0
        out.pose.orientation.y = 0.0
        out.pose.orientation.z = math.sin(yaw / 2.0)
        out.pose.orientation.w = math.cos(yaw / 2.0)
        self.pub.publish(out)

        self._prev_gray = gray
        self._tracked_pts = pts
        self._tracked_marker_id = marker_id
        self._last_center = (u, v)
        if source != "track":
            self._tracking_gap = 0
        self._missed_frames = 0

        self._log_counter += 1
        if self._log_counter % 20 == 0:
            suffix = ""
            if fallback_score is not None:
                suffix = f" score={fallback_score:.3f}"
            self.get_logger().info(
                f"Detected marker {marker_id} [{source}] -> Pix:({u:.1f},{v:.1f}) "
                f"yaw={yaw:.2f}rad{suffix}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
