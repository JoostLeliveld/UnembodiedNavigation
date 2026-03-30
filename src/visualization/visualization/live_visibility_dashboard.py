#!/usr/bin/env python3
"""Live sanity dashboard for the warehouse visibility experiments."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from matplotlib.lines import Line2D
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Float64MultiArray
from tf2_geometry_msgs import do_transform_pose

from perception.core.detection_diagnostics import (
    DETECTION_DIAGNOSTICS_TOPIC,
    diagnostics_from_message,
)


DASHBOARD_THEME = {
    'figure_bg': '#f3efe6',
    'panel_bg': '#fffaf2',
    'panel_edge': '#d8cdbd',
    'grid': '#e7ddd0',
    'text': '#1f1a17',
    'muted': '#6e6459',
    'truth': '#7d8b99',
    'state': '#20344a',
    'belief': '#c96c2d',
    'plan': '#d94841',
    'goal': '#1f8a70',
    'accent': '#2f6690',
    'exec': '#2b7a4b',
    'warning': '#b7791f',
    'danger': '#b13a2f',
    'purple': '#7a5ea6',
    'brown': '#8a5a44',
    'ok_fill': '#dff2e4',
    'warn_fill': '#f8ebcb',
    'danger_fill': '#f6d5d1',
    'info_fill': '#e3edf5',
}


def _stamp_to_float(stamp_msg) -> float:
    return float(stamp_msg.sec) + float(stamp_msg.nanosec) * 1e-9


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _path_xy(msg: Path) -> np.ndarray:
    pts = []
    for pose_stamped in msg.poses:
        pts.append([float(pose_stamped.pose.position.x), float(pose_stamped.pose.position.y)])
    if not pts:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(pts, dtype=float)


def _plan_length(xy: np.ndarray) -> float:
    if xy.shape[0] < 2:
        return 0.0
    diffs = np.diff(xy, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _pose_covariance(msg: PoseWithCovarianceStamped) -> tuple[float, float, float]:
    cov = list(msg.pose.covariance)
    cov_x = float(cov[0]) if len(cov) > 0 else math.nan
    cov_y = float(cov[7]) if len(cov) > 7 else math.nan
    cov_yaw = float(cov[35]) if len(cov) > 35 else math.nan
    return cov_x, cov_y, cov_yaw


def _series_arrays(series: deque[tuple[float, ...]]) -> np.ndarray:
    if not series:
        return np.zeros((0, 0), dtype=float)
    return np.asarray(series, dtype=float)


@dataclass
class GridLayer:
    values: np.ndarray
    extent: tuple[float, float, float, float]
    stamp: float


class LiveVisibilityDashboard(Node):
    def __init__(self):
        super().__init__('live_visibility_dashboard')

        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)
        self.declare_parameter('history_s', 60.0)
        self.declare_parameter('redraw_hz', 4.0)
        self.declare_parameter('frame_id', 'map_bev')
        self.declare_parameter('visibility_map_topic', '/visibility_map')
        self.declare_parameter('visibility_opacity_mean_map_topic', '/visibility_opacity_mean_map')
        self.declare_parameter('visibility_opacity_conservative_map_topic', '/visibility_opacity_conservative_map')
        self.declare_parameter('plan_topic', '/plan_preview')
        self.declare_parameter('planner_diag_topic', '/planner/diagnostics')
        self.declare_parameter('planner_belief_topic', '/planner_belief')
        self.declare_parameter('efe_metrics_topic', '/efe/metrics')
        self.declare_parameter('pixel_topic', '/perception/pixel_pose')

        self.history_s = max(10.0, float(self.get_parameter('history_s').value))
        self.redraw_hz = max(1.0, float(self.get_parameter('redraw_hz').value))
        self.frame_id = str(self.get_parameter('frame_id').value).strip() or 'map_bev'
        self.visibility_map_topic = str(self.get_parameter('visibility_map_topic').value).strip() or '/visibility_map'
        self.visibility_opacity_mean_map_topic = (
            str(self.get_parameter('visibility_opacity_mean_map_topic').value).strip()
            or '/visibility_opacity_mean_map'
        )
        self.visibility_opacity_conservative_map_topic = (
            str(self.get_parameter('visibility_opacity_conservative_map_topic').value).strip()
            or '/visibility_opacity_conservative_map'
        )
        self.plan_topic = str(self.get_parameter('plan_topic').value).strip() or '/plan_preview'
        self.planner_diag_topic = str(self.get_parameter('planner_diag_topic').value).strip() or '/planner/diagnostics'
        self.planner_belief_topic = (
            str(self.get_parameter('planner_belief_topic').value).strip() or '/planner_belief'
        )
        self.efe_metrics_topic = str(self.get_parameter('efe_metrics_topic').value).strip() or '/efe/metrics'
        self.pixel_topic = str(self.get_parameter('pixel_topic').value).strip() or '/perception/pixel_pose'
        self._theme = DASHBOARD_THEME.copy()

        self._lock = threading.RLock()
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._visibility_layer: GridLayer | None = None
        self._opacity_mean_layer: GridLayer | None = None
        self._opacity_cons_layer: GridLayer | None = None
        self._goal_xy: tuple[float, float] | None = None
        self._latest_plan_xy = np.zeros((0, 2), dtype=float)
        self._latest_state = (math.nan, math.nan, math.nan)
        self._latest_belief = (math.nan, math.nan, math.nan)
        self._latest_truth = (math.nan, math.nan, math.nan)
        self._last_diag_stamp = math.nan
        self._last_pixel_stamp = math.nan

        self._state_traj: deque[tuple[float, float, float]] = deque()
        self._truth_traj: deque[tuple[float, float, float]] = deque()
        self._state_series: deque[tuple[float, float, float, float, float, float, float, float]] = deque()
        self._belief_series: deque[tuple[float, float, float, float, float, float, float, float, float]] = deque()
        self._detect_series: deque[tuple[float, float, float, float, float, float]] = deque()
        self._plan_series: deque[tuple[float, float, float]] = deque()
        self._efe_series: deque[tuple[float, ...]] = deque()
        self._diag_series: deque[tuple[float, ...]] = deque()
        self._cmd_series: deque[tuple[float, float, float]] = deque()

        state_qos = QoSProfile(depth=5)
        goal_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        path_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.create_subscription(PoseWithCovarianceStamped, '/state/bev', self._state_cb, state_qos)
        self.create_subscription(
            PoseWithCovarianceStamped, self.planner_belief_topic, self._planner_belief_cb, state_qos
        )
        self.create_subscription(Odometry, '/odom', self._odom_cb, state_qos)
        self.create_subscription(PoseStamped, '/goal_bev', self._goal_cb, goal_qos)
        self.create_subscription(Path, self.plan_topic, self._plan_cb, path_qos)
        self.create_subscription(Float64MultiArray, self.efe_metrics_topic, self._efe_cb, 10)
        self.create_subscription(Float64MultiArray, self.planner_diag_topic, self._planner_diag_cb, 10)
        self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)
        self.create_subscription(Float64MultiArray, DETECTION_DIAGNOSTICS_TOPIC, self._detection_cb, 10)
        self.create_subscription(PoseStamped, self.pixel_topic, self._pixel_cb, 10)
        self.create_subscription(OccupancyGrid, self.visibility_map_topic, self._visibility_map_cb, path_qos)
        self.create_subscription(
            OccupancyGrid,
            self.visibility_opacity_mean_map_topic,
            self._opacity_mean_cb,
            path_qos,
        )
        self.create_subscription(
            OccupancyGrid,
            self.visibility_opacity_conservative_map_topic,
            self._opacity_cons_cb,
            path_qos,
        )

        plt.ion()
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'axes.titleweight': 'bold',
            'axes.labelcolor': self._theme['muted'],
            'xtick.color': self._theme['muted'],
            'ytick.color': self._theme['muted'],
        })
        self.fig, self.axes = plt.subplots(
            3,
            3,
            figsize=(18.6, 10.9),
            constrained_layout=False,
            gridspec_kw={'height_ratios': [1.15, 0.95, 1.0]},
        )
        self.fig.patch.set_facecolor(self._theme['figure_bg'])
        self.fig.subplots_adjust(left=0.045, right=0.985, bottom=0.06, top=0.84, hspace=0.34, wspace=0.18)
        self._visibility_age_ax = self.axes[1, 0].twinx()
        self._mission_length_ax = self.axes[1, 1].twinx()
        self._belief_error_ax = self.axes[1, 2].twinx()
        self._efe_total_ax = self.axes[2, 0].twinx()
        self._planner_diag_ax = self.axes[2, 1].twinx()
        self._command_sanity_ax = self.axes[2, 2].twinx()
        self._secondary_axes = (
            self._visibility_age_ax,
            self._mission_length_ax,
            self._belief_error_ax,
            self._efe_total_ax,
            self._planner_diag_ax,
            self._command_sanity_ax,
        )
        for ax in self._secondary_axes:
            ax.set_facecolor('none')

        title = 'Warehouse Visibility Control Room'
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass
        self.fig.suptitle(title, fontsize=19, fontweight='bold', y=0.975, color=self._theme['text'])
        self._header_kicker = self.fig.text(
            0.5,
            0.948,
            'LIVE SANITY DASHBOARD',
            ha='center',
            va='center',
            fontsize=9,
            color=self._theme['muted'],
        )
        self._status_cards = [
            self.fig.text(0.14, 0.916, '', ha='center', va='center', fontsize=9.2, color=self._theme['text']),
            self.fig.text(0.38, 0.916, '', ha='center', va='center', fontsize=9.2, color=self._theme['text']),
            self.fig.text(0.62, 0.916, '', ha='center', va='center', fontsize=9.2, color=self._theme['text']),
            self.fig.text(0.86, 0.916, '', ha='center', va='center', fontsize=9.2, color=self._theme['text']),
        ]
        self._header_summary = self.fig.text(
            0.5,
            0.886,
            '',
            ha='center',
            va='center',
            fontsize=9.5,
            color=self._theme['text'],
        )
        map_handles = [
            Line2D([0], [0], color=self._theme['truth'], linewidth=1.7, label='truth'),
            Line2D([0], [0], color=self._theme['state'], linewidth=2.3, label='state'),
            Line2D([0], [0], color=self._theme['plan'], linewidth=2.0, linestyle=(0, (5, 3)), label='plan preview'),
            Line2D(
                [0],
                [0],
                marker='*',
                color='none',
                markerfacecolor=self._theme['goal'],
                markeredgecolor=self._theme['panel_bg'],
                markeredgewidth=1.2,
                markersize=12,
                label='goal',
            ),
        ]
        self._map_legend = self.fig.legend(
            handles=map_handles,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.858),
            ncol=4,
            fontsize=8,
            frameon=True,
            fancybox=True,
            framealpha=0.96,
            borderpad=0.45,
            handlelength=2.3,
            columnspacing=1.4,
        )
        legend_frame = self._map_legend.get_frame()
        legend_frame.set_facecolor(self._theme['panel_bg'])
        legend_frame.set_edgecolor(self._theme['panel_edge'])
        for ax in self.axes[0, :]:
            ax.set_box_aspect(1.0)

    def _style_axis(
        self,
        ax,
        *,
        title: str,
        xlabel: str | None = None,
        ylabel: str | None = None,
        grid: bool = True,
    ) -> None:
        ax.set_facecolor(self._theme['panel_bg'])
        for spine in ax.spines.values():
            spine.set_color(self._theme['panel_edge'])
            spine.set_linewidth(1.0)
        ax.tick_params(colors=self._theme['muted'], labelsize=8.5)
        ax.grid(grid, color=self._theme['grid'], linewidth=0.8, alpha=0.9)
        ax.set_title(title, loc='left', fontsize=11.6, fontweight='bold', color=self._theme['text'], pad=18)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=9, color=self._theme['muted'])
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=9, color=self._theme['muted'])

    def _style_twin_axis(self, ax, *, ylabel: str | None = None) -> None:
        for side in ('right', 'top'):
            ax.spines[side].set_color(self._theme['panel_edge'])
            ax.spines[side].set_linewidth(1.0)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(colors=self._theme['muted'], labelsize=8.0)
        ax.grid(False)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=8.5, color=self._theme['muted'])

    def _panel_caption(self, ax, text: str) -> None:
        ax.text(
            0.0,
            1.01,
            text,
            transform=ax.transAxes,
            ha='left',
            va='bottom',
            fontsize=8.2,
            color=self._theme['muted'],
            clip_on=False,
        )

    def _empty_panel(self, ax, message: str) -> None:
        ax.text(
            0.5,
            0.5,
            message,
            ha='center',
            va='center',
            transform=ax.transAxes,
            fontsize=10,
            color=self._theme['muted'],
        )

    def _time_window(self, *arrays: np.ndarray, warmup_end_s: float = math.nan) -> tuple[float, float, float]:
        finite_max = [float(arr[-1, 0]) for arr in arrays if arr.size and np.isfinite(arr[-1, 0])]
        xmax = max(finite_max) if finite_max else 10.0
        max_window_s = min(self.history_s, 26.0)
        xmin = max(0.0, xmax - max_window_s)
        if np.isfinite(warmup_end_s):
            xmin = max(xmin, max(0.0, warmup_end_s - 2.5))
        xmin = min(xmin, max(0.0, xmax - 10.0))
        xmax = max(xmax, xmin + 10.0)
        return xmin, xmax, warmup_end_s

    def _apply_time_window(self, ax, time_window: tuple[float, float, float]) -> None:
        xmin, xmax, warmup_end_s = time_window
        ax.set_xlim(xmin, xmax)
        if np.isfinite(warmup_end_s) and warmup_end_s > xmin + 0.2:
            ax.axvspan(xmin, warmup_end_s, color=self._theme['grid'], alpha=0.45, linewidth=0.0, zorder=0)
            ax.axvline(warmup_end_s, color=self._theme['panel_edge'], linewidth=1.0, linestyle='--')
            ax.text(
                xmin + 0.02 * max(xmax - xmin, 1.0),
                0.95,
                'warm-up / no active plan',
                transform=ax.get_xaxis_transform(),
                ha='left',
                va='top',
                fontsize=8.0,
                color=self._theme['muted'],
            )

    def _legend(
        self,
        ax,
        handles,
        labels,
        *,
        ncol: int = 1,
        loc: str = 'upper right',
        bbox_to_anchor=None,
    ) -> None:
        if not handles:
            return
        legend = ax.legend(
            handles,
            labels,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            ncol=ncol,
            fontsize=7.7,
            frameon=True,
            fancybox=True,
            framealpha=0.94,
            borderpad=0.45,
            handlelength=2.0,
            columnspacing=1.0,
        )
        frame = legend.get_frame()
        frame.set_facecolor(self._theme['panel_bg'])
        frame.set_edgecolor(self._theme['panel_edge'])

    def _metric_status(
        self,
        value: float,
        *,
        warn: float,
        danger: float,
        inverse: bool = False,
    ) -> tuple[str, str]:
        if not np.isfinite(value):
            return 'n/a', self._theme['info_fill']
        if inverse:
            if value <= danger:
                return 'critical', self._theme['danger_fill']
            if value <= warn:
                return 'watch', self._theme['warn_fill']
            return 'stable', self._theme['ok_fill']
        if value >= danger:
            return 'critical', self._theme['danger_fill']
        if value >= warn:
            return 'watch', self._theme['warn_fill']
        return 'stable', self._theme['ok_fill']

    def _set_status_card(self, idx: int, title: str, detail: str, fill_color: str) -> None:
        if idx < 0 or idx >= len(self._status_cards):
            return
        text = self._status_cards[idx]
        text.set_text(f'{title}\n{detail}')
        text.set_bbox({
            'boxstyle': 'round,pad=0.35',
            'facecolor': fill_color,
            'edgecolor': self._theme['panel_edge'],
            'linewidth': 1.0,
        })

    def _map_scale_note(self, ax, label: str, low_label: str, high_label: str) -> None:
        ax.text(
            1.0,
            1.01,
            f'{label}: 0 {low_label} | 1 {high_label}',
            transform=ax.transAxes,
            ha='right',
            va='bottom',
            fontsize=7.8,
            color=self._theme['muted'],
            clip_on=False,
        )

    def _now_s(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _trim(self, dq: deque[tuple[float, ...]], newest_t: float) -> None:
        cutoff = newest_t - self.history_s
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def _append(self, dq: deque[tuple[float, ...]], row: tuple[float, ...]) -> None:
        dq.append(row)
        self._trim(dq, row[0])

    def _grid_from_msg(self, msg: OccupancyGrid, *, invert: bool) -> GridLayer | None:
        width = int(msg.info.width)
        height = int(msg.info.height)
        if width <= 0 or height <= 0:
            return None
        raw = np.asarray(msg.data, dtype=float)
        if raw.size != width * height:
            return None
        values = raw.reshape(height, width)
        values = np.where(values < 0.0, np.nan, values / 100.0)
        if invert:
            values = 1.0 - values
        resolution = float(msg.info.resolution)
        x0 = float(msg.info.origin.position.x) + 0.5 * resolution
        y0 = float(msg.info.origin.position.y) + 0.5 * resolution
        x1 = x0 + resolution * max(width - 1, 0)
        y1 = y0 + resolution * max(height - 1, 0)
        stamp = _stamp_to_float(msg.header.stamp)
        return GridLayer(values=values, extent=(x0, x1, y0, y1), stamp=stamp)

    def _sample_layer(self, layer: GridLayer | None, xq: float, yq: float) -> float:
        if layer is None or not np.isfinite(xq) or not np.isfinite(yq):
            return math.nan
        grid = np.asarray(layer.values, dtype=float)
        height, width = grid.shape
        if width < 2 or height < 2:
            return math.nan
        x0, x1, y0, y1 = layer.extent
        xs = np.linspace(x0, x1, width)
        ys = np.linspace(y0, y1, height)
        ix = int(np.clip(np.searchsorted(xs, xq, side='right') - 1, 0, width - 2))
        iy = int(np.clip(np.searchsorted(ys, yq, side='right') - 1, 0, height - 2))
        xa, xb = xs[ix], xs[ix + 1]
        ya, yb = ys[iy], ys[iy + 1]
        tx = 0.0 if abs(xb - xa) < 1e-12 else (xq - xa) / (xb - xa)
        ty = 0.0 if abs(yb - ya) < 1e-12 else (yq - ya) / (yb - ya)
        tx = float(np.clip(tx, 0.0, 1.0))
        ty = float(np.clip(ty, 0.0, 1.0))
        z00 = grid[iy, ix]
        z10 = grid[iy, ix + 1]
        z01 = grid[iy + 1, ix]
        z11 = grid[iy + 1, ix + 1]
        z0 = (1.0 - tx) * z00 + tx * z10
        z1 = (1.0 - tx) * z01 + tx * z11
        return float((1.0 - ty) * z0 + ty * z1)

    def _transform_odom_pose(self, msg: Odometry) -> tuple[float, float, float] | None:
        source_frame = (msg.header.frame_id or 'odom').strip() or 'odom'
        pose = msg.pose.pose
        if source_frame != self.frame_id:
            try:
                tf_msg = self._tf_buffer.lookup_transform(self.frame_id, source_frame, rclpy.time.Time())
                pose = do_transform_pose(msg.pose.pose, tf_msg)
            except Exception:
                return None
        return (
            float(pose.position.x),
            float(pose.position.y),
            _yaw_from_quaternion(pose.orientation),
        )

    def _state_cb(self, msg: PoseWithCovarianceStamped) -> None:
        stamp = _stamp_to_float(msg.header.stamp)
        pose = msg.pose.pose
        x = float(pose.position.x)
        y = float(pose.position.y)
        yaw = _yaw_from_quaternion(pose.orientation)
        cov_x, cov_y, cov_yaw = _pose_covariance(msg)
        with self._lock:
            self._latest_state = (x, y, yaw)
            self._append(self._state_traj, (stamp, x, y))
            goal_dist = math.nan
            if self._goal_xy is not None:
                goal_dist = math.hypot(x - self._goal_xy[0], y - self._goal_xy[1])
            p_vis = self._sample_layer(self._visibility_layer, x, y)
            state_err = math.nan
            if np.isfinite(self._latest_truth[0]) and np.isfinite(self._latest_truth[1]):
                state_err = math.hypot(x - self._latest_truth[0], y - self._latest_truth[1])
            pixel_age = math.nan
            if np.isfinite(self._last_pixel_stamp):
                pixel_age = max(stamp - self._last_pixel_stamp, 0.0)
            self._append(
                self._state_series,
                (stamp, goal_dist, p_vis, cov_x, cov_y, cov_yaw, state_err, pixel_age),
            )

    def _odom_cb(self, msg: Odometry) -> None:
        stamp = _stamp_to_float(msg.header.stamp)
        pose = self._transform_odom_pose(msg)
        if pose is None:
            return
        with self._lock:
            self._latest_truth = pose
            self._append(self._truth_traj, (stamp, pose[0], pose[1]))

    def _planner_belief_cb(self, msg: PoseWithCovarianceStamped) -> None:
        stamp = _stamp_to_float(msg.header.stamp)
        pose = msg.pose.pose
        x = float(pose.position.x)
        y = float(pose.position.y)
        yaw = _yaw_from_quaternion(pose.orientation)
        cov_x, cov_y, cov_yaw = _pose_covariance(msg)
        with self._lock:
            self._latest_belief = (x, y, yaw)
            goal_dist = math.nan
            if self._goal_xy is not None:
                goal_dist = math.hypot(x - self._goal_xy[0], y - self._goal_xy[1])
            state_err = math.nan
            if np.isfinite(self._latest_truth[0]) and np.isfinite(self._latest_truth[1]):
                state_err = math.hypot(x - self._latest_truth[0], y - self._latest_truth[1])
            self._append(
                self._belief_series,
                (stamp, goal_dist, cov_x, cov_y, cov_yaw, state_err, x, y, yaw),
            )

    def _goal_cb(self, msg: PoseStamped) -> None:
        with self._lock:
            self._goal_xy = (float(msg.pose.position.x), float(msg.pose.position.y))

    def _plan_cb(self, msg: Path) -> None:
        stamp = _stamp_to_float(msg.header.stamp)
        xy = _path_xy(msg)
        length = _plan_length(xy)
        with self._lock:
            self._latest_plan_xy = xy
            self._append(self._plan_series, (stamp, float(xy.shape[0]), length))

    def _efe_cb(self, msg: Float64MultiArray) -> None:
        vals = list(msg.data)
        while len(vals) < 12:
            vals.append(math.nan)
        stamp = self._now_s()
        with self._lock:
            self._append(self._efe_series, tuple([stamp] + [float(v) for v in vals[:12]]))

    def _planner_diag_cb(self, msg: Float64MultiArray) -> None:
        vals = list(msg.data)
        while len(vals) < 12:
            vals.append(math.nan)
        stamp = self._now_s()
        with self._lock:
            self._append(self._diag_series, tuple([stamp] + [float(v) for v in vals[:12]]))

    def _cmd_cb(self, msg: Twist) -> None:
        stamp = self._now_s()
        with self._lock:
            self._append(self._cmd_series, (stamp, float(msg.linear.x), float(msg.angular.z)))

    def _detection_cb(self, msg: Float64MultiArray) -> None:
        diag = diagnostics_from_message(msg)
        stamp = float(diag['stamp']) if math.isfinite(diag['stamp']) else self._now_s()
        with self._lock:
            self._last_diag_stamp = stamp
            self._append(
                self._detect_series,
                (
                    stamp,
                    1.0 if diag['detected'] else 0.0,
                    float(diag['u_mid']),
                    float(diag['v_mid']),
                    float(diag['border_margin_px']),
                    float(diag['separation_px']),
                ),
            )

    def _pixel_cb(self, msg: PoseStamped) -> None:
        with self._lock:
            self._last_pixel_stamp = _stamp_to_float(msg.header.stamp)

    def _visibility_map_cb(self, msg: OccupancyGrid) -> None:
        layer = self._grid_from_msg(msg, invert=True)
        if layer is None:
            return
        with self._lock:
            self._visibility_layer = layer

    def _opacity_mean_cb(self, msg: OccupancyGrid) -> None:
        layer = self._grid_from_msg(msg, invert=False)
        if layer is None:
            return
        with self._lock:
            self._opacity_mean_layer = layer

    def _opacity_cons_cb(self, msg: OccupancyGrid) -> None:
        layer = self._grid_from_msg(msg, invert=False)
        if layer is None:
            return
        with self._lock:
            self._opacity_cons_layer = layer

    def _draw_map_panel(
        self,
        ax,
        title: str,
        subtitle: str,
        layer: GridLayer | None,
        cmap: str,
        *,
        scale_label: str,
        low_label: str,
        high_label: str,
    ) -> None:
        ax.clear()
        self._style_axis(ax, title=title, xlabel='x [m]', ylabel='y [m]', grid=False)
        if layer is not None:
            ax.imshow(
                layer.values,
                extent=layer.extent,
                origin='lower',
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                aspect='equal',
            )
        else:
            self._empty_panel(ax, 'Waiting for map topic')
        truth = _series_arrays(self._truth_traj)
        if truth.size:
            ax.plot(
                truth[:, 1],
                truth[:, 2],
                color=self._theme['truth'],
                linewidth=1.6,
                alpha=0.95,
            )
        state = _series_arrays(self._state_traj)
        if state.size:
            ax.plot(
                state[:, 1],
                state[:, 2],
                color=self._theme['state'],
                linewidth=2.3,
            )
            ax.scatter(
                state[-1, 1],
                state[-1, 2],
                color=self._theme['plan'],
                edgecolors=self._theme['panel_bg'],
                linewidths=1.2,
                s=42,
                zorder=6,
            )
        if self._latest_plan_xy.size:
            ax.plot(
                self._latest_plan_xy[:, 0],
                self._latest_plan_xy[:, 1],
                color=self._theme['plan'],
                linewidth=2.0,
                linestyle=(0, (5, 3)),
            )
        if self._goal_xy is not None:
            ax.scatter(
                self._goal_xy[0],
                self._goal_xy[1],
                marker='*',
                s=170,
                color=self._theme['goal'],
                edgecolors=self._theme['panel_bg'],
                linewidths=1.4,
                zorder=7,
            )
        ax.set_aspect('equal', adjustable='box')
        ax.margins(x=0.02, y=0.02)
        self._panel_caption(ax, subtitle)
        self._map_scale_note(ax, scale_label, low_label, high_label)

    def _plot_line_panel(
        self,
        ax,
        title: str,
        subtitle: str,
        series_specs,
        *,
        ylabel: str | None = None,
        ylim=None,
        time_window: tuple[float, float, float] | None = None,
    ) -> None:
        ax.clear()
        self._style_axis(ax, title=title, xlabel='time [s]', ylabel=ylabel, grid=True)
        have_any = False
        handles = []
        labels = []
        for arr, x_idx, y_idx, kwargs in series_specs:
            if arr.size and arr.shape[1] > max(x_idx, y_idx):
                handle = ax.plot(arr[:, x_idx], arr[:, y_idx], **kwargs)[0]
                handles.append(handle)
                labels.append(kwargs.get('label', 'series'))
                have_any = True
        if ylim is not None:
            ax.set_ylim(*ylim)
        if have_any:
            if time_window is None:
                time_window = self._time_window(*[arr for arr, _, _, _ in series_specs])
            self._apply_time_window(ax, time_window)
            self._legend(
                ax,
                handles,
                labels,
                ncol=2 if len(handles) > 3 else 1,
                bbox_to_anchor=(0.995, 0.995),
            )
        else:
            self._empty_panel(ax, 'Waiting for data')
        self._panel_caption(ax, subtitle)

    def redraw(self) -> None:
        with self._lock:
            visibility_layer = self._visibility_layer
            opacity_mean_layer = self._opacity_mean_layer
            opacity_cons_layer = self._opacity_cons_layer
            state_series = _series_arrays(self._state_series)
            belief_series = _series_arrays(self._belief_series)
            detect_series = _series_arrays(self._detect_series)
            plan_series = _series_arrays(self._plan_series)
            efe_series = _series_arrays(self._efe_series)
            diag_series = _series_arrays(self._diag_series)
            cmd_series = _series_arrays(self._cmd_series)
            goal_xy = self._goal_xy
            latest_state = self._latest_state
            latest_truth = self._latest_truth
            last_pixel_stamp = self._last_pixel_stamp

        first_stamps = [
            float(arr[0, 0])
            for arr in (state_series, belief_series, detect_series, plan_series, efe_series, diag_series, cmd_series)
            if arr.size and np.isfinite(arr[0, 0])
        ]
        t0 = float(min(first_stamps)) if first_stamps else self._now_s()

        def _shift(arr: np.ndarray) -> np.ndarray:
            if not arr.size:
                return arr
            out = arr.copy()
            out[:, 0] = out[:, 0] - t0
            return out

        state_series = _shift(state_series)
        belief_series = _shift(belief_series)
        detect_series = _shift(detect_series)
        plan_series = _shift(plan_series)
        efe_series = _shift(efe_series)
        diag_series = _shift(diag_series)
        cmd_series = _shift(cmd_series)
        warmup_end_s = float(plan_series[0, 0]) if (plan_series.size and np.isfinite(plan_series[0, 0])) else math.nan
        time_window = self._time_window(
            state_series,
            belief_series,
            detect_series,
            plan_series,
            efe_series,
            diag_series,
            cmd_series,
            warmup_end_s=warmup_end_s,
        )

        self._draw_map_panel(
            self.axes[0, 0],
            'Visibility Field',
            'Camera observability over the floor plan',
            visibility_layer,
            'cividis',
            scale_label='visibility',
            low_label='occluded',
            high_label='clear',
        )
        self._draw_map_panel(
            self.axes[0, 1],
            'Mean Obstacle Opacity',
            'Average learned obstacle density',
            opacity_mean_layer,
            'magma_r',
            scale_label='opacity',
            low_label='open',
            high_label='blocked',
        )
        self._draw_map_panel(
            self.axes[0, 2],
            'Conservative Opacity',
            'Risk-biased occupancy for safer plans',
            opacity_cons_layer,
            'inferno_r',
            scale_label='opacity',
            low_label='permissive',
            high_label='guarded',
        )

        ax = self.axes[1, 0]
        ax2 = self._visibility_age_ax
        ax.clear()
        ax2.clear()
        self._style_axis(ax, title='Visibility & Freshness', xlabel='time [s]', ylabel='visibility / detection', grid=True)
        self._style_twin_axis(ax2, ylabel='age [s]')
        self._panel_caption(ax, 'Planner visibility confidence on the left, estimator freshness on the right')
        ax.axhspan(0.0, 0.35, color=self._theme['danger_fill'], alpha=0.35, zorder=0)
        ax.axhspan(0.35, 0.55, color=self._theme['warn_fill'], alpha=0.28, zorder=0)
        ax.axhline(0.55, color=self._theme['warning'], linewidth=1.0, linestyle='--', alpha=0.75)
        ax.axhline(0.8, color=self._theme['panel_edge'], linewidth=0.9, linestyle=':', alpha=0.8)
        handles = []
        labels = []
        if state_series.size:
            handle = ax.plot(
                state_series[:, 0],
                state_series[:, 2],
                linewidth=2.1,
                label='p_vis exec',
                color=self._theme['exec'],
            )[0]
            handles.append(handle)
            labels.append('p_vis exec')
        if diag_series.size and diag_series.shape[1] > 8:
            handle = ax.plot(
                diag_series[:, 0],
                diag_series[:, 7],
                linewidth=1.8,
                label='p_vis plan',
                color=self._theme['accent'],
            )[0]
            handles.append(handle)
            labels.append('p_vis plan')
            handle = ax.plot(
                diag_series[:, 0],
                diag_series[:, 8],
                linewidth=1.8,
                label='p_vis eff',
                color=self._theme['belief'],
            )[0]
            handles.append(handle)
            labels.append('p_vis eff')
        if detect_series.size:
            handle = ax.step(
                detect_series[:, 0],
                detect_series[:, 1],
                where='post',
                linewidth=1.7,
                label='detected',
                color=self._theme['text'],
            )[0]
            handles.append(handle)
            labels.append('detected')
            ax.fill_between(
                detect_series[:, 0],
                0.0,
                detect_series[:, 1],
                step='post',
                color=self._theme['text'],
                alpha=0.06,
            )
        if state_series.size or detect_series.size:
            ax.set_ylim(-0.05, 1.05)
            self._apply_time_window(ax, time_window)
        if state_series.size:
            handle = ax2.plot(
                state_series[:, 0],
                state_series[:, 7],
                linewidth=1.6,
                label='pixel age [s]',
                color=self._theme['danger'],
                alpha=0.9,
            )[0]
            handles.append(handle)
            labels.append('pixel age [s]')
        if diag_series.size and diag_series.shape[1] > 12:
            handle = ax2.plot(
                diag_series[:, 0],
                diag_series[:, 12],
                linewidth=1.5,
                linestyle='--',
                label='belief age [s]',
                color=self._theme['purple'],
            )[0]
            handles.append(handle)
            labels.append('belief age [s]')
        age_arrays = []
        if state_series.size:
            age_arrays.append(state_series[:, 7])
        if diag_series.size and diag_series.shape[1] > 12:
            age_arrays.append(diag_series[:, 12])
        finite_ages = [arr[np.isfinite(arr)] for arr in age_arrays if np.asarray(arr).size]
        max_age = max([1.2] + [float(np.nanmax(arr)) for arr in finite_ages if arr.size])
        ax2.set_ylim(0.0, max(1.2, max_age * 1.08))
        ax2.axhspan(0.5, min(1.0, ax2.get_ylim()[1]), color=self._theme['warn_fill'], alpha=0.18, zorder=0)
        if ax2.get_ylim()[1] > 1.0:
            ax2.axhspan(1.0, ax2.get_ylim()[1], color=self._theme['danger_fill'], alpha=0.16, zorder=0)
        ax2.axhline(0.5, color=self._theme['warning'], linewidth=0.95, linestyle='--', alpha=0.75)
        ax2.axhline(1.0, color=self._theme['danger'], linewidth=0.95, linestyle=':', alpha=0.75)
        if handles:
            self._legend(ax, handles, labels, ncol=2, bbox_to_anchor=(0.995, 0.995))
        else:
            self._empty_panel(ax, 'Waiting for state, detections, and planner diagnostics')

        ax = self.axes[1, 1]
        ax2 = self._mission_length_ax
        ax.clear()
        ax2.clear()
        self._style_axis(ax, title='Mission Progress', xlabel='time [s]', ylabel='goal distance [m]', grid=True)
        self._style_twin_axis(ax2, ylabel='preview length [m]')
        self._panel_caption(ax, 'Goal distance stays primary; preview length is kept on a separate scale')
        handles = []
        labels = []
        ax.axhspan(0.0, 0.5, color=self._theme['ok_fill'], alpha=0.32, zorder=0)
        ax.axhline(0.5, color=self._theme['goal'], linewidth=0.95, linestyle='--', alpha=0.75)
        if state_series.size:
            handle = ax.plot(
                state_series[:, 0],
                state_series[:, 1],
                linewidth=2.2,
                label='goal dist [m]',
                color=self._theme['accent'],
            )[0]
            handles.append(handle)
            labels.append('goal dist [m]')
            finite_goal = state_series[:, 1][np.isfinite(state_series[:, 1])]
            if finite_goal.size:
                ax.set_ylim(0.0, max(1.0, float(np.nanmax(finite_goal)) * 1.08))
        if plan_series.size:
            handle = ax2.plot(
                plan_series[:, 0],
                plan_series[:, 2],
                linewidth=1.9,
                label='preview length [m]',
                color=self._theme['belief'],
            )[0]
            handles.append(handle)
            labels.append('preview length [m]')
            finite_length = plan_series[:, 2][np.isfinite(plan_series[:, 2])]
            if finite_length.size:
                ax2.set_ylim(0.0, max(0.5, float(np.nanmax(finite_length)) * 1.15))
        if handles:
            self._apply_time_window(ax, time_window)
            self._legend(ax, handles, labels, ncol=1, bbox_to_anchor=(0.995, 0.995))
        else:
            self._empty_panel(ax, 'Waiting for mission progress data')

        ax = self.axes[1, 2]
        ax2 = self._belief_error_ax
        ax.clear()
        ax2.clear()
        self._style_axis(ax, title='Belief Quality', xlabel='time [s]', ylabel='belief variance', grid=True)
        self._style_twin_axis(ax2, ylabel='state error [m]')
        self._panel_caption(ax, 'Belief covariance on the left, divergence from truth on the right')
        handles = []
        labels = []
        if belief_series.size:
            for idx, label, color in (
                (2, 'cov_x', self._theme['accent']),
                (3, 'cov_y', self._theme['belief']),
                (4, 'cov_yaw', self._theme['exec']),
            ):
                handle = ax.plot(
                    belief_series[:, 0],
                    belief_series[:, idx],
                    linewidth=1.75,
                    label=label,
                    color=color,
                )[0]
                handles.append(handle)
                labels.append(label)
            finite_cov = belief_series[:, 2:5][np.isfinite(belief_series[:, 2:5])]
            if finite_cov.size:
                ax.set_ylim(0.0, max(0.05, float(np.nanmax(finite_cov)) * 1.15))
            finite_err = belief_series[:, 5][np.isfinite(belief_series[:, 5])]
            if finite_err.size:
                max_err = max(0.2, float(np.nanmax(finite_err)) * 1.15)
                ax2.set_ylim(0.0, max_err)
                ax2.axhspan(0.25, min(0.4, max_err), color=self._theme['warn_fill'], alpha=0.18, zorder=0)
                if max_err > 0.4:
                    ax2.axhspan(0.4, max_err, color=self._theme['danger_fill'], alpha=0.16, zorder=0)
                ax2.axhline(0.25, color=self._theme['warning'], linewidth=0.95, linestyle='--', alpha=0.75)
                ax2.axhline(0.4, color=self._theme['danger'], linewidth=0.95, linestyle=':', alpha=0.75)
                handle = ax2.plot(
                    belief_series[:, 0],
                    belief_series[:, 5],
                    linewidth=1.7,
                    label='state err [m]',
                    color=self._theme['text'],
                )[0]
                handles.append(handle)
                labels.append('state err [m]')
        if handles:
            self._apply_time_window(ax, time_window)
            self._legend(ax, handles, labels, ncol=2, bbox_to_anchor=(0.995, 0.995))
        else:
            self._empty_panel(ax, 'Waiting for planner belief data')

        ax = self.axes[2, 0]
        ax2 = self._efe_total_ax
        ax.clear()
        ax2.clear()
        self._style_axis(ax, title='EFE Cost Mix', xlabel='time [s]', ylabel='share of |cost|', grid=True)
        self._style_twin_axis(ax2, ylabel='total cost')
        self._panel_caption(ax, 'Component shares are normalized so smaller terms stay readable')
        handles = []
        labels = []
        if efe_series.size and efe_series.shape[1] > 6:
            component_matrix = np.abs(np.column_stack([
                efe_series[:, 2],
                efe_series[:, 3],
                efe_series[:, 4],
                efe_series[:, 5],
                efe_series[:, 6],
            ]))
            denom = np.sum(component_matrix, axis=1)
            shares = np.zeros_like(component_matrix)
            valid = denom > 1e-9
            shares[valid] = component_matrix[valid] / denom[valid, None]
            stack_handles = ax.stackplot(
                efe_series[:, 0],
                shares.T,
                colors=[
                    self._theme['accent'],
                    self._theme['belief'],
                    self._theme['exec'],
                    self._theme['danger'],
                    self._theme['brown'],
                ],
                alpha=0.78,
            )
            ax.set_ylim(0.0, 1.0)
            for handle, label in zip(
                stack_handles,
                ('risk share', 'ambiguity share', 'control share', 'visibility share', 'obstacle share'),
            ):
                handles.append(handle)
                labels.append(label)
            total_handle = ax2.plot(
                efe_series[:, 0],
                efe_series[:, 1],
                linewidth=2.0,
                color=self._theme['purple'],
                label='total cost',
            )[0]
            handles.insert(0, total_handle)
            labels.insert(0, 'total cost')
            finite_total = efe_series[:, 1][np.isfinite(efe_series[:, 1])]
            if finite_total.size:
                total_min = float(np.nanmin(finite_total))
                total_max = float(np.nanmax(finite_total))
                if math.isfinite(total_min) and math.isfinite(total_max):
                    pad = 0.08 * max(1.0, total_max - total_min)
                    ax2.set_ylim(total_min - pad, total_max + pad)
            self._apply_time_window(ax, time_window)
            self._legend(ax, handles, labels, ncol=2, bbox_to_anchor=(0.995, 0.995))
        else:
            self._empty_panel(ax, 'Waiting for EFE metrics')

        ax = self.axes[2, 1]
        ax2 = self._planner_diag_ax
        ax.clear()
        ax2.clear()
        self._style_axis(ax, title='Planner Health', xlabel='time [s]', ylabel='latency [ms]', grid=True)
        self._style_twin_axis(ax2, ylabel='optimizer work')
        self._panel_caption(ax, 'Latency against the 2 Hz budget, plus solver effort and missing-data events')
        handles = []
        labels = []
        if diag_series.size:
            plan_ms = diag_series[:, 5]
            solve_ms = diag_series[:, 6]
            finite_ms = np.concatenate([
                plan_ms[np.isfinite(plan_ms)],
                solve_ms[np.isfinite(solve_ms)],
            ]) if (np.isfinite(plan_ms).any() or np.isfinite(solve_ms).any()) else np.asarray([])
            ymax_ms = max(650.0, float(np.nanmax(finite_ms)) * 1.08) if finite_ms.size else 1500.0
            ax.set_ylim(0.0, ymax_ms)
            ax.axhspan(500.0, min(1000.0, ymax_ms), color=self._theme['warn_fill'], alpha=0.22, zorder=0)
            if ymax_ms > 1000.0:
                ax.axhspan(1000.0, ymax_ms, color=self._theme['danger_fill'], alpha=0.16, zorder=0)
            ax.axhline(500.0, color=self._theme['warning'], linewidth=1.0, linestyle='--', alpha=0.8)
            ax.axhline(1000.0, color=self._theme['danger'], linewidth=1.0, linestyle=':', alpha=0.8)
            handle = ax.plot(diag_series[:, 0], plan_ms, linewidth=2.1, label='plan ms', color=self._theme['accent'])[0]
            handles.append(handle)
            labels.append('plan ms')
            ax.fill_between(diag_series[:, 0], 0.0, plan_ms, color=self._theme['accent'], alpha=0.08)
            handle = ax.plot(diag_series[:, 0], solve_ms, linewidth=1.8, label='solve ms', color=self._theme['belief'])[0]
            handles.append(handle)
            labels.append('solve ms')
            handle = ax2.plot(
                diag_series[:, 0],
                diag_series[:, 3],
                linewidth=1.6,
                linestyle='--',
                label='nit',
                color=self._theme['text'],
            )[0]
            handles.append(handle)
            labels.append('nit')
            handle = ax2.plot(
                diag_series[:, 0],
                diag_series[:, 4],
                linewidth=1.6,
                linestyle=':',
                label='nfev',
                color=self._theme['purple'],
            )[0]
            handles.append(handle)
            labels.append('nfev')
            fail_mask = diag_series[:, 1] < 0.5
            if np.any(fail_mask):
                handle = ax.scatter(
                    diag_series[:, 0],
                    np.where(fail_mask, plan_ms, np.nan),
                    s=34,
                    marker='x',
                    color=self._theme['danger'],
                    linewidths=1.5,
                    label='solver fallback',
                )
                handles.append(handle)
                labels.append('solver fallback')
            if diag_series.shape[1] > 11:
                meas_missing_mask = diag_series[:, 11] < 0.5
                if np.any(meas_missing_mask):
                    handle = ax.scatter(
                        diag_series[:, 0],
                        np.where(meas_missing_mask, 0.08 * ymax_ms, np.nan),
                        s=36,
                        marker='o',
                        facecolors='none',
                        edgecolors=self._theme['warning'],
                        linewidths=1.3,
                        label='no measurement',
                    )
                    handles.append(handle)
                    labels.append('no measurement')
            self._apply_time_window(ax, time_window)
        if handles:
            self._legend(ax, handles, labels, ncol=2, bbox_to_anchor=(0.995, 0.995))
        else:
            self._empty_panel(ax, 'Waiting for planner diagnostics')

        ax = self.axes[2, 2]
        ax2 = self._command_sanity_ax
        ax.clear()
        ax2.clear()
        self._style_axis(ax, title='Commands & View Margin', xlabel='time [s]', ylabel='command', grid=True)
        self._style_twin_axis(ax2, ylabel='pixels')
        self._panel_caption(ax, 'Command stream on the left; image border clearance on the right')
        handles = []
        labels = []
        if cmd_series.size:
            ax.axhline(0.0, color=self._theme['panel_edge'], linewidth=0.9, linestyle='--', alpha=0.8)
            handle = ax.plot(
                cmd_series[:, 0],
                cmd_series[:, 1],
                linewidth=1.9,
                label='cmd v',
                color=self._theme['accent'],
            )[0]
            handles.append(handle)
            labels.append('cmd v')
            handle = ax.plot(
                cmd_series[:, 0],
                cmd_series[:, 2],
                linewidth=1.9,
                label='cmd w',
                color=self._theme['belief'],
            )[0]
            handles.append(handle)
            labels.append('cmd w')
            finite_cmd = cmd_series[:, 1:3][np.isfinite(cmd_series[:, 1:3])]
            if finite_cmd.size:
                cmd_bound = max(0.25, float(np.nanmax(np.abs(finite_cmd))) * 1.15)
                ax.set_ylim(-cmd_bound, cmd_bound)
        if detect_series.size:
            finite_px = detect_series[:, 4:6][np.isfinite(detect_series[:, 4:6])]
            max_px = max(140.0, float(np.nanmax(finite_px)) * 1.08) if finite_px.size else 220.0
            warn_margin_px = 180.0
            danger_margin_px = 120.0
            ax2.set_ylim(0.0, max_px)
            ax2.axhspan(0.0, min(danger_margin_px, max_px), color=self._theme['danger_fill'], alpha=0.16, zorder=0)
            if max_px > danger_margin_px:
                ax2.axhspan(
                    danger_margin_px,
                    min(warn_margin_px, max_px),
                    color=self._theme['warn_fill'],
                    alpha=0.16,
                    zorder=0,
                )
            ax2.axhline(danger_margin_px, color=self._theme['danger'], linewidth=0.95, linestyle=':', alpha=0.8)
            ax2.axhline(warn_margin_px, color=self._theme['warning'], linewidth=0.95, linestyle='--', alpha=0.8)
            handle = ax2.plot(
                detect_series[:, 0],
                detect_series[:, 4],
                linewidth=1.8,
                label='border px',
                color=self._theme['exec'],
            )[0]
            handles.append(handle)
            labels.append('border px')
            handle = ax2.plot(
                detect_series[:, 0],
                detect_series[:, 5],
                linewidth=1.6,
                linestyle=':',
                label='separation px',
                color=self._theme['danger'],
            )[0]
            handles.append(handle)
            labels.append('separation px')
        if handles:
            self._apply_time_window(ax, time_window)
            self._legend(ax, handles, labels, ncol=2, bbox_to_anchor=(0.995, 0.995))
        else:
            self._empty_panel(ax, 'Waiting for commands and image diagnostics')

        current_goal_dist = state_series[-1, 1] if state_series.size else math.nan
        current_plan_length = plan_series[-1, 2] if plan_series.size else math.nan
        current_p_vis = state_series[-1, 2] if state_series.size else (
            diag_series[-1, 8] if (diag_series.size and diag_series.shape[1] > 8) else math.nan
        )
        current_plan_vis = diag_series[-1, 8] if (diag_series.size and diag_series.shape[1] > 8) else math.nan
        latest_detected = detect_series[-1, 1] if detect_series.size else math.nan
        current_pixel_age = state_series[-1, 7] if state_series.size else math.nan
        latest_success = diag_series[-1, 1] if diag_series.size else math.nan
        latest_status = diag_series[-1, 2] if diag_series.size else math.nan
        latest_plan_ms = diag_series[-1, 5] if (diag_series.size and diag_series.shape[1] > 5) else math.nan
        latest_solve_ms = diag_series[-1, 6] if (diag_series.size and diag_series.shape[1] > 6) else math.nan
        latest_nit = diag_series[-1, 3] if (diag_series.size and diag_series.shape[1] > 3) else math.nan
        latest_nfev = diag_series[-1, 4] if (diag_series.size and diag_series.shape[1] > 4) else math.nan
        latest_meas_available = diag_series[-1, 11] if (diag_series.size and diag_series.shape[1] > 11) else math.nan
        latest_belief_age = diag_series[-1, 12] if (diag_series.size and diag_series.shape[1] > 12) else math.nan
        detection_text = 'detection n/a'
        if np.isfinite(latest_detected):
            detection_text = 'detected' if latest_detected >= 0.5 else 'not seen'
        visibility_label, visibility_color = self._metric_status(current_p_vis, warn=0.55, danger=0.35, inverse=True)
        if np.isfinite(latest_detected) and latest_detected < 0.5:
            visibility_label = 'lost'
            visibility_color = self._theme['danger_fill']
        planner_label, planner_color = self._metric_status(latest_plan_ms, warn=500.0, danger=1000.0)
        if np.isfinite(latest_success) and latest_success < 0.5:
            planner_label = 'fallback'
            planner_color = self._theme['danger_fill']
        freshness_label, freshness_color = self._metric_status(current_pixel_age, warn=0.5, danger=1.0)
        if np.isfinite(latest_meas_available) and latest_meas_available < 0.5:
            freshness_label = 'stale'
            freshness_color = self._theme['danger_fill']
        self._set_status_card(
            0,
            'Visibility',
            (
                f'{visibility_label} | {current_p_vis:.2f} exec | {current_plan_vis:.2f} plan | {detection_text}'
                if np.isfinite(current_p_vis) or np.isfinite(current_plan_vis) or np.isfinite(latest_detected)
                else 'waiting for visibility'
            ),
            visibility_color,
        )
        self._set_status_card(
            1,
            'Planner',
            (
                f'{planner_label} | {latest_plan_ms:.0f} ms | nit {latest_nit:.0f} | nfev {latest_nfev:.0f}'
                if np.isfinite(latest_plan_ms)
                else 'waiting for planner'
            ),
            planner_color,
        )
        self._set_status_card(
            2,
            'Mission',
            (
                f'goal {current_goal_dist:.2f} m | preview {current_plan_length:.2f} m'
                if np.isfinite(current_goal_dist) or np.isfinite(current_plan_length)
                else 'waiting for goal and plan'
            ),
            self._theme['info_fill'],
        )
        self._set_status_card(
            3,
            'Freshness',
            (
                f'{freshness_label} | pixel {current_pixel_age:.2f} s | belief {latest_belief_age:.2f} s | '
                f'{"meas" if latest_meas_available >= 0.5 else "no meas"}'
                if np.isfinite(current_pixel_age) or np.isfinite(latest_belief_age) or np.isfinite(latest_meas_available)
                else 'waiting for measurement timing'
            ),
            freshness_color,
        )
        summary_parts = [
            f'window {time_window[0]:.1f}-{time_window[1]:.1f} s',
            f'warm-up ends {warmup_end_s:.1f} s' if np.isfinite(warmup_end_s) else 'warm-up active',
            f'solve {latest_solve_ms:.0f} ms' if np.isfinite(latest_solve_ms) else 'solve n/a',
            f'status {latest_status:.0f}' if np.isfinite(latest_status) else 'status n/a',
        ]
        if goal_xy is not None:
            summary_parts.append(f'goal ({goal_xy[0]:.2f}, {goal_xy[1]:.2f})')
        if np.isfinite(latest_state[0]):
            summary_parts.append(f'state ({latest_state[0]:.2f}, {latest_state[1]:.2f})')
        if np.isfinite(latest_truth[0]):
            summary_parts.append(f'truth ({latest_truth[0]:.2f}, {latest_truth[1]:.2f})')
        self._header_summary.set_text(' | '.join(summary_parts))
        self.fig.canvas.draw_idle()

    def run_ui_loop(self) -> None:
        pause_s = 1.0 / self.redraw_hz
        while rclpy.ok() and plt.fignum_exists(self.fig.number):
            self.redraw()
            plt.pause(pause_s)


def main(args=None):
    rclpy.init(args=args)
    node = LiveVisibilityDashboard()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        node.run_ui_loop()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
