#!/usr/bin/env python3
# [DEPRECATED_LEGACY_CLEANUP] Legacy/exploratory/diagnostic script or module. Distracting from paper-facing F85-F88 runtime.
"""Estimate process noise Q and observation noise R from captured CSV."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import pathlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from unav_common.camera_model import ObliqueCameraModel


def _wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _unicycle_step(state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
    x, y, theta = state
    v, w = control
    x = x + v * dt * math.cos(theta)
    y = y + v * dt * math.sin(theta)
    theta = _wrap_angle(theta + w * dt)
    return np.array([x, y, theta], dtype=float)


@dataclass
class OdomSample:
    t: float
    x: float
    y: float
    yaw: float
    v: float
    w: float


@dataclass
class PixelSample:
    t: float
    u: float
    v: float
    yaw: float


def _parse_capture_csv(path: pathlib.Path) -> Tuple[List[OdomSample], List[PixelSample]]:
    odom: List[OdomSample] = []
    pixels: List[PixelSample] = []

    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        for row in rows:
            src = (row.get("source") or "").strip().lower()
            t = float(row["stamp"])
            if src == "odom":
                odom.append(
                    OdomSample(
                        t=t,
                        x=float(row["odom_x"]),
                        y=float(row["odom_y"]),
                        yaw=float(row["odom_yaw"]),
                        v=float(row["cmd_v"]) if row["cmd_v"] else 0.0,
                        w=float(row["cmd_w"]) if row["cmd_w"] else 0.0,
                    )
                )
            elif src == "pixel":
                pixels.append(
                    PixelSample(
                        t=t,
                        u=float(row["pix_u"]),
                        v=float(row["pix_v"]),
                        yaw=float(row["pix_yaw"]) if row["pix_yaw"] else 0.0,
                    )
                )
    odom.sort(key=lambda s: s.t)
    pixels.sort(key=lambda s: s.t)
    return odom, pixels


def _load_intrinsics(world_profiles_yaml: pathlib.Path, world: str) -> Dict[str, float]:
    data = yaml.safe_load(world_profiles_yaml.read_text(encoding="utf-8")) or {}
    global_intr = data.get("camera_intrinsics", {}) or {}
    worlds = data.get("worlds", {}) or {}
    local = (worlds.get(world, {}) or {}).get("camera_intrinsics", {}) or {}
    intr = dict(global_intr)
    intr.update(local)
    required = ("img_width", "img_height", "fov_h_rad")
    for key in required:
        if key not in intr:
            raise RuntimeError(f"Missing '{key}' in intrinsics for world '{world}'")
    return {
        "img_width": int(intr["img_width"]),
        "img_height": int(intr["img_height"]),
        "fov_h_rad": float(intr["fov_h_rad"]),
    }


def _parse_camera_pose(world_sdf_path: pathlib.Path, camera_model: str = "external_camera") -> Sequence[float]:
    tree = ET.parse(world_sdf_path)
    root = tree.getroot()
    for include in root.iter():
        if not include.tag.endswith("include"):
            continue
        uri_node = None
        pose_node = None
        for child in list(include):
            if child.tag.endswith("uri"):
                uri_node = child
            elif child.tag.endswith("pose"):
                pose_node = child
        if uri_node is None or not (uri_node.text or "").strip().startswith("model://"):
            continue
        uri = uri_node.text.strip()[len("model://") :]
        model_name = uri.split("/")[0]
        if model_name != camera_model:
            continue
        if pose_node is None or not (pose_node.text or "").strip():
            raise RuntimeError(f"Camera include '{camera_model}' is missing pose in {world_sdf_path}")
        vals = [float(v) for v in pose_node.text.replace(",", " ").split()]
        if len(vals) != 6:
            raise RuntimeError(f"Camera pose must have 6 entries, got {len(vals)}")
        return vals
    raise RuntimeError(f"Camera model '{camera_model}' not found in {world_sdf_path}")


def _compute_look_at(cam_pos: Sequence[float], roll: float, pitch: float, yaw: float) -> Sequence[float]:
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    forward = [cp * cy, cp * sy, -sp]
    if abs(forward[2]) < 1e-9:
        raise RuntimeError("Camera forward vector parallel to ground plane")
    t = -cam_pos[2] / forward[2]
    if t <= 0:
        raise RuntimeError("Camera forward ray does not intersect ground in front")
    return [cam_pos[0] + t * forward[0], cam_pos[1] + t * forward[1], 0.0]


def _nearest_odom_sample(odom: Sequence[OdomSample], times: Sequence[float], t: float) -> Tuple[Optional[OdomSample], float]:
    if not odom:
        return None, float("inf")
    idx = bisect.bisect_left(times, t)
    candidates = []
    if idx < len(odom):
        candidates.append(odom[idx])
    if idx > 0:
        candidates.append(odom[idx - 1])
    best = min(candidates, key=lambda s: abs(s.t - t))
    return best, abs(best.t - t)


def _cov(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.zeros((0, 0), dtype=float)
    if arr.shape[0] < 2:
        return np.zeros((arr.shape[1], arr.shape[1]), dtype=float)
    return np.cov(arr, rowvar=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate Q/R from capture CSV.")
    parser.add_argument("--input", required=True, type=str, help="Capture CSV from capture_noise_data.py")
    parser.add_argument("--world", required=True, type=str, help="World filename, e.g. arena10.world.sdf")
    parser.add_argument(
        "--world-profiles",
        default="src/experiments/config/world_profiles.yaml",
        type=str,
        help="Path to world_profiles.yaml",
    )
    parser.add_argument(
        "--world-dir",
        default="src/sim/gazebo_worlds/worlds",
        type=str,
        help="Directory containing world SDFs",
    )
    parser.add_argument("--camera-model", default="external_camera", type=str)
    parser.add_argument("--obs-mode", default="uv", choices=("uv", "uvt"))
    parser.add_argument(
        "--odom-origin-x",
        type=float,
        default=0.0,
        help="Translation x of odom origin expressed in map_bev (map->odom transform).",
    )
    parser.add_argument(
        "--odom-origin-y",
        type=float,
        default=0.0,
        help="Translation y of odom origin expressed in map_bev (map->odom transform).",
    )
    parser.add_argument(
        "--odom-origin-yaw",
        type=float,
        default=0.0,
        help="Yaw of odom frame relative to map_bev (map->odom transform).",
    )
    parser.add_argument("--sync-tol-s", type=float, default=0.08, help="Max odom-pixel timestamp mismatch")
    parser.add_argument("--min-dt-s", type=float, default=1e-3)
    parser.add_argument("--max-dt-s", type=float, default=0.30)
    parser.add_argument(
        "--target-dt-s",
        type=float,
        default=0.2,
        help="Planner prediction dt used to scale Q from sample dt (set <=0 to disable scaling).",
    )
    parser.add_argument("--output", default="", type=str, help="Optional JSON output path")
    args = parser.parse_args()

    in_path = pathlib.Path(args.input).resolve()
    world_profiles = pathlib.Path(args.world_profiles).resolve()
    world_sdf = pathlib.Path(args.world_dir).resolve() / args.world

    odom, pixels = _parse_capture_csv(in_path)
    if len(odom) < 3:
        raise RuntimeError(f"Need >=3 odom samples, got {len(odom)}")

    intr = _load_intrinsics(world_profiles, args.world)
    cam_pose = _parse_camera_pose(world_sdf, camera_model=args.camera_model)
    cam_pos = [cam_pose[0], cam_pose[1], cam_pose[2]]
    look_at = _compute_look_at(cam_pos, cam_pose[3], cam_pose[4], cam_pose[5])
    camera = ObliqueCameraModel(
        cam_pos=cam_pos,
        look_at=look_at,
        img_width=intr["img_width"],
        img_height=intr["img_height"],
        fov_h_rad=intr["fov_h_rad"],
    )

    # --- Q estimation from one-step dynamics residuals ---
    q_errs: List[np.ndarray] = []
    dts: List[float] = []
    for i in range(len(odom) - 1):
        a = odom[i]
        b = odom[i + 1]
        dt = b.t - a.t
        if dt < args.min_dt_s or dt > args.max_dt_s:
            continue
        xk = np.array([a.x, a.y, a.yaw], dtype=float)
        uk = np.array([a.v, a.w], dtype=float)
        pred = _unicycle_step(xk, uk, dt)
        obs = np.array([b.x, b.y, b.yaw], dtype=float)
        err = obs - pred
        err[2] = _wrap_angle(err[2])
        q_errs.append(err)
        dts.append(dt)
    q_arr = np.asarray(q_errs, dtype=float) if q_errs else np.zeros((0, 3), dtype=float)
    q_cov = _cov(q_arr)
    q_dt_mean = float(np.mean(dts)) if dts else None
    q_cov_scaled = q_cov.copy()
    q_scale = 1.0
    if (
        q_cov.size
        and q_dt_mean is not None
        and q_dt_mean > 1e-9
        and args.target_dt_s is not None
        and args.target_dt_s > 0.0
    ):
        # Discrete-time process covariance scales linearly with dt.
        q_scale = float(args.target_dt_s) / q_dt_mean
        q_cov_scaled = q_cov * q_scale

    # --- R estimation from pixel residuals against camera model ---
    odom_times = [o.t for o in odom]
    r_errs: List[np.ndarray] = []
    used_pixels = 0
    c0 = math.cos(args.odom_origin_yaw)
    s0 = math.sin(args.odom_origin_yaw)
    for p in pixels:
        nearest, dt_abs = _nearest_odom_sample(odom, odom_times, p.t)
        if nearest is None or dt_abs > args.sync_tol_s:
            continue
        # Convert odom-frame state into map_bev frame used by camera model.
        x_map = c0 * nearest.x - s0 * nearest.y + args.odom_origin_x
        y_map = s0 * nearest.x + c0 * nearest.y + args.odom_origin_y
        yaw_map = _wrap_angle(nearest.yaw + args.odom_origin_yaw)
        state = np.array([x_map, y_map, yaw_map], dtype=float)
        if args.obs_mode == "uv":
            pred = np.asarray(camera.g_uv(state), dtype=float).reshape(-1)
            meas = np.array([p.u, p.v], dtype=float)
            err = meas - pred
        else:
            pred = np.asarray(camera.g(state), dtype=float).reshape(-1)
            meas = np.array([p.u, p.v, p.yaw], dtype=float)
            err = meas - pred
            if err.size >= 3:
                err[2] = _wrap_angle(err[2])
        r_errs.append(err)
        used_pixels += 1
    r_arr = np.asarray(r_errs, dtype=float) if r_errs else np.zeros((0, 2 if args.obs_mode == "uv" else 3), dtype=float)
    r_cov = _cov(r_arr)

    q_xy_var = float(max((q_cov_scaled[0, 0] + q_cov_scaled[1, 1]) * 0.5, 0.0)) if q_cov_scaled.size else 0.0
    q_th_var = float(max(q_cov_scaled[2, 2], 0.0)) if q_cov_scaled.size else 0.0
    r_uv_var = float(max((r_cov[0, 0] + r_cov[1, 1]) * 0.5, 0.0)) if r_cov.size else 0.0
    r_yaw_var = float(max(r_cov[2, 2], 0.0)) if (r_cov.size and r_cov.shape[0] > 2) else 0.0

    result = {
        "input_csv": str(in_path),
        "world": args.world,
        "obs_mode": args.obs_mode,
        "sample_counts": {
            "odom_rows": len(odom),
            "pixel_rows": len(pixels),
            "q_pairs_used": int(q_arr.shape[0]),
            "r_pairs_used": int(used_pixels),
        },
        "timing": {
            "q_dt_mean": q_dt_mean,
            "q_dt_std": float(np.std(dts)) if dts else None,
            "q_scale_to_target_dt": q_scale,
            "target_dt_s": float(args.target_dt_s) if args.target_dt_s and args.target_dt_s > 0.0 else None,
            "sync_tol_s": float(args.sync_tol_s),
        },
        "Q_covariance_estimate_raw": q_cov.tolist() if q_cov.size else [[0.0, 0.0, 0.0]] * 3,
        "Q_covariance_estimate_scaled": q_cov_scaled.tolist() if q_cov_scaled.size else [[0.0, 0.0, 0.0]] * 3,
        "R_covariance_estimate": r_cov.tolist() if r_cov.size else [[0.0, 0.0], [0.0, 0.0]],
        "recommended_params": {
            "process_noise_xy": math.sqrt(q_xy_var),
            "process_noise_theta": math.sqrt(q_th_var),
            "obs_noise_uv": math.sqrt(r_uv_var),
            "obs_noise_yaw": math.sqrt(r_yaw_var) if args.obs_mode == "uvt" else None,
        },
    }

    print(json.dumps(result, indent=2))
    override = (
        "launch_override="
        f"process_noise_xy:={result['recommended_params']['process_noise_xy']:.6f} "
        f"process_noise_theta:={result['recommended_params']['process_noise_theta']:.6f} "
        f"obs_noise_uv:={result['recommended_params']['obs_noise_uv']:.6f}"
    )
    if result["recommended_params"]["obs_noise_yaw"] is not None:
        override += f" obs_noise_yaw:={result['recommended_params']['obs_noise_yaw']:.6f}"
    print(override)

    if args.output:
        out = pathlib.Path(args.output).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"output_json={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
