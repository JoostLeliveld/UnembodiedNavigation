#!/usr/bin/env python3
"""Shared helpers for the C2 diagnostics: run loading, camera model, GP rho,
truth interpolation, frame matching, and the planner-belief projection
reconstruction (P_planner) that is NOT logged anywhere.
"""
import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/joostleliveld/Thesis/UnembodiedNavigation")
WORLD_PROFILES = ROOT / "src/experiments/config/world_profiles.yaml"
GP_NPZ = ROOT / "paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz"


# ----------------------------------------------------------------------------- CSV
def load_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path, na_values=["NaN", "nan", "", "inf", "-inf"], low_memory=False)


def load_run(run_dir):
    run_dir = Path(run_dir)
    perc = load_csv(run_dir / "perception.csv")
    exp = load_csv(run_dir / "experiment.csv")
    summary = {}
    sp = run_dir / "run_summary.json"
    if sp.exists():
        summary = json.loads(sp.read_text())
    return perc, exp, summary


# ----------------------------------------------------------------------------- camera
def build_camera(world_file="warehouse_aws.world.sdf", profiles_path=None):
    from unav_common.camera_model import ObliqueCameraModel
    from experiments.core.world_profiles import load_profile, compute_look_at_from_pose
    profiles_path = str(profiles_path or WORLD_PROFILES)
    _p, intr, _w, cp = load_profile(profiles_path, world_file)
    cam = ObliqueCameraModel(
        cam_pos=[cp[0], cp[1], cp[2]],
        look_at=compute_look_at_from_pose([cp[0], cp[1], cp[2]], cp[3], cp[4], cp[5]),
        img_width=intr["img_width"], img_height=intr["img_height"], fov_h_rad=intr["fov_h_rad"],
    )
    return cam


# ----------------------------------------------------------------------------- GP rho (occlusion)
_GP = None


def gp():
    global _GP
    if _GP is None:
        g = np.load(GP_NPZ, allow_pickle=True)
        _GP = {
            "xs": np.asarray(g["xs"], float),
            "ys": np.asarray(g["ys"], float),
            "rho": np.asarray(g["P_conservative_plan_map"], float),
        }
        _GP["extent"] = (float(_GP["xs"][0]), float(_GP["xs"][-1]),
                         float(_GP["ys"][0]), float(_GP["ys"][-1]))
    return _GP


def sample_rho(x, y):
    g = gp()
    xs, ys, rho = g["xs"], g["ys"], g["rho"]
    xi = np.clip(np.searchsorted(xs, np.asarray(x, float)), 0, len(xs) - 1)
    yi = np.clip(np.searchsorted(ys, np.asarray(y, float)), 0, len(ys) - 1)
    return rho[yi, xi]


# ----------------------------------------------------------------------------- truth interp
def truth_interp_fns(exp):
    """Return (x(t), y(t), yaw(t)) interpolators from experiment.csv truth."""
    d = exp.dropna(subset=["stamp", "odom_map_x", "odom_map_y"]).sort_values("stamp")
    t = d["stamp"].to_numpy(float)
    tx = d["odom_map_x"].to_numpy(float)
    ty = d["odom_map_y"].to_numpy(float)
    tyaw = np.unwrap(d["odom_map_yaw"].to_numpy(float)) if "odom_map_yaw" in d else np.zeros_like(t)

    def fx(s):
        return float(np.interp(s, t, tx))

    def fy(s):
        return float(np.interp(s, t, ty))

    def fyaw(s):
        return float(np.interp(s, t, tyaw))

    return fx, fy, fyaw


# ----------------------------------------------------------------------------- frame matching
def index_frames(frames_dir):
    """Map capture-stamp (rounded ms) -> raw frame path."""
    frames_dir = Path(frames_dir)
    paths = glob.glob(str(frames_dir / "raw" / "*.png"))
    if not paths:
        paths = glob.glob(str(frames_dir / "*.png"))
    out = {}
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            st = float(base.split("_")[-1])
        except ValueError:
            continue
        out[round(st, 3)] = p
    return out


def frame_at(frames_idx, stamp, tol=0.25):
    if not frames_idx:
        return None, None
    keys = sorted(frames_idx)
    k = min(keys, key=lambda j: abs(j - stamp))
    if abs(k - stamp) <= tol:
        return frames_idx[k], k
    return None, None


# ----------------------------------------------------------------------------- P_planner reconstruction
def add_planner_projection(perc, bev_y_offset=0.05):
    """Reconstruct the planner's effective measurement world position.

    The planner (unicycle_planner_node._pixel_cb) takes the RAW pixel, applies
    ONLY a bev_y_calibration_offset_m shift in world-y (via pixel_to_world ->
    +offset in y -> world_to_pixel), then runs its EKF in pixel space. The net
    world position it converges the belief toward is therefore:
        P_planner = (pred_world_x, pred_world_y + offset)
    whereas /state/bev uses the full 6-param affine (pred_world_*_calibrated).
    This column is computed nowhere in the pipeline; reconstruct it here.
    """
    out = perc.copy()
    if "pred_world_x" in out and "pred_world_y" in out:
        out["planner_world_x"] = out["pred_world_x"]
        out["planner_world_y"] = out["pred_world_y"] + bev_y_offset
        if {"true_x", "true_y"}.issubset(out.columns):
            out["planner_loc_error_m"] = np.hypot(
                out["planner_world_x"] - out["true_x"],
                out["planner_world_y"] - out["true_y"],
            )
    return out


# ----------------------------------------------------------------------------- driveable prisms
def driveable_prisms(run_dir):
    """Parse the driveable geometry prisms from the run manifest (keep-in rects)."""
    run_dir = Path(run_dir)
    man = run_dir / "run_manifest.json"
    if not man.exists():
        return []
    try:
        m = json.loads(man.read_text())
    except Exception:
        return []
    # search for driveable_geometry_json anywhere in the manifest
    def find_geo(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "driveable_geometry_json" and isinstance(v, str):
                    try:
                        return json.loads(v)
                    except Exception:
                        return None
                r = find_geo(v)
                if r:
                    return r
        elif isinstance(obj, list):
            for v in obj:
                r = find_geo(v)
                if r:
                    return r
        return None
    geo = find_geo(m)
    if not geo:
        return []
    return [(float(p["xmin"]), float(p["xmax"]), float(p["ymin"]), float(p["ymax"]))
            for p in geo.get("prisms", [])]


def detection_coverage(perc, stamps, window_s=0.6):
    """Boolean array aligned to `stamps`: was the robot ACTUALLY detected by the
    camera within +/- window_s of each stamp? This is the honest 'is the camera
    seeing the robot' signal -- distinct from the GP reliability rho (which is a
    learned prior, low even where the camera still detects, just inaccurately)."""
    stamps = np.asarray(stamps, float)
    if perc is None or "detected" not in perc or "diag_stamp" not in perc:
        return np.ones_like(stamps, dtype=bool)
    det = perc[perc["detected"] == 1]["diag_stamp"].dropna().to_numpy(float)
    if det.size == 0:
        return np.zeros_like(stamps, dtype=bool)
    out = np.zeros_like(stamps, dtype=bool)
    for i, s in enumerate(stamps):
        out[i] = bool(np.min(np.abs(det - s)) <= window_s)
    return out


def first_cmd_time(exp):
    if exp is None or "cmd_v" not in exp:
        return None
    w = exp.get("cmd_w", pd.Series(np.zeros(len(exp))))
    moving = exp[(exp["cmd_v"].abs() > 1e-4) | (w.abs() > 1e-4)]
    return float(moving["stamp"].iloc[0]) if not moving.empty else None
