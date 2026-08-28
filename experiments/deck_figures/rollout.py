"""Predict how the robot's position uncertainty evolves along a route.

This is the calculation the planner itself performs, run offline on a candidate route so it
can be drawn.  It is a *prediction* from the commissioned model, not a recorded drive.

Two ingredients are measured: how likely each camera is to give a usable sighting at each
place, and how accurate that sighting would be (the pixel noise pushed through the imaging
geometry).  One ingredient is assumed and stated on the figure: how fast onboard odometry
drifts without help.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources as src
import style as D
sys.path.insert(0, str(D.REPO / "experiments/measurement_commissioning"))
from camera import camera_models  # noqa: E402
from observation import jacobian  # noqa: E402

DATASET = D.REPO / "logs/perception_datasets/warehouse_v2_yolo_shared_20260822"
SIGMA_PX = json.loads((D.REPO / "logs/studies/measurement_commissioning/calibration.json")
                      .read_text())["calibration"]["sigma_px"]
ODOM_DRIFT_CM_PER_M = 1.5      # stated assumption: onboard drift without external help
STEP_M = 0.20


def resample(poly, step=STEP_M):
    poly = np.asarray(poly, dtype=float)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.arange(0, cum[-1], step)
    return t, np.column_stack([np.interp(t, cum, poly[:, 0]), np.interp(t, cum, poly[:, 1])])


def rollout(poly, fields, cams, yaw=0.0):
    """Return distance, predicted 1-sigma position uncertainty (cm), and support at each step."""
    t, P = resample(poly)
    keys = {c: np.array(sorted(f)) for c, f in fields.items()}
    vals = {c: np.array([fields[c][tuple(k)] for k in keys[c]]) for c in fields}
    cov = np.eye(2) * (0.02 ** 2)          # start well localized: 2 cm
    sig, sup = [], []
    for i, (x, y) in enumerate(P):
        cov = cov + np.eye(2) * (ODOM_DRIFT_CM_PER_M / 100.0 * STEP_M) ** 2
        info = np.linalg.inv(cov)
        total_p = 0.0
        for c in fields:
            d = np.linalg.norm(keys[c] - np.array([x, y]), axis=1)
            j = int(np.argmin(d))
            p = float(vals[c][j]) if d[j] < 0.75 else 0.0
            if p <= 0:
                continue
            total_p = max(total_p, p)
            J = jacobian(cams[c], x, y, yaw)
            Ji = np.linalg.inv(J)
            R = Ji @ (SIGMA_PX ** 2 * np.eye(2)) @ Ji.T
            info = info + p * np.linalg.inv(R)      # expected information from this camera
        cov = np.linalg.inv(info)
        sig.append(math.sqrt(np.trace(cov) / 2) * 100.0)
        sup.append(total_p)
    return t, P, np.array(sig), np.array(sup)


def load_cams():
    return camera_models(DATASET)
