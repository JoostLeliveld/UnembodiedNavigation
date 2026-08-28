"""The one real moment the hull storyline is told on, and the pieces every panel needs.

Pose (0.67, -5.48) at 300 deg: four of the five cameras returned a detection there, and their
ellipses run from 0.9 cm to 6.0 cm, so one moment carries the whole argument. Chosen for that
spread and for camera B and E crossing at 90 degrees -- stated here rather than searched for
again in every figure.

Nothing in this module fits anything. sigma_px is read from the frozen calibration.
"""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "deck_figures"))
sys.path.insert(0, str(HERE.parents[2] / "deck_figures/observation"))
import style as D                                            # noqa: E402
from _common import CAMS, DATA, index                        # noqa: E402
from observation import h, jacobian, predicted_box           # noqa: E402

OUT = D.REPO / "logs/studies/fusion_on_fixed_routes/00_hull_observation"
OUT.mkdir(parents=True, exist_ok=True)
DETECTOR_CSV = "detector_readings_halfopen_detect_20260825.csv"
SIGMA_PX = json.loads((D.REPO / "logs/studies/measurement_commissioning/calibration.json")
                      .read_text())["calibration"]["sigma_px"]
#: the moment, as it appears in the capture index (strings, so no coordinate rounding)
POSE = ("0.6687499999999993", "-5.477777777777778", "5.235987755982988")


def _detections(camera_id):
    path = DATA / camera_id / DETECTOR_CSV
    return {r["image"]: r for r in csv.DictReader(open(path)) if r["detected"] == "1"}


def moment():
    """Every camera that saw the robot at this pose, with its frame and both boxes."""
    rows = [r for r in index()
            if (r["robot_x"], r["robot_y"], r["robot_yaw"]) == POSE and r["image"]]
    if not rows:
        raise SystemExit(f"pose {POSE} not in the capture index")
    x, y, yaw = (float(v) for v in POSE)
    out = {}
    for r in rows:
        cam_id = r["camera_id"]
        det = _detections(cam_id).get(r["image"])
        if det is None:
            continue
        cam = CAMS[cam_id]
        box = predicted_box(cam, x, y, yaw)
        pred = h(cam, x, y, yaw)
        if box is None or pred is None:
            continue
        dbox = (float(det["x0"]), float(det["y0"]), float(det["x1"]), float(det["y1"]))
        seen = np.array([0.5 * (dbox[0] + dbox[2]), dbox[3]])
        J = jacobian(cam, x, y, yaw)
        Ji = np.linalg.inv(J)
        out[cam_id] = dict(
            cam=cam, image=DATA / cam_id / r["image"], pred_box=box, det_box=dbox,
            pred_uv=np.array(pred), det_uv=seen, J=J, Ji=Ji,
            residual_px=seen - np.array(pred),
            # this camera's own estimate of the robot, and its covariance
            est=np.array([x, y]) + Ji @ (seen - np.array(pred)),
            cov=Ji @ (SIGMA_PX ** 2 * np.eye(2)) @ Ji.T,
            range_m=float(np.linalg.norm(cam.cam_pos[:2] - np.array([x, y]))),
            conf=float(det["confidence"]))
    return (x, y, yaw), dict(sorted(out.items()))


def solve_position(cam, uv, yaw, start):
    """Which robot position would put the predicted box bottom-centre on this pixel?

    Gauss-Newton on h(x, y | yaw) = uv. This is the calculation that CANNOT be done from a
    pixel alone: it needs a heading, and a different heading gives a different answer.
    """
    p = np.array(start, dtype=float)
    for _ in range(60):
        cur = h(cam, p[0], p[1], yaw)
        if cur is None:
            return None
        step = np.linalg.inv(jacobian(cam, p[0], p[1], yaw)) @ (np.asarray(uv) - np.array(cur))
        p = p + step
        if np.linalg.norm(step) < 1e-6:
            break
    return p


def ellipse(cov, n_sigma=1.0, n=200):
    """Points of the n-sigma ellipse of a 2x2 covariance, in the same units."""
    w, V = np.linalg.eigh(cov)
    t = np.linspace(0, 2 * math.pi, n)
    return (V @ (np.sqrt(np.maximum(w, 0))[:, None] * np.array([np.cos(t), np.sin(t)])) * n_sigma).T


def crop(im, box, pad_x=1.05, pad_y=1.25):
    """A window around a predicted box, clipped to the frame."""
    cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    half = max(box[2] - box[0], box[3] - box[1], 40.0)
    x0 = int(max(cx - half * pad_x, 0)); x1 = int(min(cx + half * pad_x, im.shape[1]))
    y0 = int(max(cy - half * pad_y, 0)); y1 = int(min(cy + half * pad_y, im.shape[0]))
    return x0, y0, x1, y1
