"""Push the measured pixel scatter of the marked-point reading onto the floor.

The reading is: predict two marker keypoints, back-project both onto the marker plane
(z = 0.21 m), take the heading from the front-to-rear vector and base_link from the pair.
So its floor covariance is not a single-pixel projection -- it is that four-number
function's Jacobian applied to the 4x4 pixel residual covariance, including the
front/rear correlation, which the data says is real (-0.40 in the u components).

Everything here is derived from measurements already recorded in
logs/studies/keypoint_measurement/v4_retrained/per_sample.csv plus the camera in the
capture manifest. Nothing is fitted.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reading_data as rd  # noqa: E402

MARKER_Z = 0.21           # capture manifest: keypoint_marker_world_z
MID_OFFSET = 0.5 * (0.040 + (-0.100))   # midpoint of the two markers, in base_link
STEP_PX = 0.05            # central-difference step for the numeric Jacobian


def read_base(camera, pixels) -> np.ndarray:
    """The reading itself: four pixels -> base_link (x, y) in metres."""
    front = np.asarray(camera.pixel_to_world_at_z(pixels[0], pixels[1], MARKER_Z), float)[:2]
    rear = np.asarray(camera.pixel_to_world_at_z(pixels[2], pixels[3], MARKER_Z), float)[:2]
    heading = math.atan2(front[1] - rear[1], front[0] - rear[0])
    return 0.5 * (front + rear) - MID_OFFSET * np.array([math.cos(heading),
                                                         math.sin(heading)])


def jacobian(camera, pixels) -> np.ndarray:
    """d(base x, base y) / d(front u, front v, rear u, rear v), by central differences."""
    jac = np.zeros((2, 4))
    for k in range(4):
        up, dn = np.array(pixels, float), np.array(pixels, float)
        up[k] += STEP_PX
        dn[k] -= STEP_PX
        jac[:, k] = (read_base(camera, up) - read_base(camera, dn)) / (2 * STEP_PX)
    return jac


def both_rendered_mask(reading) -> np.ndarray:
    """Readings where BOTH marker disks actually rendered, among those detected."""
    rows = [row for row, ok in zip(reading.rows, reading.detected) if ok]
    return np.array([int(r['front_rendered']) + int(r['rear_rendered']) == 2 for r in rows])


def pixel_residuals(reading) -> np.ndarray:
    """(N,4) predicted-minus-projected keypoint residuals, detected readings only."""
    rows = [row for row, ok in zip(reading.rows, reading.detected) if ok]
    return np.array([[float(r['res_front_u']), float(r['res_front_v']),
                      float(r['res_rear_u']), float(r['res_rear_v'])] for r in rows])


def pixel_covariance(reading, mask=None) -> np.ndarray:
    """The 4x4 pixel residual covariance about its own mean, in px^2."""
    res = pixel_residuals(reading)
    if mask is not None:
        res = res[mask]
    return np.cov(res - res.mean(axis=0), rowvar=False, ddof=1)


def ground_truth_pixels(reading) -> np.ndarray:
    rows = [row for row, ok in zip(reading.rows, reading.detected) if ok]
    return np.array([[float(r['gt_front_u']), float(r['gt_front_v']),
                      float(r['gt_rear_u']), float(r['gt_rear_v'])] for r in rows])


def predicted_floor_covariance(reading, pixel_cov: np.ndarray) -> np.ndarray:
    """(N,2,2) covariance in cm^2 that the pixel scatter alone predicts, per reading."""
    camera = rd.camera()
    gt = ground_truth_pixels(reading)
    out = np.zeros((len(gt), 2, 2))
    for i, pixels in enumerate(gt):
        jac = jacobian(camera, pixels)
        out[i] = 1e4 * (jac @ pixel_cov @ jac.T)     # m^2 -> cm^2
    return out


def whiten(errors_cm: np.ndarray, cov_cm2: np.ndarray) -> np.ndarray:
    """Each error divided by the covariance the geometry predicts for it.

    A calibrated prediction gives a round unit cloud, i.e. mean |z|^2 / 2 == 1.
    """
    out = np.zeros_like(errors_cm)
    for i, (e, cov) in enumerate(zip(errors_cm, cov_cm2)):
        vals, vecs = np.linalg.eigh(cov)
        inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-12))) @ vecs.T
        out[i] = inv_sqrt @ e
    return out
