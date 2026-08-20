"""Forward model and Jacobians for reading the robot through a possibly-drifted camera.

The reading is the marked-point reading: the two marker disks on the robot are
predicted as keypoints, so ONE sighting supplies FOUR numbers (two pixels x two
coordinates) about THREE unknowns (x, y, heading). That surplus is the whole
reason a camera's own mounting error can be estimated without ground truth.

Nothing here is fitted. The camera comes from the capture manifest, the marker
offsets come from the capture manifest, and the drift parameterisation is the
pre-existing `reliability.calibration_perturbation` used by the drift-lifecycle
study, imported rather than reimplemented.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for pkg in ("unav_common", "reliability", "experiments"):
    p = str(REPO / "src" / pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.core.world_profiles import compute_look_at_from_pose  # noqa: E402
from reliability.calibration_perturbation import (  # noqa: E402
    CalibrationPerturbation,
    PinholeGroundCamera,
    perturb,
)
from unav_common.camera_model import ObliqueCameraModel  # noqa: E402

# The six mounting degrees of freedom a wall bracket can lose. Intrinsics are not
# included: a lens does not change focal length because a bracket sagged.
PARAM_NAMES = ("pan_deg", "tilt_deg", "roll_deg", "tx_m", "ty_m", "tz_m")
N_PARAM = len(PARAM_NAMES)

# Step sizes for the central differences. Small enough to be linear, large enough
# to stay far above double-precision cancellation on ~1e3 px values.
STEP_POSE = (1.0e-4, 1.0e-4, 1.0e-4)          # m, m, rad
STEP_PARAM = (1.0e-3, 1.0e-3, 1.0e-3, 1.0e-4, 1.0e-4, 1.0e-4)   # deg, deg, deg, m, m, m


def load_capture(dataset_dir: Path):
    """Cameras and marker geometry exactly as the capture built them."""
    manifest = json.loads((dataset_dir / "capture_manifest.json").read_text())
    marker_z = float(manifest["keypoint_marker_world_z"])
    geom = manifest["marker_geometry"]
    offsets = (float(geom["front_x"]), float(geom["rear_x"]))

    cameras: dict[str, PinholeGroundCamera] = {}
    for name, spec in manifest["cameras"].items():
        pose = [float(v) for v in spec["pose"]]
        look_at = compute_look_at_from_pose(pose[:3], pose[3], pose[4], pose[5])
        oblique = ObliqueCameraModel(
            cam_pos=pose[:3], look_at=look_at,
            img_width=int(spec["img_width"]), img_height=int(spec["img_height"]),
            fov_h_rad=float(spec["fov_h_rad"]),
        )
        cameras[name] = PinholeGroundCamera(
            center_m=(pose[0], pose[1], pose[2]),
            rotation_cw=tuple(tuple(float(v) for v in row) for row in oblique.R),
            fx=float(oblique.K[0, 0]), fy=float(oblique.K[1, 1]),
            cx=float(oblique.K[0, 2]), cy=float(oblique.K[1, 2]),
            width=int(spec["img_width"]), height=int(spec["img_height"]),
            ground_z_m=marker_z,
        )
    return cameras, marker_z, offsets


def marker_world(pose, offset_x: float, marker_z: float) -> np.ndarray:
    """Where a marker disk sits in the world, given the robot pose."""
    x, y, yaw = pose
    return np.array([x + math.cos(yaw) * offset_x, y + math.sin(yaw) * offset_x, marker_z])


def drifted(camera: PinholeGroundCamera, theta) -> PinholeGroundCamera:
    """The camera as it physically is, given a six-vector of mounting error."""
    pan, tilt, roll, tx, ty, tz = (float(v) for v in theta)
    return perturb(camera, CalibrationPerturbation(
        yaw_deg=pan, pitch_deg=tilt, roll_deg=roll, tx_m=tx, ty_m=ty, tz_m=tz))


def predict(camera: PinholeGroundCamera, pose, theta, marker_z: float, offsets) -> np.ndarray:
    """The four pixels this camera reports for this robot pose: [fu, fv, ru, rv]."""
    cam = camera if theta is None else drifted(camera, theta)
    out = np.empty(4)
    for k, off in enumerate(offsets):
        uv = cam.world_to_pixel(marker_world(pose, off, marker_z)[:2], marker_z)
        if uv is None:
            return None
        out[2 * k], out[2 * k + 1] = uv
    return out


def _central(fn, base, steps) -> np.ndarray:
    jac = np.empty((4, len(base)))
    for k, h in enumerate(steps):
        up, dn = np.array(base, float), np.array(base, float)
        up[k] += h
        dn[k] -= h
        jac[:, k] = (fn(up) - fn(dn)) / (2.0 * h)
    return jac


def jacobians(camera: PinholeGroundCamera, pose, marker_z: float, offsets):
    """d(4 pixels)/d(pose) [4x3] and d(4 pixels)/d(mounting error) [4x6], at zero drift."""
    zero = np.zeros(N_PARAM)
    j_pose = _central(lambda p: predict(camera, p, zero, marker_z, offsets),
                      np.asarray(pose, float), STEP_POSE)
    j_theta = _central(lambda t: predict(camera, pose, t, marker_z, offsets),
                       zero, STEP_PARAM)
    return j_pose, j_theta
