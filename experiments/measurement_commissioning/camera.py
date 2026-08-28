"""The cameras: rebuilt from the capture manifests.  Optics, and nothing about the robot.

Kept apart from everything else because it is the one piece that is true regardless of what
is being looked at.  ``capture.verify_reconstruction`` checks the result against predictions
already sealed into the capture and refuses to continue if they disagree, so a convention
slip cannot reach the numbers.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src/unav_common") not in sys.path:
    sys.path.insert(0, str(REPO / "src/unav_common"))

from unav_common.camera_model import ObliqueCameraModel  # noqa: E402

IMG_W, IMG_H = 1280, 720


def camera_models(dataset_dir: Path) -> dict[str, ObliqueCameraModel]:
    """Rebuild each camera from the pose and intrinsics sealed into the capture manifest.

    The manifest stores the mount as ``[x, y, z, roll, pitch, yaw]`` while
    ``ObliqueCameraModel`` wants a look-at point, so the viewing direction is turned into
    the point where it meets the floor.  ``data.verify_camera_reconstruction`` checks the
    result against predictions already stored in the capture and refuses to continue if
    they disagree, so a silent convention slip cannot reach the numbers.
    """
    cams = {}
    for c in "ABCDE":
        m = json.loads((dataset_dir / f"camera_{c}/dataset_manifest.json").read_text())
        x, y, z, _roll, pitch, yaw = m["camera_pose_xyz_rpy"]
        intr = m["camera_intrinsics"]
        direction = np.array([math.cos(pitch) * math.cos(yaw),
                              math.cos(pitch) * math.sin(yaw),
                              -math.sin(pitch)])
        look_at = np.array([x, y, z]) + direction * (z / math.sin(pitch))
        cams[f"camera_{c}"] = ObliqueCameraModel(
            cam_pos=(x, y, z), look_at=tuple(look_at),
            img_width=intr["img_width"], img_height=intr["img_height"],
            fov_h_rad=intr["fov_h_rad"])
    return cams
