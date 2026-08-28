"""Shared loading for the box-versus-centre figures."""
import csv, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sources as src            # noqa: E402
import style as D                # noqa: E402
sys.path.insert(0, str(D.REPO / "experiments/measurement_commissioning"))
from camera import camera_models            # noqa: E402
from observation import h, heading_jacobian, predicted_box  # noqa: E402

OUT = D.REPO / "logs/studies/deck_figures/observation"
OUT.mkdir(parents=True, exist_ok=True)
DATA = D.REPO / "logs/perception_datasets/warehouse_v2_yolo_shared_20260822"
CAMS = camera_models(DATA)


def index():
    """The capture index: every attempted sighting with its pose and image."""
    return list(csv.DictReader(open(DATA / "localization_calibration_index_hull.csv")))


def gap_vector(cam, x, y, yaw):
    """Where the box bottom-centre lands minus where the robot is, in centimetres."""
    p = h(cam, x, y, yaw)
    if p is None:
        return None
    gx, gy = cam.pixel_to_world(p[0], p[1])[:2]
    return np.array([(gx - x) * 100.0, (gy - y) * 100.0])
