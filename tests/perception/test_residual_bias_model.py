import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "perception"
sys.path.insert(0, str(SCRIPT_DIR))

from residual_bias_model import CAMERAS, FEATURE_NAMES, feature_vector


def _row():
    return {
        "image_width": "1920", "image_height": "1080",
        "robot_yaw": str(math.pi / 2), "baseline_bearing_rad": "0",
        "box_bottom_u": "960", "box_bottom_v": "900",
        "box_x1": "900", "box_x2": "1020", "box_y1": "800", "box_y2": "900",
        "box_confidence": "0.8", "baseline_range_m": "9", "camera": "camera_C",
    }


def test_feature_contract_and_heading_encoding():
    values = feature_vector(_row())
    assert values.shape == (len(FEATURE_NAMES),)
    assert np.isclose(values[0], 0.5)
    assert np.isclose(values[6], 1.0)
    assert np.isclose(values[7], 0.0, atol=1e-6)
    assert values[8 + CAMERAS.index("camera_C")] == 1.0


def test_heading_ablation_removes_both_periodic_features():
    values = feature_vector(_row(), disable_heading=True)
    assert values[6] == 0.0
    assert values[7] == 0.0


def test_heading_wrap_is_continuous():
    row = _row(); row["baseline_bearing_rad"] = str(math.pi)
    left = feature_vector(row, heading_rad=-math.pi + 1e-6)
    right = feature_vector(row, heading_rad=math.pi - 1e-6)
    assert np.allclose(left[6:8], right[6:8], atol=3e-6)
