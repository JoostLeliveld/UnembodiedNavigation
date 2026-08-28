import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "perception"
sys.path.insert(0, str(SCRIPT_DIR))

from shape_conditioned_update_model import feature_names, feature_vector


def row():
    return dict(image_width="1920", image_height="1080", prior_x="1", prior_y="2",
                prior_yaw=str(math.pi/2), box_x1="900", box_y1="700", box_x2="1020",
                box_y2="900", box_confidence="0.8", raw_ground_x="1.2", raw_ground_y="2.1",
                camera="camera_A", hull_u0="910", hull_v0="710", hull_u1="1010", hull_v1="890",
                h_ground_x="1.1", h_ground_y="2.05", dhx_dx="1", dhx_dy="0.1",
                dhy_dx="0.2", dhy_dy="1.1")


def test_shape_features_extend_common_contract():
    blind = feature_vector(row(), use_shape=False); shape = feature_vector(row(), use_shape=True)
    assert blind.shape == (len(feature_names(False)),)
    assert shape.shape == (len(feature_names(True)),)
    assert np.allclose(shape[:len(blind)], blind)
    assert len(shape) > len(blind)


def test_shape_delta_bottom_is_normalized():
    values = feature_vector(row(), use_shape=True)
    offset = len(feature_names(False))
    assert np.isclose(values[offset + 4], 0.0)
    assert np.isclose(values[offset + 5], 10 / 1080)
