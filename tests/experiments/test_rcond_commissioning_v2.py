from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.stats import multivariate_t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/rcond_commissioning_v2"))
import analyse_campaign as A  # noqa: E402


def row(error, repeat, session="session_0", anchor="xy_000_000", yaw_idx=0):
    return {
        "error": np.asarray(error, float), "repeat_idx": repeat, "session_id": session,
        "anchor_id": anchor, "yaw_idx": yaw_idx, "camera": "external_camera",
        "range_m": 4.0, "yaw_rad": 0.0, "gt_front_u": 640.0, "gt_rear_u": 630.0,
    }


def test_group_key_keeps_session_anchor_heading_together():
    assert A.group_key(row([0, 0], 0)) == ("session_0", "external_camera", "xy_000_000", 0)
    assert A.group_key(row([0, 0], 1)) == A.group_key(row([0, 0], 0))


def test_nearest_psd_clips_negative_component():
    value = A.nearest_psd(np.asarray([[1.0, 2.0], [2.0, 1.0]]), minimum=0.0)
    assert np.linalg.eigvalsh(value).min() >= -1e-12


def test_student_likelihood_is_finite():
    assert np.isfinite(A.student_nll(np.asarray([0.01, -0.02]), np.eye(2) * 0.001))


def test_student_nll_matches_scipy_parameterization():
    error = np.asarray([0.01, -0.02])
    covariance = np.asarray([[0.001, 0.0002], [0.0002, 0.002]])
    scale = covariance * (A.STUDENT_DF - 2.0) / A.STUDENT_DF
    expected = -multivariate_t.logpdf(error, loc=np.zeros(2), shape=scale, df=A.STUDENT_DF)
    assert A.student_nll(error, covariance) == pytest.approx(expected)


def test_capture_script_preserves_repeat_group_metadata():
    source = (ROOT / "scripts/perception/capture_projected_keypoint_dataset.py").read_text()
    for token in ("--repeats", "anchor_id", "session_id", "repeat_idx", "nominal_yaw_rad"):
        assert token in source
