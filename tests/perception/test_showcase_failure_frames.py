from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


def _load_showcase_module():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "perception"
        / "showcase_failure_frames.py"
    )
    spec = importlib.util.spec_from_file_location("showcase_failure_frames", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeCamera:
    def pixel_to_world(self, u, v):
        return float(u) / 10.0, float(v) / 10.0

    def pixel_to_world_at_z(self, u, v, z_plane):
        return float(u) / 10.0 + float(z_plane), float(v) / 10.0

    def world_to_pixel(self, x, y, z=0.0):
        return float(x) * 10.0, float(y) * 10.0, True


def test_collect_frame_files_rejects_mixed_capture_prefixes(tmp_path) -> None:
    mod = _load_showcase_module()
    (tmp_path / "frame_111_1.000.png").write_bytes(b"")
    (tmp_path / "frame_222_1.100.png").write_bytes(b"")

    with pytest.raises(ValueError, match="mixes"):
        mod.collect_frame_files(tmp_path)


def test_candidate_detector_error_uses_diag_stamp_not_stale_pixel_pose(tmp_path) -> None:
    mod = _load_showcase_module()
    (tmp_path / "frame_111_4.000.png").write_bytes(b"")
    frames = mod.collect_frame_files(tmp_path)
    truth = np.asarray([[0.0, 0.0, 0.0], [10.0, 10.0, 0.0]], dtype=float)
    rows = [{
        "diag_stamp": "4.0",
        "log_stamp": "4.5",
        "pixel_pose_stamp": "1.0",
        "obs_u": "20.0",
        "obs_v": "0.0",
        "pixel_pose_u": "999.0",
        "pixel_pose_v": "999.0",
        "yolo_detected_after_threshold": "1.0",
        "localization_error_calibrated_m": "123.0",
    }]

    candidates = mod.build_candidates(
        rows,
        truth,
        frames,
        _FakeCamera(),
        {
            "bev_y_calibration_offset_m": 0.0,
            "bev_affine_values": None,
            "bbox_contact_z_m": 0.0,
        },
        max_frame_dt=0.05,
    )

    assert len(candidates) == 1
    c = candidates[0]
    assert math.isclose(c["truth_capture_x"], 4.0)
    assert math.isclose(c["det_world_x"], 2.0)
    assert math.isclose(c["detector_capture_error_m"], 2.0)
    assert math.isclose(c["logger_localization_error_calibrated_m"], 123.0)
    assert math.isclose(c["source_stamp_mismatch_s"], 3.0)


def test_affine_calibration_replaces_constant_y_offset() -> None:
    mod = _load_showcase_module()
    calibration = {
        "bev_y_calibration_offset_m": 10.0,
        "bev_affine_values": [1.0, 0.0, 2.0, 0.0, 1.0, 3.0],
    }

    assert mod.apply_bev_calibration(4.0, 5.0, calibration) == (6.0, 8.0)
