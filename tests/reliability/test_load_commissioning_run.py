"""Validate the real-data bridge: captured commissioning run -> replay frames.

`experiments/multicamera_fusion_extension/tools/load_commissioning_run.py` is the
single link between REAL captured multi-camera runs and the (otherwise
synthetic-validated) offline replay/evaluation apparatus. This test exercises it
on a REAL capture fixture on disk and asserts the three properties that must hold
before any containment result is trusted:

1. it builds non-empty replay frames + evaluation frames with real detections;
2. schema + timestamp alignment (observations sit on their frame stamp, SPD cov,
   known camera ids, time-ordered frames);
3. the GROUND-TRUTH FIREWALL: GT lives only in EvaluationFrames, never in an
   operational frame/observation (odometry != truth), and the operational
   contracts reject an evaluation-only key structurally;
4. the bridge output is actually consumable by `run_replay`.

Skips cleanly when no real capture fixture is present (portable), so it never
fabricates data — per the no-synthetic-data rule it only asserts on real logs.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))
sys.path.insert(0, str(ROOT / "experiments" / "multicamera_fusion_extension" / "tools"))

import load_commissioning_run as bridge  # noqa: E402
from reliability.contracts import CameraQuality, LeakageError  # noqa: E402
from reliability.replay import ReplayConfig, ReplayMode, run_replay  # noqa: E402

# Real capture fixtures that carry the full raw/ + evaluation_only/ layout.
_CANDIDATES = (
    "logs/studies/multicamera_commissioning_bigwarehouse/gt_validation_smoke2_20260716",
    "logs/studies/multicamera_commissioning_bigwarehouse/gt_validation_smoke_20260716",
)


def _fixture() -> Path | None:
    for rel in _CANDIDATES:
        cand = ROOT / rel
        if (cand / "raw" / "experiment.csv").is_file() and (
            cand / "evaluation_only" / "ground_truth.csv"
        ).is_file():
            return cand
    return None


FIX = _fixture()
requires_fixture = pytest.mark.skipif(
    FIX is None, reason="no real commissioning capture fixture on disk"
)


@pytest.fixture(scope="module")
def loaded():
    return bridge.load_run(FIX)


@requires_fixture
def test_bridge_builds_frames_with_real_detections(loaded):
    assert len(loaded.frames) > 0
    assert len(loaded.evaluation_frames) > 0
    # real multi-camera detections were attached (not an empty/degenerate load)
    assert sum(loaded.observation_counts.values()) > 0
    assert any(fr.observations for fr in loaded.frames)


@requires_fixture
def test_frame_schema_and_timestamp_alignment(loaded):
    times = [fr.timestamp_s for fr in loaded.frames]
    assert times == sorted(times)
    assert all(math.isfinite(t) for t in times)
    for fr in loaded.frames:
        assert len(fr.odometry_xy_m) == 2
        assert all(math.isfinite(v) for v in fr.odometry_xy_m)
        for obs in fr.observations:
            # each detection is attached to its frame's timestamp
            assert abs(obs.timestamp_s - fr.timestamp_s) <= 1e-9
            assert obs.camera_id in bridge.DEFAULT_CAMERAS
            (a, b), (c, d) = obs.covariance_m2
            assert a > 0.0 and d > 0.0 and (a * d - b * c) > 0.0  # SPD
            assert all(math.isfinite(v) for v in obs.xy_m)


@requires_fixture
def test_ground_truth_firewall(loaded):
    frame_ts = {round(fr.timestamp_s, 9) for fr in loaded.frames}
    eval_ts = {round(ef.timestamp_s, 9) for ef in loaded.evaluation_frames}
    # ground truth only appears aligned to operational frame times, never elsewhere
    assert eval_ts.issubset(frame_ts)

    # operational odometry must NOT be the ground-truth pose (GT never leaks in)
    gt_by_t = {round(ef.timestamp_s, 9): tuple(ef.truth_xy_m) for ef in loaded.evaluation_frames}
    leaks = 0
    matched = 0
    for fr in loaded.frames:
        g = gt_by_t.get(round(fr.timestamp_s, 9))
        if g is None:
            continue
        matched += 1
        if tuple(fr.odometry_xy_m) == g:
            leaks += 1
    assert matched > 0
    assert leaks == 0

    # the operational contract rejects an evaluation-only key structurally
    with pytest.raises(LeakageError):
        CameraQuality.from_dict({"camera_id": "camera_A", "gt_x": 1.0})


@requires_fixture
def test_bridge_output_is_replay_consumable(loaded):
    result = run_replay(
        loaded.frames,
        ReplayConfig(mode=ReplayMode.SINGLE_FIXED_R),
        evaluation_frames=loaded.evaluation_frames,
    )
    assert len(result.steps) > 0
    assert result.metrics is not None
