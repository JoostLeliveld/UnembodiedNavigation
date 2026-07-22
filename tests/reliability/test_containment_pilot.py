"""Real-data smoke test for the Paper-2 containment harness.

`experiments/multicamera_fusion_extension/tools/run_containment_pilot.py` wires the
validated bridge to the E4 subset sweep + Δ_fault fault-injection. This runs it
END-TO-END on a real capture fixture and asserts the harness produces a finite
fusion gain and structured, finite Δ_fault values — the plumbing that lets the
Paper-2 R3/R4 tables fill from a real handover capture.

Numbers here are detector-limited PLUMBING, not evidence. Skips cleanly when no
real capture fixture is on disk (never fabricates data).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))
sys.path.insert(0, str(ROOT / "experiments" / "multicamera_fusion_extension" / "tools"))

import run_containment_pilot as pilot  # noqa: E402
from reliability.calibration_perturbation import PinholeGroundCamera  # noqa: E402
from reliability.contracts import CameraQuality  # noqa: E402
from reliability.fusion import MapObservation  # noqa: E402
from reliability.replay import EvaluationFrame, ReplayConfig, ReplayFrame, ReplayMode  # noqa: E402

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
requires_fixture = pytest.mark.skipif(FIX is None, reason="no real commissioning capture fixture on disk")


@pytest.fixture(scope="module")
def res():
    return pilot.run_pilot(FIX, bias_levels_m=(0.2, 0.5))


@requires_fixture
def test_subset_sweep_runs_and_reports_gain(res):
    sub = res["subset"]
    assert len(res["cameras"]) >= 2
    assert math.isfinite(sub["best_single_p95_m"])
    assert math.isfinite(sub["full_set_p95_m"])
    assert math.isfinite(sub["fusion_gain_p95_m"])
    # at least every single-camera subset scored; full set present
    assert len(sub["per_subset_p95_m"]) >= len(res["cameras"])
    full_key = "|".join(sub["full_set"])
    assert full_key in sub["per_subset_p95_m"]


@requires_fixture
def test_delta_fault_structured_and_finite(res):
    assert set(res["delta_fault"]) == set(res["cameras"])
    for _cam, rows in res["delta_fault"].items():
        assert len(rows) == 2  # two bias levels requested
        for row in rows:
            assert math.isfinite(row["p95_dropped_m"])
            assert math.isfinite(row["p95_with_bad_m"])
            assert math.isfinite(row["delta_fault_m"])
            # dropped baseline is bias-independent -> identical across a camera's rows
        dropped = {row["p95_dropped_m"] for row in rows}
        assert len(dropped) == 1


@requires_fixture
def test_healthy_full_set_error_is_sane(res):
    # real belief-vs-GT error is small (campaign-metrics: p95 ~0.13 m); a heavy tail
    # would signal the bridge grabbed a stale/wrong position column.
    assert 0.0 < res["p95_healthy_full_m"] < 1.0


# --------------------------------------------------------------------------- #
# Fixture-free plumbing tests for the fault models + detection wiring.
# Synthetic frames; correctness only, not evidence.
# --------------------------------------------------------------------------- #
def _obs(camera_id, t, xy):
    return MapObservation(
        camera_id=camera_id, timestamp_s=float(t), xy_m=(float(xy[0]), float(xy[1])),
        covariance_m2=((0.05**2, 0.0), (0.0, 0.05**2)),
        quality=CameraQuality(camera_id=camera_id, p_available=0.9),
    )


def _two_camera_scenario(n=40):
    """Robot creeps along +x; camA + camB both accurate; odom perfectly anchors."""
    frames, evals = [], []
    for i in range(n):
        t = 0.1 * i
        truth = (1.0 + 0.03 * i, 0.0)
        frames.append(ReplayFrame(
            timestamp_s=t, odometry_xy_m=truth,
            observations=(_obs("camA", t, truth), _obs("camB", t, truth)),
        ))
        evals.append(EvaluationFrame(timestamp_s=t, truth_xy_m=truth))
    return frames, evals


def test_position_fault_fn_biases_only_target():
    frames, _ = _two_camera_scenario(3)
    fn = pilot.position_fault_fn()
    out = fn(frames, "camA", 0.5)
    by_cam = {o.camera_id: o for o in out[0].observations}
    assert by_cam["camA"].xy_m[0] == pytest.approx(frames[0].observations[0].xy_m[0] + 0.5)
    assert by_cam["camB"].xy_m == frames[0].observations[1].xy_m  # untouched


def test_calibration_fault_fn_reprojects_target():
    frames, _ = _two_camera_scenario(3)
    cam = PinholeGroundCamera.looking_at((0.0, 0.0, 6.0), (4.0, 0.0, 0.0), fov_h_deg=90.0)
    fn = pilot.calibration_fault_fn({"camA": cam})
    out = fn(frames, "camA", 2.0)
    by_cam = {o.camera_id: o for o in out[0].observations}
    assert by_cam["camA"].xy_m != frames[0].observations[0].xy_m  # yaw drift moved it
    assert by_cam["camB"].xy_m == frames[0].observations[1].xy_m  # untouched


def test_delta_fault_for_camera_structured():
    frames, evals = _two_camera_scenario()
    cfg = ReplayConfig(mode=ReplayMode.SEQUENTIAL_FUSION, nis_gate=9.21)
    rows = pilot.delta_fault_for_camera(
        frames, evals, "camA", cfg, fault_fn=pilot.position_fault_fn(), severities=(0.2, 0.5))
    assert len(rows) == 2
    assert {r["bias_m"] for r in rows} == {0.2, 0.5}
    assert len({r["p95_dropped_m"] for r in rows}) == 1  # dropped baseline is severity-independent
    for r in rows:
        assert math.isfinite(r["delta_fault_m"])


def test_detection_for_camera_scores_b6_and_catches_gross_drift():
    frames, evals = _two_camera_scenario()
    cfg = ReplayConfig(mode=ReplayMode.HEALTH_AWARE_FUSION, nis_gate=9.21)
    out = pilot.detection_for_camera(
        frames, evals, "camA", cfg, fault_fn=pilot.position_fault_fn(), severities=(0.3, 1.0))
    assert [s for s, _ in out] == [0.3, 1.0]
    for _s, det in out:
        assert det.faulted_camera == "camA"
        assert det.episode.startswith("camA@")
    gross = dict(out)[1.0]
    assert gross.detected is True  # a 1.0 m persistent bias is flagged by the health monitor


def test_build_camera_models_from_world(tmp_path):
    sdf = tmp_path / "world.sdf"
    sdf.write_text(
        "<sdf version='1.7'><world name='w'>"
        "<include><name>camera_A</name><uri>model://cam</uri><pose>0 0 6 0 0.9 0</pose></include>"
        "</world></sdf>",
        encoding="utf-8",
    )
    models = pilot.build_camera_models_from_world(sdf, ["camera_A"])
    assert "camera_A" in models
    assert isinstance(models["camera_A"], PinholeGroundCamera)
    # the model can project a ground point in front of it
    assert models["camera_A"].world_to_pixel((4.0, 0.0)) is not None
