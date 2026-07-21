from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))
sys.path.insert(0, str(ROOT / "experiments" / "multicamera_fusion_extension" / "tools"))

from reliability.contracts import CameraQuality  # noqa: E402
from reliability.fusion import MapObservation  # noqa: E402
from reliability.replay import EvaluationFrame, ReplayConfig, ReplayFrame, ReplayMode  # noqa: E402

from replay_sweeps import (  # noqa: E402
    all_nonempty_subsets,
    bias_camera_position,
    delay_camera,
    drop_camera_after,
    drop_camera_permanent,
    filter_frames_to_subset,
    run_calibration_drift_sweep,
    run_camera_subset_sweep,
    run_dropout_sweep,
    run_latency_sweep,
)


# Camera A: precise in x, noisy in y. Camera B: noisy in x, precise in y.
# Complementary viewpoints → fusion must beat either single camera.
_CAM = {
    "camera_A": (0.02, 0.20),
    "camera_B": (0.20, 0.02),
}
_DT = 0.1
_ODOM_DRIFT_PER_STEP = 0.003  # x-only odometry drift → x needs camera correction


def _obs(cam: str, t: float, xy) -> MapObservation:
    sx, sy = _CAM[cam]
    return MapObservation(
        camera_id=cam,
        timestamp_s=t,
        xy_m=(float(xy[0]), float(xy[1])),
        covariance_m2=((sx * sx, 0.0), (0.0, sy * sy)),
        quality=CameraQuality(camera_id=cam),
    )


def _synthetic_run(n: int = 80, *, seed: int = 0, cameras=("camera_A", "camera_B")):
    rng = random.Random(seed)
    frames, evals = [], []
    for k in range(n):
        t = _DT * k
        truth = (0.1 * k, 0.0)
        odom = (0.1 * k + _ODOM_DRIFT_PER_STEP * k, 0.0)  # drifts ahead in x
        obs = []
        for cam in cameras:
            sx, sy = _CAM[cam]
            noisy = (truth[0] + rng.gauss(0.0, sx), truth[1] + rng.gauss(0.0, sy))
            obs.append(_obs(cam, t, noisy))
        frames.append(ReplayFrame(timestamp_s=t, odometry_xy_m=odom, observations=tuple(obs)))
        evals.append(EvaluationFrame(timestamp_s=t, truth_xy_m=truth))
    return frames, evals


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def test_filter_and_drop_transforms() -> None:
    frames, _ = _synthetic_run(10)
    only_a = filter_frames_to_subset(frames, ["camera_A"])
    assert all(all(o.camera_id == "camera_A" for o in f.observations) for f in only_a)
    dropped = drop_camera_permanent(frames, "camera_A")
    assert all(all(o.camera_id != "camera_A" for o in f.observations) for f in dropped)


def test_drop_camera_after_cutoff() -> None:
    frames, _ = _synthetic_run(10)
    out = drop_camera_after(frames, "camera_A", 0.5)
    has_a = ["camera_A" in {o.camera_id for o in f.observations} for f in out]
    assert has_a[0] is True and has_a[-1] is False


def test_all_nonempty_subsets_count() -> None:
    subs = all_nonempty_subsets(["camera_A", "camera_B", "camera_C"])
    assert len(subs) == 7  # 2^3 - 1
    assert ("camera_A",) in subs and ("camera_A", "camera_B", "camera_C") in subs


def test_delay_camera_moves_observations_later() -> None:
    frames, _ = _synthetic_run(10)
    delayed = delay_camera(frames, "camera_A", 0.25)  # ~2.5 frames
    # First frames lose their camera_A obs (pushed to later frames).
    assert "camera_A" not in {o.camera_id for o in delayed[0].observations}
    total_a = sum(sum(o.camera_id == "camera_A" for o in f.observations) for f in delayed)
    # Some A observations at the tail are pushed off the end and dropped.
    assert 0 < total_a <= 10


def test_transform_guards() -> None:
    frames, _ = _synthetic_run(5)
    with pytest.raises(ValueError):
        drop_camera_after(frames, "camera_A", 1.5)
    with pytest.raises(ValueError):
        delay_camera(frames, "camera_A", -0.1)


# --------------------------------------------------------------------------- #
# E4 subset sweep
# --------------------------------------------------------------------------- #
def test_subset_sweep_fusion_beats_best_single() -> None:
    frames, evals = _synthetic_run(80, seed=1)
    result = run_camera_subset_sweep(frames, evals, camera_ids=["camera_A", "camera_B"])
    assert set(result.per_subset) == {("camera_A",), ("camera_B",), ("camera_A", "camera_B")}
    # Complementary cameras → full-set p95 strictly below the best single camera.
    assert result.full_set_p95 < result.best_single_p95
    assert result.fusion_gain_p95 > 0.0


def test_subset_sweep_requires_eval() -> None:
    frames, _ = _synthetic_run(10)
    with pytest.raises(ValueError):
        run_camera_subset_sweep(frames, [], camera_ids=["camera_A"])


# --------------------------------------------------------------------------- #
# E5 dropout & latency
# --------------------------------------------------------------------------- #
def test_dropout_sweep_degrades_with_probability() -> None:
    frames, evals = _synthetic_run(80, seed=2)
    sweep = run_dropout_sweep(frames, evals, "camera_A", [0.0, 0.25, 0.5, 0.75], seed=3)
    assert sweep.axis == "p_drop"
    # Losing the precise-x camera more often raises error at the extremes.
    assert sweep.p95_by_severity[-1] > sweep.p95_by_severity[0]
    assert sweep.error_severity_auc_p95 is not None and sweep.error_severity_auc_p95 > 0.0


def test_latency_sweep_degrades_with_delay() -> None:
    frames, evals = _synthetic_run(80, seed=4)
    sweep = run_latency_sweep(frames, evals, "camera_A", [0.0, 0.2, 0.5, 1.0])
    assert sweep.axis == "delay_s"
    assert sweep.p95_by_severity[-1] > sweep.p95_by_severity[0]


# --------------------------------------------------------------------------- #
# E6 calibration drift
# --------------------------------------------------------------------------- #
def test_drift_sweep_monotone_without_gate() -> None:
    # With NO NIS gate the biased camera is always trusted, so error rises
    # monotonically with bias — isolating the "trusting a drifted camera drags
    # the estimate" mechanism the transform is meant to expose.
    frames, evals = _synthetic_run(80, seed=5)
    levels = [0.0, 0.1, 0.2, 0.4, 0.8]
    ungated = ReplayConfig(mode=ReplayMode.SEQUENTIAL_FUSION, nis_gate=None)
    sweep = run_calibration_drift_sweep(
        frames, evals, "camera_A", levels, direction=(1.0, 0.0), config=ungated
    )
    assert sweep.axis == "bias_m"
    assert sweep.p95_by_severity[-1] > sweep.p95_by_severity[0]
    increases = sum(
        sweep.p95_by_severity[i + 1] >= sweep.p95_by_severity[i] - 1e-6
        for i in range(len(levels) - 1)
    )
    assert increases == len(levels) - 1
    assert sweep.error_severity_auc_p95 is not None and sweep.error_severity_auc_p95 > 0.0


def test_drift_sweep_gate_rejects_biased_camera() -> None:
    # With the default NIS gate, a grossly biased camera should be REJECTED,
    # so the update-acceptance rate falls relative to the unbiased baseline.
    frames, evals = _synthetic_run(80, seed=5)
    sweep = run_calibration_drift_sweep(frames, evals, "camera_A", [0.0, 0.8], direction=(1.0, 0.0))
    assert sweep.metrics[-1].update_acceptance_rate < sweep.metrics[0].update_acceptance_rate


def test_drift_direction_guard() -> None:
    frames, evals = _synthetic_run(10)
    with pytest.raises(ValueError):
        run_calibration_drift_sweep(frames, evals, "camera_A", [0.0, 0.1], direction=(0.0, 0.0))
