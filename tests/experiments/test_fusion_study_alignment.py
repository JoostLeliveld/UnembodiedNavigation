"""The fusion study's shared loader must align in time and count each reading once.

Both mistakes it prevents were made independently in more than one analysis script, and
both were invisible: every number parsed, and each was wrong by a factor (2.3x on the
belief error, 4x on the sample count) that read as a property of the camera network.

Built on synthetic CSVs with a known answer, so a regression shows up as a wrong number
rather than as a crash.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

STUDY = (Path(__file__).resolve().parents[2]
         / "experiments" / "fusion_on_fixed_routes")
sys.path.insert(0, str(STUDY))
import aligned as A  # noqa: E402


#: truth runs along +x at exactly this speed, so travel is predictable to the millimetre
SPEED_M_S = 1.0
#: the belief is logged one publish cycle after the instant it describes
BELIEF_LAG_S = 0.1


def _write_run(tmp_path: Path, *, schema: int = 2, repeats: int = 4,
               rejected_batch: int | None = None) -> Path:
    """A drive where the belief is exactly right, but logged BELIEF_LAG_S late.

    Any scorer that pairs the belief with the truth on the log row will read
    SPEED_M_S * BELIEF_LAG_S = 10 cm of error that is not there.
    """
    run = tmp_path / "experiment_20260828_000000"
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text(json.dumps({"logging_schema_version": schema}))

    experiment_columns = [
        "stamp", "gt_available", "gt_x", "gt_y", "gt_yaw", "gt_stamp",
        "planner_belief_x", "planner_belief_y", "planner_belief_stamp",
        "state_x", "state_y", "state_stamp", "state_available",
        "planner_cov_x", "planner_cov_xy", "planner_cov_y", "odom_map_gt_drift_m",
    ]
    rows = []
    for i in range(60):
        t = round(1.0 + 0.1 * i, 4)
        belief_stamp = round(t - BELIEF_LAG_S, 4)
        rows.append({
            "stamp": t, "gt_available": 1.0,
            "gt_x": t * SPEED_M_S, "gt_y": 0.0, "gt_yaw": 0.0, "gt_stamp": t,
            # exactly right about where the robot was at belief_stamp
            "planner_belief_x": belief_stamp * SPEED_M_S, "planner_belief_y": 0.0,
            "planner_belief_stamp": belief_stamp,
            "state_x": belief_stamp * SPEED_M_S, "state_y": 0.0,
            "state_stamp": belief_stamp, "state_available": 1.0,
            "planner_cov_x": 0.01, "planner_cov_xy": 0.0, "planner_cov_y": 0.01,
            "odom_map_gt_drift_m": 0.24,
        })
    with open(run / "experiment.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=experiment_columns)
        writer.writeheader()
        writer.writerows(rows)

    # Two cameras, a 5 Hz detector, and a manager republishing each round `repeats`
    # times 50 ms apart -- the live 20 Hz-against-5 Hz ratio.
    obs_columns = [
        "stamp", "source_batch_id", "camera", "used", "obs_x", "obs_y",
        "obs_cov_xx", "obs_cov_xy", "obs_cov_yy", "n_candidates", "n_used",
        "fused_x", "fused_y", "fused_cov_xx", "fused_cov_xy", "fused_cov_yy",
        "gt_x", "gt_y", "gt_x_at_obs", "gt_y_at_obs", "fused_stamp",
        "obs_repeat", "range_m", "conf", "bbox_h_px", "bbox_w_px", "obs_stamp",
    ]
    obs_rows = []
    for k in range(20):
        capture = round(1.2 + 0.2 * k, 4)
        for repeat in range(repeats):
            decision = round(capture + 0.05 * (repeat + 1), 4)
            for camera, bias_m in (("A", 0.02), ("B", -0.02)):
                obs_rows.append({
                    "stamp": decision, "source_batch_id": f"batch-{k}",
                    "camera": camera, "used": 1,
                    "obs_x": capture * SPEED_M_S + bias_m, "obs_y": 0.0,
                    "obs_cov_xx": 1e-4, "obs_cov_xy": 0.0, "obs_cov_yy": 1e-4,
                    "n_candidates": 2, "n_used": 2,
                    "fused_x": decision * SPEED_M_S, "fused_y": 0.0,
                    "fused_cov_xx": 5e-5, "fused_cov_xy": 0.0, "fused_cov_yy": 5e-5,
                    "gt_x": decision * SPEED_M_S, "gt_y": 0.0,
                    "gt_x_at_obs": capture * SPEED_M_S, "gt_y_at_obs": 0.0,
                    "fused_stamp": decision,
                    "obs_repeat": repeat, "range_m": 6.0, "conf": 0.9,
                    "bbox_h_px": 80.0, "bbox_w_px": 120.0, "obs_stamp": capture,
                })
    with open(run / "fusion_observations.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=obs_columns)
        writer.writeheader()
        writer.writerows(obs_rows)
    if schema >= 4:
        assimilation_columns = [
            "source_batch_id", "correction_stamp", "apply_stamp", "status",
            "reason", "accepted", "nis", "belief_stamp_after",
        ]
        with open(run / "correction_assimilations.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=assimilation_columns)
            writer.writeheader()
            for k in range(20):
                capture = round(1.2 + 0.2 * k, 4)
                decision = round(capture + 0.05, 4)
                rejected = k == rejected_batch
                writer.writerow({
                    "source_batch_id": f"batch-{k}",
                    "correction_stamp": decision,
                    "apply_stamp": decision + 0.01,
                    "status": "rejected" if rejected else "accepted",
                    "reason": "nis_too_large" if rejected else "accepted",
                    "accepted": 0 if rejected else 1,
                    "nis": 12.0 if rejected else 1.0,
                    "belief_stamp_after": decision,
                })
    return run


def test_belief_error_is_scored_at_the_belief_stamp(tmp_path):
    """The belief is perfect; only the log-time convention makes it look wrong."""
    run = _write_run(tmp_path)
    result = A.aligned_error_cm(run, "belief")
    aligned = result["aligned_cm"][np.isfinite(result["aligned_cm"])]
    logtime = result["logtime_cm"][np.isfinite(result["logtime_cm"])]

    assert np.median(aligned) == pytest.approx(0.0, abs=1e-6)
    # 0.1 s at 1 m/s = 10 cm of pure lag, which is what the old convention reported.
    assert np.median(logtime) == pytest.approx(BELIEF_LAG_S * SPEED_M_S * 100.0, abs=1e-6)
    assert np.nanmedian(result["lag_s"]) == pytest.approx(BELIEF_LAG_S, abs=1e-6)


def test_truth_is_never_extrapolated_past_the_recorded_interval(tmp_path):
    run = _write_run(tmp_path)
    truth = A.truth_series(run)
    gx, gy = truth.at([truth.t[0] - 1.0, truth.t[-1] + 1.0, float("nan")])
    assert np.isnan(gx).all() and np.isnan(gy).all()


def test_readings_are_counted_once_per_detection(tmp_path):
    """20 rounds x 2 cameras = 40 readings, however many times each is republished."""
    run = _write_run(tmp_path, repeats=4)
    deduped = A.readings(run)
    assert len(deduped) == 40
    assert len(A.observations(run)) == 160
    assert len(A.readings(run, dedupe=False)) == 160


def test_repeat_count_does_not_change_the_measured_error(tmp_path):
    """A statistic must not move because the manager republished more often."""
    quiet = A.readings(_write_run(tmp_path / "quiet", repeats=1))
    busy = A.readings(_write_run(tmp_path / "busy", repeats=8))
    assert len(quiet) == len(busy)
    assert (np.median([r["error_cm"] for r in quiet])
            == pytest.approx(np.median([r["error_cm"] for r in busy]), abs=1e-9))


def test_each_camera_is_scored_at_its_own_capture_time(tmp_path):
    """The cameras are 2 cm off. Scored at the decision stamp they would look worse."""
    run = _write_run(tmp_path)
    errors = [r["error_cm"] for r in A.readings(run)]
    assert np.median(errors) == pytest.approx(2.0, abs=1e-6)


def test_fused_and_per_camera_are_scored_at_their_own_instants(tmp_path):
    """The fused answer describes `fused_stamp`; the cameras describe `obs_stamp`.

    Scoring both against one truth is what let the fusion rule beat the cameras it was
    combining: here the fused answer is exactly right at its own instant, and the
    cameras carry their real 2 cm.
    """
    run = _write_run(tmp_path)
    rounds = A.fused_answers(run)
    assert len(rounds) == 20
    assert np.median([r["error_cm"] for r in rounds]) == pytest.approx(0.0, abs=1e-6)
    per_camera = [c["error_cm"] for r in rounds for c in r["cameras"].values()]
    assert np.median(per_camera) == pytest.approx(2.0, abs=1e-6)


def test_schema_1_runs_still_align_from_the_10hz_truth_series(tmp_path):
    """Drives logged before the fix must still be scorable, from the log-clock series."""
    run = _write_run(tmp_path, schema=1)
    assert A.schema_version(run) == 1
    result = A.aligned_error_cm(run, "belief")
    aligned = result["aligned_cm"][np.isfinite(result["aligned_cm"])]
    assert np.median(aligned) == pytest.approx(0.0, abs=1e-6)
    assert "schema 1" in result["truth_source"]


def test_corrections_counts_detector_rounds_not_log_rows(tmp_path):
    run = _write_run(tmp_path)
    counts = A.corrections(run)
    # One detector round is one chance to correct the belief, however many cameras
    # saw the robot in it. 20 rounds x 2 cameras is 20 corrections, not 40.
    assert counts["n_detector_rounds"] == 20
    # and the old number, which is log rows with a fresh correction, is different
    assert counts["n_state_publications"] == 60
    assert counts["state_fresh_rate_hz"] == pytest.approx(60.0 / counts["duration_s"])


def test_belief_events_require_an_accepted_source_batch_assimilation(tmp_path):
    run = _write_run(tmp_path, schema=4, rejected_batch=7)
    events = A.belief_at_fusion_events(run)
    assert len(events) == 19
    assert "batch-7" not in {event["source_batch_id"] for event in events}
    assert {event["assimilation_status"] for event in events} == {"accepted"}


def test_nees_targets_are_not_interchangeable():
    """A median NEES judged against 2.0 understates the miscalibration by 44%."""
    assert A.NEES_MEAN_TARGET == 2.0
    assert A.NEES_MEDIAN_TARGET == pytest.approx(1.3863, abs=1e-4)


def test_nees_matches_the_explicit_quadratic_form():
    rng = np.random.default_rng(0)
    residuals = rng.normal(size=(50, 2))
    covariances = np.tile(np.array([[4.0, 1.0], [1.0, 9.0]]), (50, 1, 1))
    expected = np.array([r @ np.linalg.solve(covariances[0], r) for r in residuals])
    assert A.nees(residuals, covariances) == pytest.approx(expected, abs=1e-9)


def test_landed_mask_scores_a_held_message_once():
    stamps = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    assert A.landed_mask(stamps).tolist() == [True, False, False, True, False, True]
