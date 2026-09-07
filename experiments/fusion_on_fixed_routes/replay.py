"""Retired historical replay helpers, preserved only for provenance audits.

The CLI fails closed. These old helpers are not a supported accuracy evaluator;
use the manifest-bound commissioning replay through aligned.py for new analysis.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/planning"))
sys.path.insert(0, str(REPO / "src/reliability"))
from planning.core.dynamics import unicycle_process_noise  # noqa: E402
from reliability.contracts import CameraQuality  # noqa: E402
from reliability.fusion import MapObservation, joint_network_estimate_2d  # noqa: E402

DRIVES = REPO / "logs/studies/fusion_on_fixed_routes"
OUT = DRIVES / "replay"
#: the runtime's own process-noise parameters, so the replay's prediction step is the
#: filter's prediction step and not a second model invented here
PROCESS_NOISE_XY = 0.01
PROCESS_NOISE_THETA = 0.02
CAMERAS = "ABCDE"


def load(run: Path):
    """The drive as a replayable stream: dead reckoning, truth, and every camera's reading."""

    steps = []
    for row in csv.DictReader(open(run / "experiment.csv")):
        try:
            if float(row["gt_available"]) != 1.0:
                continue
            step = dict(
                t=float(row["stamp"]),
                odom=np.array([float(row["odom_map_x"]), float(row["odom_map_y"])]),
                yaw=float(row["odom_map_yaw"]),
                v=float(row["odom_v"]),
                gt=np.array([float(row["gt_x"]), float(row["gt_y"])]),
                belief=np.array([float(row["planner_belief_x"]),
                                 float(row["planner_belief_y"])]))
        except (KeyError, ValueError):
            continue
        if not (np.isfinite(step["odom"]).all() and np.isfinite(step["gt"]).all()):
            continue
        steps.append(step)
    steps.sort(key=lambda s: s["t"])

    observations = []
    for row in csv.DictReader(open(run / "fusion_observations.csv")):
        try:
            cap = float(row["obs_stamp"])
            xy = np.array([float(row["obs_x"]), float(row["obs_y"])])
            cov = np.array([[float(row["obs_cov_xx"]), float(row["obs_cov_xy"])],
                            [float(row["obs_cov_xy"]), float(row["obs_cov_yy"])]])
        except (KeyError, ValueError):
            continue
        if not (math.isfinite(cap) and np.isfinite(xy).all() and np.isfinite(cov).all()):
            continue
        if np.linalg.det(cov) <= 0.0:
            continue
        observations.append(dict(camera=row["camera"], cap=cap, xy=xy, cov=cov,
                                 used=row["used"] == "1",
                                 range_m=float(row["range_m"]) if row["range_m"] else np.nan))
    return steps, observations


def bind(steps, observations, cameras, admitted_only=True, max_range_m=None):
    """Attach each admitted reading to the log step nearest the instant it describes."""

    times = np.array([s["t"] for s in steps])
    bucket = {}
    for obs in observations:
        if obs["camera"] not in cameras:
            continue
        if admitted_only and not obs["used"]:
            continue
        if max_range_m is not None and not (obs["range_m"] < max_range_m):
            continue
        index = int(np.argmin(np.abs(times - obs["cap"])))
        if abs(times[index] - obs["cap"]) > 0.15:
            continue
        bucket.setdefault(index, {}).setdefault(obs["camera"], obs)
    # One reading per camera per step. The manager republishes each detection on about
    # four consecutive decisions, so the file holds ~4 rows per (camera, capture time);
    # `setdefault` keeps the FIRST -- the reading as the manager first computed it, not
    # the last re-projection. That is the intended choice: the later copies are the same
    # detection re-corrected against a newer belief, which is not an independent look.
    return {k: list(v.values()) for k, v in bucket.items()}


def combine(readings, rule: str):
    """The four rules, and nothing else differs between the arms."""

    means = [r["xy"] for r in readings]
    covs = [r["cov"] for r in readings]
    if rule == "best_single":
        best = int(np.argmin([np.trace(c) for c in covs]))
        return means[best], covs[best]
    if rule == "distance_angle":
        weights = np.array([1.0 / max(r["range_m"], 1.0e-3) ** 2 for r in readings])
        weights = weights / weights.sum()
        mean = sum(w * m for w, m in zip(weights, means))
        cov = sum((w ** 2) * c for w, c in zip(weights, covs))
        return mean, cov
    information = sum(np.linalg.inv(c) for c in covs)
    mean = np.linalg.solve(information, sum(np.linalg.solve(c, m) for c, m in zip(covs, means)))
    if rule == "independent":
        return mean, np.linalg.inv(information)
    if rule == "joint_network":
        batch = [MapObservation(
            camera_id=reading["camera"],
            timestamp_s=float(reading["cap"]),
            xy_m=tuple(reading["xy"]),
            covariance_m2=tuple(tuple(row) for row in reading["cov"]),
            quality=CameraQuality(camera_id=reading["camera"]),
            source="offline_replay",
        ) for reading in readings]
        joint_mean, joint_covariance = joint_network_estimate_2d(batch)
        return np.asarray(joint_mean), np.asarray(joint_covariance)
    raise ValueError(rule)


def replay(steps, attached, rule: str, initial_sigma_m: float = 0.05,
           shared_floor_m: float = 0.0, belief_floor_m: float = 0.0):
    """``shared_floor_m`` is the part of the error every reading shares, added to the
    combined R AFTER the rule has run. A filter that treats readings as independent votes
    shrinks its covariance like 1/N; the shared part does not shrink, so without this the
    belief becomes certain of a position it does not have. Unlike a per-camera inflation,
    a term added after the combination cannot be washed out by adding more cameras.

    ``belief_floor_m`` is the same idea one level up: the belief may not claim to know the
    position better than the part of the error that REPEATS. One pass of the filter over the
    stream, deterministic -- no seed, no randomness."""

    x = steps[0]["gt"].copy()
    P = np.eye(2) * initial_sigma_m ** 2
    errors, nees, claims, corrected_at = [], [], [], []
    for i, step in enumerate(steps):
        if i > 0:
            dt = step["t"] - steps[i - 1]["t"]
            x = x + (step["odom"] - steps[i - 1]["odom"])
            if dt > 0.0:
                Q = unicycle_process_noise(PROCESS_NOISE_XY, PROCESS_NOISE_THETA, dt,
                                           theta=step["yaw"], v=step["v"], base_dt=dt)
                P = P + np.asarray(Q)[:2, :2]
        readings = attached.get(i)
        if readings:
            z, R = combine(readings, rule)
            R = R + (shared_floor_m ** 2) * np.eye(2)
            S = P + R
            K = np.linalg.solve(S.T, P.T).T
            x = x + K @ (z - x)
            P = (np.eye(2) - K) @ P @ (np.eye(2) - K).T + K @ R @ K.T
            if belief_floor_m > 0.0:
                # The belief may not claim to know the position better than the part of the
                # error that repeats. Repeated looks at the same lean are not independent
                # votes, so without this P shrinks like 1/N past the systematic error.
                w, V = np.linalg.eigh(P)
                P = V @ np.diag(np.maximum(w, belief_floor_m ** 2)) @ V.T
            corrected_at.append(step["t"])
        e = x - step["gt"]
        errors.append(float(np.linalg.norm(e)))
        nees.append(float(e @ np.linalg.solve(P, e)))
        claims.append(float(math.sqrt(np.trace(P) / 2)))
    return dict(errors=np.array(errors), nees=np.array(nees), claims=np.array(claims),
                corrected_at=np.array(corrected_at))


def summarise(steps, result):
    """Accuracy, honesty and coverage, in the units the paper reports."""

    times = np.array([s["t"] for s in steps])
    gt = np.array([s["gt"] for s in steps])
    distance = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(gt, axis=0), axis=1))])
    corrected = result["corrected_at"]
    if len(corrected) > 1:
        gaps = np.diff(corrected)
        blind_m = [float(np.interp(corrected[j + 1], times, distance)
                         - np.interp(corrected[j], times, distance))
                   for j in range(len(corrected) - 1)]
        worst_gap_s, worst_gap_m = float(gaps.max()), float(max(blind_m))
    else:
        worst_gap_s = float(times[-1] - times[0])
        worst_gap_m = float(distance[-1])
    return dict(
        corrections=int(len(corrected)),
        median_error_cm=float(np.median(result["errors"]) * 100),
        p95_error_cm=float(np.percentile(result["errors"], 95) * 100),
        rmse_cm=float(np.sqrt(np.mean(result["errors"] ** 2)) * 100),
        max_error_cm=float(result["errors"].max() * 100),
        claims_cm=float(np.median(result["claims"]) * 100),
        nees=float(np.median(result["nees"]) / (2.0 * math.log(2.0)) * 2.0),
        worst_blind_s=worst_gap_s, worst_blind_m=worst_gap_m)


#: DIAGNOSTIC ONLY. A floor picked by scanning these against the calibration they are then
#: scored by is a fit, not a finding -- an earlier version of this study did exactly that and
#: it is retracted. The mechanism is measured instead by ``repeating_error.py``: a camera's
#: error is the same frame after frame for as long as the robot stays in view, so a pass
#: through a field of view carries the information of ONE reading, not sixty. The ladder is
#: kept only to show how strongly the answer depends on a number nobody has measured.
BELIEF_FLOOR_LADDER = (0.0, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 2.0)
CHOSEN_FLOOR_CM = 0.0


def main():
    raise SystemExit(
        "This historical replay is retired: its logger-tick observation pairing and "
        "directory-glob selection do not meet the evidence contract. Use "
        "experiments/icra_commissioning/replay.py with an explicit frozen selection "
        "or network_replay.py for the registered diagnostic pilot. Preserve old outputs."
    )


if __name__ == "__main__":
    main()
