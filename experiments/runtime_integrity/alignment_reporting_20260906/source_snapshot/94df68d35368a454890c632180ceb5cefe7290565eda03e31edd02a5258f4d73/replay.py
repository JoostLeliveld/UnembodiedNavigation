"""Every fusion arm on EXACTLY the same observations, by replaying one drive offline.

**Why this is allowed here, and why it is better than one drive per arm.** The route is
fixed. The robot's path, its odometry and every camera's reading are therefore inputs, not
outcomes -- so a fusion rule cannot change what the cameras saw, only what the filter does
with it. Replaying the logged observation stream through different rules removes run-to-run
variance *by construction*: the 1.6 points of calibration and 0.09 cm of error that separate
two identical closed-loop drives become exactly zero, and a 0.3 cm difference between rules
becomes measurable instead of being buried.

It also lets the network be taken apart. "What if only camera E existed?" is not a drive you
can run without rebuilding the world; it is a subset of a stream you already have.

**What it cannot answer:** anything where the belief steers the robot. A worse filter would
have driven a different path and seen different cameras. That feedback is real and this
method is blind to it, which is exactly why the closed-loop drives still exist -- this
answers "which rule uses the data best", they answer "does it survive being in the loop".

Each observation is applied at the instant the CAMERA saw it, so the pipeline delay the
runtime has to compensate for is absent here by construction.

    python3 experiments/fusion_on_fixed_routes/replay.py [DRIVE_GLOB]

Writes ``logs/studies/fusion_on_fixed_routes/replay/results.json``.
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
    pattern = sys.argv[1] if len(sys.argv) > 1 else "drives_commissioning_v*"
    runs = sorted({p.parent for g in DRIVES.glob(pattern)
                   for p in g.rglob("fusion_observations.csv")})
    runs = [r for r in runs if "obs_stamp" in open(r / "fusion_observations.csv").readline()]
    if not runs:
        raise SystemExit(f"no drives with obs_stamp under {pattern}")
    loaded = {}
    for run in runs:
        steps, observations = load(run)
        loaded[run] = (steps, observations)
    held_out = runs[-1]
    fit = runs[:-1]
    print(f"replaying {len(runs)} drives of the same fixed route "
          f"({len(fit)} to choose the floor, 1 held out)\n")

    # --- how far may the belief shrink? ------------------------------------------------
    print("DIAGNOSTIC, not a result: how much the answer moves with a covariance floor "
          "nobody\n  has measured. The measured mechanism is in repeating_error.py.")
    print(f"  {'floor':>8} {'median error':>14} {'calibration (2 = honest)':>26}")
    ladder = {}
    for floor_cm in BELIEF_FLOOR_LADDER:
        rows = []
        for run in fit:
            steps, observations = loaded[run]
            rows.append(summarise(steps, replay(
                steps, bind(steps, observations, set(CAMERAS)), "independent",
                belief_floor_m=floor_cm / 100.0)))
        error = float(np.median([r["median_error_cm"] for r in rows]))
        nees = float(np.median([r["nees"] for r in rows]))
        ladder[floor_cm] = dict(median_error_cm=error, nees=nees)
        mark = "  <- as deployed" if floor_cm == 0.0 else ""
        print(f"  {floor_cm:6.1f} cm {error:11.2f} cm {nees:22.2f}{mark}")
    print("  No value is chosen here. Picking one against the calibration column would be "
          "fitting\n  the answer to the score.")

    # --- the arms, all at the chosen floor ----------------------------------------------
    floor = CHOSEN_FLOOR_CM / 100.0
    report = {"floor_ladder": ladder, "chosen_floor_cm": CHOSEN_FLOOR_CM, "drives": {}}
    for run in runs:
        steps, observations = loaded[run]
        label = f"{run.parents[3].name}/{run.parents[2].name}"
        tag = " (HELD OUT)" if run is held_out else ""
        attached = bind(steps, observations, set(CAMERAS))
        arms = {}
        for camera in CAMERAS:
            arms[f"camera {camera} alone"] = summarise(steps, replay(
                steps, bind(steps, observations, {camera}), "independent",
                belief_floor_m=floor))
        for rule, name in (("best_single", "all five: the most confident one"),
                           ("distance_angle", "all five: weighted by distance"),
                           ("independent", "all five: precisions add"),
                           ("joint_network", "all five: one robust batch estimate")):
            arms[name] = summarise(steps, replay(steps, attached, rule, belief_floor_m=floor))
        arms["all five: precisions add, NO floor"] = summarise(
            steps, replay(steps, attached, "independent"))
        arms["no cameras at all (dead reckoning)"] = summarise(
            steps, replay(steps, {}, "independent", belief_floor_m=floor))
        print(f"\n=== {label}{tag} — {len(steps)} steps, "
              f"{len(observations)} logged readings ===")
        print(f"  {'arm':38} {'corrections':>11} {'median':>9} {'p95':>9} "
              f"{'longest blind':>21} {'claims':>8} {'calib':>7}")
        for name, s in arms.items():
            print(f"  {name:38} {s['corrections']:11d} {s['median_error_cm']:6.2f} cm "
                  f"{s['p95_error_cm']:6.2f} cm {s['worst_blind_s']:9.1f} s = "
                  f"{s['worst_blind_m']:5.2f} m {s['claims_cm']:5.2f} cm {s['nees']:7.2f}")
        report["drives"][label + tag] = arms

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
