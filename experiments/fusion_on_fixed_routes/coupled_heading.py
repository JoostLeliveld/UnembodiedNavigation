"""Does letting a position fix correct the heading, through the covariance, actually work?

The runtime deletes that channel. Every accepted camera correction does

    S[:2, 2] = 0 ;  S[2, :2] = 0 ;  S[2, 2] = S_predicted[2, 2]

so the position-heading correlation the prediction step builds is thrown away at each update,
the heading variance never comes down, and the heading mean is overwritten with map-frame
odometry. The comment in `unicycle_planner_node.py` records why: an earlier version kept the
PRIOR cross terms beside the POSTERIOR position block, which is not a covariance matrix -- it
went indefinite, produced negative innovation statistics, and diverged.

That failure was the hybrid, not the coupling. A full three-state Joseph update is positive
semi-definite by construction:

    S = (I - K H) S (I - K H)' + K R K'

so it cannot go indefinite however small R is. This script runs both on the same recorded
observation stream and reports what each is worth, including the smallest eigenvalue, so the
claim "the Joseph form stays a covariance matrix" is checked rather than asserted.

    python3 experiments/fusion_on_fixed_routes/coupled_heading.py

Writes ``logs/studies/fusion_on_fixed_routes/replay/coupled_heading.json``.
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
sys.path.insert(0, str(REPO / "experiments/fusion_on_fixed_routes"))
from planning.core.dynamics import unicycle_process_noise, unicycle_jacobian  # noqa: E402
import replay as RP                                                           # noqa: E402

DRIVES = REPO / "logs/studies/fusion_on_fixed_routes"
OUT = DRIVES / "replay"
#: the runtime's own values -- this study changes the update, not the noise model
PROCESS_NOISE_XY = 0.01
PROCESS_NOISE_THETA = 0.02
H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def stream(run: Path):
    """Dead reckoning, truth and the camera readings, as a replayable sequence."""

    steps = []
    for row in csv.DictReader(open(run / "experiment.csv")):
        try:
            if float(row["gt_available"]) != 1.0:
                continue
            step = dict(
                t=float(row["stamp"]),
                odom=np.array([float(row["odom_map_x"]), float(row["odom_map_y"]),
                               float(row["odom_map_yaw"])]),
                gt=np.array([float(row["gt_x"]), float(row["gt_y"]), float(row["gt_yaw"])]))
        except (KeyError, ValueError):
            continue
        if not (np.isfinite(step["odom"]).all() and np.isfinite(step["gt"]).all()):
            continue
        steps.append(step)
    steps.sort(key=lambda s: s["t"])
    _, observations = RP.load(run)
    return steps, observations


def run_filter(steps, attached, coupled: bool, rule: str = "independent"):
    """One pass. ``coupled=False`` reproduces the runtime: cross terms deleted, heading
    variance rolled back to its predicted value, heading mean taken from odometry."""

    m = steps[0]["gt"].copy()
    S = np.diag([0.05 ** 2, 0.05 ** 2, math.radians(2.0) ** 2])
    errors, nees, heading_error, min_eig, claims = [], [], [], [], []
    for i, step in enumerate(steps):
        if i > 0:
            previous = steps[i - 1]
            dt = step["t"] - previous["t"]
            delta = step["odom"] - previous["odom"]
            delta[2] = wrap(delta[2])
            # Odometry is a RELATIVE motion input, so its increment has to be read in the
            # robot's own frame and then re-applied using the FILTER's heading. Adding the
            # world-frame increment straight onto the state makes the position advance
            # independently of the believed heading, while the Jacobian below says it does
            # not -- and a filter whose covariance claims a coupling its mean does not have
            # will attribute position innovations to a heading that could not have caused
            # them. That is what made heading diverge in the first version of this script.
            c, s_ = math.cos(previous["odom"][2]), math.sin(previous["odom"][2])
            body = np.array([c * delta[0] + s_ * delta[1],      # forward
                             -s_ * delta[0] + c * delta[1]])    # lateral
            cf, sf = math.cos(m[2]), math.sin(m[2])
            m = m + np.array([cf * body[0] - sf * body[1],
                              sf * body[0] + cf * body[1],
                              delta[2]])
            m[2] = wrap(m[2])
            if dt > 0.0:
                speed = float(body[0] / dt)
                omega = float(delta[2] / dt)
                F = np.asarray(unicycle_jacobian(m, np.array([speed, omega]), dt), dtype=float)
                Q = np.asarray(unicycle_process_noise(PROCESS_NOISE_XY, PROCESS_NOISE_THETA,
                                                      dt, theta=float(m[2]), v=speed,
                                                      base_dt=dt), dtype=float)
                S = F @ S @ F.T + Q
        readings = attached.get(i)
        if readings:
            z, R = RP.combine(readings, rule)
            S_pred = S.copy()
            innovation_cov = H @ S @ H.T + R
            K = S @ H.T @ np.linalg.inv(innovation_cov)
            m = m + K @ (z - H @ m)
            m[2] = wrap(m[2])
            IKH = np.eye(3) - K @ H
            S = IKH @ S @ IKH.T + K @ R @ K.T          # Joseph form: PSD by construction
            if not coupled:
                # exactly what the runtime does today
                S[:2, 2] = 0.0
                S[2, :2] = 0.0
                S[2, 2] = S_pred[2, 2]
                m[2] = wrap(step["odom"][2])
            min_eig.append(float(np.linalg.eigvalsh(S).min()))
        e = m[:2] - step["gt"][:2]
        errors.append(float(np.linalg.norm(e)))
        nees.append(float(e @ np.linalg.solve(S[:2, :2], e)))
        claims.append(float(math.sqrt(np.trace(S[:2, :2]) / 2)))
        heading_error.append(abs(wrap(float(m[2] - step["gt"][2]))))
    return dict(errors=np.asarray(errors), nees=np.asarray(nees),
                heading=np.asarray(heading_error), claims=np.asarray(claims),
                min_eig=np.asarray(min_eig) if min_eig else np.zeros(1),
                yaw_sigma=float(math.degrees(math.sqrt(max(S[2, 2], 0.0)))))


def summarise(result):
    return dict(
        median_error_cm=float(np.median(result["errors"]) * 100),
        p95_error_cm=float(np.percentile(result["errors"], 95) * 100),
        claims_cm=float(np.median(result["claims"]) * 100),
        calibration=float(np.median(result["nees"]) / (2.0 * math.log(2.0)) * 2.0),
        median_heading_error_deg=float(math.degrees(np.median(result["heading"]))),
        final_heading_sigma_deg=result["yaw_sigma"],
        smallest_eigenvalue=float(result["min_eig"].min()))


def main():
    runs = [r for r in sorted({p.parent for g in DRIVES.glob("drives_commissioning_v*")
                               for p in g.rglob("fusion_observations.csv")})
            if "obs_stamp" in open(r / "fusion_observations.csv").readline()]
    report = {}
    print(f"{'drive':10} {'update':34} {'position':>10} {'claims':>9} {'calib':>7} "
          f"{'heading err':>12} {'heading sigma':>14} {'min eigenvalue':>16}")
    for run in runs:
        steps, observations = stream(run)
        attached = RP.bind(steps, observations, set("ABCDE"))
        label = f"{run.parents[3].name[-2:]}/{run.parents[2].name[:6]}"
        arms = {}
        for coupled, name in ((False, "as deployed: cross terms deleted"),
                              (True, "coupled: full Joseph 3-state update")):
            arms[name] = summarise(run_filter(steps, attached, coupled))
        for name, s in arms.items():
            print(f"{label:10} {name:34} {s['median_error_cm']:7.2f} cm "
                  f"{s['claims_cm']:6.2f} cm {s['calibration']:7.2f} "
                  f"{s['median_heading_error_deg']:9.2f} deg {s['final_heading_sigma_deg']:11.2f} deg "
                  f"{s['smallest_eigenvalue']:16.2e}")
        report[label] = arms
        print()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coupled_heading.json").write_text(json.dumps(report, indent=1))
    print("A negative smallest eigenvalue would mean the covariance stopped being one.")
    print(f"wrote {OUT / 'coupled_heading.json'}")


if __name__ == "__main__":
    main()
