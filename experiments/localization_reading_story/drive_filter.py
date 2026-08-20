#!/usr/bin/env python3
"""Run the notebook's own filter and R-learning loop over the drive, once per reading.

`score_drive_frames.py` produced two readings per frame from the same image. This builds
the state sequence each of them implies -- odometry increments on a 10 Hz grid, readings
attached to the grid step they land nearest -- and hands it to
`experiments/filter_notebook/notebook_model.learn_R`, unmodified. So the loop, the gate,
the process noise, the inverse-Wishart prior and the ELBO are the notebook's, and the only
thing that differs between the two runs is which pixel the reading came from.

Ground truth rides along in `seq.truth` for scoring, exactly as the notebook carries it,
and no filter reads it.

Run: python3 experiments/localization_reading_story/drive_filter.py <drive dir>
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reading_data as rd  # noqa: E402

sys.path.insert(0, str(rd.REPO_ROOT / 'experiments/filter_notebook'))
import notebook_model as nm  # noqa: E402

CAMERA = 'camera_A'
READINGS = {
    'box_bottom': ('box_detected', 'box_x', 'box_y'),
    'keypoint': ('kp_detected', 'kp_x', 'kp_y'),
}


def driving_window(stamps, odom, truth, *, window_s: float = 10.0, ratio: float = 0.6):
    """The stretch in which the wheels and the robot agree that it is moving.

    A route driver steers on odometry, so when the robot runs into a rack it keeps turning
    the wheels: odometry sails on while the truth stands still. Everything after that is
    not a drive, and filtering it measures the collision rather than the reading. This cuts
    at the first window where the truth covers less than 60% of the ground the odometry
    claims -- which on the comb capture lands at 150 s, twenty seconds before the truth
    stops entirely.
    """
    step = max(1, int(window_s / max(np.median(np.diff(stamps)), 1e-6)))
    for i in range(0, len(stamps) - step, max(step // 5, 1)):
        j = i + step
        claimed = float(np.hypot(*np.diff(odom[i:j], axis=0).T).sum())
        actual = float(np.hypot(*np.diff(truth[i:j], axis=0).T).sum())
        if claimed > 0.5 and actual < ratio * claimed:
            return float(stamps[0]), float(stamps[i])
    return float(stamps[0]), float(stamps[-1])


class DriveSequence:
    """The notebook's Sequence, built from a drive's CSVs instead of a capture object."""

    def __init__(self, drive: Path, reading: str, *, grid_hz: float = nm.GRID_HZ):
        flag, xk, yk = READINGS[reading]
        with (drive / 'raw/experiment.csv').open(encoding='utf-8') as handle:
            odom_rows = [r for r in csv.DictReader(handle) if r.get('odom_noisy_x')]
        stamps = np.array([float(r['stamp']) for r in odom_rows])
        odom = np.column_stack([[float(r['odom_noisy_x']) for r in odom_rows],
                                [float(r['odom_noisy_y']) for r in odom_rows]])
        order = np.argsort(stamps)
        stamps, odom = stamps[order], odom[order]

        with (drive / 'readings.csv').open(encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        rows.sort(key=lambda r: float(r['stamp']))
        seen = [r for r in rows if r[flag] == '1']

        lo = max(stamps[0], float(rows[0]['stamp']))
        hi = min(stamps[-1], float(rows[-1]['stamp']))
        grid = np.arange(lo, hi, 1.0 / grid_hz)
        self.stamps = grid
        self.dt = 1.0 / grid_hz
        self.odom = np.column_stack([np.interp(grid, stamps, odom[:, i]) for i in range(2)])
        self.u = np.vstack([np.zeros((1, 2)), np.diff(self.odom, axis=0)])
        self.cameras = (CAMERA,)

        self.y = np.full((len(grid), 2), np.nan)
        self.camera: list[str | None] = [None] * len(grid)
        for row in seen:
            stamp = float(row['stamp'])
            index = int(np.argmin(np.abs(grid - stamp)))
            if abs(grid[index] - stamp) > nm.ASSOC_TOL_S or self.camera[index] is not None:
                continue
            self.y[index] = (float(row[xk]), float(row[yk]))
            self.camera[index] = CAMERA

        # EVALUATION ONLY
        truth_stamps = np.array([float(r['stamp']) for r in rows])
        truth_xy = np.column_stack([[float(r['gt_x']) for r in rows],
                                    [float(r['gt_y']) for r in rows]])
        self.truth = np.column_stack([np.interp(grid, truth_stamps, truth_xy[:, i])
                                      for i in range(2)])
        truth_yaw = np.unwrap(np.array([float(r['gt_yaw']) for r in rows]))
        self.truth_yaw = np.interp(grid, truth_stamps, truth_yaw)   # EVALUATION ONLY
        # Cut the tail where the wheels turn but the robot does not.
        start_s, end_s = driving_window(self.stamps, self.odom, self.truth)
        keep = (self.stamps >= start_s) & (self.stamps <= end_s)
        if not keep.all():
            self.stamps, self.odom, self.u = self.stamps[keep], self.odom[keep], self.u[keep]
            self.y, self.truth = self.y[keep], self.truth[keep]
            self.truth_yaw = self.truth_yaw[keep]
            self.camera = [c for c, k in zip(self.camera, keep) if k]
            self.u[0] = 0.0
        self.window = (start_s, end_s)
        self.dropped_steps = int((~keep).sum())
        self.reading = reading
        self.n_readings = int(sum(1 for c in self.camera if c is not None))

    @property
    def n_steps(self) -> int:
        return len(self.stamps)

    @property
    def observed(self) -> np.ndarray:
        return ~np.isnan(self.y[:, 0])


def run(drive: Path, reading: str, *, passes: int = 12) -> dict:
    seq = DriveSequence(drive, reading)
    R_final, history, posterior = nm.learn_R(seq, iterations=passes)
    return {'seq': seq, 'history': history, 'R': R_final, 'posterior': posterior}


def summarise(result: dict) -> None:
    import math
    seq, history = result['seq'], result['history']
    print(f"\n== {seq.reading}: {seq.n_readings} readings on {seq.n_steps} grid steps "
          f"({seq.stamps[-1] - seq.stamps[0]:.0f} s)"
          + (f"; dropped {seq.dropped_steps} steps where the wheels turned and the robot "
             f"did not" if seq.dropped_steps else ""))
    ok = np.isfinite(seq.truth[:, 0])
    obs = seq.observed
    err = 100 * (seq.y[obs] - seq.truth[obs])
    print(f"   the readings themselves: mean ({err[:, 0].mean():+.2f}, "
          f"{err[:, 1].mean():+.2f}) cm, median miss "
          f"{np.median(np.hypot(*err.T)):.2f} cm")
    for p in (0, 1, len(history) - 1):
        R = 1e4 * history[p]['R_in'][CAMERA]
        forward = nm.kalman_filter(seq, history[p]['R_in'])
        belief = 100 * (forward['m'][ok] - seq.truth[ok])
        nees = np.array([float(e @ np.linalg.inv(1e4 * P) @ e)
                         for e, P in zip(belief, forward['P'][ok])])
        print(f"   pass {p:2d}: R {math.sqrt(R[0, 0]):.2f} x {math.sqrt(R[1, 1]):.2f} cm | "
              f"belief error median {np.median(np.hypot(*belief.T)):.2f} cm | "
              f"NEES {nees.mean():5.1f} | inside its 95% "
              f"{100 * np.mean(nees <= nm.GATE_CHI2_2DOF):.0f}% | "
              f"used {int(forward['used'].sum())}, rejected {int(forward['rejected'].sum())}")


def main() -> None:
    drive = Path(sys.argv[1]).expanduser().resolve()
    for reading in READINGS:
        summarise(run(drive, reading))


if __name__ == '__main__':
    main()
