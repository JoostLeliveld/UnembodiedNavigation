#!/usr/bin/env python3
"""Load a real captured commissioning run into offline replay inputs.

Turns the CSVs a commissioning capture writes into `ReplayFrame`s (operational:
noisy odometry + per-camera world-projected observations) and `EvaluationFrame`s
(evaluation-only ground truth), so the offline replay/evaluation apparatus can be
run on REAL data instead of synthetic fixtures.

Firewall discipline: ground truth is read ONLY into `EvaluationFrame`s, which the
replay engine uses solely for scoring (`compute_replay_metrics`). It never enters
an operational frame, an observation, or a covariance. Operational inputs come
from `raw/experiment.csv` (odom_noisy) and `raw/camera_*_perception.csv`.

Input layout (as written by the commissioning recorder):
    <run>/raw/experiment.csv                 stamp, odom_noisy_x, odom_noisy_y, cov...
    <run>/raw/camera_<X>_perception.csv      diag_stamp, detected, yolo_score_selected,
                                             pred_world_x, pred_world_y, camera_id, ...
    <run>/evaluation_only/ground_truth.csv   stamp, gt_x, gt_y, gt_yaw

Frames are built at the odometry cadence (dense belief propagation); each camera
detection is attached to the nearest odometry frame, and ground truth is resampled
onto the frame timestamps by nearest-stamp join so the replay metrics can align.
"""

from __future__ import annotations

import argparse
import bisect
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability.contracts import CameraQuality  # noqa: E402
from reliability.fusion import MapObservation  # noqa: E402
from reliability.replay import EvaluationFrame, ReplayFrame  # noqa: E402

DEFAULT_CAMERAS = ("camera_A", "camera_B", "camera_C", "camera_D")
DEFAULT_OBS_STD_M = 0.15  # placeholder per-observation std; modes may override it
DEFAULT_ASSOC_TOL_S = 0.05
DEFAULT_GT_TOL_S = 0.06


@dataclass(frozen=True)
class LoadedRun:
    frames: tuple[ReplayFrame, ...]
    evaluation_frames: tuple[EvaluationFrame, ...]
    observation_counts: dict[str, int]
    dropped_observations: dict[str, int]
    run_dir: str

    def summary(self) -> str:
        obs = ", ".join(f"{k}:{v}" for k, v in sorted(self.observation_counts.items()))
        return (
            f"{len(self.frames)} frames, {len(self.evaluation_frames)} eval frames; "
            f"observations {obs}"
        )


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _nearest_index(sorted_times: list[float], target: float) -> int | None:
    """Index of the closest value in a sorted list."""
    if not sorted_times:
        return None
    pos = bisect.bisect_left(sorted_times, target)
    best, best_d = None, float("inf")
    for cand in (pos - 1, pos, pos + 1):
        if 0 <= cand < len(sorted_times):
            d = abs(sorted_times[cand] - target)
            if d < best_d:
                best, best_d = cand, d
    return best


def load_run(
    run_dir: str | Path,
    *,
    cameras: tuple[str, ...] = DEFAULT_CAMERAS,
    obs_std_m: float = DEFAULT_OBS_STD_M,
    association_tolerance_s: float = DEFAULT_ASSOC_TOL_S,
    gt_tolerance_s: float = DEFAULT_GT_TOL_S,
) -> LoadedRun:
    """Load one captured run directory into replay + evaluation frames."""

    run = Path(run_dir)
    odom_rows = _rows(run / "raw" / "experiment.csv")
    if not odom_rows:
        raise ValueError(f"no operational odometry rows in {run}/raw/experiment.csv")

    odom: list[tuple[float, float, float]] = []
    for row in odom_rows:
        t = _float(row.get("stamp"))
        x = _float(row.get("odom_noisy_x"))
        y = _float(row.get("odom_noisy_y"))
        if t is None or x is None or y is None:
            continue
        odom.append((t, x, y))
    odom.sort(key=lambda item: item[0])
    frame_times = [item[0] for item in odom]

    # Attach detections to the nearest odometry frame.
    per_frame_obs: list[list[MapObservation]] = [[] for _ in odom]
    counts: dict[str, int] = {}
    dropped: dict[str, int] = {}
    var = float(obs_std_m) ** 2
    for cam in cameras:
        rows = _rows(run / "raw" / f"{cam}_perception.csv")
        kept = 0
        drop = 0
        for row in rows:
            if str(row.get("detected", "0")).strip() not in ("1", "true", "True"):
                continue
            t = _float(row.get("diag_stamp"))
            wx = _float(row.get("pred_world_x"))
            wy = _float(row.get("pred_world_y"))
            if t is None or wx is None or wy is None:
                continue
            idx = _nearest_index(frame_times, t)
            if idx is None or abs(frame_times[idx] - t) > association_tolerance_s:
                drop += 1
                continue
            score = _float(row.get("yolo_score_selected")) or 0.0
            score = min(max(score, 0.0), 1.0)
            per_frame_obs[idx].append(
                MapObservation(
                    camera_id=cam,
                    timestamp_s=frame_times[idx],
                    xy_m=(wx, wy),
                    covariance_m2=((var, 0.0), (0.0, var)),
                    quality=CameraQuality(camera_id=cam, p_available=score),
                    source="captured_commissioning_run",
                )
            )
            kept += 1
        counts[cam] = kept
        dropped[cam] = drop

    frames = tuple(
        ReplayFrame(timestamp_s=t, odometry_xy_m=(x, y), observations=tuple(per_frame_obs[i]))
        for i, (t, x, y) in enumerate(odom)
    )

    # Evaluation-only ground truth, resampled onto the frame timestamps.
    gt_rows = _rows(run / "evaluation_only" / "ground_truth.csv")
    gt: list[tuple[float, float, float]] = []
    for row in gt_rows:
        t = _float(row.get("stamp"))
        x = _float(row.get("gt_x"))
        y = _float(row.get("gt_y"))
        if t is None or x is None or y is None:
            continue
        gt.append((t, x, y))
    gt.sort(key=lambda item: item[0])
    gt_times = [item[0] for item in gt]

    eval_frames: list[EvaluationFrame] = []
    for t, _x, _y in odom:
        idx = _nearest_index(gt_times, t)
        if idx is None or abs(gt_times[idx] - t) > gt_tolerance_s:
            continue
        eval_frames.append(EvaluationFrame(timestamp_s=t, truth_xy_m=(gt[idx][1], gt[idx][2])))

    return LoadedRun(
        frames=frames,
        evaluation_frames=tuple(eval_frames),
        observation_counts=counts,
        dropped_observations=dropped,
        run_dir=str(run),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="captured commissioning run directory")
    args = parser.parse_args()
    loaded = load_run(args.run_dir)
    print(loaded.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
