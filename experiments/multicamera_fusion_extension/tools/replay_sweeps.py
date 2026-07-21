#!/usr/bin/env python3
"""Offline replay sweep drivers for experiments E4/E5/E6 (plan 11).

STUDY code (not the runtime package): these drivers consume recorded operational
frames plus evaluation-only truth frames the CALLER supplies, transform the
frames (mask cameras, drop/delay observations, perturb positions), re-run the
existing offline replay engine (`reliability.replay`), and aggregate per-run
metrics. They load no truth themselves — truth arrives as ``EvaluationFrame``s
from an evaluation harness — and they never fit anything.

Every sweep reuses one replay pipeline over identical detections, so any
difference between conditions is attributable to the transform, not to a retuned
filter (the §21 fusion-gate discipline). Cross-run paired confidence intervals
are the caller's job via ``reliability.campaign_statistics`` (run each recorded
run through the sweep, collect per-run metrics, then ``summarize_paired``).

- E4 camera subsets: ``run_camera_subset_sweep`` — every non-empty camera subset
  through the SAME fusion mode; a size-1 subset equals the single-camera result,
  so fusion-gain-vs-best-single is a like-for-like comparison.
- E5 dropout & latency: ``run_dropout_sweep`` / ``run_latency_sweep`` — masking
  and delayed-association transforms over a severity axis, with error-severity
  AUC.
- E6 calibration drift: ``run_calibration_drift_sweep`` — a controlled per-camera
  world-position bias (the "controlled calibration-ablation evidence" of §E6),
  images untouched, over a severity axis.

Validated in ``tests/reliability/test_replay_sweeps.py`` against synthetic frames
with known monotone behaviour, so the drivers are turn-key for real recordings.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
import math
from pathlib import Path
import random
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability.replay import (  # noqa: E402
    EvaluationFrame,
    ReplayConfig,
    ReplayFrame,
    ReplayMetrics,
    ReplayMode,
    run_replay,
)
from reliability.campaign_statistics import error_severity_auc  # noqa: E402


def default_fusion_config() -> ReplayConfig:
    """Sequential multi-camera fusion at the frozen NIS gate (2-dof 0.99)."""

    return ReplayConfig(mode=ReplayMode.SEQUENTIAL_FUSION, nis_gate=9.21)


# --------------------------------------------------------------------------- #
# Frame transforms (pure; return new frame lists, never mutate).
# --------------------------------------------------------------------------- #
def _camera_ids(frames: Sequence[ReplayFrame]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for frame in frames:
        for obs in frame.observations:
            seen.setdefault(obs.camera_id, None)
    return tuple(seen)


def filter_frames_to_subset(
    frames: Sequence[ReplayFrame], camera_ids: Sequence[str]
) -> list[ReplayFrame]:
    """Keep only observations whose camera is in ``camera_ids`` (odometry kept)."""

    keep = set(camera_ids)
    return [
        replace(f, observations=tuple(o for o in f.observations if o.camera_id in keep))
        for f in frames
    ]


def drop_camera_permanent(frames: Sequence[ReplayFrame], camera_id: str) -> list[ReplayFrame]:
    """Remove one camera for the whole run (E5 permanent outage)."""

    return [
        replace(f, observations=tuple(o for o in f.observations if o.camera_id != camera_id))
        for f in frames
    ]


def drop_camera_after(
    frames: Sequence[ReplayFrame], camera_id: str, at_fraction: float
) -> list[ReplayFrame]:
    """Disable one camera after a fraction of route progress (E5 sudden outage)."""

    if not 0.0 <= at_fraction <= 1.0:
        raise ValueError("at_fraction must be in [0, 1]")
    ordered = sorted(frames, key=lambda fr: fr.timestamp_s)
    cutoff_idx = int(math.floor(at_fraction * len(ordered)))
    out = []
    for i, f in enumerate(ordered):
        if i >= cutoff_idx:
            out.append(
                replace(f, observations=tuple(o for o in f.observations if o.camera_id != camera_id))
            )
        else:
            out.append(f)
    return out


def drop_camera_intermittent(
    frames: Sequence[ReplayFrame], camera_id: str, p_drop: float, *, seed: int = 0
) -> list[ReplayFrame]:
    """Independently drop each of a camera's observations with prob ``p_drop``."""

    if not 0.0 <= p_drop <= 1.0:
        raise ValueError("p_drop must be in [0, 1]")
    rng = random.Random(seed)
    out = []
    for f in sorted(frames, key=lambda fr: fr.timestamp_s):
        kept = []
        for o in f.observations:
            if o.camera_id == camera_id and rng.random() < p_drop:
                continue
            kept.append(o)
        out.append(replace(f, observations=tuple(kept)))
    return out


def delay_camera(
    frames: Sequence[ReplayFrame], camera_id: str, delay_s: float, *, tolerance_s: float = 1.0e-6
) -> list[ReplayFrame]:
    """Delay one camera's observations by ``delay_s`` (E5 latency).

    Each delayed observation keeps its measured position (the robot's location at
    capture time) but is re-associated with the frame whose timestamp is closest
    to capture-time + delay. Because the belief has since propagated on odometry,
    a stale observation injects the motion-over-delay as innovation — the honest
    offline model of measurement latency. Observations delayed past the last
    frame are dropped (arrived after the run ended).
    """

    if delay_s < 0.0:
        raise ValueError("delay_s must be non-negative")
    ordered = sorted(frames, key=lambda fr: fr.timestamp_s)
    times = [f.timestamp_s for f in ordered]
    kept_by_idx: list[list] = [[] for _ in ordered]
    for i, f in enumerate(ordered):
        for o in f.observations:
            if o.camera_id != camera_id or delay_s == 0.0:
                kept_by_idx[i].append(o)
                continue
            target = o.timestamp_s + delay_s
            # nearest frame at or after target; fall back to nearest overall
            j = _nearest_frame_index(times, target)
            if j is None or times[j] < target - tolerance_s and times[-1] < target - tolerance_s:
                continue  # arrives after the run ends → dropped
            kept_by_idx[j].append(replace(o, timestamp_s=times[j]))
    return [replace(f, observations=tuple(kept_by_idx[i])) for i, f in enumerate(ordered)]


def _nearest_frame_index(times: Sequence[float], target: float) -> int | None:
    if not times:
        return None
    # smallest index whose time >= target; else the last frame
    for idx, t in enumerate(times):
        if t >= target:
            return idx
    return len(times) - 1


def bias_camera_position(
    frames: Sequence[ReplayFrame], camera_id: str, bias_xy: Sequence[float]
) -> list[ReplayFrame]:
    """Add a constant world-frame offset to one camera's observations (E6 drift).

    Models a lost-calibration camera whose projected positions are systematically
    displaced. Images are untouched — this is controlled calibration-ablation
    evidence, not a physical move.
    """

    bx, by = float(bias_xy[0]), float(bias_xy[1])
    out = []
    for f in frames:
        obs = tuple(
            replace(o, xy_m=(o.xy_m[0] + bx, o.xy_m[1] + by)) if o.camera_id == camera_id else o
            for o in f.observations
        )
        out.append(replace(f, observations=obs))
    return out


# --------------------------------------------------------------------------- #
# E4 — camera-subset sweep.
# --------------------------------------------------------------------------- #
def all_nonempty_subsets(camera_ids: Sequence[str]) -> list[tuple[str, ...]]:
    ids = sorted(set(camera_ids))
    subsets: list[tuple[str, ...]] = []
    for r in range(1, len(ids) + 1):
        subsets.extend(combinations(ids, r))
    return subsets


@dataclass(frozen=True)
class SubsetSweepResult:
    per_subset: dict[tuple[str, ...], ReplayMetrics]
    full_set: tuple[str, ...]
    best_single_camera: tuple[str, ...]
    best_single_p95: float
    full_set_p95: float
    fusion_gain_p95: float  # best single-camera p95 − full-set p95 (positive ⇒ fusion helps)


def run_camera_subset_sweep(
    frames: Sequence[ReplayFrame],
    evaluation_frames: Sequence[EvaluationFrame],
    *,
    camera_ids: Sequence[str] | None = None,
    config: ReplayConfig | None = None,
    subsets: Sequence[Sequence[str]] | None = None,
) -> SubsetSweepResult:
    """Run every non-empty camera subset through the same fusion mode (E4)."""

    if not evaluation_frames:
        raise ValueError("subset sweep needs evaluation frames to score error")
    cfg = config or default_fusion_config()
    ids = tuple(camera_ids) if camera_ids is not None else _camera_ids(frames)
    if not ids:
        raise ValueError("no cameras present in frames")
    wanted = [tuple(sorted(s)) for s in subsets] if subsets is not None else all_nonempty_subsets(ids)

    per_subset: dict[tuple[str, ...], ReplayMetrics] = {}
    for subset in wanted:
        masked = filter_frames_to_subset(frames, subset)
        result = run_replay(masked, cfg, evaluation_frames=evaluation_frames)
        if result.metrics is not None:
            per_subset[subset] = result.metrics

    singles = {s: m for s, m in per_subset.items() if len(s) == 1 and math.isfinite(m.p95_error_m)}
    if not singles:
        raise ValueError("no single-camera subset produced finite metrics")
    best_single = min(singles, key=lambda s: singles[s].p95_error_m)
    full = tuple(sorted(ids))
    full_p95 = per_subset[full].p95_error_m if full in per_subset else math.nan
    best_single_p95 = singles[best_single].p95_error_m
    gain = (best_single_p95 - full_p95) if math.isfinite(full_p95) else math.nan
    return SubsetSweepResult(
        per_subset=per_subset,
        full_set=full,
        best_single_camera=best_single,
        best_single_p95=best_single_p95,
        full_set_p95=full_p95,
        fusion_gain_p95=gain,
    )


# --------------------------------------------------------------------------- #
# E5 / E6 — severity sweeps (dropout, latency, drift).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SeveritySweepResult:
    axis: str
    severities: tuple[float, ...]
    metrics: tuple[ReplayMetrics, ...]
    p95_by_severity: tuple[float, ...]
    max_by_severity: tuple[float, ...]
    error_severity_auc_p95: float | None  # None when any level lacks finite p95


def _severity_sweep(
    axis: str,
    severities: Sequence[float],
    frames_for_level,
    evaluation_frames: Sequence[EvaluationFrame],
    config: ReplayConfig,
) -> SeveritySweepResult:
    if len(severities) < 1:
        raise ValueError("need at least one severity level")
    metrics: list[ReplayMetrics] = []
    p95s: list[float] = []
    maxes: list[float] = []
    for level in severities:
        result = run_replay(frames_for_level(level), config, evaluation_frames=evaluation_frames)
        if result.metrics is None:
            raise ValueError(f"level {level} produced no metrics (missing evaluation overlap?)")
        metrics.append(result.metrics)
        p95s.append(result.metrics.p95_error_m)
        maxes.append(result.metrics.max_error_m)
    auc: float | None = None
    if len(severities) >= 2 and all(math.isfinite(v) for v in p95s):
        auc = error_severity_auc(list(severities), p95s)
    return SeveritySweepResult(
        axis=axis,
        severities=tuple(float(s) for s in severities),
        metrics=tuple(metrics),
        p95_by_severity=tuple(p95s),
        max_by_severity=tuple(maxes),
        error_severity_auc_p95=auc,
    )


def run_dropout_sweep(
    frames: Sequence[ReplayFrame],
    evaluation_frames: Sequence[EvaluationFrame],
    camera_id: str,
    p_levels: Sequence[float],
    *,
    config: ReplayConfig | None = None,
    seed: int = 0,
) -> SeveritySweepResult:
    """Intermittent dropout of one camera at each probability level (E5)."""

    cfg = config or default_fusion_config()
    return _severity_sweep(
        "p_drop",
        p_levels,
        lambda p: drop_camera_intermittent(frames, camera_id, p, seed=seed),
        evaluation_frames,
        cfg,
    )


def run_latency_sweep(
    frames: Sequence[ReplayFrame],
    evaluation_frames: Sequence[EvaluationFrame],
    camera_id: str,
    delays_s: Sequence[float],
    *,
    config: ReplayConfig | None = None,
) -> SeveritySweepResult:
    """Delayed association of one camera at each delay level (E5)."""

    cfg = config or default_fusion_config()
    return _severity_sweep(
        "delay_s",
        delays_s,
        lambda d: delay_camera(frames, camera_id, d),
        evaluation_frames,
        cfg,
    )


def run_calibration_drift_sweep(
    frames: Sequence[ReplayFrame],
    evaluation_frames: Sequence[EvaluationFrame],
    camera_id: str,
    bias_levels_m: Sequence[float],
    *,
    direction: Sequence[float] = (1.0, 0.0),
    config: ReplayConfig | None = None,
) -> SeveritySweepResult:
    """Constant world-position bias of one camera at each magnitude level (E6).

    ``bias_levels_m`` are severity magnitudes applied along the unit ``direction``.
    """

    cfg = config or default_fusion_config()
    norm = math.hypot(float(direction[0]), float(direction[1]))
    if norm == 0.0:
        raise ValueError("direction must be non-zero")
    ux, uy = direction[0] / norm, direction[1] / norm
    return _severity_sweep(
        "bias_m",
        bias_levels_m,
        lambda b: bias_camera_position(frames, camera_id, (b * ux, b * uy)),
        evaluation_frames,
        cfg,
    )
