"""Score every estimate against the truth at the instant that estimate describes.

One loader, used by `score.py`, `per_camera_error.py` and `repeating_error.py`, because
the two mistakes it exists to prevent were each made independently in more than one of
them.

**Mistake 1: pairing a timestamped estimate with a later truth.** Every publisher here
stamps what it produces, and the logger samples at 10 Hz. The belief publishes at 10 Hz
too, so the row that records it is written one full cycle after the instant it describes:
measured 0.1000 s median. At 0.22 m/s that is 2.2 cm of robot travel added to a real
median error of 1.1 cm. Recomputed on the six-arm drives, aligning the truth moved the
median belief error from 2.75 cm to 1.13 cm and mean NEES from 6.78 to 3.54 -- and it
moved the raw-box arms by under 1 cm, because 24 cm of error swamps 2 cm of lag. So the
misalignment flattered nothing and discredited nothing; it just made the good arms look
2.3x worse than they are and the whole network look overconfident.

**Mistake 2: counting each reading about four times.** The camera manager decides at
20 Hz and the detector produces 5 Hz, so one physical detection is republished on about
four consecutive decisions and written to `fusion_observations.csv` four times. Measured
25656 rows for 6418 distinct readings across five drives. Those rows are not even
identical -- the hull correction is re-applied against a newer belief each tick -- so
they are a mixture of one reading's successive re-estimates, weighted by how long the
manager kept re-fusing it.

Both are fixed here rather than in each caller, and both work on runs written before the
logger recorded the fields (`logging_schema_version` 1) by re-deriving alignment from the
10 Hz truth series. Where the logger's own aligned columns exist they are preferred: they
were computed against the full-rate truth buffer instead of an interpolated 10 Hz one.

Ground truth is used to score and for nothing else.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


class TruthSeries:
    """The run's ground-truth path, on ground truth's own clock."""

    def __init__(self, t, x, y, yaw, source: str):
        order = np.argsort(t)
        self.t = np.asarray(t, dtype=float)[order]
        self.x = np.asarray(x, dtype=float)[order]
        self.y = np.asarray(y, dtype=float)[order]
        self.yaw = np.unwrap(np.asarray(yaw, dtype=float)[order]) if len(yaw) else np.array([])
        self.source = source

    def __len__(self):
        return int(self.t.size)

    def at(self, stamps):
        """Truth at `stamps`. NaN outside the recorded interval -- never clamped.

        Clamping to an endpoint is how a misaligned comparison hides: it returns a real
        pose from the wrong instant and every downstream statistic accepts it.
        """
        s = np.asarray(stamps, dtype=float)
        gx = np.interp(s, self.t, self.x)
        gy = np.interp(s, self.t, self.y)
        outside = (s < self.t[0]) | (s > self.t[-1]) | ~np.isfinite(s)
        gx = np.where(outside, np.nan, gx)
        gy = np.where(outside, np.nan, gy)
        return gx, gy

    def yaw_at(self, stamps):
        s = np.asarray(stamps, dtype=float)
        gyaw = np.interp(s, self.t, self.yaw)
        outside = (s < self.t[0]) | (s > self.t[-1]) | ~np.isfinite(s)
        return np.where(outside, np.nan, gyaw)


def schema_version(run: Path) -> int:
    """Which logging conventions this run was written under."""

    manifest = Path(run) / "run_manifest.json"
    if not manifest.is_file():
        return 1
    try:
        return int(json.loads(manifest.read_text()).get("logging_schema_version", 1))
    except (ValueError, TypeError, OSError):
        return 1


def _float(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def rows(run: Path) -> list[dict]:
    return list(csv.DictReader(open(Path(run) / "experiment.csv")))


def truth_series(run: Path, table: list[dict] | None = None) -> TruthSeries:
    """The truth path, timestamped as well as the run allows.

    Schema 2 logs `gt_stamp`, the stamp the pose itself carried, so the series sits on
    ground truth's own clock. Schema 1 has only the log clock, which is late by however
    long the held value had been sitting there -- bounded by the 10 Hz log tick.
    """
    table = rows(run) if table is None else table
    t, x, y, yaw = [], [], [], []
    use_gt_stamp = schema_version(run) >= 2 and table and "gt_stamp" in table[0]
    for row in table:
        if _float(row, "gt_available") != 1.0:
            continue
        stamp = _float(row, "gt_stamp") if use_gt_stamp else _float(row, "stamp")
        gx, gy = _float(row, "gt_x"), _float(row, "gt_y")
        if not (math.isfinite(stamp) and math.isfinite(gx) and math.isfinite(gy)):
            continue
        t.append(stamp)
        x.append(gx)
        y.append(gy)
        yaw.append(_float(row, "gt_yaw"))
    if not t:
        raise SystemExit(f"{run}: no usable ground truth")
    return TruthSeries(t, x, y, yaw,
                       "gt_stamp" if use_gt_stamp else "log_stamp (schema 1)")


def aligned_error_cm(run: Path, kind: str, table: list[dict] | None = None) -> dict:
    """Error of one estimate against the truth at the estimate's OWN stamp.

    `kind` is 'belief' (the planner's belief) or 'state' (the correction the filter is
    holding). Returns centimetres, plus the same quantity scored at log time so the size
    of the old artefact stays visible instead of being asserted.
    """
    if kind not in {"belief", "state"}:
        raise ValueError("kind must be 'belief' or 'state'")
    table = rows(run) if table is None else table
    truth = truth_series(run, table)
    xkey, ykey, skey = (
        ("planner_belief_x", "planner_belief_y", "planner_belief_stamp")
        if kind == "belief" else ("state_x", "state_y", "state_stamp"))

    est_x = np.array([_float(r, xkey) for r in table])
    est_y = np.array([_float(r, ykey) for r in table])
    est_s = np.array([_float(r, skey) for r in table])
    log_s = np.array([_float(r, "stamp") for r in table])

    have = np.isfinite(est_x) & np.isfinite(est_y) & np.isfinite(est_s)
    gx_a, gy_a = truth.at(est_s)
    gx_l, gy_l = truth.at(log_s)
    aligned = np.where(have, np.hypot(est_x - gx_a, est_y - gy_a), np.nan) * 100.0
    logtime = np.where(have, np.hypot(est_x - gx_l, est_y - gy_l), np.nan) * 100.0
    lag = np.where(have, log_s - est_s, np.nan)
    return dict(aligned_cm=aligned, logtime_cm=logtime, stamp=est_s, log_stamp=log_s,
                x=est_x, y=est_y, gt_x=gx_a, gt_y=gy_a, have=have, lag_s=lag,
                truth_source=truth.source)


def landed_mask(stamps) -> np.ndarray:
    """True on the first row that carries each distinct message stamp.

    A held message is re-logged every tick, and re-scoring it against a robot that has
    moved on measures the robot's travel, not the sensor.
    """
    s = np.asarray(stamps, dtype=float)
    if s.size == 0:
        return np.zeros(0, dtype=bool)
    first = np.concatenate([[True], np.diff(s) > 1.0e-9])
    return first & np.isfinite(s)


def corrections(run: Path, table: list[dict] | None = None) -> dict:
    """When a fresh correction actually landed, and how long the gaps were.

    `n_state_publications` is what the old `corrections` field counted: log rows whose
    `state_stamp` changed. The manager publishes at 20 Hz and the logger samples at
    10 Hz, so that number is neither the corrections published nor the camera readings
    behind them -- it is the fraction of the log that had a fresh correction, times the
    duration. `n_detector_rounds` is the honest count where the run logs capture times.
    """
    table = rows(run) if table is None else table
    state_s = np.array([_float(r, "state_stamp") for r in table])
    available = np.array([_float(r, "state_available") == 1.0 for r in table])
    unique = np.array(sorted({float(s) for s, ok in zip(state_s, available)
                              if ok and math.isfinite(s)}))
    gaps = np.diff(unique) if unique.size > 1 else np.array([])
    log_s = np.array([_float(r, "stamp") for r in table])
    finite_log = log_s[np.isfinite(log_s)]
    duration_s = float(finite_log[-1] - finite_log[0]) if finite_log.size > 1 else math.nan

    n_detector_rounds = None
    obs = observations(run)
    if obs:
        source_batches = {o["source_batch_id"] for o in obs if o["source_batch_id"]}
        if source_batches:
            n_detector_rounds = len(source_batches)
        else:
            # Legacy fallback: a tuple of the camera capture stamps identifies one
            # detector round more honestly than counting each camera as a round.
            by_decision = {}
            for item in obs:
                decision = round(item["decision_stamp"], 6)
                if math.isfinite(item["obs_stamp"]):
                    by_decision.setdefault(decision, set()).add(
                        round(item["obs_stamp"], 6))
            n_detector_rounds = len({
                tuple(sorted(stamps)) for stamps in by_decision.values() if stamps
            }) or None

    return dict(
        n_state_publications=int(unique.size),
        n_state_publications_note=(
            "log rows with a fresh /state/bev stamp, at the 10 Hz log rate -- an "
            "availability fraction, not a count of corrections"),
        n_detector_rounds=n_detector_rounds,
        duration_s=duration_s,
        state_fresh_rate_hz=(float(unique.size) / duration_s
                             if duration_s and math.isfinite(duration_s) else math.nan),
        longest_gap_s=float(gaps.max()) if gaps.size else math.nan,
        median_gap_s=float(np.median(gaps)) if gaps.size else math.nan,
    )


def observations(run: Path) -> list[dict]:
    """Raw `fusion_observations.csv`, one dict per row, floats parsed. No filtering."""

    path = Path(run) / "fusion_observations.csv"
    if not path.is_file():
        return []
    out = []
    for row in csv.DictReader(open(path)):
        entry = dict(
            camera=row.get("camera", ""),
            source_batch_id=row.get("source_batch_id", ""),
            used=row.get("used") == "1",
            decision_stamp=_float(row, "stamp"),
            common_capture_stamp=_float(row, "common_capture_stamp"),
            obs_stamp=_float(row, "obs_stamp"),
            obs_x=_float(row, "obs_x"),
            obs_y=_float(row, "obs_y"),
            fused_x=_float(row, "fused_x"),
            fused_y=_float(row, "fused_y"),
            fused_stamp=_float(row, "fused_stamp"),
            n_candidates=_float(row, "n_candidates"),
            n_used=_float(row, "n_used"),
            conf=_float(row, "conf"),
            bbox_h_px=_float(row, "bbox_h_px"),
            bbox_w_px=_float(row, "bbox_w_px"),
            range_m=_float(row, "range_m"),
            obs_repeat=_float(row, "obs_repeat"),
        )
        entry["cov"] = np.array([
            [_float(row, "obs_cov_xx"), _float(row, "obs_cov_xy")],
            [_float(row, "obs_cov_xy"), _float(row, "obs_cov_yy")]])
        entry["fused_cov"] = np.array([
            [_float(row, "fused_cov_xx"), _float(row, "fused_cov_xy")],
            [_float(row, "fused_cov_xy"), _float(row, "fused_cov_yy")]])
        entry["aligned_xy"] = np.array([
            _float(row, "aligned_x"), _float(row, "aligned_y")])
        entry["aligned_cov"] = np.array([
            [_float(row, "aligned_cov_xx"), _float(row, "aligned_cov_xy")],
            [_float(row, "aligned_cov_xy"), _float(row, "aligned_cov_yy")]])
        out.append(entry)
    return out


def readings(run: Path, *, admitted_only: bool = True, dedupe: bool = True,
             require_capture_time: bool = True) -> list[dict]:
    """One entry per camera reading, scored against the truth when the camera saw it.

    `dedupe` keeps the FIRST row for each (camera, capture time): the reading as the
    manager first computed it. Without it every statistic below counts each reading
    about four times and weights cameras by how long the manager re-fused them.

    `admitted_only` restricts to readings the arm's rule used. That is what the runtime
    consumed, but it conditions on passing the disagreement gate -- on agreeing with the
    other cameras -- so it is not a clean per-camera error. Pass False for the
    unconditional distribution; callers that care should report both.
    """
    obs = observations(run)
    if not obs:
        return []
    have_capture = any(math.isfinite(o["obs_stamp"]) for o in obs)
    if require_capture_time and not have_capture:
        return []
    truth = truth_series(run)

    seen = set()
    out = []
    for o in obs:
        if admitted_only and not o["used"]:
            continue
        if not (math.isfinite(o["obs_x"]) and math.isfinite(o["obs_y"])):
            continue
        cap = o["obs_stamp"]
        if not math.isfinite(cap):
            continue
        if dedupe:
            key = (o["camera"], round(cap, 6))
            if key in seen:
                continue
            seen.add(key)
        gx, gy = truth.at([cap])
        if not (math.isfinite(gx[0]) and math.isfinite(gy[0])):
            continue
        entry = dict(o)
        entry["truth"] = np.array([float(gx[0]), float(gy[0])])
        entry["truth_yaw"] = float(truth.yaw_at([cap])[0])
        entry["error"] = np.array([o["obs_x"], o["obs_y"]]) - entry["truth"]
        entry["error_cm"] = float(np.linalg.norm(entry["error"]) * 100.0)
        out.append(entry)
    out.sort(key=lambda r: r["obs_stamp"])
    return out


def fused_answers(run: Path, *, dedupe: bool = True) -> list[dict]:
    """The fused correction per decision, scored at the instant IT describes.

    The manager propagates the fused answer to `now` and re-stamps it, while leaving
    each camera's reading at capture time. Scoring both against one truth -- as the
    fusion-quality check used to -- charges every individual camera ~200 ms of robot
    travel and charges the fused answer none, which is a comparison the fusion rule wins
    by convention.
    """
    obs = observations(run)
    if not obs:
        return []
    truth = truth_series(run)
    by_decision: dict[object, list[dict]] = {}
    for o in obs:
        key = o["source_batch_id"] or round(o["decision_stamp"], 6)
        by_decision.setdefault(key, []).append(o)

    out = []
    seen_rounds = set()
    for round_key, group in by_decision.items():
        head = group[0]
        caps = tuple(sorted({round(o["obs_stamp"], 6) for o in group
                             if math.isfinite(o["obs_stamp"])}))
        identity = head["source_batch_id"] or caps
        if dedupe and identity:
            if identity in seen_rounds:
                continue
            seen_rounds.add(identity)
        # The fused answer describes `fused_stamp` where the run records it, and the
        # newest capture time where it does not (schema 1 predates the field).
        when = head["fused_stamp"]
        if not math.isfinite(when):
            when = max(caps) if caps else head["decision_stamp"]
        gx, gy = truth.at([when])
        if not (math.isfinite(head["fused_x"]) and math.isfinite(gx[0])):
            continue
        fused_truth = np.array([float(gx[0]), float(gy[0])])
        cameras = {}
        for o in group:
            cgx, cgy = truth.at([o["obs_stamp"]])
            if not math.isfinite(cgx[0]):
                continue
            cameras[o["camera"]] = dict(
                error_cm=float(np.linalg.norm(
                    np.array([o["obs_x"], o["obs_y"]])
                    - np.array([float(cgx[0]), float(cgy[0])])) * 100.0),
                used=o["used"])
        out.append(dict(
            source_batch_id=head["source_batch_id"],
            decision_stamp=head["decision_stamp"], fused_stamp=when,
            fused_xy=np.array([head["fused_x"], head["fused_y"]]),
            fused_cov=head["fused_cov"], truth=fused_truth,
            error_cm=float(np.linalg.norm(
                np.array([head["fused_x"], head["fused_y"]]) - fused_truth) * 100.0),
            n_candidates=head["n_candidates"], n_used=head["n_used"],
            cameras=cameras))
    return out


def belief_at_fusion_events(run: Path, table: list[dict] | None = None) -> list[dict]:
    """Belief immediately after each unique detector batch correction.

    This gives correction-count and belief panels the same event weighting. A 10 Hz logger
    must never turn a 5 Hz correction into two experimental observations.
    """

    table = rows(run) if table is None else table
    belief = aligned_error_cm(run, "belief", table)
    cov = np.array([[[_float(r, "planner_cov_x"), _float(r, "planner_cov_xy")],
                     [_float(r, "planner_cov_xy"), _float(r, "planner_cov_y")]]
                    for r in table])
    valid_indices = np.flatnonzero(belief["have"])
    if not valid_indices.size:
        return []
    stamps = belief["stamp"][valid_indices]
    out = []
    for event in fused_answers(run):
        target = float(event["fused_stamp"])
        after = np.flatnonzero(stamps > target + 1.0e-9)
        if not after.size:
            continue
        idx = int(valid_indices[int(after[0])])
        lag = float(belief["stamp"][idx] - target)
        if not math.isfinite(lag) or lag > 0.30:
            continue
        covariance = cov[idx]
        if not np.isfinite(covariance).all():
            continue
        out.append({
            "source_batch_id": event["source_batch_id"],
            "n_candidates": event["n_candidates"],
            "error_cm": float(belief["aligned_cm"][idx]),
            "stated_sigma_cm": float(
                np.sqrt(np.trace(covariance) / 2.0) * 100.0),
            "belief_lag_after_fusion_s": lag,
        })
    return out


def nees(residuals, covariances) -> np.ndarray:
    """Normalised squared error, in closed form, for 2x2 covariances."""

    r = np.asarray(residuals, dtype=float).reshape(-1, 2)
    c = np.asarray(covariances, dtype=float).reshape(-1, 2, 2)
    a, b, d = c[:, 0, 0], c[:, 0, 1], c[:, 1, 1]
    det = a * d - b * b
    ok = det > 0.0
    out = np.full(r.shape[0], np.nan)
    out[ok] = ((d[ok] * r[ok, 0] ** 2
                - 2.0 * b[ok] * r[ok, 0] * r[ok, 1]
                + a[ok] * r[ok, 1] ** 2) / det[ok])
    return out


#: Mean of a 2-D chi-square: the target for a MEAN NEES.
NEES_MEAN_TARGET = 2.0
#: Median of a 2-D chi-square: the target for a MEDIAN NEES. Not 2.0 -- comparing a
#: median NEES against 2.0 understates the miscalibration by 44%.
NEES_MEDIAN_TARGET = 2.0 * math.log(2.0)
#: 95th percentile of a 2-D chi-square, for ellipse coverage.
CHI2_95_2D = 5.991
