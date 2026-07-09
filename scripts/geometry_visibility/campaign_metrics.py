#!/usr/bin/env python3
"""Canonical, self-checking metrics loader for visibility-comparison campaign logs.

WHY THIS EXISTS
---------------
Each per-run CSV has ~216 columns with SIX overlapping position fields. Picking the
wrong one silently yields garbage — e.g. `state_x/state_y` is STALE (updates rarely,
frozen for long spans) so ||state - gt|| reaches 4.5 m while the true belief error is
<0.35 m. Using it as "belief" produced a fake heavy-tailed error and a bogus
"spreading helps" result. This module exposes ONLY verified-canonical fields and
asserts them on load, so analysis code cannot grab a stale column.

CANONICAL FIELDS (use these):
  belief          = planner_belief_x / planner_belief_y   (== est_x/est_y)
  truth           = gt_x / gt_y                            (GT-bridge ground truth)
  belief_error_m  = belief_error_gt_m                      (== ||belief - truth||, verified)
  reported_sigma_m= state_sigma_major_m                    (reported 1σ major axis;
                    NOTE: OVERCONFIDENT — median ~0.017 m vs actual error median ~0.05 m)
  detection       = perception.csv: detected {0,1}, yolo_score_raw

DO NOT USE as belief/truth:
  state_x / state_y (experiment or perception) -> STALE (frozen); ||state-gt|| up to 4.5 m
  truth_x / truth_y                            -> wheel-odom truth (pre-GT-bridge); use gt_*

See docs/campaign_log_metrics.md. Every loader runs assert_canonical() which fails
loudly if planner_belief stops reproducing the belief_error_gt_m column.
"""
from __future__ import annotations
import csv, glob, pathlib
import numpy as np

DEPRECATED_STALE = ("state_x", "state_y", "truth_x", "truth_y")  # never use as belief/truth


def _f(row, key):
    try:
        return float(row.get(key, ""))
    except Exception:
        return np.nan


def _arr(rows, key):
    return np.array([_f(r, key) for r in rows])


def load_run(experiment_csv: str) -> dict:
    """Load one run's per-timestep canonical metrics from an experiment.csv path."""
    rows = list(csv.DictReader(open(experiment_csv)))
    out = {
        "stamp": _arr(rows, "stamp"),
        "belief_x": _arr(rows, "planner_belief_x"), "belief_y": _arr(rows, "planner_belief_y"),
        "truth_x": _arr(rows, "gt_x"), "truth_y": _arr(rows, "gt_y"),
        "belief_error_m": _arr(rows, "belief_error_gt_m"),
        "reported_sigma_m": _arr(rows, "state_sigma_major_m"),
    }
    assert_canonical(out, source=experiment_csv)
    return out


def assert_canonical(run: dict, source: str = "", tol: float = 1e-3) -> None:
    """Fail loudly if the canonical belief no longer reproduces the GT-error column."""
    bx, by, gx, gy = run["belief_x"], run["belief_y"], run["truth_x"], run["truth_y"]
    col = run["belief_error_m"]
    m = np.isfinite(bx) & np.isfinite(gx) & np.isfinite(col)
    if m.sum() == 0:
        return
    recomputed = np.hypot(bx[m] - gx[m], by[m] - gy[m])
    bad = np.max(np.abs(recomputed - col[m]))
    if bad > tol:
        raise AssertionError(
            f"CANONICAL CHECK FAILED for {source}: ||planner_belief - gt|| disagrees with "
            f"belief_error_gt_m by {bad:.3f} m (>tol {tol}). Do not trust the belief field — "
            f"check for a stale/renamed column before computing metrics."
        )


def load_detections(campaign_dir: str, stamp_tol_s: float = 0.3) -> list[dict]:
    """Per-detection events across a campaign: canonical belief+truth from experiment.csv
    (matched to each perception detection by stamp), plus detected/yolo_score."""
    events = []
    for pf in sorted(glob.glob(str(pathlib.Path(campaign_dir) / "*/*/*/*/perception.csv"))):
        run = load_run(str(pathlib.Path(pf).parent / "experiment.csv"))
        est = run["stamp"]
        for r in csv.DictReader(open(pf)):
            if r.get("detected") not in ("0", "1"):
                continue
            t = _f(r, "log_stamp")
            if not np.isfinite(t) or len(est) == 0:
                continue
            j = int(np.argmin(np.abs(est - t)))
            if abs(est[j] - t) > stamp_tol_s or not np.isfinite(run["belief_x"][j]) or not np.isfinite(run["truth_x"][j]):
                continue
            events.append(dict(
                belief=(run["belief_x"][j], run["belief_y"][j]),
                truth=(run["truth_x"][j], run["truth_y"][j]),
                belief_error_m=float(run["belief_error_m"][j]),
                reported_sigma_m=float(run["reported_sigma_m"][j]) if np.isfinite(run["reported_sigma_m"][j]) else np.nan,
                detected=int(r["detected"]),
                yolo_score=_f(r, "yolo_score_raw"),
            ))
    return events


if __name__ == "__main__":  # smoke test / self-check on the honest campaign
    camp = pathlib.Path(__file__).resolve().parents[2] / "logs/visibility_comparison/honest_campaign_v1"
    ev = load_detections(str(camp))
    err = np.array([e["belief_error_m"] for e in ev])
    print(f"canonical check PASSED on {len(glob.glob(str(camp/'*/*/*/*/experiment.csv')))} runs")
    print(f"{len(ev)} detections; belief-vs-GT error p50 {np.percentile(err,50):.3f} "
          f"p95 {np.percentile(err,95):.3f} max {err.max():.3f} m")
