#!/usr/bin/env python3
"""Score ONE arm's drive: what it did, what it claimed, and where it broke.

    python3 experiments/fusion_on_fixed_routes/score.py [ARM ...]

Reads the drive under logs/studies/fusion_on_fixed_routes/drives/... and writes
numbers.json into that arm's own storyline folder. One arm at a time, on purpose: the arms
are not pooled and nothing here compares them.

Ground truth is used to form errors and to score. It is never an input to the filter, the
manager or the planner.

**Every error here is scored against the truth at the instant the estimate describes**,
via `aligned.py`. Scoring against the truth at LOG time -- what this script used to do --
added a fixed 0.1 s of robot travel to the belief error and 0.05 s to the correction
error, which inflated the median belief error 2.3x and the mean NEES 1.9x on the four
fusion arms while barely touching the raw-box arms. Both numbers are reported so the
artefact stays visible in the output rather than in a memo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import aligned as A  # noqa: E402

#: Which campaign to score. Coverage alone is not a calibration metric -- a wide enough
#: ellipse contains the truth every time -- so every run is also scored by NEES, whose
#: target is 2.0 for the MEAN of a 2-D belief and 1.386 for the median.
DRIVES_ROOT = REPO / "logs/studies/fusion_on_fixed_routes" / os.environ.get(
    "FUSION_DRIVES", "drives")
FROZEN_RUNS = Path(os.environ.get(
    "FUSION_RUN_MANIFEST",
    str(REPO / "logs/studies/fusion_on_fixed_routes/frozen_runs.json"),
)).expanduser()
STORY_ROOT = REPO / "logs/studies/fusion_on_fixed_routes"
#: every route the arms drive, in the order they are reported
TASKS = ("fusion_network_traverse", "fusion_overlap_rich", "fusion_overlap_sparse",
         "fusion_long_traverse")
#: arm id -> its own folder, so each method's numbers live with its own figures
FOLDER = {
    "F1": "01_best_single_camera", "F2": "02_distance_angle",
    "F3": "03_independent_fusion", "F4": "04_joint_network_estimator",
    "O1": "05_raw_box", "O2": "06_fixed_offset",
}


def _fusion_quality(run: Path, *, max_reference_gap_s=None) -> dict:
    """Did combining the cameras beat the best one that was on the table?

    Scoring only -- the rule cannot see which camera is closest to the truth. But if the
    combined answer is regularly worse than a camera the rule already had, that is the
    fusion rule losing information rather than adding it, and it is invisible in any
    summary error.

    The fused answer and each camera are scored at their OWN instants, which are not the
    same instant: the manager propagates the fused correction forward and re-stamps it
    while the per-camera readings stay at capture time. Scoring both against one truth
    handed fusion a ~200 ms head start on every camera it was being compared with.
    """

    start, stop = A.mission_interval(run)
    rounds = [r for r in A.fused_answers(run, max_reference_gap_s=max_reference_gap_s)
              if start <= r["fused_stamp"] <= stop]
    if not rounds:
        return {"logged": False,
                "note": "this drive predates per-camera observation logging, or logs no "
                        "capture time, so no reading can be scored at its own instant"}
    worse, chose_best, total = 0, 0, 0
    fused_cm, best_cm = [], []
    for entry in rounds:
        cameras = entry["cameras"]
        if not cameras:
            continue
        total += 1
        closest = min(cameras, key=lambda c: cameras[c]["error_cm"])
        fused_cm.append(entry["error_cm"])
        best_cm.append(cameras[closest]["error_cm"])
        if entry["error_cm"] > cameras[closest]["error_cm"] * 1.5 + 1.0:
            worse += 1
        if cameras[closest]["used"]:
            chose_best += 1
    if not total:
        return {"logged": False, "note": "no usable rows"}
    return {
        "logged": True,
        "rounds": total,
        "rounds_note": "one entry per detector round, not per manager decision -- the "
                       "manager republishes each round about four times",
        "worse_than_best_available_camera": round(worse / total, 3),
        "used_the_closest_camera": round(chose_best / total, 3),
        "used_the_closest_camera_note": (
            "informative only for a rule that picks ONE camera; a rule that uses every "
            "available camera scores 1.0 by construction"),
        "median_fused_error_cm": round(float(np.median(fused_cm)), 2),
        "median_best_available_camera_error_cm": round(float(np.median(best_cm)), 2),
        "scoring": "fused answer at its own fused_stamp, each camera at its own capture "
                   "time",
    }


def story_dir(task: str, arm: str) -> Path:
    """Each route gets its own tree, so one route's figures can never be read as another's."""

    return STORY_ROOT / task / FOLDER[arm]


def _selected_runs(arm: str, task: str = TASKS[0]) -> list[Path]:
    """Return explicitly frozen evidence runs; directory recency is never provenance."""

    try:
        frozen = json.loads(FROZEN_RUNS.read_text(encoding="utf-8"))
        if int(frozen.get("schema_version", 0)) != 1:
            raise ValueError("schema_version must be 1")
        entry = frozen["runs"][task][arm]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"{FROZEN_RUNS}: missing explicit frozen run selection for {task}/{arm}: {exc}. "
            "Do not substitute the newest-looking directory."
        )
    values = entry.get("analysis", []) if isinstance(entry, dict) else entry
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list) or not values:
        raise SystemExit(f"{FROZEN_RUNS}: {task}/{arm} analysis must be a non-empty list")
    expected = {
        "F1": ("best_single", "hull"), "F2": ("distance_angle", "hull"),
        "F3": ("independent", "hull"), "F4": ("joint_network", "hull"),
        "O1": ("joint_network", "raw_box"),
        "O2": ("joint_network", "fixed_offset"),
    }[arm]
    selected, seeds, provenance = [], set(), set()
    required=("run_manifest.json","run_summary.json","experiment.csv","fusion_observations.csv","correction_assimilations.csv")
    for value in values:
        record=value if isinstance(value,dict) else {"run":value,"files":frozen.get("artifacts",{}).get(str(value),{})}
        run = Path(str(record['run'])).expanduser()
        if not run.is_absolute():
            run = REPO / run
        manifest_path, summary_path = run / "run_manifest.json", run / "run_summary.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid frozen run {run}: {exc}")
        if int(manifest.get("logging_schema_version", 0) or 0) < 4:
            raise SystemExit(
                f"{run}: schema 4 or newer is required for source-batch assimilation"
            )
        try:
            A.verify_frozen_entry(dict(record,run=str(run),task=task,seed=manifest.get('seed')),required,repo=REPO)
        except (ValueError,OSError) as exc:
            raise SystemExit(f'{run}: invalid frozen artifact selection: {exc}') from exc
        if manifest.get("task") != task:
            raise SystemExit(f"{run}: task identity mismatch")
        actual = (manifest.get("manager_fusion_rule"),
                  manifest.get("manager_observation_model"))
        if actual != expected:
            raise SystemExit(f"{run}: arm identity {actual!r} does not match {arm} {expected!r}")
        if manifest.get("goal_termination_reference") != "planner_belief":
            raise SystemExit(f"{run}: ground-truth-independent termination is required")
        if not summary.get("completed") or not summary.get("valid_run", False):
            raise SystemExit(f"{run}: frozen evidence must be completed and valid")
        try:
            A.validate_run_ledger(run)
        except ValueError as exc:
            raise SystemExit(f"{run}: {exc}") from exc
        seed = int(manifest.get("seed", -1))
        if seed in seeds:
            raise SystemExit(f"{run}: duplicate seed {seed} in {task}/{arm}")
        seeds.add(seed)
        provenance.add(comparison_identity(manifest))
        selected.append(run.resolve())
    # The manifest declares which seeds it froze, so a deliberately small set is
    # allowed and an accidentally incomplete one is still caught. Hardcoding 0..4 made
    # every design except five-seed unscoreable, including a single-seed first look.
    declared = frozen.get("seeds")
    if declared is None:
        raise SystemExit(
            f"{FROZEN_RUNS}: declare the seed set as a top-level \"seeds\" list. "
            "An implicit seed count is how a partial campaign gets reported as a whole one."
        )
    expected_seeds = {int(s) for s in declared}
    if not isinstance(declared, list) or not expected_seeds or len(expected_seeds) != len(declared):
        raise SystemExit("frozen seed list must be nonempty and unique")
    if seeds != expected_seeds:
        raise SystemExit(
            f"{task}/{arm}: frozen seeds are {sorted(seeds)}; "
            f"the manifest declares {sorted(expected_seeds)}"
        )
    if len(expected_seeds) < 2:
        print(f"  NOTE {task}/{arm}: one seed. Time samples within a drive are correlated, "
              "so this shows the shape of a result and cannot support a comparison "
              "between arms.")
    if len(provenance) != 1:
        raise SystemExit(f"{task}/{arm}: selected runs do not share one source/artifact identity")
    return selected


def comparison_identity(manifest):
    """Common collection identity; treatment-specific fusion/model labels are separate."""
    required=("campaign_config_sha256", "git_sha", "git_diff_sha256", "git_untracked_content_sha256",
              "yolo_model_sha256", "visibility_geometry_sha256", "collision_geometry_sha256",
              "process_noise_xy", "process_noise_theta", "use_odom_for_predict", "odom_topic")
    absent=[k for k in required if k not in manifest or manifest[k] in (None, "")]
    if absent:raise SystemExit(f"missing common source/configuration identity: {absent}")
    return tuple(manifest[k] for k in required)


def showcase_run(arm: str, task: str = TASKS[0]) -> Path:
    """The one drive a storyline panel illustrates, named explicitly in the manifest.

    Never "the latest": it must be one of the frozen runs, and if the manifest does not
    name a showcase the first frozen run is used so the choice is still reproducible.
    """

    runs = _selected_runs(arm, task)
    frozen = json.loads(FROZEN_RUNS.read_text(encoding="utf-8"))
    entry = frozen["runs"][task][arm]
    showcase = entry.get("showcase") if isinstance(entry, dict) else None
    if showcase:
        chosen = Path(str(showcase)).expanduser()
        if not chosen.is_absolute():
            chosen = REPO / chosen
        chosen = chosen.resolve()
        if chosen not in runs:
            raise SystemExit(f"{task}/{arm}: showcase must also be in the analysis list")
        return chosen
    return runs[0]


def _f(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan
    return value


def _route_polyline(task: str = TASKS[0]):
    record = json.loads(
        (Path(__file__).resolve().parent / "routes" / f"{task}.json").read_text())
    return np.asarray(json.loads(record["polyline_canonical_json"]), dtype=float)


def _distance_to_polyline(points, poly):
    """Shortest distance from each point to the commanded route, in metres."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom[denom == 0.0] = 1.0e-12
    out = np.empty(len(points))
    for i, p in enumerate(points):
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / denom, 0.0, 1.0)
        proj = a + t[:, None] * ab
        out[i] = np.min(np.linalg.norm(p - proj, axis=1))
    return out


def _score_one(run: Path, arm: str, task: str = TASKS[0], *, max_reference_gap_s=None) -> dict:
    table = A.rows(run)
    summary = json.loads((run / "run_summary.json").read_text())
    A.validate_run_ledger(run)
    start, stop = A.mission_interval(run)
    truth = A.truth_series(run, table, max_reference_gap_s=max_reference_gap_s)

    # --- what it did: how far the belief actually was from the truth -----------
    belief = A.aligned_error_cm(run, "belief", table, max_reference_gap_s=max_reference_gap_s)
    population = (A.landed_mask(belief["stamp"]) & belief["have"] &
                  (belief["stamp"] >= start) & (belief["stamp"] <= stop))
    supported = population & np.isfinite(belief["aligned_cm"])
    error_cm = belief["aligned_cm"][supported]
    error_logtime_cm = belief["logtime_cm"][population & np.isfinite(belief["logtime_cm"])]

    # --- is it honest: does the truth fall inside the stated 95% ellipse? ------
    cov = np.array([[[_f(r, "planner_cov_x"), _f(r, "planner_cov_xy")],
                     [_f(r, "planner_cov_xy"), _f(r, "planner_cov_y")]] for r in table])
    resid = np.stack([belief["gt_x"] - belief["x"], belief["gt_y"] - belief["y"]], axis=1)
    usable = supported
    A.validate_covariances(cov[population])
    nees = A.nees(resid[usable], cov[usable])
    nees = nees[np.isfinite(nees)]
    # The stated 1-sigma is reported as a MEDIAN. Its mean is meaningless here: during a
    # correction outage the belief's stated sigma reaches metres, so the mean of a
    # 150 s drive is set by a few seconds of it. Reported beside a median error, a mean
    # sigma of 28 cm sat next to a median error of 2.75 cm and neither described the
    # same part of the run.
    stated_sigma_cm = np.sqrt(np.trace(cov[usable], axis1=1, axis2=2) / 2.0) * 100.0
    inside = nees <= A.CHI2_95_2D

    # --- the correction's own error, scored ONCE per correction -----------------
    # state_error scores whatever correction the filter is holding, so during an outage
    # it re-scores an ageing message against a moving robot and reports the robot's own
    # travel as measurement error.
    fused = [r for r in A.fused_answers(run, max_reference_gap_s=max_reference_gap_s)
             if start <= r["fused_stamp"] <= stop]
    correction_cm = np.array([r["error_cm"] for r in fused])
    correction_logtime_cm = np.array([])  # No later logger row substitutes for a fused event.
    corrections = A.corrections(run, table)
    accounting = A.correction_accounting(run)

    # --- did it stay on the commanded route? ----------------------------------
    mission_truth = (truth.t >= start) & (truth.t <= stop)
    path = np.stack([truth.x[mission_truth], truth.y[mission_truth]], axis=1)
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1) if len(path) > 1 else np.array([0.0])
    off_route = _distance_to_polyline(path, _route_polyline(task)) if len(path) else np.array([np.nan])

    def pct(values, q):
        return round(float(np.percentile(values, q)), 2) if len(values) else None

    return {
        "arm": arm,
        "task": task,
        "run": str(run.relative_to(REPO)),
        "logging_schema_version": A.schema_version(run),
        "truth_clock": belief["truth_source"],
        "reference_max_gap_s": max_reference_gap_s,
        "reference_method": belief["reference_method"],
        "accounting": accounting,
        "fusion": _fusion_quality(run, max_reference_gap_s=max_reference_gap_s),
        "completion": summary.get("completion_reason"),
        "duration_s": round(float(summary.get("elapsed_after_first_cmd_s", float("nan"))), 1),
        "belief_error_cm": {
            "median": pct(error_cm, 50), "p95": pct(error_cm, 95),
            "worst": round(float(error_cm.max()), 2) if error_cm.size else None,
            "n_samples": int(error_cm.size),
            "n_unique_mission_beliefs": int(population.sum()),
            "n_reference_unscoreable": int((population & ~supported).sum()),
            "median_scored_at_log_time": pct(error_logtime_cm, 50),
            "note": "scored against the truth at the belief's own stamp. The log-time "
                    "figure beside it is the old definition, late by one publish cycle "
                    f"(median {np.nanmedian(belief['lag_s']):.3f} s here).",
        },
        "honesty": {
            "truth_inside_stated_95pct_ellipse": (
                round(float(np.mean(inside)), 3) if inside.size else None),
            "median_stated_1sigma_cm": (
                round(float(np.median(stated_sigma_cm)), 2) if stated_sigma_cm.size else None),
            "p95_stated_1sigma_cm": (
                round(float(np.percentile(stated_sigma_cm, 95)), 2)
                if stated_sigma_cm.size else None),
            "n_samples": int(nees.size),
            "nees_mean": round(float(np.mean(nees)), 2) if nees.size else None,
            "nees_mean_target": A.NEES_MEAN_TARGET,
            "nees_median": round(float(np.median(nees)), 2) if nees.size else None,
            "nees_median_target": round(A.NEES_MEDIAN_TARGET, 3),
            "note": "coverage alone is not calibration -- a wide enough ellipse contains "
                    "the truth every time. NEES is the metric: compare the MEAN against "
                    "2.0 and the MEDIAN against 1.386, never the median against 2.0. "
                    "Read the error and the stated sigma beside both. The stated sigma "
                    "here is still dominated by the pi^2 heading variance the runtime "
                    "carries, so it is not yet a camera-network property.",
        },
        "correction_error_cm": {
            "median": pct(correction_cm, 50),
            "p95": pct(correction_cm, 95),
            "worst": round(float(correction_cm.max()), 2) if correction_cm.size else None,
            "n": int(correction_cm.size),
            "median_scored_at_log_time": pct(correction_logtime_cm, 50),
            "p95_scored_at_log_time": pct(correction_logtime_cm, 95),
            "note": "Unique published fused corrections, including refused corrections, scored at fused_stamp. "
                    "This population is separate from accepted updates and public beliefs.",
        },
        "corrections": {
            "detector_rounds": corrections["n_detector_rounds"],
            "state_publications_seen": corrections["n_state_publications"],
            "state_fresh_rate_hz": (round(corrections["state_fresh_rate_hz"], 2)
                                    if math.isfinite(corrections["state_fresh_rate_hz"])
                                    else None),
            "longest_gap_s": accounting["longest_correction_gap_s"],
            "median_gap_s": accounting["median_correction_gap_s"],
            "correction_dropped_fraction": accounting["correction_dropped_fraction"],
            "note": "detector_rounds is the count of distinct camera readings. "
                    "state_publications_seen is log rows with a fresh correction at the "
                    "10 Hz log rate, so it is an availability fraction times duration, "
                    "not a count of corrections -- compare arms by the rate, not the "
                    "count, because the drives differ in length.",
        },
        "driving": {
            "ground_truth_path_m": round(float(steps.sum()), 2),
            "final_goal_distance_m": round(
                float(summary.get("final_goal_distance", float("nan"))), 2),
            "final_goal_distance_reference": summary.get(
                "final_goal_distance_reference", "wheel_odometry (schema 1)"),
            "minimum_goal_distance_m": round(
                float(summary.get("minimum_goal_distance", float("nan"))), 2),
            "max_offset_from_commanded_route_m": (
                round(float(np.nanmax(off_route)), 3) if off_route.size else None),
            "median_offset_from_commanded_route_m": (
                round(float(np.nanmedian(off_route)), 3) if off_route.size else None),
            "note": "ground truth is evaluation-only. Goal and stuck termination use the "
                    "planner belief; physical contact and timeout are independent stops.",
        },
        "odometry": {
            "final_odom_vs_truth_drift_m": round(
                float(_f(table[-1], "odom_map_gt_drift_m")), 3),
        },
        "caveat": "single-run diagnostic; paper-facing summaries aggregate frozen seeds.",
    }


def _finite_json(value):
    if isinstance(value,dict):return {k:_finite_json(v) for k,v in value.items()}
    if isinstance(value,list):return [_finite_json(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def aggregate_reports(reports):
    """Median of each per-run statistic, with complete denominators and failures."""
    reports=_finite_json(reports)
    if not reports:
        raise ValueError("no selected reports")
    result = {key: reports[0][key] for key in ("arm", "task", "logging_schema_version",
              "truth_clock", "reference_max_gap_s", "reference_method")}
    result.update(runs=[r["run"] for r in reports], n_runs=len(reports), per_run=reports)
    result["completion_counts"] = {str(k): sum(r["completion"] == k for r in reports)
                                   for k in {r["completion"] for r in reports}}
    result["completion"] = next(iter(result["completion_counts"])) if len(result["completion_counts"]) == 1 else "mixed"

    def aggregate_values(values):
        finite = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
        # An unavailable run statistic must not disappear from a displayed denominator.
        return (float(np.median(finite)) if len(finite) == len(reports) else None,
                len(finite), [min(finite), max(finite)] if finite else None)

    result["duration_s"], result["duration_n_runs"], _ = aggregate_values([r["duration_s"] for r in reports])
    for name in ("belief_error_cm", "honesty", "correction_error_cm", "corrections", "accounting", "driving", "odometry", "fusion"):
        sections = [r[name] for r in reports]
        combined = {}
        for key in set().union(*(s.keys() for s in sections)):
            values = [s.get(key) for s in sections]
            if key == "logged":
                combined[key] = all(values)
            elif key in ("n", "n_samples", "rounds", "n_unique_mission_beliefs", "n_reference_unscoreable"):
                combined[key] = sum(v or 0 for v in values)
            elif any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                value, count, spread = aggregate_values(values)
                combined[key], combined[key+"_n_runs"], combined[key+"_run_range"] = value, count, spread
            elif all(v == values[0] for v in values):
                combined[key] = values[0]
        combined["per_run"] = sections
        result[name] = combined
    result["aggregation"] = ("Median of per-run statistics only when every selected run contributes; "
        "otherwise unavailable with contributing-run counts. Sample counts sum correlated observations. "
        "Failures and unscoreable runs remain in per_run and completion_counts.")
    result["caveat"] = "One drive is one experimental unit; no navigation gain follows from offline replay."
    return result


def score(arm: str, task: str = TASKS[0], *, max_reference_gap_s=None) -> dict:
    runs = _selected_runs(arm, task)
    result = aggregate_reports([_score_one(run, arm, task, max_reference_gap_s=max_reference_gap_s) for run in runs])
    result["selection_sha256"] = hashlib.sha256(FROZEN_RUNS.read_bytes()).hexdigest()
    result["analysis_sources"] = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (Path(__file__), Path(A.__file__))}
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", nargs="*")
    parser.add_argument("--task", action="append", choices=TASKS)
    parser.add_argument("--max-reference-gap-s", type=float)
    parser.add_argument("--out", type=Path, default=STORY_ROOT)
    args = parser.parse_args(argv)
    if any(a not in FOLDER for a in args.arms):parser.error("unknown arm")
    failed = False
    for task in args.task or list(TASKS):
        for arm in args.arms or list(FOLDER):
            try:
                numbers = score(arm, task, max_reference_gap_s=args.max_reference_gap_s)
                out = args.out / task / FOLDER[arm]
                destination = out / "numbers.json"
                if destination.exists() and json.loads(destination.read_text()) != numbers:
                    raise ValueError(f"{destination}: existing report differs; preserve it and choose a new output tree")
                out.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(numbers, indent=2, allow_nan=False) + "\n")
                print(f"{task}/{arm}: {numbers['completion_counts']}; {numbers['n_runs']} selected runs; {destination}")
            except (SystemExit, ValueError) as exc:
                print(f"{task}/{arm}: {exc}", file=sys.stderr)
                failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
