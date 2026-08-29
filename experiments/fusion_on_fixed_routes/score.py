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
    "F3": "03_independent_fusion", "F4": "04_network_fusion",
    "O1": "05_raw_box", "O2": "06_fixed_offset",
}


def _fusion_quality(run: Path) -> dict:
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

    rounds = A.fused_answers(run)
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
        "F3": ("independent", "hull"), "F4": ("network", "hull"),
        "O1": ("network", "raw_box"), "O2": ("network", "fixed_offset"),
    }[arm]
    selected, seeds, provenance = [], set(), set()
    for value in values:
        run = Path(str(value)).expanduser()
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
        assimilation = A.assimilations(run)
        correction_batches = {
            row["source_batch_id"] for row in A.fused_answers(run)
            if row["source_batch_id"]
        }
        assimilation_batches = {row["source_batch_id"] for row in assimilation}
        if assimilation_batches != correction_batches:
            missing = sorted(correction_batches - assimilation_batches)
            extra = sorted(assimilation_batches - correction_batches)
            raise SystemExit(
                f"{run}: correction/assimilation identity mismatch; "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
        dropped = [row for row in assimilation if row["status"] == "dropped"]
        if dropped:
            raise SystemExit(
                f"{run}: {len(dropped)} correction(s) were dropped by the filter"
            )
        seed = int(manifest.get("seed", -1))
        if seed in seeds:
            raise SystemExit(f"{run}: duplicate seed {seed} in {task}/{arm}")
        seeds.add(seed)
        provenance.add(tuple(manifest.get(key) for key in (
            "campaign_config_sha256", "git_sha", "git_diff_sha256",
            "git_untracked_content_sha256", "yolo_model_sha256",
            "manager_commissioned_calibration_sha256",
        )))
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


def _score_one(run: Path, arm: str, task: str = TASKS[0]) -> dict:
    table = A.rows(run)
    summary = json.loads((run / "run_summary.json").read_text())
    truth = A.truth_series(run, table)

    # --- what it did: how far the belief actually was from the truth -----------
    belief = A.aligned_error_cm(run, "belief", table)
    error_cm = belief["aligned_cm"][np.isfinite(belief["aligned_cm"])]
    error_logtime_cm = belief["logtime_cm"][np.isfinite(belief["logtime_cm"])]

    # --- is it honest: does the truth fall inside the stated 95% ellipse? ------
    cov = np.array([[[_f(r, "planner_cov_x"), _f(r, "planner_cov_xy")],
                     [_f(r, "planner_cov_xy"), _f(r, "planner_cov_y")]] for r in table])
    resid = np.stack([belief["gt_x"] - belief["x"], belief["gt_y"] - belief["y"]], axis=1)
    usable = (belief["have"] & np.isfinite(resid).all(axis=1)
              & (cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] ** 2 > 0.0))
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
    state = A.aligned_error_cm(run, "state", table)
    landed = A.landed_mask(state["stamp"])
    correction_cm = state["aligned_cm"][landed & np.isfinite(state["aligned_cm"])]
    correction_logtime_cm = state["logtime_cm"][
        landed & np.isfinite(state["logtime_cm"])]

    corrections = A.corrections(run, table)

    # --- did it stay on the commanded route? ----------------------------------
    path = np.stack([truth.x, truth.y], axis=1)
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
        "fusion": _fusion_quality(run),
        "completion": summary.get("completion_reason"),
        "duration_s": round(float(summary.get("elapsed_after_first_cmd_s", float("nan"))), 1),
        "belief_error_cm": {
            "median": pct(error_cm, 50), "p95": pct(error_cm, 95),
            "worst": round(float(error_cm.max()), 2) if error_cm.size else None,
            "n_samples": int(error_cm.size),
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
            "note": "each correction scored once, when it was published, against the "
                    "truth at its own stamp. Scoring a held correction against a moving "
                    "robot measures the robot's travel, not the sensor: on these drives "
                    "that turned a 4.9 cm p95 into an apparent 124 cm one.",
        },
        "corrections": {
            "detector_rounds": corrections["n_detector_rounds"],
            "state_publications_seen": corrections["n_state_publications"],
            "state_fresh_rate_hz": (round(corrections["state_fresh_rate_hz"], 2)
                                    if math.isfinite(corrections["state_fresh_rate_hz"])
                                    else None),
            "longest_gap_s": (round(corrections["longest_gap_s"], 2)
                              if math.isfinite(corrections["longest_gap_s"]) else None),
            "median_gap_s": (round(corrections["median_gap_s"], 2)
                             if math.isfinite(corrections["median_gap_s"]) else None),
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


def score(arm: str, task: str = TASKS[0]) -> dict:
    """Aggregate the predeclared runs at the run level, never as pseudo-replicate rows."""

    reports = [_score_one(run, arm, task) for run in _selected_runs(arm, task)]
    result = json.loads(json.dumps(reports[0]))
    result["runs"] = [report["run"] for report in reports]
    result["n_runs"] = len(reports)
    result.pop("run", None)
    result["completion_counts"] = {
        key: sum(report["completion"] == key for report in reports)
        for key in sorted({report["completion"] for report in reports})
    }

    def aggregate_section(name: str, keys: tuple[str, ...]) -> None:
        section = result[name]
        section["per_run"] = []
        for report in reports:
            section["per_run"].append({key: report[name].get(key) for key in keys})
        for key in keys:
            values = [report[name].get(key) for report in reports]
            finite = [float(value) for value in values
                      if value is not None and math.isfinite(float(value))]
            if not finite:
                section[key] = None
                continue
            section[key] = round(float(np.median(finite)), 3)
            section[f"{key}_run_range"] = [
                round(float(min(finite)), 3), round(float(max(finite)), 3)
            ]

    aggregate_section("belief_error_cm", ("median", "p95", "worst"))
    result["belief_error_cm"]["n_samples"] = sum(
        report["belief_error_cm"]["n_samples"] for report in reports)
    aggregate_section("honesty", (
        "truth_inside_stated_95pct_ellipse", "median_stated_1sigma_cm",
        "p95_stated_1sigma_cm", "nees_mean", "nees_median",
    ))
    result["honesty"]["n_samples"] = sum(
        report["honesty"]["n_samples"] for report in reports)
    aggregate_section("correction_error_cm", ("median", "p95", "worst"))
    result["correction_error_cm"]["n"] = sum(
        report["correction_error_cm"]["n"] for report in reports)
    if all(report["fusion"].get("logged") for report in reports):
        aggregate_section("fusion", (
            "worse_than_best_available_camera", "used_the_closest_camera",
            "median_fused_error_cm", "median_best_available_camera_error_cm",
        ))
        result["fusion"]["rounds"] = sum(
            int(report["fusion"]["rounds"]) for report in reports)
    result["aggregation"] = (
        "Each displayed point estimate is the median of the per-run statistic across "
        f"{len(reports)} frozen seeds. Samples within a drive are not treated as independent runs."
    )
    result["caveat"] = (
        f"{len(reports)} paired seeds. Report run-level spread and paired arm contrasts; "
        "do not use camera frames as the experimental sample size."
    )
    return result


def main() -> int:
    args = [a for a in sys.argv[1:]]
    tasks = [a.split("=", 1)[1] for a in args if a.startswith("--task=")] or list(TASKS)
    arms = [a for a in args if not a.startswith("--")] or list(FOLDER)
    for task in tasks:
        for arm in arms:
            if arm not in FOLDER:
                raise SystemExit(f"unknown arm {arm!r}; expected one of {list(FOLDER)}")
            try:
                numbers = score(arm, task)
            except SystemExit as exc:
                print(f"{task}/{arm}: {exc}")
                continue
            out = story_dir(task, arm)
            out.mkdir(parents=True, exist_ok=True)
            (out / "numbers.json").write_text(json.dumps(numbers, indent=2) + "\n")
            e, h, fq = numbers["belief_error_cm"], numbers["honesty"], numbers["fusion"]
            print(f"{task:24s} {arm} {numbers['completion']:>12s}  median {e['median']} cm  "
                  f"p95 {e['p95']} cm | inside 95% "
                  f"{h['truth_inside_stated_95pct_ellipse']} at a median stated "
                  f"{h['median_stated_1sigma_cm']} cm, NEES mean {h['nees_mean']} "
                  f"(target 2), median {h['nees_median']} (target 1.386)"
                  + (f" | worse than its best camera "
                     f"{fq['worse_than_best_available_camera']*100:.0f}%"
                     if fq.get("logged") else ""))
            print(f"{'':24s}    at log time this drive would read median "
                  f"{e['median_scored_at_log_time']} cm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
