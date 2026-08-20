#!/usr/bin/env python3
"""Evaluate the online-EFE readiness gate against one real Gazebo run.

This script LAUNCHES NOTHING. It reads a finished run directory and reports
pass/fail per criterion in ``gate.yaml``. Before launching the run itself:

    pgrep -a "ros2 launch|ign gazebo|run_visibility_campaign"

Non-empty means do not launch — a second Gazebo collides with the live one on the
same ROS topics and corrupts both.

Usage:
    python3 experiments/availability_paper/e0_online_efe_readiness/check_gate.py \
        --run logs/visibility_comparison/<campaign>/<task>/C1/seed0/experiment_*/

Notes on what is trusted. Belief and truth come from ``campaign_metrics.load_run``,
which asserts the canonical columns; ``state_x/state_y`` and ``truth_x/truth_y``
are stale in these logs and are never read here. ``run_summary.json`` only exists
once a run ends, so a missing summary means the run is still going or died, not
that it failed the gate.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import common as C  # noqa: E402

sys.path.insert(0, str(C.REPO / "scripts/geometry_visibility"))
import campaign_metrics as CM  # noqa: E402

GATE = HERE / "gate.yaml"
FUSED_FIELD = C.REPO / "logs/visibility_comparison/spawn_grid_20260727/fused_planner_four_camera.npz"


def load_fused_field() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The frozen four-camera planning field the campaign itself consumes."""

    if not FUSED_FIELD.is_file():
        raise RuntimeError(f"Frozen fused planning field missing: {FUSED_FIELD}")
    data = np.load(FUSED_FIELD, allow_pickle=True)
    for key in ("P_union_4cam_map", "P_best_4cam_map", "P_mean_map"):
        if key in data.files:
            return np.asarray(data["xs"], float), np.asarray(data["ys"], float), np.asarray(data[key], float)
    raise RuntimeError(f"No usable probability map in {FUSED_FIELD}; keys: {list(data.files)}")


def check(run_dir: Path, goal_xy: tuple[float, float] | None) -> list[dict]:
    spec = yaml.safe_load(GATE.read_text())
    results: list[dict] = []

    def record(cid: str, passed: bool | None, detail: str) -> None:
        name = next((c["name"] for c in spec["criteria"] if c["id"] == cid), cid)
        results.append({"id": cid, "name": name, "passed": passed, "detail": detail})

    experiment_csv = run_dir / "experiment.csv"
    perception_csv = run_dir / "perception.csv"
    summary_json = run_dir / "run_summary.json"

    # G1 — did camera corrections actually reach the filter?
    #
    # NOT from perception.csv. That file was empty in the first passing gate run
    # (header, no rows) even though 818 of 861 timesteps carried an accepted
    # correction, so it is not a reliable witness. experiment.csv's
    # pixel_corr_accepted is the record the filter actually acted on, and it is the
    # same column E4's primary endpoint is computed from.
    if experiment_csv.is_file() and experiment_csv.stat().st_size > 0:
        with open(experiment_csv, newline="") as fh:
            rows = list(csv.DictReader(fh))
        accepted = sum(
            1 for r in rows if str(r.get("pixel_corr_accepted", "")).strip() in {"1", "1.0", "True", "true"}
        )
        rate = accepted / len(rows) if rows else 0.0
        record(
            "G1",
            accepted > 0,
            f"{accepted}/{len(rows)} timesteps had an accepted camera correction ({rate:.0%})",
        )
    else:
        record("G1", None, "experiment.csv absent or empty")

    # G2 — did the selected global plan actually reach the goal?
    #
    # This is the pilot's real failure and it is NOT in run_summary.json. It lives in
    # global_plan_meta.json: `terminal_goal_distance_pred` is how far the selected
    # plan ends from the goal. `rollout_valid` is not sufficient — the pilot's plans
    # were dynamically valid and ended 12.9 m short.
    plan_meta = run_dir / "global_plan_meta.json"
    tolerance = float(spec["criteria"][1].get("pass_if_max_terminal_pred_m", 0.5))
    if plan_meta.is_file():
        meta = json.loads(plan_meta.read_text())
        pred = meta.get("terminal_goal_distance_pred")
        valid = bool(meta.get("rollout_valid", False))
        selected = str(meta.get("selected_source", "?"))
        if pred is None:
            record("G2", None, f"global_plan_meta.json has no terminal_goal_distance_pred (selected {selected})")
        else:
            record(
                "G2",
                valid and float(pred) <= tolerance,
                f"selected plan ends {float(pred):.2f} m from goal "
                f"(needs <= {tolerance} m), rollout_valid={valid}, source={selected}",
            )
    else:
        record("G2", None, "global_plan_meta.json absent")

    # G3 — did the local controller reject step zero?
    #
    # Also not in run_summary.json: the reject code is emitted to the ROS text logs,
    # which live once per CAMPAIGN rather than per run. Walk up to find them. If they
    # cannot be found this is N/A, never PASS — a gate that reports PASS on something
    # it did not measure is worse than no gate.
    ros_logs = None
    for parent in run_dir.parents:
        candidate = parent / "_ros_logs"
        if candidate.is_dir():
            ros_logs = candidate
            break
    if ros_logs is None:
        record("G3", None, "no _ros_logs directory found above the run; step-zero rejects not measurable")
    else:
        hits = 0
        for log in ros_logs.glob("*.log"):
            try:
                hits += log.read_text(errors="ignore").count("driveable_clearance_violation_step_0")
            except OSError:
                continue
        record(
            "G3",
            hits == 0,
            f"driveable_clearance_violation_step_0 appears {hits}x in {ros_logs.name} "
            "(campaign-wide, not per-run)",
        )

    # G5 — completion reason.
    if summary_json.is_file():
        summary = json.loads(summary_json.read_text())
        reason = str(summary.get("completion_reason", ""))
        # run_visibility_campaign documents exactly: goal_reached,
        # timeout_after_first_cmd, collision. An earlier version of this checker
        # tested for "goal" and so failed a run that had actually succeeded.
        record("G5", reason == "goal_reached", f"completion_reason={reason!r}")
    else:
        record("G5", None, "run_summary.json absent; the run has not ended")

    # G4/G6 need the trajectory.
    if experiment_csv.is_file() and experiment_csv.stat().st_size > 0:
        run = CM.load_run(str(experiment_csv))
        belief = np.column_stack([run["belief_x"], run["belief_y"]])
        finite = np.isfinite(belief).all(axis=1)
        belief = belief[finite]
        if goal_xy is not None and belief.size:
            final = float(np.linalg.norm(belief[-1] - np.asarray(goal_xy, float)))
            record("G4", final <= float(spec["criteria"][3]["pass_if_max_m"]), f"final distance {final:.3f} m")
        else:
            record("G4", None, "pass --goal x y to evaluate arrival")

        if belief.size:
            xs, ys, field = load_fused_field()
            p_along = C.sample_field_at(field, xs, ys, belief)
            threshold = float(next(c for c in spec["criteria"] if c["id"] == "G6")["pass_if_max"])
            record(
                "G6",
                float(np.min(p_along)) <= threshold,
                f"min fused p_use along executed path {np.min(p_along):.4f} "
                f"(needs <= {threshold} to discriminate the arms)",
            )
        else:
            record("G6", None, "no finite belief samples")
    else:
        record("G4", None, "experiment.csv absent or empty")
        record("G6", None, "experiment.csv absent or empty")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run directory (glob allowed)")
    parser.add_argument("--goal", nargs=2, type=float, default=None, metavar=("X", "Y"))
    args = parser.parse_args()

    matches = sorted(Path(p) for p in glob.glob(args.run))
    if not matches:
        raise SystemExit(f"no run directory matched {args.run!r}")
    run_dir = matches[-1]

    results = check(run_dir, tuple(args.goal) if args.goal else None)
    print(f"Online-EFE readiness gate — {run_dir}\n")
    width = max(len(r["name"]) for r in results) + 2
    for r in results:
        mark = {True: "PASS", False: "FAIL", None: "N/A "}[r["passed"]]
        print(f"  [{mark}] {r['id']} {r['name']:<{width}} {r['detail']}")

    failed = [r["id"] for r in results if r["passed"] is False]
    unknown = [r["id"] for r in results if r["passed"] is None]
    print()
    if failed:
        print(f"GATE FAILED on {', '.join(failed)}. No campaign seeds may be allocated.")
        raise SystemExit(1)
    if unknown:
        print(f"GATE INCONCLUSIVE: {', '.join(unknown)} could not be measured.")
        raise SystemExit(2)
    print("GATE PASSED. EXP-AVAIL-CL may be promoted from BLOCKED to READY.")


if __name__ == "__main__":
    main()
