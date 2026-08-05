#!/usr/bin/env python3
"""Paired analysis of the closed-loop calibration arms (v2 vs v3).

Matched (task, seed) pairs, one changed key. Everything is scored per pair and then
aggregated, because with 15 runs per arm an unpaired comparison would be dominated
by which seeds happened to land in each arm.

Belief/truth columns come from ``campaign_metrics.load_run`` and nothing else --
``state_x/y`` is stale and ``truth_x/y`` is wheel-odometry, and both are present in
these CSVs. The loader asserts that the belief still reproduces ``belief_error_gt_m``.

Ground truth is evaluation only: it scores outcomes, it never entered a run.

Outputs -> logs/studies/closed_loop_calibration/exp1_v2_vs_v3/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
sys.path.insert(0, str(REPO / "scripts" / "geometry_visibility"))
import campaign_metrics as cm  # noqa: E402  (THE canonical loader)

OUT = REPO / "logs/studies/closed_loop_calibration/exp1_v2_vs_v3"

ARMS = {"clv2": "v2 (deployed, along-bearing only)", "clv3": "v3 (2-DOF, gated cross term)"}

#: Outcome fields read straight from run_summary.json.
SUMMARY_FIELDS = (
    "completed",
    "goal_region_success",
    "collision_any",
    "collision_contact",
    "inside_no_go",
    "final_goal_distance",
    "elapsed_after_first_cmd_s",
)


def load_arm(log_root: Path) -> dict[tuple[str, int], dict]:
    """One record per (task, seed) run under a campaign log root."""
    out = {}
    for summary_path in sorted(log_root.glob("*/*/*/*/run_summary.json")):
        experiment_csv = summary_path.parent / "experiment.csv"
        if not experiment_csv.is_file():
            continue
        task = summary_path.parents[3].name
        seed_dir = summary_path.parents[1].name
        try:
            seed = int(seed_dir.replace("seed", ""))
        except ValueError:
            continue

        summary = json.loads(summary_path.read_text())
        run = cm.load_run(str(experiment_csv))  # asserts canonical columns

        error = run["belief_error_m"]
        finite = error[np.isfinite(error)]
        sigma = run["reported_sigma_m"]
        sigma_finite = sigma[np.isfinite(sigma)]

        record = {name: summary.get(name) for name in SUMMARY_FIELDS}
        record.update(
            {
                "task": task,
                "seed": seed,
                "steps": int(error.size),
                "belief_error_median_m": float(np.median(finite)) if finite.size else math.nan,
                "belief_error_p95_m": float(np.percentile(finite, 95)) if finite.size else math.nan,
                "belief_error_max_m": float(np.max(finite)) if finite.size else math.nan,
                "reported_sigma_median_m": (
                    float(np.median(sigma_finite)) if sigma_finite.size else math.nan
                ),
                "run_dir": str(summary_path.parent.relative_to(REPO)),
            }
        )
        out[(task, seed)] = record
    return out


def _rate(records, field) -> float:
    values = [bool(r[field]) for r in records if r.get(field) is not None]
    return float(np.mean(values)) if values else math.nan


def paired(a: dict, b: dict) -> dict:
    """Per-pair deltas (v3 minus v2) over the (task, seed) keys present in BOTH arms."""
    keys = sorted(set(a) & set(b))
    missing_a = sorted(set(b) - set(a))
    missing_b = sorted(set(a) - set(b))
    deltas = []
    for key in keys:
        left, right = a[key], b[key]
        entry = {"task": key[0], "seed": key[1]}
        for field in (
            "belief_error_median_m", "belief_error_p95_m", "belief_error_max_m",
            "final_goal_distance", "elapsed_after_first_cmd_s", "reported_sigma_median_m",
        ):
            lv, rv = left.get(field), right.get(field)
            entry[f"d_{field}"] = (
                float(rv) - float(lv)
                if isinstance(lv, (int, float)) and isinstance(rv, (int, float))
                and math.isfinite(lv) and math.isfinite(rv)
                else math.nan
            )
        for field in ("completed", "goal_region_success", "collision_any", "inside_no_go"):
            entry[f"{field}_v2"] = left.get(field)
            entry[f"{field}_v3"] = right.get(field)
            entry[f"{field}_flipped"] = bool(left.get(field)) != bool(right.get(field))
        deltas.append(entry)
    return {
        "n_pairs": len(keys),
        "unmatched_in_v2_only": [f"{t}/seed{s}" for t, s in missing_b],
        "unmatched_in_v3_only": [f"{t}/seed{s}" for t, s in missing_a],
        "pairs": deltas,
    }


def summarize_arm(records: dict) -> dict:
    values = list(records.values())
    if not values:
        return {"n_runs": 0}
    return {
        "n_runs": len(values),
        "clean_goal_rate": _rate(values, "goal_region_success"),
        "completed_rate": _rate(values, "completed"),
        "contact_rate": _rate(values, "collision_any"),
        "nogo_breach_rate": _rate(values, "inside_no_go"),
        "belief_error_median_m": float(
            np.median([v["belief_error_median_m"] for v in values])
        ),
        "belief_error_p95_m": float(np.median([v["belief_error_p95_m"] for v in values])),
        "final_goal_distance_median_m": float(
            np.median([
                v["final_goal_distance"] for v in values
                if isinstance(v["final_goal_distance"], (int, float))
            ])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-root", default="logs/visibility_comparison/clv2")
    parser.add_argument("--v3-root", default="logs/visibility_comparison/clv3")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    roots = {"clv2": REPO / args.v2_root, "clv3": REPO / args.v3_root}
    for name, root in roots.items():
        if not root.is_dir():
            raise SystemExit(f"{name}: no campaign log at {root} -- run the arm first")

    arms = {name: load_arm(root) for name, root in roots.items()}
    for name, records in arms.items():
        if not records:
            raise SystemExit(f"{name}: no runs found under {roots[name]}")

    comparison = paired(arms["clv2"], arms["clv3"])

    payload = {
        "study": "closed_loop_calibration",
        "experiment": "exp1_v2_vs_v3",
        "arms": ARMS,
        "arm_summary": {name: summarize_arm(records) for name, records in arms.items()},
        "paired": comparison,
        "runs": {
            name: [records[k] for k in sorted(records)] for name, records in arms.items()
        },
        "ground_truth_use": "evaluation only (breach, contact, goal, belief error)",
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "v2_vs_v3.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
    )

    print(f"{'arm':<7} {'runs':>5} {'clean goal':>11} {'contacts':>9} {'no-go':>7} "
          f"{'bel err med':>12} {'bel err p95':>12}")
    for name in ("clv2", "clv3"):
        s = payload["arm_summary"][name]
        print(
            f"{name:<7} {s['n_runs']:>5} {s['clean_goal_rate']:>11.2f} "
            f"{s['contact_rate']:>9.2f} {s['nogo_breach_rate']:>7.2f} "
            f"{s['belief_error_median_m']:>12.4f} {s['belief_error_p95_m']:>12.4f}"
        )

    print(f"\n{comparison['n_pairs']} matched pairs")
    for label in ("unmatched_in_v2_only", "unmatched_in_v3_only"):
        if comparison[label]:
            print(f"  !! {label}: {comparison[label]}")

    if comparison["pairs"]:
        for field in ("d_belief_error_median_m", "d_belief_error_p95_m", "d_final_goal_distance"):
            values = np.array([p[field] for p in comparison["pairs"]], dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                wins = int(np.sum(values < 0))
                print(
                    f"  {field:<28} median {np.median(values):+.4f}  "
                    f"v3 better on {wins}/{values.size}"
                )
        flips = [
            p for p in comparison["pairs"]
            if p["collision_any_flipped"] or p["inside_no_go_flipped"]
            or p["goal_region_success_flipped"]
        ]
        print(f"  outcome flips: {len(flips)}")
        for flip in flips:
            print(
                f"    {flip['task']}/seed{flip['seed']}: "
                f"goal {flip['goal_region_success_v2']}->{flip['goal_region_success_v3']} "
                f"contact {flip['collision_any_v2']}->{flip['collision_any_v3']} "
                f"nogo {flip['inside_no_go_v2']}->{flip['inside_no_go_v3']}"
            )

    print(f"\n-> {out_dir / 'v2_vs_v3.json'}")


if __name__ == "__main__":
    main()
