#!/usr/bin/env python3
"""Paired analysis of the closed-loop calibration arms (v2 vs v3).

Matched (task, seed) pairs, one changed key. Everything is scored per pair and then
aggregated, because with 15 runs per arm an unpaired comparison would be dominated
by which seeds happened to land in each arm.

Belief/truth columns come from ``campaign_metrics.load_run`` and nothing else --
``state_x/y`` is stale and ``truth_x/y`` is wheel-odometry, and both are present in
these CSVs. The loader asserts that the belief still reproduces ``belief_error_gt_m``.

Ground truth is evaluation only: it scores outcomes, it never entered a run.

Belief honesty is scored with 2-DOF NEES against the logged planner covariance, and the
correction path is scored with the logged NIS, acceptance flag and correction age. The NIS
*threshold* is read back from the runs rather than assumed, because the runtime gate (9.21)
and the offline gate (5.991) are different policies and neither supersedes the other.

Paired deltas carry clustered bootstrap intervals from ``reliability.campaign_statistics``
(route -> seed nesting); the flat bootstrap would pretend the three routes are exchangeable.

The pair matrix is checked fail-closed: an incomplete or unmatched matrix aborts instead of
reporting a comparison over whichever runs happened to survive.

Outputs -> logs/studies/closed_loop_calibration/exp1_v2_vs_v3/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
sys.path.insert(0, str(REPO / "scripts" / "geometry_visibility"))
import campaign_metrics as cm  # noqa: E402  (THE canonical loader)

sys.path.insert(0, str(REPO / "src" / "reliability"))
from reliability.campaign_statistics import Leaf, hierarchical_bootstrap  # noqa: E402

OUT = REPO / "logs/studies/closed_loop_calibration/exp1_v2_vs_v3"

ARMS = {"clv2": "v2 (deployed, along-bearing only)", "clv3": "v3 (2-DOF, gated cross term)"}

#: chi-square(2) upper 95% point -- the stated-ellipse containment test for a 2-D belief.
CHI2_2_95 = 5.991464547107979
#: median of chi-square(2); a calibrated 2-DOF filter sits here, not at the mean of 2.
CHI2_2_MEDIAN = 1.3862943611198906

#: Fields whose paired delta is summarized with a clustered bootstrap interval.
DELTA_FIELDS = (
    "belief_error_median_m",
    "belief_error_p95_m",
    "final_goal_distance",
    "nees_median",
    "outside_95_ellipse_rate",
    "correction_accept_rate",
)

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


def _column(rows: list[dict], key: str) -> np.ndarray:
    out = np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        try:
            out[i] = float(row.get(key, ""))
        except (TypeError, ValueError):
            pass
    return out


def belief_honesty(experiment_csv: Path, run: dict) -> dict:
    """2-DOF NEES of the canonical belief against the logged planner covariance.

    ``run`` supplies belief/truth from the canonical loader; only the covariance columns are
    read here, and they are diagnostics rather than an alternative belief source.
    """
    rows = list(csv.DictReader(open(experiment_csv)))
    if len(rows) != run["belief_x"].size:
        raise AssertionError(
            f"{experiment_csv}: diagnostic rows {len(rows)} != canonical rows "
            f"{run['belief_x'].size}; the covariance columns cannot be aligned to the belief"
        )

    pxx, pxy, pyy = (_column(rows, k) for k in ("planner_cov_x", "planner_cov_xy", "planner_cov_y"))
    ex = run["belief_x"] - run["truth_x"]
    ey = run["belief_y"] - run["truth_y"]

    det = pxx * pyy - pxy * pxy
    ok = np.isfinite(det) & (det > 0) & (pxx > 0) & (pyy > 0) & np.isfinite(ex) & np.isfinite(ey)
    nees = np.full(ex.shape, np.nan)
    nees[ok] = (
        pyy[ok] * ex[ok] ** 2 - 2.0 * pxy[ok] * ex[ok] * ey[ok] + pxx[ok] * ey[ok] ** 2
    ) / det[ok]

    valid = nees[np.isfinite(nees)]
    return {
        "nees_steps": int(valid.size),
        "nees_median": float(np.median(valid)) if valid.size else math.nan,
        "nees_mean": float(np.mean(valid)) if valid.size else math.nan,
        "outside_95_ellipse_rate": (
            float(np.mean(valid > CHI2_2_95)) if valid.size else math.nan
        ),
        "nondefinite_cov_fraction": float(np.mean(~ok)) if ok.size else math.nan,
    }


def correction_path(experiment_csv: Path) -> dict:
    """Correction acceptance, NIS and staleness, with the gate threshold read back."""
    rows = list(csv.DictReader(open(experiment_csv)))
    nis = _column(rows, "pixel_corr_nis")
    accepted = _column(rows, "pixel_corr_accepted")
    age = _column(rows, "planner_pixel_correction_age_s")
    threshold = _column(rows, "pixel_corr_nis_threshold")

    attempted = np.isfinite(nis)
    acc = accepted[attempted]
    acc = acc[np.isfinite(acc)]
    age_valid = age[np.isfinite(age)]
    thresholds = np.unique(threshold[np.isfinite(threshold)])

    return {
        "corrections_attempted": int(attempted.sum()),
        "correction_accept_rate": float(np.mean(acc > 0.5)) if acc.size else math.nan,
        "correction_nis_median": (
            float(np.median(nis[attempted])) if attempted.any() else math.nan
        ),
        "correction_age_median_s": float(np.median(age_valid)) if age_valid.size else math.nan,
        "correction_age_p95_s": (
            float(np.percentile(age_valid, 95)) if age_valid.size else math.nan
        ),
        "nis_threshold_observed": [float(t) for t in thresholds],
    }


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
        record.update(belief_honesty(experiment_csv, run))
        record.update(correction_path(experiment_csv))
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
            "nees_median", "nees_mean", "outside_95_ellipse_rate",
            "correction_accept_rate", "correction_nis_median", "correction_age_median_s",
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


def _median_of(values: list[dict], field: str) -> float:
    finite = [
        float(v[field]) for v in values
        if isinstance(v.get(field), (int, float)) and math.isfinite(float(v[field]))
    ]
    return float(np.median(finite)) if finite else math.nan


def summarize_arm(records: dict) -> dict:
    values = list(records.values())
    if not values:
        return {"n_runs": 0}
    thresholds = sorted({t for v in values for t in v.get("nis_threshold_observed", [])})
    return {
        "n_runs": len(values),
        "clean_goal_rate": _rate(values, "goal_region_success"),
        "completed_rate": _rate(values, "completed"),
        "contact_rate": _rate(values, "collision_any"),
        "nogo_breach_rate": _rate(values, "inside_no_go"),
        "belief_error_median_m": _median_of(values, "belief_error_median_m"),
        "belief_error_p95_m": _median_of(values, "belief_error_p95_m"),
        "final_goal_distance_median_m": _median_of(values, "final_goal_distance"),
        "nees_median": _median_of(values, "nees_median"),
        "nees_mean": _median_of(values, "nees_mean"),
        "outside_95_ellipse_rate": _median_of(values, "outside_95_ellipse_rate"),
        "correction_accept_rate": _median_of(values, "correction_accept_rate"),
        "correction_nis_median": _median_of(values, "correction_nis_median"),
        "correction_age_median_s": _median_of(values, "correction_age_median_s"),
        "correction_age_p95_s": _median_of(values, "correction_age_p95_s"),
        "nis_threshold_observed": thresholds,
    }


def bootstrap_deltas(pairs: list[dict], seed: int = 0, n_boot: int = 2000) -> dict:
    """Clustered (route -> seed) bootstrap interval for each paired delta field."""
    out = {}
    for field in DELTA_FIELDS:
        leaves = [
            Leaf(route=p["task"], seed=str(p["seed"]), episode="run", delta=p[f"d_{field}"])
            for p in pairs
            if isinstance(p.get(f"d_{field}"), (int, float)) and math.isfinite(p[f"d_{field}"])
        ]
        if not leaves:
            continue
        result = hierarchical_bootstrap(leaves, n_boot=n_boot, seed=seed, lower_is_better=True)
        out[field] = {
            "mean_delta": result.point,
            "ci_low": result.low,
            "ci_high": result.high,
            "n_units": result.n_units,
            "n_boot": result.n_boot,
            "proportion_v3_better": result.proportion_favorable,
            "excludes_zero": bool(result.low > 0.0 or result.high < 0.0),
        }
    return out


def write_plots(arms: dict, comparison: dict, boot: dict, out_dir: Path) -> list[str]:
    """Paired-delta intervals and the per-arm NEES position, as one review figure."""
    fields = [f for f in DELTA_FIELDS if f in boot]
    if not fields:
        return []

    fig, (ax_delta, ax_nees) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    y = np.arange(len(fields))
    point = [boot[f]["mean_delta"] for f in fields]
    low = [boot[f]["mean_delta"] - boot[f]["ci_low"] for f in fields]
    high = [boot[f]["ci_high"] - boot[f]["mean_delta"] for f in fields]
    ax_delta.errorbar(point, y, xerr=[low, high], fmt="o", capsize=4, color="#1f77b4")
    ax_delta.axvline(0.0, color="0.4", lw=1, ls="--")
    ax_delta.set_yticks(y)
    ax_delta.set_yticklabels(fields, fontsize=8)
    ax_delta.invert_yaxis()
    ax_delta.set_xlabel("paired delta, v3 - v2 (clustered 95% CI)")
    ax_delta.set_title(f"{comparison['n_pairs']} matched pairs")

    for offset, (name, records) in enumerate(sorted(arms.items())):
        values = [
            r["nees_median"] for r in records.values()
            if isinstance(r.get("nees_median"), (int, float)) and math.isfinite(r["nees_median"])
        ]
        if values:
            ax_nees.scatter(
                np.full(len(values), offset) + np.random.default_rng(0).normal(0, 0.04, len(values)),
                values, label=name, alpha=0.8,
            )
    ax_nees.axhline(CHI2_2_MEDIAN, color="green", ls="--", lw=1, label="calibrated median (1.386)")
    ax_nees.axhline(CHI2_2_95, color="red", ls=":", lw=1, label="chi2(2) 95% (5.991)")
    ax_nees.set_xticks(range(len(arms)))
    ax_nees.set_xticklabels(sorted(arms), fontsize=9)
    ax_nees.set_ylabel("per-run median NEES")
    ax_nees.set_title("belief honesty")
    ax_nees.legend(fontsize=7)

    fig.tight_layout()
    path = out_dir / "v2_vs_v3.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    try:
        return [str(path.relative_to(REPO))]
    except ValueError:  # --out pointed outside the repo (scratch runs)
        return [str(path)]


def check_matrix(comparison: dict, arms: dict, expect_pairs: int) -> list[str]:
    """Fail-closed completeness gate; returns the blocking reasons."""
    problems = []
    if comparison["n_pairs"] != expect_pairs:
        problems.append(
            f"expected {expect_pairs} matched pairs, found {comparison['n_pairs']}"
        )
    for label in ("unmatched_in_v2_only", "unmatched_in_v3_only"):
        if comparison[label]:
            problems.append(f"{label}: {comparison[label]}")
    for name, records in arms.items():
        if len(records) != expect_pairs:
            problems.append(f"{name}: {len(records)} runs, expected {expect_pairs}")
    thresholds = {t for r in arms.values() for v in r.values() for t in v["nis_threshold_observed"]}
    if len(thresholds) > 1:
        problems.append(f"runs disagree on the NIS gate threshold: {sorted(thresholds)}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-root", default="logs/visibility_comparison/clv2")
    parser.add_argument("--v3-root", default="logs/visibility_comparison/clv3")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument(
        "--expect-pairs", type=int, default=15,
        help="matched (task, seed) pairs the frozen matrix must contain; 0 disables the gate",
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
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
    boot = bootstrap_deltas(comparison["pairs"], seed=args.bootstrap_seed, n_boot=args.n_boot)
    problems = check_matrix(comparison, arms, args.expect_pairs) if args.expect_pairs else []

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = write_plots(arms, comparison, boot, out_dir)

    payload = {
        "study": "closed_loop_calibration",
        "experiment": "exp1_v2_vs_v3",
        "arms": ARMS,
        "arm_summary": {name: summarize_arm(records) for name, records in arms.items()},
        "paired": comparison,
        "paired_bootstrap": boot,
        "bootstrap_config": {
            "kind": "hierarchical (route -> seed)",
            "n_boot": args.n_boot,
            "seed": args.bootstrap_seed,
            "ci": 0.95,
        },
        "matrix_gate": {
            "expected_pairs": args.expect_pairs,
            "passed": not problems,
            "problems": problems,
        },
        "figures": figures,
        "runs": {
            name: [records[k] for k in sorted(records)] for name, records in arms.items()
        },
        "ground_truth_use": "evaluation only (breach, contact, goal, belief error, NEES)",
    }
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

    print(f"\n{'arm':<7} {'NEES med':>9} {'outside 95%':>12} {'accept':>8} "
          f"{'NIS med':>8} {'age med s':>10} {'NIS gate':>10}")
    for name in ("clv2", "clv3"):
        s = payload["arm_summary"][name]
        gate = ",".join(f"{t:g}" for t in s["nis_threshold_observed"]) or "-"
        print(
            f"{name:<7} {s['nees_median']:>9.2f} {s['outside_95_ellipse_rate']:>12.3f} "
            f"{s['correction_accept_rate']:>8.3f} {s['correction_nis_median']:>8.2f} "
            f"{s['correction_age_median_s']:>10.3f} {gate:>10}"
        )
    print(f"  calibrated 2-DOF reference: median NEES {CHI2_2_MEDIAN:.3f}, "
          f"outside-95% rate 0.05")

    print(f"\n{comparison['n_pairs']} matched pairs")
    for label in ("unmatched_in_v2_only", "unmatched_in_v3_only"):
        if comparison[label]:
            print(f"  !! {label}: {comparison[label]}")

    if boot:
        print("\npaired delta v3 - v2, clustered 95% CI (negative favours v3)")
        for field, b in boot.items():
            flag = "  *" if b["excludes_zero"] else ""
            print(
                f"  {field:<26} {b['mean_delta']:+.4f}  "
                f"[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]  "
                f"v3 better on {b['proportion_v3_better']:.0%}{flag}"
            )

    if comparison["pairs"]:
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
    for figure in figures:
        print(f"-> {figure}")

    if problems:
        print("\nMATRIX GATE FAILED -- this comparison is not reportable:")
        for problem in problems:
            print(f"  !! {problem}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
