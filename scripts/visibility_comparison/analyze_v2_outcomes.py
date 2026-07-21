#!/usr/bin/env python3
"""Summarise honest_campaign_v2 (3 conditions, standard-termination policy).

Reads per-run run_summary.json (GT-based, pre-computed outcomes — no raw position
columns, so it sidesteps the campaign-metrics column trap) and reports, per task
and per condition (C0 geometry-only baseline / C1 constant-R EFE / C2 learned-R EFE):

  - clean goal reaches, physical contacts, geometric grazes (non-terminal under
    the v2 policy), region-exits, valid runs
  - minimum GT clearance distribution (the continuous safety metric that replaces
    graze-as-terminal-failure)
  - belief-vs-GT error, path length, final goal distance

Usage:
  python3 scripts/visibility_comparison/analyze_v2_outcomes.py \
      [logs/visibility_comparison/honest_campaign_v2]
"""
import json
import glob
import os
import sys
import statistics
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else "logs/visibility_comparison/honest_campaign_v2"
COND_ORDER = ["C0", "C1", "C2"]
COND_LABEL = {"C0": "C0 geom-only", "C1": "C1 const-R", "C2": "C2 learned-R"}
# descriptive task names for the paper
TASK_LABEL = {
    "route_apron_to_a3_mid": "apron-A3",
    "route_apron_to_a2_mid": "apron-A2",
    "route_west_to_a1_upper": "west-A1",
    "control_west_to_a1_low": "control",
}
TASK_ORDER = list(TASK_LABEL)


def _num(d, k):
    v = d.get(k)
    return float(v) if isinstance(v, (int, float)) else None


def load_runs(root):
    """Return {(task,cond,seed): summary_dict}, keeping the latest valid run per cell."""
    runs = {}
    for rs in glob.glob(f"{root}/**/run_summary.json", recursive=True):
        parts = rs.split(os.sep)
        try:
            i = parts.index(os.path.basename(root))
            task, cond = parts[i + 1], parts[i + 2]
            seed = next(p for p in parts if p.startswith("seed"))
        except (ValueError, StopIteration):
            continue
        if cond not in COND_ORDER:
            continue
        key = (task, cond, seed)
        # keep the run with the latest experiment_* dir name (dedup any re-runs)
        prev = runs.get(key)
        if prev is None or rs > prev[0]:
            runs[key] = (rs, json.load(open(rs)))
    return {k: v[1] for k, v in runs.items()}


def min_clearance(d):
    vals = [x for x in (_num(d, "min_wall_distance_m"), _num(d, "min_obstacle_distance_m"))
            if x is not None]
    return min(vals) if vals else None


def main():
    runs = load_runs(ROOT)
    if not runs:
        print(f"No run_summary.json found under {ROOT}")
        return
    # per (task, cond) aggregation
    agg = defaultdict(list)
    for (task, cond, seed), d in runs.items():
        agg[(task, cond)].append(d)

    print(f"\n=== honest_campaign_v2 outcomes  ({len(runs)} runs under {ROOT}) ===\n")
    hdr = f"{'task':10s} {'cond':13s} {'n':>2s} {'goal':>5s} {'contact':>7s} {'graze':>5s} {'exit':>4s} {'minClr_m':>9s} {'beliefGT':>8s} {'valid':>5s}"
    print(hdr)
    print("-" * len(hdr))
    totals = defaultdict(lambda: defaultdict(int))
    clr_by_cond = defaultdict(list)
    for task in TASK_ORDER:
        for cond in COND_ORDER:
            ds = agg.get((task, cond), [])
            if not ds:
                continue
            n = len(ds)
            goal = sum(1 for d in ds if d.get("goal_region_success"))
            contact = sum(1 for d in ds if d.get("collision_contact"))
            graze = sum(1 for d in ds if d.get("collision_geom"))
            exit_ = sum(1 for d in ds if d.get("inside_no_go"))
            valid = sum(1 for d in ds if d.get("valid_run"))
            clrs = [c for c in (min_clearance(d) for d in ds) if c is not None]
            clr_by_cond[cond].extend(clrs)
            min_clr = min(clrs) if clrs else float("nan")
            be = [x for x in (_num(d, "mean_belief_error_gt_after_first_cmd_m") or
                              _num(d, "mean_belief_error_gt_m") for d in ds) if x is not None]
            be_mean = statistics.mean(be) if be else float("nan")
            print(f"{TASK_LABEL[task]:10s} {COND_LABEL[cond]:13s} {n:>2d} "
                  f"{goal:>2d}/{n} {contact:>5d}   {graze:>5d} {exit_:>4d} "
                  f"{min_clr:>9.3f} {be_mean:>8.3f} {valid:>3d}/{n}")
            for k, v in dict(n=n, goal=goal, contact=contact, graze=graze,
                             exit=exit_, valid=valid).items():
                totals[cond][k] += v
        print()

    print("=== per-condition totals ===")
    for cond in COND_ORDER:
        t = totals[cond]
        clrs = sorted(clr_by_cond[cond])
        p5 = clrs[max(0, int(0.05 * len(clrs)) - 0)] if clrs else float("nan")
        print(f"{COND_LABEL[cond]:14s}: goal {t['goal']}/{t['n']}  "
              f"physical-contact {t['contact']}/{t['n']}  graze {t['graze']}/{t['n']}  "
              f"region-exit {t['exit']}/{t['n']}  valid {t['valid']}/{t['n']}  "
              f"| min-clear over all runs = {min(clrs) if clrs else float('nan'):.3f} m, "
              f"p5 = {p5:.3f} m")


if __name__ == "__main__":
    main()
