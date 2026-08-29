#!/usr/bin/env python3
"""Live / post-hoc campaign monitor + triage. Given a campaign log-root, report
per-run status and FLAG anomalies so a bad run can be caught and re-run before
the whole campaign is wasted. Safe to call repeatedly while the campaign runs.

Per run it reports: did the global solve produce a plan, the chosen route, the
first-command sim stamp, the execution outcome (goal / collision / stuck /
interrupted), validity, the post-first-command belief error against GROUND TRUTH, the
fraction of corrections the filter refused and the longest stretch with no usable
correction. It then prints per-condition counts and an ANOMALY list:
  * compute-fail   : ran but never produced a plan/command (the dominant past
                     failure; if many, the machine is contended -> restart fresh)
  * route-mismatch : online chosen route != the offline expected route (needs
                     an offline route-sanity artifact present; skipped when absent)
  * invalid        : run_summary.valid_run is False for a non-outcome reason

This is a triage view, never evidence. Scoring goes through
experiments/fusion_on_fixed_routes/aligned.py against a frozen run manifest.

Usage:
    python monitor_campaign.py logs/visibility_comparison/robustness_campaign_v2
    python monitor_campaign.py <log-root> --expect 5   # expected seeds/condition
"""
import sys, json, glob, os, math, argparse
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OFFLINE = ROOT / "logs/paper_figures/offline_plan_sanity.json"


def _num(x, nd=2):
    try:
        if x is None:
            return None
        f = float(x)
        return None if math.isnan(f) else round(f, nd)
    except (TypeError, ValueError):
        return None


def _offline_routes():
    if not OFFLINE.exists():
        return {}
    d = json.loads(OFFLINE.read_text())
    return {(name, c): d[name][c]["route"] for name in d for c in ("C1", "C2")}


def collect(log_root: Path):
    off = _offline_routes()
    rows = []
    for rs_p in sorted(glob.glob(str(log_root / "*/*/seed*/*/run_summary.json"))):
        run = os.path.dirname(rs_p)
        parts = Path(rs_p).relative_to(log_root).parts
        task, cond, seed = parts[0], parts[1], parts[2].replace("seed", "")
        rs = json.loads(Path(rs_p).read_text())
        meta_p = os.path.join(run, "global_plan_meta.json")
        meta = json.loads(Path(meta_p).read_text()) if os.path.exists(meta_p) else {}
        route = str(meta.get("selected_source", "")).replace("solver:route:", "").replace("solver:", "")
        rows.append(dict(
            task=task.split("_")[0], cond=cond, seed=seed,
            has_plan=bool(meta), route=route, off=off.get((task, cond), "?"),
            fcmd=_num(rs.get("first_cmd_stamp"), 1),
            completion=(rs.get("completion_reason") or "")[:18],
            crashed=bool(rs.get("crashed")), collision=bool(rs.get("collision_any")),
            fgoal=_num(rs.get("final_goal_distance")),
            valid=rs.get("valid_run"), invalid=(rs.get("invalid_reason") or "")[:18],
            # Against GROUND TRUTH. The odometry column next to it in run_summary is
            # belief-vs-wheel-odometry, which is not an error at all -- it read 0.297 m
            # where thetrue error was 0.032 m, a ninefold overstatement, under a heading
            # that said "truth".
            gt_err=_num(rs.get("mean_belief_error_gt_after_first_cmd_m"), 3),
            drop_frac=_num(rs.get("correction_dropped_fraction"), 3),
            blind_s=_num(rs.get("longest_correction_gap_s"), 1),
        ))
    return rows, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_root", type=str)
    ap.add_argument("--expect", type=int, default=5, help="expected seeds per task/condition")
    args = ap.parse_args()
    log_root = Path(args.log_root)
    if not log_root.is_absolute():
        log_root = ROOT / log_root
    rows, off = collect(log_root)
    if not rows:
        print(f"No runs found yet under {log_root}"); return

    hdr = (f"{'tsk':4} {'c':3} {'s':1} {'plan':4} {'route':16} {'fcmd':5} "
           f"{'completion':18} {'col':3} {'fgl':5} {'val':3} {'gt_err_m':8} "
           f"{'refused':7} {'blind_s':7}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        rm = "" if (r["off"] == "?" or not r["has_plan"]) else ("" if r["route"] == r["off"] else " <-MISMATCH")
        print(f"{r['task'][:4]:4} {r['cond']:3} {r['seed']:1} {str(r['has_plan'])[0]:4} "
              f"{r['route'][:16]:16} {str(r['fcmd']):5} {r['completion']:18} "
              f"{str(r['collision'])[0]:3} {str(r['fgoal']):5} "
              f"{str(r['valid'])[0] if r['valid'] is not None else '?':3} "
              f"{str(r['gt_err']):8} {str(r['drop_frac']):7} {str(r['blind_s']):7}{rm}")

    # ---- per-condition counts, whatever conditions this campaign actually has ----
    print()
    for c in sorted({r["cond"] for r in rows}):
        rs = [r for r in rows if r["cond"] == c]
        goal = sum(1 for r in rs if r["completion"].startswith("goal_reached"))
        coll = sum(1 for r in rs if r["collision"])
        print(f"  {c}: {len(rs)} runs | goal_reached {goal} | collisions {coll}")

    # ---- anomalies ----
    def _id(r, suffix=""):
        return "{}/{}/s{}{}".format(r["task"], r["cond"], r["seed"], suffix)
    compute_fail = [r for r in rows if not r["has_plan"]]
    mismatch = [r for r in rows if r["has_plan"] and r["off"] != "?" and r["route"] != r["off"]]
    invalid = [r for r in rows if r["valid"] is False]
    cf_ids = ", ".join(_id(r) for r in compute_fail)
    mm_ids = ", ".join(_id(r, "=" + r["route"]) for r in mismatch)
    iv_ids = ", ".join(_id(r, ":" + r["invalid"]) for r in invalid)
    print("\nANOMALIES:")
    print(f"  compute-fail (no plan/command): {len(compute_fail)}" + (f"  -> {cf_ids}" if cf_ids else ""))
    print(f"  route-mismatch vs offline:      {len(mismatch)}" + (f"  -> {mm_ids}" if mm_ids else ""))
    print(f"  invalid (non-outcome):          {len(invalid)}" + (f"  -> {iv_ids}" if iv_ids else ""))
    if not OFFLINE.exists():
        print("  (no offline route artifact -> route-mismatch check skipped)")

    # ---- progress vs expected ----
    per = defaultdict(int)
    for r in rows:
        per[(r["task"], r["cond"])] += 1
    print(f"\nPROGRESS: {len(rows)} runs logged; per task/condition seed counts:")
    for k in sorted(per):
        flag = "" if per[k] >= args.expect else f"  (expect {args.expect})"
        print(f"  {k[0]:6} {k[1]}: {per[k]}{flag}")

    if compute_fail:
        print("\nTRIAGE: compute-fails present. If just 1-2 and the rest are clean, simply re-run those "
              "seeds (re-launch with --resume). If many, the machine is contended -> stop, reboot fresh "
              "(only VS Code), relaunch. Plans themselves are fine when produced (see route column).")


if __name__ == "__main__":
    main()
