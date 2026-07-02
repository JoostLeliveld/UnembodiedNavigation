#!/usr/bin/env python3
"""Compare two gate configs (baseline vs gate-off) on the same route/seeds.

For each run reports outcome + the mechanism signals: correction age, jump/NIS
rejections, how long the belief stayed diverged, and the worst excursion. The
hypothesis: gate-off ACCEPTS the recovery corrections, so the belief re-locks
faster -> shorter/smaller divergence -> safer.

Usage: python compare_gate.py <baseline_dir> <gateoff_dir>
"""
import sys, glob, json
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

DIV = 0.5

def num(d, c):
    return pd.to_numeric(d.get(c), errors="coerce") if c in d else pd.Series(np.nan, index=d.index)

def run_stats(exp_csv):
    d = pd.read_csv(exp_csv, low_memory=False)
    t = num(d, "stamp"); be = num(d, "belief_error_odom_m"); w = num(d, "cmd_w").abs()
    acc = num(d, "pixel_corr_accepted"); age = num(d, "planner_pixel_correction_age_s")
    reason = d.get("pixel_corr_reject_reason")
    cr = num(d, "first_crash_stamp").dropna(); ct = float(cr.iloc[0]) if len(cr) else np.inf
    keep = (t <= ct) & be.notna()
    dt = float(np.median(np.diff(t.values[:60])))
    diverged = keep & (be > DIV)
    ev = acc.notna() & keep
    njump = int((reason[ev] == "jump_too_large").sum()) if reason is not None else 0
    nnis = int((reason[ev] == "nis_too_large").sum()) if reason is not None else 0
    # accept rate while diverged
    dv = ev & (be > DIV)
    acc_div = float(acc[dv].mean()) if dv.sum() else np.nan
    return dict(
        be_med=float(be[keep].median()),
        be_max=float(be[keep].max()),
        age_med=float(age[keep].median()),
        diverged_time_s=float(diverged.sum()) * dt,
        n_jump_rej=njump, n_nis_rej=nnis,
        acc_rate_diverged=acc_div,
        n_div_samples=int(diverged.sum()),
    )

def load(campaign_dir):
    rows = {}
    for f in sorted(glob.glob(str(Path(campaign_dir) / "**" / "experiment_*" / "experiment.csv"), recursive=True)):
        seed = "seed" + f.split("seed")[1][0]
        s = run_stats(f)
        summ = Path(f).with_name("run_summary.json")
        if summ.exists():
            j = json.load(open(summ))
            s["outcome"] = j.get("completion_reason")
            s["coll"] = bool(j.get("collision_any") or j.get("crashed"))
            s["min_goal"] = float(j.get("minimum_goal_distance", float("nan")))
            s["goal"] = bool(j.get("goal_region_success")) or (
                s["min_goal"] <= float(j.get("goal_success_radius", 0.25)))
        rows[seed] = s
    return rows

base = load(sys.argv[1]); off = load(sys.argv[2])
print(f"{'seed':6} {'config':9} {'outcome':16} {'minGoal':>7} {'be_max':>7} {'age':>5} {'div_s':>6} {'jumpRej':>7} {'nisRej':>6} {'accDiv':>6}")
for seed in sorted(set(base) | set(off)):
    for tag, src in [("BASELINE", base), ("GATE-OFF", off)]:
        r = src.get(seed)
        if not r: continue
        print(f"{seed:6} {tag:9} {str(r.get('outcome')):16} {r.get('min_goal',float('nan')):7.3f} {r['be_max']:7.3f} "
              f"{r['age_med']:5.2f} {r['diverged_time_s']:6.2f} {r['n_jump_rej']:7d} {r['n_nis_rej']:6d} "
              f"{(r['acc_rate_diverged'] if r['acc_rate_diverged']==r['acc_rate_diverged'] else float('nan')):6.2f}")
    print()

def agg(src, key):
    vals = [r[key] for r in src.values() if r.get(key) == r.get(key)]
    return np.median(vals) if vals else float("nan")
print("MEDIANS  baseline -> gate-off")
for k in ["be_max", "diverged_time_s", "n_jump_rej", "acc_rate_diverged"]:
    print(f"  {k:18} {agg(base,k):7.3f} -> {agg(off,k):7.3f}")
bc = sum(1 for r in base.values() if r.get('coll'))
oc = sum(1 for r in off.values() if r.get('coll'))
bg = sum(1 for r in base.values() if r.get('goal'))
og = sum(1 for r in off.values() if r.get('goal'))
print(f"  collisions          {bc} -> {oc}   |   goals  {bg} -> {og}   (n={len(base)})")
