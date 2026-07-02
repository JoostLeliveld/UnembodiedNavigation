#!/usr/bin/env python3
"""Definitive, campaign-wide proof of the current limitation and why C2 != 100%.

Builds the full causal chain from the 40-run keepin_clean campaign's own
per-timestep logs (experiment.csv), trimming post-crash samples:

  L1 hardware floor : the camera correction is ~1.2 s stale (ros_gz bridge),
                      identical for C1 and C2 and invariant to tuning.
  L2 turn coupling  : a fast turn during that 1.2 s window dead-reckons the pose,
                      opening a belief-vs-truth gap (pooled, monotone in |w|).
  L3 gate runaway   : once the gap exceeds the jump/NIS gate, the camera
                      correction that would HEAL it is rejected (jump_too_large /
                      nis_too_large) -> belief keeps dead-reckoning -> diverges.
  L4 collision      : robot steers on a 0.5-3 m-wrong pose -> clips the rack.
                      All 18 collisions are preceded by a ~14x belief excursion.
  Why C2 : its correct visibility-aware detours execute MORE total turning, so it
           enters the diverged/runaway regime ~8x more often -> 11 vs 7 collisions.
           Per-turn mechanism is identical; C2 is punished for doing the right
           thing on a machine whose perception cannot keep up in turns.

Outputs: figures/why_c2_not_100.png  +  why_c2_not_100_stats.json  in the campaign dir.
Usage:   python prove_c2_limit.py [campaign_dir]
"""
import sys, json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(sys.argv[1] if len(sys.argv) > 1
            else "logs/visibility_comparison/robustness_keepin_clean_20260619")
TURN = 0.7
PRECRASH = 3.0
DIV = 0.5     # belief-error threshold for "diverged"

def num(d, c):
    return pd.to_numeric(d.get(c), errors="coerce") if c in d else pd.Series(np.nan, index=d.index)

samp = []   # per-timestep, pre-crash
runs = []
for f in sorted(glob.glob(str(ROOT / "*/C*/seed*/experiment_*/experiment.csv"))):
    p = Path(f).parts
    route, cond, seed = p[-5], p[-4], p[-3]
    d = pd.read_csv(f, low_memory=False)
    if len(d) < 20:
        continue
    t = num(d, "stamp"); be = num(d, "belief_error_odom_m"); w = num(d, "cmd_w").abs()
    age = num(d, "planner_pixel_correction_age_s"); acc = num(d, "pixel_corr_accepted")
    coll = num(d, "collision_any").fillna(0).max() > 0.5
    cr = num(d, "first_crash_stamp").dropna(); crash_t = float(cr.iloc[0]) if len(cr) else np.inf
    dt = float(np.median(np.diff(t.values[:60])))
    keep = (t <= crash_t) & be.notna() & w.notna()
    sub = pd.DataFrame(dict(cond=cond, route=route, be=be, w=w, age=age, acc=acc))[keep.values]
    samp.append(sub)
    pre = np.nan
    if coll and np.isfinite(crash_t):
        m = (t >= crash_t - PRECRASH) & (t <= crash_t) & be.notna()
        pre = float(be[m].max()) if m.any() else np.nan
    runs.append(dict(route=route, cond=cond, seed=seed,
                     collision=int(coll), be_med=float(be[keep].median()),
                     pre_crash_be=pre, age_med=float(age.median()),
                     total_turn_s=float((w[keep] >= TURN).sum()) * dt))

S = pd.concat(samp, ignore_index=True)
R = pd.DataFrame(runs)

# -------- stats --------
def acc_in(cond, lo, hi):
    m = (S.cond == cond) & S.acc.notna() & (S.be >= lo) & (S.be < hi)
    return float(S.acc[m].mean()), int(m.sum())

stats = {
    "n_runs": int(len(R)),
    "straight_be_med_m": float(S[S.w < 0.2].be.median()),
    "turn_be_med_m": float(S[S.w >= TURN].be.median()),
    "turn_be_p95_m": float(S[S.w >= TURN].be.quantile(0.95)),
    "collisions": {c: int(((R.cond == c) & (R.collision == 1)).sum()) for c in ["C1", "C2"]},
    "precrash_be_med_m": float(R[R.collision == 1].pre_crash_be.median()),
    "precrash_ratio_med": float((R[R.collision == 1].pre_crash_be / R[R.collision == 1].be_med).median()),
    "age_med_s": {c: float(R[R.cond == c].age_med.median()) for c in ["C1", "C2"]},
    "total_turn_s_med": {c: float(R[R.cond == c].total_turn_s.median()) for c in ["C1", "C2"]},
    "gate_accept_goodbelief": {c: acc_in(c, 0.0, 0.15) for c in ["C1", "C2"]},
    "gate_accept_diverged": {c: acc_in(c, DIV, 1e9) for c in ["C1", "C2"]},
    "diverged_turn_samples": {c: int(((S.cond == c) & (S.w >= TURN) & (S.be > DIV)).sum())
                              for c in ["C1", "C2"]},
}
(ROOT / "why_c2_not_100_stats.json").write_text(json.dumps(stats, indent=2))

# -------- figure --------
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
C = {"C1": "#1f77b4", "C2": "#d62728"}

# A: belief error vs turn rate (pooled, both conditions)
bins = [0.0, 0.2, 0.5, 0.7, 0.9, 1.01]
labels = ["0-0.2", "0.2-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.0"]
S["wb"] = pd.cut(S.w, bins, right=False)
med = S.groupby("wb").be.median().values
p95 = S.groupby("wb").be.quantile(0.95).values
x = np.arange(len(labels))
ax[0, 0].bar(x, med, color="#888", label="median")
ax[0, 0].plot(x, p95, "o-", color="#d62728", label="p95")
ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels(labels)
ax[0, 0].set_xlabel("|cmd_w|  (rad/s)"); ax[0, 0].set_ylabel("belief-vs-truth error (m)")
ax[0, 0].set_title("L1+L2  belief error explodes in turns\n(40 runs, pooled, pre-crash)")
ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)

# B: collision onset — run-median vs pre-crash belief error
Cc = R[R.collision == 1]
xb = np.arange(len(Cc))
ax[0, 1].bar(xb - .2, Cc.be_med, .4, label="run-median", color="#888")
ax[0, 1].bar(xb + .2, Cc.pre_crash_be, .4, label="3 s pre-crash (max)",
             color=[C[c] for c in Cc.cond])
ax[0, 1].set_xlabel("collision run (n=%d)" % len(Cc)); ax[0, 1].set_ylabel("belief error (m)")
ax[0, 1].set_title("L4  every collision is preceded by a\nbelief excursion (~%.0fx baseline)"
                   % stats["precrash_ratio_med"])
ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)

# C: NIS/jump-gate runaway — acceptance vs belief regime
regimes = [("good\n<0.15 m", 0.0, 0.15), ("0.15-0.5 m", 0.15, 0.5), ("diverged\n>0.5 m", 0.5, 1e9)]
xr = np.arange(len(regimes))
for c in ["C1", "C2"]:
    acc = [acc_in(c, lo, hi)[0] for _, lo, hi in regimes]
    ax[1, 0].plot(xr, acc, "o-", color=C[c], label=c, lw=2)
ax[1, 0].set_xticks(xr); ax[1, 0].set_xticklabels([r[0] for r in regimes])
ax[1, 0].set_xlabel("belief error regime"); ax[1, 0].set_ylabel("camera correction accept rate")
ax[1, 0].set_title("L3  the gate REJECTS the recovery once\nbelief diverges (runaway)")
ax[1, 0].set_ylim(0, 1.05); ax[1, 0].legend(); ax[1, 0].grid(alpha=.3)

# D: why C2 — more turning -> more diverged-regime time -> more collisions
mets = ["total turn (s)", "diverged-in-turn\nsamples /100", "collisions"]
v1 = [stats["total_turn_s_med"]["C1"], stats["diverged_turn_samples"]["C1"] / 100, stats["collisions"]["C1"]]
v2 = [stats["total_turn_s_med"]["C2"], stats["diverged_turn_samples"]["C2"] / 100, stats["collisions"]["C2"]]
xm = np.arange(len(mets))
ax[1, 1].bar(xm - .2, v1, .4, color=C["C1"], label="C1")
ax[1, 1].bar(xm + .2, v2, .4, color=C["C2"], label="C2")
ax[1, 1].set_xticks(xm); ax[1, 1].set_xticklabels(mets)
ax[1, 1].set_title("Why C2 != 100%%: same mechanism, MORE exposure\n(correction age identical: %.2f vs %.2f s)"
                   % (stats["age_med_s"]["C1"], stats["age_med_s"]["C2"]))
ax[1, 1].legend(); ax[1, 1].grid(alpha=.3)

fig.suptitle("Why C2 is not 100%: hardware-bound stale-belief-in-turns + gate runaway "
             "(keepin_clean, 40 runs)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = ROOT / "figures" / "why_c2_not_100.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=130)
print("wrote", out)
print(json.dumps(stats, indent=2))
