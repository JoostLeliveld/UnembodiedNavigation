#!/usr/bin/env python3
"""The cleanest single-run failure story: EKF update cycle working -> then failing.

Panel A (zoom, a STRAIGHT segment): belief error sawtooth -- it grows while the
belief dead-reckons, and drops each time a camera correction is ACCEPTED. This is
the predict/update cycle working: the filter is sound at rest.

Panel B (whole run): the same machinery in turns. Hard turns shaded; accepted
corrections (green) vs rejected (red x). In a turn the 1.2 s-stale correction
cannot keep up, the gap opens, the gate then REJECTS the recovery (red) -> runaway
-> collision (vertical line).

Usage: python plot_failure_story.py <run_experiment.csv> [out.png]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

f = sys.argv[1]
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f).with_name("failure_story.png")
d = pd.read_csv(f, low_memory=False)
def num(c): return pd.to_numeric(d.get(c), errors="coerce")
t = num("stamp"); be = num("belief_error_odom_m"); w = num("cmd_w").abs()
acc = num("pixel_corr_accepted"); age = num("planner_pixel_correction_age_s")
cr = num("first_crash_stamp").dropna(); ct = float(cr.iloc[0]) if len(cr) else None
t0 = t.iloc[0]; T = t - t0
end = (ct - t0) if ct else T.max()

# correction events
ev = acc.notna()
acc_t, acc_be = T[ev & (acc == 1)], be[ev & (acc == 1)]
rej_t, rej_be = T[ev & (acc == 0)], be[ev & (acc == 0)]

def shade_turns(ax):
    inturn = (w >= 0.7).fillna(False).values
    tt = T.values
    s = None
    for i, v in enumerate(inturn):
        if v and s is None: s = tt[i]
        if (not v or i == len(inturn) - 1) and s is not None:
            ax.axvspan(s, tt[i], color="orange", alpha=0.12, lw=0)
            s = None

fig, (axA, axB) = plt.subplots(2, 1, figsize=(12, 8.5))

# ---- Panel A: zoom into a clean straight segment (sawtooth) ----
# pick the longest low-turn window in the first 60% of the run
mask = (w < 0.2).fillna(False).values & (T.values < 0.6 * end)
# find longest run of True
best = (0, 0); s = None
for i, v in enumerate(mask):
    if v and s is None: s = i
    if (not v or i == len(mask) - 1) and s is not None:
        if i - s > best[1] - best[0]: best = (s, i)
        s = None
zs, ze = T.values[best[0]], T.values[min(best[1], len(T) - 1)]
zs, ze = zs, min(zs + 14, ze)  # cap window to ~14 s for readability
zm = (T >= zs) & (T <= ze)
axA.plot(T[zm], be[zm], color="#1f77b4", lw=1.5, label="belief-vs-truth error")
za = (acc_t >= zs) & (acc_t <= ze)
axA.scatter(acc_t[za], acc_be[za], s=45, color="#2ca02c", zorder=5,
            label="camera correction ACCEPTED", edgecolor="k", linewidth=.4)
shade_turns(axA)
axA.set_title("A.  The EKF is healthy in straights: belief error stays < 0.06 m and every camera correction is accepted")
axA.set_ylabel("belief error (m)"); axA.set_xlim(zs, ze)
axA.legend(loc="upper right", fontsize=9); axA.grid(alpha=.3)

# ---- Panel B: whole run -> the failure ----
axB.plot(T, be, color="#1f77b4", lw=1.3, label="belief-vs-truth error")
shade_turns(axB)
axB.scatter(acc_t, acc_be, s=22, color="#2ca02c", zorder=4, label="correction accepted")
axB.scatter(rej_t, rej_be, s=55, color="#d62728", marker="x", zorder=5,
            label="correction REJECTED (jump/NIS gate)")
axB.axhline(0.5, ls="--", color="#888", lw=1, label="divergence threshold")
if ct:
    axB.axvline(end, color="k", lw=2)
    axB.text(end, axB.get_ylim()[1] * .9, " COLLISION", fontweight="bold", va="top")
axB.fill_between([0, 0], [0], [0], color="orange", alpha=.12, label="hard turn (|cmd_w|>=0.7)")
axB.set_title("B.  The SAME machinery in turns: stale correction can't keep up -> gap opens -> gate rejects the recovery -> runaway -> crash")
axB.set_ylabel("belief error (m)"); axB.set_xlabel("time (s)")
axB.set_xlim(0, end * 1.02); axB.legend(loc="upper left", fontsize=9); axB.grid(alpha=.3)

tag = "/".join(Path(f).parts[-5:-2])
fig.suptitle(f"How the belief update works, and why the run fails — {tag}",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(out, dpi=130)
print("wrote", out)
print(f"  zoom window {zs:.1f}-{ze:.1f}s | crash at {end:.1f}s | "
      f"{len(rej_t)} rejected, {len(acc_t)} accepted corrections")
