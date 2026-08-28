"""SOLUTION 3: nothing more elaborate beats it."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, matplotlib.pyplot as plt
from _common import ladder, OUT
import style as D

L = ladder()["ladder"]
names = [m["model"] for m in L]; npar = [m["parameters"] for m in L]
sur = [m["negative_log_likelihood"] for m in L]
cov = [m["coverage_95"] * 100 for m in L]
# adopt the simplest model within a small margin of the best -- the rule in PLAN.md.
# Here the top two are 0.004 apart, which is a tie, so the one-number model wins.
MARGIN = 0.05
within = [i for i, s in enumerate(sur) if s <= min(sur) + MARGIN]
best = min(within, key=lambda i: npar[i])
print(f"    within {MARGIN} of the best: " +
      ", ".join(f"{names[i]} ({npar[i]})" for i in within))
fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.0), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1.3, 1]})
ax = axes[0]
ys = np.arange(len(names))[::-1]
cols = [D.GOOD if i == best else D.MUTED for i in range(len(names))]
ax.barh(ys, [-s for s in sur], color=cols, height=0.6, edgecolor="white", lw=2)
for y, s, n, i in zip(ys, sur, npar, range(len(names))):
    ax.text(-s + 0.04, y, f"{n} number{'s' if n != 1 else ''}", va="center", fontsize=12,
            color=D.GOOD if i == best else D.INK2,
            fontweight="bold" if i == best else "normal")
ax.set_yticks(ys); ax.set_yticklabels(names, fontsize=12.5)
ax.set_xlim(4.2, 5.95); ax.set_xlabel("how well it predicts the errors actually seen  (further right is better)", fontsize=12)
ax.grid(True, axis="x", color="#e8e7e2", lw=0.7); ax.set_axisbelow(True)
for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
ax.set_title("More parameters do not help", loc="left", fontsize=17, color=D.INK, pad=30)
ax.text(0, 1.005, "fitted on half the floor positions, scored on the other half.  The top two are\n"
        "0.004 apart — a tie — so the simpler one is kept.",
        transform=ax.transAxes, fontsize=12, color=D.INK2, va="bottom")
ax = axes[1]
ax.scatter(npar, cov, s=190, color=[D.GOOD if i == best else D.MUTED for i in range(len(names))],
           edgecolor="white", lw=2, zorder=4)
ax.axhline(95, color=D.INK, lw=2.2, ls=(0, (5, 4)))
ax.text(0.98, 0.90, "what every one of them promises", transform=ax.transAxes, ha="right",
        fontsize=12, color=D.INK)
for n, c, nm, i in zip(npar, cov, names, range(len(names))):
    if i in (best, 0, len(names) - 1):
        ax.annotate(nm.replace(", ", ",\n"), (n, c), textcoords="offset points",
                    xytext=(9, -6), fontsize=11,
                    color=D.GOOD if i == best else D.INK2)
ax.set_xscale("symlog"); ax.set_xlabel("numbers fitted", fontsize=12.5)
ax.set_ylabel("how often the robot is really inside the stated 95% ellipse", fontsize=12.5)
ax.set_ylim(84, 97); ax.grid(True, color="#e8e7e2", lw=0.7); ax.set_axisbelow(True)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.set_title("And the elaborate ones get worse", loc="left", fontsize=17, color=D.INK, pad=30)
ax.text(0, 1.005, "a map fitted over the floor has forty-one numbers and is the least honest of all",
        transform=ax.transAxes, fontsize=12, color=D.INK2, va="bottom")
fig.savefig(OUT / "06_more_parameters_dont_help.png", dpi=175, bbox_inches="tight")
print("06: best is", names[best], "with", npar[best], "number(s)")
