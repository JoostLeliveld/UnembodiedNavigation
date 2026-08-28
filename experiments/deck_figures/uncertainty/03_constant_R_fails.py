"""WHY 3: what one constant covariance actually does."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, matplotlib.pyplot as plt
from _common import ladder, OUT
import style as D

L = ladder()["constant_by_distance"]
bands = [b["band"] for b in L]
cov = [b["constant"]["coverage_95"] * 100 for b in L]
nis = [b["constant"]["normalised_squared_error"] for b in L]
fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6), constrained_layout=True)
xs = np.arange(len(bands))
ax = axes[0]
cols = [D.BAD if c < 90 else (D.GOOD if c < 98 else "#c98500") for c in cov]
ax.bar(xs, cov, color=cols, width=0.62, edgecolor="white", lw=2)
ax.axhline(95, color=D.INK, lw=2.2, ls=(0, (5, 4)))
ax.text(len(bands) - 0.4, 96, "what it promises: 95%", ha="right", fontsize=12, color=D.INK)
for x, c in zip(xs, cov): ax.text(x, c + 1.4, f"{c:.0f}%", ha="center", fontsize=13, fontweight="bold", color=D.INK)
ax.set_xticks(xs); ax.set_xticklabels(bands, fontsize=11.5)
ax.set_ylabel("how often the robot is really inside\nthe stated 95% ellipse", fontsize=12.5)
ax.set_ylim(0, 108); ax.grid(True, axis="y", color="#e8e7e2", lw=0.7); ax.set_axisbelow(True)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.set_title("Too cautious up close, dangerously sure far away", loc="left", fontsize=15.5, color=D.INK, pad=28)
ax.text(0, 1.005, "it promises 95% everywhere and delivers 100% near, 61% far",
        transform=ax.transAxes, fontsize=12, color=D.INK2, va="bottom")
ax = axes[1]
ax.bar(xs, nis, color=[D.BAD if v > 3 else (D.GOOD if v > 1.4 else "#c98500") for v in nis],
       width=0.62, edgecolor="white", lw=2)
ax.axhline(2.0, color=D.INK, lw=2.2, ls=(0, (5, 4)))
ax.text(len(bands) - 0.4, 2.25, "an honest covariance sits here", ha="right", fontsize=12, color=D.INK)
for x, v in zip(xs, nis): ax.text(x, v + 0.2, f"{v:.1f}", ha="center", fontsize=13, fontweight="bold", color=D.INK)
ax.set_xticks(xs); ax.set_xticklabels(bands, fontsize=11.5)
ax.set_ylabel("stated uncertainty vs the error actually seen", fontsize=12.5)
ax.grid(True, axis="y", color="#e8e7e2", lw=0.7); ax.set_axisbelow(True)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.set_title("Eight times too large near, four times too small far", loc="left", fontsize=15.5, color=D.INK, pad=28)
ax.text(0, 1.005, "below the line the filter ignores good corrections; above it, it trusts bad ones",
        transform=ax.transAxes, fontsize=12, color=D.INK2, va="bottom")
fig.suptitle("What happens if you measure the error once, in centimetres, and use it everywhere",
             x=0.006, ha="left", fontsize=18, color=D.INK)
fig.savefig(OUT / "03_constant_R_fails.png", dpi=180, bbox_inches="tight")
print("03:", " ".join(f"{b}={c:.0f}%" for b, c in zip(bands, cov)))
