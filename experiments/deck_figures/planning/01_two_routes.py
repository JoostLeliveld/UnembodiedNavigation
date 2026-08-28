"""Slides 1, 2 and 11: two routes to the same goal, and what they cost in camera support."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "warehouse_v2_sketches"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src/experiments"))
import sources as src
import style as D, rollout as BR
import route_tasks as RT

TASK = ("aisleA2_s", "bay_B_e")
OUT = D.REPO / "logs/studies/deck_figures/planning"; OUT.mkdir(parents=True, exist_ok=True)
lay = D.layout(); agg, _ = src.support_field(); fields = src.per_camera_fields()
xs, ys, mask, _ = RT.driveable()
ia = RT.snap(xs, ys, mask, RT.WAYPOINTS[TASK[0]]); ib = RT.snap(xs, ys, mask, RT.WAYPOINTS[TASK[1]])
routes = RT.diverse_routes(mask, ia, ib, xs, ys)
keys = {c: np.array(sorted(f)) for c, f in fields.items()}
vals = {c: np.array([fields[c][tuple(k)] for k in keys[c]]) for c in fields}

def analyse(poly):
    t, P = BR.resample(poly)
    percam = {}
    for c in fields:
        d = np.linalg.norm(P[:, None, :] - keys[c][None, :, :], axis=2)
        j = np.argmin(d, axis=1)
        percam[c] = np.where(d[np.arange(len(P)), j] < 0.75, vals[c][j], 0.0)
    best = np.max(np.stack([percam[c] for c in sorted(fields)]), axis=0)
    gap, run = [], 0.0
    for s in best:
        run = 0.0 if s >= 0.25 else run + 0.2
        gap.append(run)
    return dict(t=t, P=P, percam=percam, best=best, gap=np.array(gap),
                L=float(t[-1]), blind=float(max(gap)),
                blind_frac=float(np.mean(best < 0.25)))

res = [analyse(r) for r in routes]
short = int(np.argmin([r["L"] for r in res]))
best_i = int(np.argmin([r["blind"] for r in res]))
A, B = res[short], res[best_i]
for i, r in enumerate(res):
    print(f"  route {i}: {r['L']:.1f} m, worst blind {r['blind']:.1f} m, {r['blind_frac']*100:.0f}% of the way unseen")

fig = plt.figure(figsize=(16.5, 7.6), constrained_layout=True)
gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1, 1])

ax = fig.add_subplot(gs[:, 0])
D.draw_warehouse(ax, lay); D.draw_support(ax, agg, hatch_zero=True)
for r, col, lab in ((A, D.BAD, f"Route A — shortest, {A['L']:.0f} m"),
                    (B, D.GOOD, f"Route B — best supported, {B['L']:.0f} m")):
    ax.plot(r["P"][:, 0], r["P"][:, 1], color="white", lw=9.5, zorder=6, solid_capstyle="round")
    ax.plot(r["P"][:, 0], r["P"][:, 1], color=col, lw=5.5, zorder=7, solid_capstyle="round", label=lab)
sx, sy = A["P"][0]; gx, gy = A["P"][-1]
ax.plot(sx, sy, "o", ms=17, color=D.INK, zorder=11); ax.text(sx - 0.5, sy - 1.3, "start", ha="center", fontsize=13, color=D.INK, zorder=11)
ax.plot(gx, gy, "*", ms=27, color=D.INK, zorder=11); ax.text(gx, gy + 1.2, "goal", ha="center", fontsize=13, color=D.INK, zorder=11)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.01), fontsize=13, frameon=False, ncol=1)
ax.set_title("Two ways to the same goal", loc="left", fontsize=20, color=D.INK, pad=8)

ax = fig.add_subplot(gs[0, 1])
for row, (r, lab) in enumerate(((A, "Route A"), (B, "Route B"))):
    y = -row * 1.25
    for ci, c in enumerate(sorted(fields)):
        on = r["percam"][c] >= 0.25
        ax.fill_between(r["t"], y + ci * 0.19, y + ci * 0.19 + 0.16, where=on,
                        color=D.CAM_COLOUR[c[-1]], lw=0, step="mid")
    ax.text(-0.9, y + 0.42, lab, ha="right", va="center", fontsize=12.5, color=D.INK)
for ci, c in enumerate(sorted(fields)):
    ax.text(-0.25, -1.25 + ci * 0.19 + 0.08, c[-1], ha="right", va="center",
            fontsize=11, color=D.CAM_COLOUR[c[-1]], fontweight="bold")
ax.set_xlim(-3.2, max(A["L"], B["L"])); ax.set_ylim(-1.45, 1.15)
ax.set_yticks([]); ax.set_xlabel("distance travelled (m)", fontsize=12)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("Which camera is watching, and where it hands over",
             loc="left", fontsize=15, color=D.INK)

ax = fig.add_subplot(gs[1, 1])
for r, col, lab in ((A, D.BAD, "Route A"), (B, D.GOOD, "Route B")):
    ax.fill_between(r["t"], 0, r["gap"], color=col, alpha=0.20, lw=0)
    ax.plot(r["t"], r["gap"], color=col, lw=3.0, label=f"{lab}: worst {r['blind']:.1f} m")
ax.set_xlabel("distance travelled (m)", fontsize=12)
ax.set_ylabel("distance driven since the\nlast usable sighting (m)", fontsize=12)
ax.grid(True, color="#e8e7e2", lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=12, loc="upper left")
ax.set_title("How far the robot drives unseen", loc="left", fontsize=15, color=D.INK)

fig.text(0.012, -0.025,
         f"Route B is {(B['L']/A['L']-1)*100:.0f}% longer and never goes more than {B['blind']:.1f} m unseen, "
         f"against {A['blind']:.1f} m for the shortest route.  Measured from 386 floor positions x 6 headings x 5 cameras "
         f"— nothing here assumes how fast the robot drifts.",
         fontsize=12, color=D.INK2)
fig.savefig(OUT / "01_two_routes.png", dpi=175, bbox_inches="tight")
print("wrote 01_two_routes.png")
