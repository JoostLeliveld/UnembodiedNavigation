"""The route every fusion arm drives, and the camera support it will meet.

The fusion experiment fixes ONE route, so the only thing that differs between arms is how
the cameras' readings are combined. This figure shows that route before any arm is named:
where it goes, which cameras are expected to watch it, and how many watch at once.

The task is `fusion_network_traverse` in `src/experiments/config/tasks.yaml` -- the west dock
door, three metres from camera A, to the east cross aisle under camera D. Starting under
camera A is deliberate: it is the only start that lets all five cameras contribute to one
traverse. The candidate corridors
come from the lane geometry alone (`route_tasks.py`), never from a support field. Camera
support is the commissioned per-camera usable-sighting rate, read from
`logs/studies/measurement_commissioning/`.

No simulator, ~1 minute. Writes logs/studies/deck_figures/fusion/01_the_route.png.
"""
import json
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

TASK = ("dock_w", "xaisle_e")          # tasks.yaml: fusion_network_traverse
P_USABLE = 0.25                        # a camera counts as watching here at this rate or better
NEAR_M = 0.75                          # a route step reads the commissioned cell within this radius
OUT = D.REPO / "logs/studies/deck_figures/fusion"
OUT.mkdir(parents=True, exist_ok=True)

lay = D.layout()
fields = src.per_camera_fields()
keys = {c: np.array(sorted(f)) for c, f in fields.items()}
vals = {c: np.array([fields[c][tuple(k)] for k in keys[c]]) for c in fields}
CAMS = sorted(fields)

# The DRIVEN route is read from the frozen artifact, never regenerated here: the arms execute
# that polyline, and a figure that redrew it from a different erosion would show a route
# nobody drove. The other corridors are regenerated at the same erosion, for context only.
FROZEN = json.loads(
    (D.REPO / "experiments/fusion_on_fixed_routes/route_fusion_network_traverse.json").read_text())
driven_poly = json.loads(FROZEN["polyline_canonical_json"])
RT.CLEARANCE_M = float(FROZEN["grid_erosion_m"])
xs, ys, mask, _ = RT.driveable()
ia = RT.snap(xs, ys, mask, RT.WAYPOINTS[TASK[0]])
ib = RT.snap(xs, ys, mask, RT.WAYPOINTS[TASK[1]])
routes = RT.diverse_routes(mask, ia, ib, xs, ys)
routes = [driven_poly] + [r for i, r in enumerate(routes) if i != FROZEN["candidate_index"]]


def analyse(poly):
    t, P = BR.resample(poly)
    step = float(t[1] - t[0])
    percam = {}
    for c in CAMS:
        d = np.linalg.norm(P[:, None, :] - keys[c][None, :, :], axis=2)
        j = np.argmin(d, axis=1)
        percam[c] = np.where(d[np.arange(len(P)), j] < NEAR_M, vals[c][j], 0.0)
    on = np.stack([percam[c] >= P_USABLE for c in CAMS])
    n = on.sum(axis=0)
    run, worst = 0.0, 0.0
    for k in n:
        run = 0.0 if k >= 1 else run + step
        worst = max(worst, run)
    return dict(t=t, P=P, percam=percam, on=on, n=n, step=step, L=float(t[-1]),
                blind_m=float(worst), frac0=float(np.mean(n == 0)),
                frac2=float(np.mean(n >= 2)), frac3=float(np.mean(n >= 3)))


res = [analyse(r) for r in routes]
order = np.argsort([r["L"] for r in res])
driven = 0                      # index 0 is the frozen polyline every arm executes
A = res[driven]
for i, r in enumerate(res):
    print(f"  candidate {i}: {r['L']:5.1f} m | 2+ cameras {r['frac2']*100:3.0f}% | "
          f"none {r['frac0']*100:3.0f}% | worst blind {r['blind_m']:4.1f} m"
          f"{'   [DRIVEN]' if i == driven else ''}")
regime = {k: float(np.sum(A["n"] == k) * A["step"]) for k in range(int(A["n"].max()) + 1)}
print("  metres at each camera count:",
      " ".join(f"{k}:{v:.1f}m" for k, v in regime.items()))

fig = plt.figure(figsize=(16.0, 8.0), constrained_layout=True)
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05], height_ratios=[1, 1])

# ---- the route on the floor -------------------------------------------------
ax = fig.add_subplot(gs[:, 0])
D.draw_warehouse(ax, lay)
for i, r in enumerate(res):
    if i == driven:
        continue
    ax.plot(r["P"][:, 0], r["P"][:, 1], color=D.MUTED, lw=2.2, ls=(0, (5, 4)), zorder=6,
            solid_capstyle="round")
ax.plot(A["P"][:, 0], A["P"][:, 1], color="white", lw=10.0, zorder=7, solid_capstyle="round")
ax.plot(A["P"][:, 0], A["P"][:, 1], color=D.ROBOT, lw=5.5, zorder=8, solid_capstyle="round",
        label=f"the driven route — {A['L']:.0f} m, identical for every arm")
ax.plot([], [], color=D.MUTED, lw=2.2, ls=(0, (5, 4)),
        label=f"the {len(res)-1} other corridors the lane geometry offers — not driven here")
sx, sy = A["P"][0]; gx, gy = A["P"][-1]
ax.plot(sx, sy, "o", ms=16, color=D.INK, zorder=11)
ax.text(sx, sy - 1.35, "start", ha="center", fontsize=13.5, color=D.INK, zorder=11)
ax.plot(gx, gy, "*", ms=26, color=D.INK, zorder=11)
ax.text(gx, gy + 1.15, "goal", ha="center", fontsize=13.5, color=D.INK, zorder=11)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.01), fontsize=12.5, frameon=False)
ax.set_title("One route, driven the same way by every arm", loc="left",
             fontsize=20, color=D.INK, pad=8)

# ---- who is watching --------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
for ci, c in enumerate(CAMS):
    on = A["on"][ci]
    ax.fill_between(A["t"], ci * 0.9, ci * 0.9 + 0.72, where=on, step="mid",
                    color=D.CAM_COLOUR[c[-1]], lw=0)
    metres = float(on.sum() * A["step"])
    ax.text(-0.4, ci * 0.9 + 0.36, f"camera {c[-1]}", ha="right", va="center",
            fontsize=12.5, color=D.CAM_COLOUR[c[-1]], fontweight="bold")
    ax.text(A["L"] + 0.4, ci * 0.9 + 0.36, f"{metres:.0f} m", ha="left", va="center",
            fontsize=11.5, color=D.INK2)
ax.set_xlim(-3.6, A["L"] + 2.4); ax.set_ylim(-0.35, len(CAMS) * 0.9)
ax.set_yticks([]); ax.set_xlabel("distance along the route (m)", fontsize=12)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Which cameras are expected to see the robot, and where they hand over",
             loc="left", fontsize=15, color=D.INK)

# ---- how many at once -------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
top = int(A["n"].max())
ax.fill_between(A["t"], 0, np.where(A["n"] == 0, top + 0.5, 0), step="mid",
                color=D.BAD, alpha=0.18, lw=0)
ax.fill_between(A["t"], 0, A["n"], step="mid", color=D.GOOD, alpha=0.22, lw=0)
ax.step(A["t"], A["n"], where="mid", color=D.GOOD, lw=3.0)
ax.axhline(2, color=D.INK2, lw=1.0, ls=(0, (4, 3)))
ax.text(A["L"] * 0.5, 2.12, "two cameras", ha="center", va="bottom",
        fontsize=11, color=D.INK2)
for k, m in regime.items():
    ax.text(A["L"] + 0.35, k + 0.16, f"{m:.1f} m", ha="left", va="center",
            fontsize=11, color=D.BAD if k == 0 else D.INK2)
ax.set_xlim(0, A["L"] + 2.2); ax.set_ylim(0, top + 0.55)
ax.set_yticks(range(0, top + 1))
ax.set_xlabel("distance along the route (m)", fontsize=12)
ax.set_ylabel("cameras expected to give a\nusable sighting at the same time", fontsize=12)
ax.grid(True, color="#e8e7e2", lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title(f"The route spends {regime.get(1, 0):.0f} m with one camera, "
             f"{A['frac2']*100:.0f}% of the way with two or more, "
             f"and {regime.get(0, 0):.1f} m with none",
             loc="left", fontsize=15, color=D.INK)

others = ", ".join(sorted("%.1f m" % res[i]["L"] for i in range(1, len(res))))
fig.text(0.008, -0.012,
         "Route: the task fusion_network_traverse — the west dock door, 3 m from camera A, to the east cross\n"
         f"aisle under camera D, {A['L']:.1f} m, hash-bound in route_fusion_network_traverse.json; the other {len(res)-1}\n"
         f"corridors the lane geometry offers are {others}.\n"
         "Candidates come from the lane geometry alone; no availability or uncertainty field is read.\n"
         "Support is the commissioned usable-sighting rate per camera, 386 floor positions x 6 headings x 5 cameras;\n"
         f"a camera counts as watching where that rate is {P_USABLE:.0%} or better. Longest single stretch with no\n"
         f"camera: {A['blind_m']:.1f} m, out of {regime.get(0, 0):.1f} m with none in total.\n"
         "This is what the commissioned model expects the route to meet — not a recorded drive.",
         fontsize=10.5, color=D.INK2, va="top", linespacing=1.5)

fig.savefig(OUT / "01_the_route.png", dpi=170, bbox_inches="tight")
print(f"wrote {OUT/'01_the_route.png'}")
