"""The bias correction, in four panels: what we collect, what we see, what it removes, how little it takes."""
import csv, json, math, sys, collections
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sources as src
import style as D
sys.path.insert(0, str(D.REPO / "experiments/measurement_commissioning"))
from camera import camera_models  # noqa: E402
from observation import jacobian  # noqa: E402

DATA = D.REPO / "logs/perception_datasets/warehouse_v2_yolo_shared_20260822"
OUT = D.REPO / "logs/studies/deck_figures/observation"; OUT.mkdir(parents=True, exist_ok=True)
cal = json.loads((D.REPO / "logs/studies/measurement_commissioning/calibration.json").read_text())["calibration"]
CU, CV = cal["coefficients_du"], cal["coefficients_dv"]
cams = camera_models(DATA)

# The spots the correction was fitted on, matched by the id commission.py assigned.
# Never by coordinate: the grid lands on rounding ties and every file formats them
# differently, which has silently corrupted this three times.
bias_ids = {int(r["position_id"])
            for r in csv.DictReader(open(D.REPO / "logs/studies/measurement_commissioning/offset_positions.csv"))}

S = []
for r in csv.DictReader(open(D.REPO / "logs/studies/measurement_commissioning/sightings.csv")):
    x, y, yaw, rng = float(r["x"]), float(r["y"]), float(r["yaw"]), float(r["range_m"])
    Ji = np.linalg.inv(jacobian(cams[r["camera"]], x, y, yaw))
    d = np.array([float(r["du_px"]), float(r["dv_px"])])
    b = np.array([np.polyval(CU[::-1], rng), np.polyval(CV[::-1], rng)])
    S.append(dict(x=x, y=y, rng=rng, du=d[0], dv=d[1], Ji=Ji,
                  raw=(Ji @ d) * 100.0, cor=(Ji @ (d - b)) * 100.0,
                  fit=int(r["position_id"]) in bias_ids))
FIT = [s for s in S if s["fit"]]; HELD = [s for s in S if not s["fit"]]

fig = plt.figure(figsize=(15.5, 11.0), constrained_layout=True)
gs = fig.add_gridspec(2, 2)

# ---- 1. where we park the robot -------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
D.draw_warehouse(ax, D.layout(), camera_labels=True)
spots = sorted({(s["x"], s["y"]) for s in FIT})
allp = sorted({(s["x"], s["y"]) for s in S})
ax.scatter([p[0] for p in allp], [p[1] for p in allp], s=9, color=D.MUTED, alpha=0.45, lw=0, zorder=3)
ax.scatter([p[0] for p in spots], [p[1] for p in spots], s=190, marker="X",
           color=D.BAD, edgecolor="white", lw=2.0, zorder=8)
ax.set_title(f"1.  Park the robot on {len(spots)} marked spots",
             loc="left", fontsize=16, color=D.INK, pad=34)
ax.text(0.0, 1.005, "chosen from a floor plan: in each distance band, the spots\n"
        "the most cameras can see.  Grey = every other placement.",
        transform=ax.transAxes, fontsize=12, color=D.INK2, va="bottom")

# ---- 2. the error against distance, and the curve fitted to it -------------------
ax = fig.add_subplot(gs[0, 1])
rr = np.linspace(2, 24, 200)
for arr, col, lab, mk in ((np.array([[s["rng"], s["dv"]] for s in HELD]), "#c9c8c1", "every other sighting", "."),
                          (np.array([[s["rng"], s["dv"]] for s in FIT]), D.BAD, f"the {len(FIT)} sightings we fit on", "o")):
    ax.scatter(arr[:, 0], arr[:, 1], s=(8 if mk == "." else 42), color=col,
               alpha=(0.30 if mk == "." else 0.95), lw=0, zorder=(2 if mk == "." else 4), label=lab)
ax.plot(rr, np.polyval(CV[::-1], rr), color=D.ROBOT, lw=3.6, zorder=6,
        label="the fitted curve  =  three of the six numbers")
ax.axhline(0, color=D.MUTED, lw=1.2, zorder=1)
ax.set_xlabel("distance from camera to robot (m)", fontsize=13)
ax.set_ylabel("how far the detector's box edge sits\nfrom where it should be  (pixels)", fontsize=13)
ax.set_ylim(-4, 4); ax.grid(True, color="#e8e7e2", lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=11.5, loc="lower left")
ax.set_title("2.  The error is not random, it drifts with distance",
             loc="left", fontsize=16, color=D.INK, pad=34)
ax.text(0.0, 1.005, "one curve per axis; this is the vertical one, and it is\n"
        "three of the six numbers", transform=ax.transAxes, fontsize=12, color=D.INK2, va="bottom")

# ---- 3. what it removes, on positions never used ---------------------------------
ax = fig.add_subplot(gs[1, 0])
edges = np.array([0, 5, 7, 9, 11, 13, 15, 17, 19, 24])
for key, col, lab in (("raw", D.BAD, "before"), ("cor", D.GOOD, "after")):
    xs, ys, es = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = [s for s in HELD if lo <= s["rng"] < hi]
        if len(sub) < 25: continue
        e = np.array([s[key] for s in sub])
        xs.append((lo + hi) / 2); ys.append(np.linalg.norm(e.mean(axis=0)))
        es.append(1.96 * np.linalg.norm(e.std(axis=0) / math.sqrt(len(e))))
    xs, ys, es = np.array(xs), np.array(ys), np.array(es)
    ax.fill_between(xs, ys - es, ys + es, color=col, alpha=0.18, lw=0)
    ax.plot(xs, ys, color=col, lw=3.0, marker="o", ms=8, label=lab)
ax.axhline(0, color=D.MUTED, lw=1.2)
ax.set_xlabel("distance from camera to robot (m)", fontsize=13)
ax.set_ylabel("how far the average sighting\nlands from the truth  (cm)", fontsize=13)
ax.grid(True, color="#e8e7e2", lw=0.6); ax.set_axisbelow(True); ax.set_ylim(bottom=0)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=12.5, loc="upper left")
raw_all = np.linalg.norm(np.array([s["raw"] for s in HELD]).mean(axis=0))
cor_all = np.linalg.norm(np.array([s["cor"] for s in HELD]).mean(axis=0))
ax.set_title(f"3.  What it removes, on {len({(s['x'],s['y']) for s in HELD})} positions it never saw",
             loc="left", fontsize=16, color=D.INK, pad=34)
ax.text(0.0, 1.005, f"overall {raw_all:.2f} cm to {cor_all:.2f} cm, against 2.2 cm of\n"
        f"random scatter.  Shaded = 95% confidence.",
        transform=ax.transAxes, fontsize=12, color=D.INK2, va="bottom")

# ---- 4. how few spots you need ---------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
N = [3, 5, 8, 12, 20, 35, 60, 100]
med = [0.294, 0.233, 0.209, 0.215, 0.217, 0.222, 0.225, 0.226]
hi = [0.96, 0.21, 0.13, 0.09, 0.06, 0.04, 0.02, 0.005]
ax.fill_between(N, med, np.array(med) + np.array(hi), color=D.ROBOT, alpha=0.16, lw=0)
ax.plot(N, med, color=D.ROBOT, lw=3.2, marker="o", ms=8, zorder=5)
ax.axhline(0.556, color=D.BAD, lw=2.4, ls=(0, (5, 4)))
ax.text(101, 0.556, "  no correction at all", color=D.BAD, fontsize=12, va="center")
ax.axvspan(2.5, 6, color=D.BAD, alpha=0.09, lw=0)
ax.text(4.2, 0.86, "a lottery\ndown here", color=D.BAD, fontsize=11.5, ha="center")
ax.plot([20], [0.217], "o", ms=17, mfc="none", mec=D.GOOD, mew=3.2, zorder=7)
ax.annotate("we use 20", xy=(20, 0.217), xytext=(30, 0.44), fontsize=13, color=D.GOOD,
            fontweight="bold", arrowprops=dict(arrowstyle="-|>", color=D.GOOD, lw=2.2, shrinkB=12))
ax.set_xscale("log"); ax.set_xticks(N); ax.set_xticklabels([str(n) for n in N], fontsize=12)
ax.set_xlim(2.5, 150); ax.set_ylim(0, 1.15)
ax.set_xlabel("spots the robot is parked on", fontsize=13)
ax.set_ylabel("error left behind, on positions\nnever used for fitting  (cm)", fontsize=13)
ax.grid(True, color="#e8e7e2", lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("4.  Almost no data is needed",
             loc="left", fontsize=16, color=D.INK, pad=34)
ax.text(0.0, 1.005, "the answer stops moving at about ten spots; past that,\n"
        "more parking buys confidence, not accuracy",
        transform=ax.transAxes, fontsize=12, color=D.INK2, va="bottom")

fig.suptitle("What is left once the box is predicted properly: half a centimetre",
             x=0.006, ha="left", fontsize=21, color=D.INK)
fig.savefig(OUT / "05_what_is_left.png", dpi=155, bbox_inches="tight")
print("05: wrote 05_what_is_left.png")
print(f"  fit on {len({(s['x'],s['y']) for s in FIT})} positions / {len(FIT)} sightings")
print(f"  held out {len({(s['x'],s['y']) for s in HELD})} positions / {len(HELD)} sightings")
print(f"  {raw_all:.3f} -> {cor_all:.3f} cm")
