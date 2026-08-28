"""Why the planner cannot route on detector confidence, in four panels.

Confidence was the previous method's reliability signal.  It is not dropped here, it is
tested: this figure shows what it is made of, where it does not exist, and what happens
when it is used as the probability a planner needs.

Read from the frozen commissioning capture.  Detector readings are joined to the trial
table by (camera, image filename) -- never by coordinate; see the README for the
10 %-of-sightings trap that join has already caused once.
"""
import csv, collections, math, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sources as src
import style as D
sys.path.insert(0, str(D.REPO / "scripts/shared"))
from metrics import auroc, brier, ece  # noqa: E402  -- never hand-roll these

DATA = D.REPO / "logs/perception_datasets/warehouse_v2_yolo_shared_20260822"
READINGS = "detector_readings_halfopen_detect_20260825.csv"
OUT = D.REPO / "logs/studies/deck_figures/confidence"; OUT.mkdir(parents=True, exist_ok=True)
MAP_CAM = "camera_E"      # the best-covered camera: make the point on the strongest case
CELL = 1.0                # panels 3 and 4 pool trials into one-square-metre cells
SUB = dict(fontsize=11.5, color=D.INK2, va="bottom")
TITLE = dict(loc="left", fontsize=15.5, color=D.INK, pad=46)

# ---- the detector's own output, on every frame it was run on ----------------------
conf = {}
for p in sorted(DATA.glob(f"camera_*/{READINGS}")):
    for r in csv.DictReader(open(p)):
        if int(r["detected"]):
            conf[(p.parent.name, r["image"])] = float(r["confidence"])

trials = list(csv.DictReader(open(D.REPO / "logs/studies/measurement_commissioning/availability.csv")))
for t in trials:
    t["conf"] = conf.get((t["camera"], t["image"])) if t["image"] else None

POS = collections.defaultdict(lambda: {"n": 0, "usable": 0, "fired": 0})
for t in trials:
    d = POS[(t["camera"], float(t["x"]), float(t["y"]))]
    d["n"] += 1; d["usable"] += int(t["usable"]); d["fired"] += t["conf"] is not None

CELLS = collections.defaultdict(lambda: {"n": 0, "usable": 0, "conf": []})
for t in trials:
    c = CELLS[(t["camera"], math.floor(float(t["x"]) / CELL), math.floor(float(t["y"]) / CELL))]
    c["n"] += 1; c["usable"] += int(t["usable"])
    if t["conf"] is not None: c["conf"].append(t["conf"])

# The previous method's field: average the score over headings, a miss counting as zero.
# That is what the old builder produced, so that is what gets compared.
rec = [dict(cam=k[0], i=k[1], j=k[2], n=v["n"], rate=v["usable"] / v["n"],
            score=sum(v["conf"]) / v["n"])
       for k, v in CELLS.items() if v["n"] >= 4]
rate = np.array([r["rate"] for r in rec]); score = np.array([r["score"] for r in rec])
slope, icpt = np.polyfit(rate, score, 1)
r2 = 1 - ((score - (slope * rate + icpt)) ** 2).sum() / ((score - score.mean()) ** 2).sum()
habit = float(np.mean(list(conf.values())))

fig = plt.figure(figsize=(16.0, 12.2), constrained_layout=True)
gs = fig.add_gridspec(2, 2)

# ---- 1. the signal does not exist where the route needs it ------------------------
ax = fig.add_subplot(gs[0, 0])
D.draw_warehouse(ax, D.layout(), camera_labels=True)
field = {(x, y): d["usable"] / d["n"] for (cam, x, y), d in POS.items() if cam == MAP_CAM}
silent = [(x, y) for (cam, x, y), d in POS.items() if cam == MAP_CAM and d["fired"] == 0]
sm = D.draw_support(ax, field, cell=0.6687, hatch_zero=False)
for (x, y) in silent:
    ax.add_patch(plt.Rectangle((x - 0.334, y - 0.334), 0.6687, 0.6687, facecolor="none",
                               edgecolor=D.BAD, lw=0.7, hatch="////", zorder=4))
cb = fig.colorbar(sm, ax=ax, fraction=0.032, pad=0.02)
cb.set_label("chance of a usable sighting", fontsize=11)
ax.legend(handles=[Patch(facecolor="white", edgecolor=D.BAD, lw=0.9, hatch="////",
                         label="never fired here:\nno number to average")],
          loc="upper left", frameon=True, framealpha=0.94, edgecolor="none",
          fontsize=11, labelcolor=D.BAD, borderpad=0.6).set_zorder(12)
frac = len(silent) / len(field)
ax.set_title("1.  Silent where routing needs an answer", **TITLE)
ax.text(0.0, 1.005, f"camera E is the best-covered camera and still has\n"
        f"no confidence value at {frac:.0%} of places — {57}% over all five.\n"
        f"A route is planned through places not yet reached.",
        transform=ax.transAxes, **SUB)

# ---- 2. confident does not mean usable --------------------------------------------
ax = fig.add_subplot(gs[0, 1])
cu = np.array([t["conf"] for t in trials if t["conf"] is not None and int(t["usable"])])
cn = np.array([t["conf"] for t in trials if t["conf"] is not None and not int(t["usable"])])
bins = np.linspace(0.25, 1.0, 31)
ax.hist(cu, bins=bins, color=D.GOOD, alpha=0.85, lw=0, label=f"kept, usable ({len(cu)})")
ax.hist(cn, bins=bins, color=D.BAD, alpha=0.80, lw=0, label=f"thrown out ({len(cn)})")
ax.axvline(0.9, color=D.INK, lw=1.6, ls=(0, (4, 3)))
over = (cn > 0.9).mean()
ax.annotate(f"{over:.0%} of the boxes we throw\naway still score above 0.9",
            xy=(0.928, len(cn) * 0.16), xytext=(0.40, 0.42), textcoords="axes fraction",
            fontsize=12.5, color=D.BAD, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=D.BAD, lw=2.0, shrinkB=6))
ax.set_xlabel("what the detector said its box was worth", fontsize=12.5)
ax.set_ylabel("number of frames", fontsize=12.5)
ax.set_xlim(0.25, 1.0); ax.grid(True, axis="y", color="#e8e7e2", lw=0.6); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=11.5, loc="upper left")
ax.set_title("2.  A confident box is not a usable measurement", **TITLE)
ax.text(0.0, 1.005, "confidence answers 'is a robot in this picture'.\n"
        "Usability also asks whether the box matches the robot\n"
        "we predicted — the question that decides if a fix arrives.",
        transform=ax.transAxes, **SUB)

# ---- 3. it is the same map with the axis squashed ---------------------------------
ax = fig.add_subplot(gs[1, 0])
ax.scatter(rate, score, s=26, color=D.OLD, alpha=0.35, lw=0, zorder=3)
xx = np.linspace(0, 1, 50)
ax.plot(xx, slope * xx + icpt, color=D.OLD, lw=3.2, zorder=5,
        label=f"old score = {slope:.2f} × chance {icpt:+.2f}")
ax.plot(xx, xx, color=D.MUTED, lw=2.0, ls=(0, (5, 4)), zorder=4, label="an honest probability")
n0 = sum(1 for r in rec if r["rate"] == 0 and r["score"] > 0.5)
ax.annotate(f"{n0} squares that never gave one\nusable sighting still score 0.5+",
            xy=(0.015, 0.80), xytext=(0.20, 0.94), fontsize=11.5, color=D.BAD, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=D.BAD, lw=2.0, shrinkB=4))
ax.set_xlabel("measured chance of a usable sighting", fontsize=12.5)
ax.set_ylabel("the previous method's score", fontsize=12.5)
ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.06); ax.set_aspect("equal")
ax.grid(True, color="#e8e7e2", lw=0.6); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=11.5, loc="lower right")
ax.set_title("3.  The old score is that chance, squashed and lifted", **TITLE)
ax.text(0.0, 1.005, f"one dot = one camera, one square metre.  Slope and offset\n"
        f"are the detector's habit of scoring {habit:.2f} whenever it fires —\n"
        f"retrain it and the field moves, though nothing else did.  R² {r2:.2f}",
        transform=ax.transAxes, **SUB)

# ---- 4. so it states the wrong odds ------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
fit = {(r["cam"], r["i"], r["j"]): r for r in rec if (r["i"] + r["j"]) % 2 == 0}
held = [r for r in rec if (r["i"] + r["j"]) % 2 == 1]
def predict(r, key):
    best, bd = None, 1e9
    for (cam, i, j), t in fit.items():
        if cam != r["cam"]: continue
        d = (i - r["i"]) ** 2 + (j - r["j"]) ** 2
        if d < bd: bd, best = d, t
    return best[key]
EDGES = [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]
summary, PROM = {}, {}
for key, col, lab in (("score", D.OLD, "previous: confidence"),
                      ("rate", D.GOOD, "proposed: measured chance")):
    # Expand each held-out square back into its trials, so the scores are computed on
    # 0/1 outcomes by the shared library rather than on cell averages.
    y, pr = [], []
    for r in held:
        k = int(round(r["rate"] * r["n"])); pv = predict(r, key)
        y += [1] * k + [0] * (r["n"] - k); pr += [pv] * r["n"]
    y, pr = np.array(y), np.array(pr)
    summary[key] = (brier(y, pr), ece(y, pr, bins=5), auroc(y, pr))
    PROM[key] = (float(pr.mean()), float(y.mean()))
    px, py = [], []
    for lo, hi in EDGES:
        m = (pr >= lo) & (pr < hi)
        if m.sum() < 20: continue
        px.append(pr[m].mean()); py.append(y[m].mean())
    ax.plot(px, py, color=col, lw=3.2, marker="o", ms=10, zorder=5, label=lab)
ax.plot([0, 1], [0, 1], color=D.MUTED, lw=2.0, ls=(0, (5, 4)), zorder=3)
ax.text(0.66, 0.715, "promises kept", color=D.MUTED, fontsize=11.5, rotation=40, ha="center")
ax.annotate("promises 0.73, a sighting\narrives a third of the time", xy=(0.733, 0.353),
            xytext=(0.40, 0.03), fontsize=12, color=D.OLD, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=D.OLD, lw=2.0, shrinkB=8))
ax.set_xlabel("what the field promises", fontsize=12.5)
ax.set_ylabel("how often a sighting actually arrived", fontsize=12.5)
ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.06); ax.set_aspect("equal")
ax.grid(True, color="#e8e7e2", lw=0.6); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=11.5, loc="upper left")
ax.set_title("4.  It knows which places are better, not the odds", **TITLE)
ax.text(0.0, 1.005, f"{len(held)} held-out squares.  It ranks places slightly better than\n"
        f"the measured chance ({summary['score'][2]:.2f} against {summary['rate'][2]:.2f}) "
        f"and still scores worse overall,\n"
        f"because it promises {PROM['score'][0]:.2f} and delivers {PROM['score'][1]:.2f}.  "
        f"The planner multiplies by this.",
        transform=ax.transAxes, **SUB)

fig.suptitle("Detector confidence knows which places are better — it does not know the odds, "
             "and it is silent where a route needs it",
             x=0.006, ha="left", fontsize=19, color=D.INK)
fig.savefig(OUT / "01_why_not_confidence.png", dpi=155, bbox_inches="tight")
print("wrote 01_why_not_confidence.png")
print(f"  confidence defined on {len(conf)}/{len(trials)} trials = {len(conf)/len(trials):.3f}")
nsil = sum(1 for v in POS.values() if v['fired'] == 0)
print(f"  camera x position with no confidence value: {nsil}/{len(POS)} = {nsil/len(POS):.3f}")
print(f"  usable fires median conf {np.median(cu):.3f}; thrown-out median {np.median(cn):.3f}; "
      f"thrown out above 0.9 {over:.3f}")
print(f"  old field = {slope:.3f} x rate {icpt:+.3f}, R2 {r2:.3f}; {n0} dead squares score >0.5")
for k, (b, e, a) in summary.items():
    print(f"  {k:6s} held-out Brier {b:.4f}  calibration error {e:.4f}  AUC {a:.4f}  "
          f"promises {PROM[k][0]:.3f} delivers {PROM[k][1]:.3f}")
