#!/usr/bin/env python3
"""Pipeline-mechanics diagnostic: how camera/inference/projection/EKF/control interact.

Panels:
 (a) EKF predict-correct cycle (zoom): truth/belief/state/camera-measurement in x & y,
     with correction events — shows odom-predict drift vs camera-correct snap and the latency lag.
 (b) Stage cadence raster: camera frames, accepted/rejected corrections, control commands —
     reveals rates and frame drops.
 (c) Correction latency (capture->applied age) over time + the detector inference time.
 (d) Belief error vs the naive latency-lag prediction (speed*latency) — shows replay compensation.

Usage: python3 scripts/paper_figures/diagnose_pipeline_mechanics.py <run_dir> [out.png] [t_lo t_hi]
"""
import csv, json, math, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/pipeline_mechanics.png")
ZOOM = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (10.0, 22.0)


def f(r, k):
    try:
        v = float(r[k]); return v if math.isfinite(v) else math.nan
    except (KeyError, ValueError, TypeError):
        return math.nan


exp = list(csv.DictReader(open(RUN / "experiment.csv")))
per = list(csv.DictReader(open(RUN / "perception.csv")))
t0 = float(json.load(open(RUN / "run_summary.json"))["first_cmd_stamp"])


def arr(rows, keys, tkey="stamp"):
    out = []
    for r in rows:
        t = f(r, tkey)
        if math.isfinite(t) and t - t0 >= -1.0:
            out.append([t - t0] + [f(r, k) for k in keys])
    return np.array(out) if out else np.zeros((0, 1 + len(keys)))


E = arr(exp, ["truth_x", "truth_y", "planner_belief_x", "planner_belief_y", "state_x", "state_y",
              "truth_belief_error_m", "planner_pixel_correction_age_s", "cmd_v", "cmd_stamp",
              "planner_pixel_correction_stamp", "pixel_corr_accepted"])
iT, iTX, iTY, iBX, iBY, iSX, iSY, iERR, iAGE, iV, iCMD, iCORR, iACC = range(13)
# camera measurements (perception)
C = arr(per, ["pred_world_x_calibrated", "pred_world_y_calibrated", "yolo_detected_after_threshold",
              "yolo_inference_ms"], "log_stamp")
det = C[C[:, 3] >= 0.5]

# correction events (stamp change) + accepted flag
corr_t, corr_acc = [], []
last = None
for r in exp:
    t = f(r, "stamp")
    if not math.isfinite(t) or t - t0 < -1.0:
        continue
    cs = f(r, "planner_pixel_correction_stamp")
    if math.isfinite(cs) and cs != last:
        last = cs
        corr_t.append(t - t0); corr_acc.append(f(r, "pixel_corr_accepted") >= 0.5)
corr_t = np.array(corr_t); corr_acc = np.array(corr_acc, bool)
# control command events (cmd_stamp change)
cmd_t = []
last = None
for r in exp:
    t = f(r, "stamp")
    if not math.isfinite(t) or t - t0 < -1.0:
        continue
    cs = f(r, "cmd_stamp")
    if math.isfinite(cs) and cs != last:
        last = cs; cmd_t.append(t - t0)
cmd_t = np.array(cmd_t)

fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(4, 1, height_ratios=[1.5, 0.6, 1, 1], hspace=0.33)
axA = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1], sharex=axA)
axC = fig.add_subplot(gs[2], sharex=axA); axD = fig.add_subplot(gs[3], sharex=axA)


def zwin(t):
    return (t >= ZOOM[0]) & (t <= ZOOM[1])


# (a) EKF cycle — plot y-coordinate (most motion is in y here)
m = zwin(E[:, iT])
axA.plot(E[m, iT], E[m, iTY], "-", color="k", lw=2.2, label="truth y")
axA.plot(E[m, iT], E[m, iBY], "-", color="#7b3294", lw=1.6, label="belief y (EKF)")
axA.plot(E[m, iT], E[m, iSY], "-", color="#ff7f0e", lw=0.9, alpha=0.6, label="state y (BEV node)")
md = zwin(det[:, 0])
axA.plot(det[md, 0], det[md, 2 - 1 + 1], "v", color="#0099cc", ms=6, alpha=0.8, label="camera meas y (pred_world)")
for t, a in zip(corr_t, corr_acc):
    if ZOOM[0] <= t <= ZOOM[1]:
        axA.axvline(t, color=("green" if a else "red"), ls=":", lw=0.7, alpha=0.4)
axA.set_ylabel("y (m)"); axA.legend(fontsize=8, ncol=2, loc="best")
axA.set_title(f"(a) EKF predict–correct cycle [zoom {ZOOM[0]:.0f}–{ZOOM[1]:.0f}s]: belief tracks truth between camera corrections (green=accepted, red=rejected)", fontsize=10)
axA.grid(alpha=0.25)

# (b) cadence raster
axB.plot(det[:, 0], np.full(len(det), 3), "|", color="#0099cc", ms=7, label="camera detections")
axB.plot(corr_t[corr_acc], np.full(corr_acc.sum(), 2), "|", color="green", ms=7, label="corrections accepted")
if (~corr_acc).any():
    axB.plot(corr_t[~corr_acc], np.full((~corr_acc).sum(), 2), "|", color="red", ms=7, label="corrections rejected")
axB.plot(cmd_t, np.full(len(cmd_t), 1), "|", color="black", ms=4, alpha=0.3, label="control cmds")
axB.set_yticks([1, 2, 3]); axB.set_yticklabels(["control", "correction", "camera"]); axB.set_ylim(0.5, 3.5)
axB.legend(fontsize=7, ncol=4, loc="upper right"); axB.set_title("(b) stage cadence (full run)", fontsize=10)

# (c) latency
axC.plot(E[:, iT], E[:, iAGE], "-", color="crimson", lw=1.0, label="correction age (capture→applied)")
if len(det):
    axC.plot(det[:, 0], det[:, 4] / 1000.0, ".", color="navy", ms=3, alpha=0.5, label="YOLO inference (s)")
axC.axhline(np.nanmedian(E[:, iAGE]), color="crimson", ls="--", lw=0.8, alpha=0.6,
            label=f"median age {np.nanmedian(E[:, iAGE]):.2f}s")
axC.set_ylabel("seconds"); axC.legend(fontsize=8, ncol=3); axC.grid(alpha=0.25)
axC.set_title("(c) correction latency vs detector inference — most latency is NOT inference", fontsize=10)

# (d) belief error vs naive lag prediction
lag = E[:, iV] * E[:, iAGE]  # naive lag if uncompensated
axD.plot(E[:, iT], E[:, iERR], "-", color="#7b3294", lw=1.4, label="actual belief error")
axD.plot(E[:, iT], lag, "-", color="gray", lw=1.0, alpha=0.7, label="naive lag = speed × latency (if NO replay)")
axD.plot(E[:, iT], E[:, iV] * 0.18, "-", color="green", lw=0.8, alpha=0.5, label="speed (scaled)")
axD.set_ylabel("m"); axD.set_xlabel("time after first command (s)"); axD.legend(fontsize=8, ncol=3); axD.grid(alpha=0.25)
axD.set_ylim(0, None)
axD.set_title("(d) belief error stays well below the naive lag → motion-replay compensates latency (spikes at stops)", fontsize=10)

fig.suptitle(f"Pipeline mechanics — {RUN.parts[-4]}/{RUN.parts[-3]} ({json.load(open(RUN/'run_summary.json')).get('completion_reason')})",
             fontsize=13, fontweight="bold")
fig.savefig(OUT, dpi=90, bbox_inches="tight")
print(f"wrote {OUT}")
print(f"camera dets={len(det)} corrections={len(corr_t)} (acc {corr_acc.sum()}) controls={len(cmd_t)} "
      f"| median age {np.nanmedian(E[:,iAGE]):.2f}s inf {np.nanmedian(det[:,4]):.0f}ms")
