#!/usr/bin/env python3
"""Comprehensive pipeline diagnostic for a single run (truth/belief/state/camera/timing).

Built to investigate why the paired-mechanism route can look erratic while the
localization error looks tame: it plots EVERY relevant quantity with explicit,
verified definitions so timing/definition mismatches are visible.

Definitions (verified against experiment.csv / perception.csv):
  truth            : truth_x/y                          (Gazebo ground truth)
  belief (EKF)     : planner_belief_x/y  (== est_x/y)   ; truth_belief_error_m == |truth-belief|
  state (BEV node) : state_x/y                           (raw-ish overhead-cam state, noisier)
  camera meas      : perception.csv pred_world_x/y_calibrated (the measurement fed to the EKF)
  detection        : perception.csv yolo_detected_after_threshold>=0.5 at log_stamp
  camera update    : row where planner_pixel_correction_stamp changes (measurement_available is NaN -> unusable)
  latency          : planner_pixel_correction_age_s (meas->applied); replay = pixel_corr_cmd_replay_duration_s
  uncertainty      : 2sigma from planner_cov_x/xy/y
Time base: t = stamp - first_cmd_stamp (perception log_stamp shares the sim clock).

Usage: python3 scripts/paper_figures/diagnose_taskA_pipeline.py <run_dir> [out.png]
"""
import csv, json, math, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

RUN = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/diag_taskA.png")
GPZ = Path("paper_artifacts/gp/aws_gp_v3/yolo_score_raw_gp.npz")


def f(r, k):
    try:
        v = float(r[k]); return v if math.isfinite(v) else math.nan
    except (KeyError, ValueError, TypeError):
        return math.nan


exp = list(csv.DictReader(open(RUN / "experiment.csv")))
per = list(csv.DictReader(open(RUN / "perception.csv")))
summ = json.load(open(RUN / "run_summary.json"))
t0 = float(summ["first_cmd_stamp"])
goal = (f(exp[-1], "goal_x"), f(exp[-1], "goal_y"))

# ---- experiment time series ----
def col(rows, key, tkey="stamp"):
    out = []
    for r in rows:
        t = f(r, tkey)
        if math.isfinite(t) and t - t0 >= -0.05:
            out.append((t - t0, f(r, key)))
    return np.array(out) if out else np.zeros((0, 2))

tx = np.array([[f(r, "stamp") - t0, f(r, "truth_x"), f(r, "truth_y")] for r in exp
               if math.isfinite(f(r, "stamp")) and f(r, "stamp") - t0 >= -0.05
               and math.isfinite(f(r, "truth_x"))])
bel = np.array([[f(r, "stamp") - t0, f(r, "planner_belief_x"), f(r, "planner_belief_y")] for r in exp
                if math.isfinite(f(r, "stamp")) and f(r, "stamp") - t0 >= -0.05
                and math.isfinite(f(r, "planner_belief_x"))])
sta = np.array([[f(r, "stamp") - t0, f(r, "state_x"), f(r, "state_y")] for r in exp
                if math.isfinite(f(r, "stamp")) and f(r, "stamp") - t0 >= -0.05
                and math.isfinite(f(r, "state_x"))])
err_b = col(exp, "truth_belief_error_m")
err_s = col(exp, "truth_state_error_m")
r2s = np.array([[f(r, "stamp") - t0,
                 2.0 * math.sqrt(max(0.0, float(np.linalg.eigvalsh(
                     np.array([[f(r, "planner_cov_x"), f(r, "planner_cov_xy")],
                               [f(r, "planner_cov_xy"), f(r, "planner_cov_y")]]))[-1])))]
                for r in exp if math.isfinite(f(r, "stamp")) and f(r, "stamp") - t0 >= -0.05
                and math.isfinite(f(r, "planner_cov_x"))])
age = col(exp, "planner_pixel_correction_age_s")
replay = col(exp, "pixel_corr_cmd_replay_duration_s")
nis = col(exp, "pixel_corr_nis")
cmdv = col(exp, "cmd_v")
pvis = col(exp, "p_vis_plan")

# camera updates: correction stamp changes; accepted/rejected
upd_t, upd_acc = [], []
last = None
for r in exp:
    t = f(r, "stamp")
    if not math.isfinite(t) or t - t0 < -0.05:
        continue
    cs = f(r, "planner_pixel_correction_stamp")
    if math.isfinite(cs) and cs != last:
        last = cs
        upd_t.append(t - t0); upd_acc.append(f(r, "pixel_corr_accepted") >= 0.5)
upd_t = np.array(upd_t); upd_acc = np.array(upd_acc, dtype=bool)

# ---- perception (camera measurement) ----
cam = np.array([[f(r, "log_stamp") - t0, f(r, "pred_world_x_calibrated"), f(r, "pred_world_y_calibrated"),
                 f(r, "localization_error_calibrated_m"), f(r, "yolo_detected_after_threshold")]
                for r in per if math.isfinite(f(r, "log_stamp")) and f(r, "log_stamp") - t0 >= -0.05])
det = cam[cam[:, 4] >= 0.5] if len(cam) else cam
miss = cam[cam[:, 4] < 0.5] if len(cam) else cam
inf_ms = col(per, "yolo_inference_ms", "log_stamp")

# detection gaps (occlusion) shading
det_t = det[:, 0] if len(det) else np.array([])
gaps = []
for i in range(len(det_t) - 1):
    if det_t[i + 1] - det_t[i] > 0.6:
        gaps.append((det_t[i], det_t[i + 1]))

# ---- GP background ----
gp = np.load(GPZ, allow_pickle=True)
xs, ys, P = gp["xs"], gp["ys"], gp["P_conservative_plan_map"]
extent = (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]))
racks = [(p["xmin"], p["xmax"], p["ymin"], p["ymax"])
         for p in json.loads(str(gp["geometry_json"]).strip("[]'")).get("prisms", [])]

# ================= FIGURE =================
fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(4, 2, width_ratios=[1.05, 1.0], height_ratios=[1, 1, 1, 1], hspace=0.42, wspace=0.18)
axmap = fig.add_subplot(gs[:, 0])
ax1 = fig.add_subplot(gs[0, 1]); ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
ax3 = fig.add_subplot(gs[2, 1], sharex=ax1); ax4 = fig.add_subplot(gs[3, 1], sharex=ax1)


def shade_gaps(ax):
    for a, b in gaps:
        ax.axvspan(a, b, color="orange", alpha=0.18, lw=0)


# --- MAP ---
im = axmap.imshow(P, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=0.9, aspect="equal", zorder=0)
for (x0, x1, y0, y1) in racks:
    axmap.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#7a7a7a", edgecolor="#1f1f1f", lw=0.5, alpha=0.85, zorder=3))
if len(cam):
    axmap.scatter(det[:, 1], det[:, 2], s=10, c="cyan", edgecolor="none", alpha=0.5, zorder=4, label="camera meas (pred_world)")
if len(sta):
    axmap.plot(sta[:, 1], sta[:, 2], "-", color="#ff7f0e", lw=0.8, alpha=0.6, zorder=5, label="state (BEV node)")
if len(bel):
    axmap.plot(bel[:, 1], bel[:, 2], "-", color="#7b3294", lw=1.6, alpha=0.95, zorder=7, label="belief (EKF)")
if len(tx):
    axmap.plot(tx[:, 1], tx[:, 2], "-", color="k", lw=2.4, alpha=0.9, zorder=6, label="truth")
axmap.scatter([tx[0, 1]], [tx[0, 2]], s=90, c="#16a34a", edgecolor="k", zorder=9, label="start")
axmap.scatter([goal[0]], [goal[1]], s=160, marker="*", c="#e41a1c", edgecolor="k", zorder=9, label="goal")
axmap.set_xlim(min(tx[:,1].min(), goal[0]) - 0.6, max(tx[:,1].max(), goal[0]) + 0.6)
axmap.set_ylim(min(tx[:,2].min(), goal[1]) - 0.6, max(tx[:,2].max(), goal[1]) + 0.6)
axmap.set_title(f"{RUN.parts[-4]}/{RUN.parts[-3]} — {summ.get('completion_reason')}", fontweight="bold")
axmap.set_xlabel("x (m)"); axmap.set_ylabel("y (m)"); axmap.legend(fontsize=7.5, loc="best"); axmap.set_aspect("equal")
fig.colorbar(im, ax=axmap, fraction=0.04, pad=0.02, label=r"$\rho_{\rm plan}$ (detection-rate GP)")

# --- (1) position error ---
shade_gaps(ax1)
if len(err_b): ax1.plot(err_b[:, 0], err_b[:, 1], "-", color="#7b3294", lw=1.5, label="truth–belief")
if len(err_s): ax1.plot(err_s[:, 0], err_s[:, 1], "-", color="#ff7f0e", lw=1.0, alpha=0.7, label="truth–state")
if len(cam): ax1.plot(det[:, 0], det[:, 3], ".", color="cyan", ms=3, alpha=0.6, label="truth–camera meas")
if len(r2s): ax1.plot(r2s[:, 0], r2s[:, 1], "--", color="0.4", lw=1.0, label="2σ belief radius")
ax1.axhline(0.125, color="r", ls=":", lw=0.8, label="robot radius 0.125 m")
ax1.set_ylabel("position error (m)"); ax1.set_ylim(0, None); ax1.legend(fontsize=7, ncol=2); ax1.grid(alpha=0.25)
ax1.set_title("(1) localization error: belief vs state vs raw camera measurement", fontsize=10)

# --- (2) timing / latency ---
shade_gaps(ax2)
if len(age): ax2.plot(age[:, 0], age[:, 1], "-", color="crimson", lw=1.2, label="correction age (meas→applied)")
if len(replay): ax2.plot(replay[:, 0], replay[:, 1], "-", color="green", lw=1.0, label="motion-replay duration")
if len(inf_ms): ax2.plot(inf_ms[:, 0], inf_ms[:, 1] / 1000.0, ".", color="navy", ms=3, alpha=0.5, label="YOLO inference (s)")
ax2.set_ylabel("seconds"); ax2.legend(fontsize=7, ncol=2); ax2.grid(alpha=0.25)
ax2.set_title("(2) timing: correction latency vs replay compensation vs detector inference", fontsize=10)

# --- (3) camera updates / NIS ---
shade_gaps(ax3)
if len(nis): ax3.plot(nis[:, 0], nis[:, 1], "-", color="teal", lw=1.0, label="NIS")
ax3.axhline(9.21, color="r", ls=":", lw=0.8, label="NIS gate 9.21")
if len(upd_t):
    ax3.plot(upd_t[upd_acc], np.full(upd_acc.sum(), -0.6), "|", color="green", ms=8, label="update accepted")
    if (~upd_acc).any():
        ax3.plot(upd_t[~upd_acc], np.full((~upd_acc).sum(), -0.6), "|", color="red", ms=8, label="update rejected")
ax3.set_ylabel("NIS"); ax3.set_ylim(-1.2, max(12, np.nanmax(nis[:,1]) if len(nis) else 12)); ax3.legend(fontsize=7, ncol=2); ax3.grid(alpha=0.25)
ax3.set_title("(3) camera updates: NIS + accept/reject (orange = detection gap >0.6 s)", fontsize=10)

# --- (4) motion + visibility ---
shade_gaps(ax4)
if len(cmdv): ax4.plot(cmdv[:, 0], cmdv[:, 1], "-", color="black", lw=1.0, label="cmd_v (m/s)")
if len(pvis): ax4.plot(pvis[:, 0], pvis[:, 1], "-", color="purple", lw=1.0, alpha=0.6, label="p_vis_plan")
ax4.set_ylabel("v / p_vis"); ax4.set_xlabel("time after first command (s)"); ax4.legend(fontsize=7); ax4.grid(alpha=0.25)
ax4.set_title("(4) commanded speed + planned visibility", fontsize=10)

fig.suptitle(f"Task-A pipeline diagnostic — {RUN.name}", fontsize=13, fontweight="bold")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=85, bbox_inches="tight")
print(f"wrote {OUT}")
# quick numeric summary
print(f"detector inf med {np.nanmedian(inf_ms[:,1]):.0f}ms; corr_age med {np.nanmedian(age[:,1]):.2f}s; "
      f"replay med {np.nanmedian(replay[:,1]):.2f}s; updates {len(upd_t)} ({upd_acc.sum()} acc); "
      f"det {len(det)}/{len(cam)}; belief err med {np.nanmedian(err_b[:,1]):.3f}")
