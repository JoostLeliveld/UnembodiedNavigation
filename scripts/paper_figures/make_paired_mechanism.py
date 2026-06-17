#!/usr/bin/env python3
"""Paired representative run on a route-choice task.

The default source is the clean verification run archived under
`_paper_runs/paired_mechanism_clean_verify`, generated from the locked campaign
configuration with the same runtime settings used by the paper campaign: YOLO
masks off, keep-in warning-band no-go cost, lane-graph route seeds, the
warehouse visibility GP, and the same noise/controller parameters.

Three panels:
  (a) C1 route map: global plan, executed trajectory, belief mean, and uncertainty band,
  (b) C2 route map: global plan, executed trajectory, belief mean, and uncertainty band,
  (c) truth-belief position error with the 2-sigma belief radius.
"""
import sys, json, glob, math, csv, shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.patches import Circle, Patch, Rectangle
from matplotlib.lines import Line2D

import os
ROOT = Path(__file__).resolve().parents[2]
CAMP_NAME = os.environ.get("PAIRED_CAMP", "_paper_runs/paired_mechanism_clean_verify")
CAMP = ROOT/"logs/visibility_comparison"/CAMP_NAME
TASK = os.environ.get("PAIRED_TASK", "route_apron_to_a3_mid")
SEED = int(os.environ.get("PAIRED_SEED", "0"))
GPZ = ROOT/os.environ.get("PAIRED_GP", "paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz")
LOCKED_CONFIG = ROOT/"scripts/visibility_comparison/warehouse_visibility_campaign.yaml"
_OUTNAME = os.environ.get("PAIRED_OUT", "paired_mechanism_taskA")
OUT = ROOT/f"paper_artifacts/figures/{_OUTNAME}.pdf"
COPY_TO_THESIS = os.environ.get("PAIRED_COPY_TO_THESIS", "1") not in ("0", "false", "False") and _OUTNAME == "paired_mechanism_taskA"
PAPER = ROOT.parent/f"thesis-report/figures/campaign/{_OUTNAME}.pdf"
DATA_DIR = ROOT/f"paper_artifacts/figures/{_OUTNAME}_data"
CAM = (0.0, -5.5)
C1C, C2C = "#e8000b", "#1a5fd6"
BELIEF_C = "#7b3294"
# START, GOAL and the MAP_XLIM/MAP_YLIM window are derived PER TASK from the run
# data after the conditions are loaded (see below). They used to be hardcoded to
# F31 (start (3.3,-1.0), goal (1.0,1.75), window (0.45,3.75)x(-2.85,2.20)) and so
# were wrong for every other task -- e.g. b2/b6 live around x=-5..-3 and fell
# entirely outside the old window. Env overrides: PAIRED_START / PAIRED_GOAL
# ("x,y"), PAIRED_TITLE_A / PAIRED_TITLE_B for the two map-panel titles.
START = GOAL = (math.nan, math.nan)
MAP_XLIM = MAP_YLIM = None

LEGACY_TASK_DIRS = {
    "route_apron_to_a3_mid": "F31_b1_apron_a3_mid",
    "route_apron_to_a2_mid": "b5_a4_apron_to_a2_mid",
    "route_west_to_a1_upper": "b2_a0_west_to_a1_upper",
    "control_west_to_a1_low": "b6_a0_west_to_a1_low_control",
}


def _run_dir(cond):
    task_dirs = [TASK]
    legacy = LEGACY_TASK_DIRS.get(TASK)
    if legacy:
        task_dirs.append(legacy)
    matches = []
    for task_dir in task_dirs:
        matches = sorted(glob.glob(str(CAMP/task_dir/cond/f"seed{SEED}"/"experiment_*")))
        if matches:
            break
    if not matches:
        raise FileNotFoundError(f"No run directory for {CAMP}/{TASK}/{cond}/seed{SEED}")
    return matches[-1]


def _read_csv(path):
    return list(csv.DictReader(open(path)))


def _f(row, key):
    try:
        v = float(row[key])
        return v if math.isfinite(v) else math.nan
    except (KeyError, ValueError, TypeError):
        return math.nan


def _rel(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _locked_config_value(key, fallback=None):
    if not LOCKED_CONFIG.exists():
        return fallback
    try:
        for line in LOCKED_CONFIG.read_text(encoding="utf-8").splitlines():
            stripped = line.split("#", 1)[0].strip()
            if stripped.startswith(f"{key}:"):
                raw = stripped.split(":", 1)[1].strip().strip("'\"")
                try:
                    return float(raw)
                except ValueError:
                    return raw
    except OSError:
        return fallback
    return fallback


def _copy_source_data(selected_runs, provenance):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files_to_copy = (
        "experiment.csv",
        "perception.csv",
        "plan_samples.csv",
        "global_plan.csv",
        "global_waypoints.csv",
        "global_plan_meta.json",
        "run_manifest.json",
        "run_summary.json",
        "tasks.yaml",
    )
    copied = []
    for cond, run in selected_runs.items():
        src_dir = Path(run["run_dir"])
        dst_dir = DATA_DIR / cond
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in files_to_copy:
            src = src_dir / name
            if src.exists():
                dst = dst_dir / name
                shutil.copy2(src, dst)
                copied.append(_rel(dst))
    campaign_log = CAMP / "campaign_log.json"
    if campaign_log.exists():
        dst = DATA_DIR / "campaign_log.json"
        shutil.copy2(campaign_log, dst)
        copied.append(_rel(dst))
    locked_config = LOCKED_CONFIG
    if locked_config.exists():
        dst = DATA_DIR / "warehouse_visibility_campaign.yaml"
        shutil.copy2(locked_config, dst)
        copied.append(_rel(dst))
    gp_manifest = GPZ.parent / "gp_manifest.json"
    if gp_manifest.exists():
        dst = DATA_DIR / "gp_manifest.json"
        shutil.copy2(gp_manifest, dst)
        copied.append(_rel(dst))
    provenance["copied_source_files"] = copied
    (DATA_DIR / "README.md").write_text(
        f"# {_OUTNAME} source data\n\n"
        "This directory contains the exact run files used to generate "
        f"`{_OUTNAME}.pdf`.\n\n"
        "Source campaign: `logs/visibility_comparison/{camp}`\n\n"
        "Task: `{task}`, seed: `{seed}`\n\n"
        "Regenerate the figure with:\n\n"
        "```bash\n"
        "cd /home/joostleliveld/Thesis/UnembodiedNavigation\n"
        "PAIRED_CAMP={camp} PAIRED_TASK={task} PAIRED_SEED={seed} \\\n"
        "  python3 scripts/paper_figures/make_paired_mechanism.py\n"
        "```\n\n"
        "The selected runs use the locked keep-in warning-band no-go setting, "
        "YOLO masks disabled, and the warehouse visibility GP artifact. The "
        "full runtime values are recorded in each condition's `run_manifest.json`.\n"
        .format(camp=CAMP_NAME, task=TASK, seed=SEED),
        encoding="utf-8",
    )
    provenance_path = OUT.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    shutil.copy2(provenance_path, DATA_DIR / provenance_path.name)
    return provenance_path


gp = np.load(GPZ, allow_pickle=True)
xs, ys, P = gp["xs"], gp["ys"], gp["P_conservative_plan_map"]
extent = (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]))
racks = [(p["xmin"], p["xmax"], p["ymin"], p["ymax"])
         for p in json.loads(str(gp["geometry_json"]).strip("[]'")).get("prisms", [])]


def sample_rho(x, y):
    """GP planner reliability at (x,y) -- C2's predictive rho_plan along the path."""
    i = int(np.clip(np.searchsorted(xs, x), 0, len(xs) - 1))
    j = int(np.clip(np.searchsorted(ys, y), 0, len(ys) - 1))
    return float(P[j, i])


def load_cond(cond):
    rd = _run_dir(cond)
    exp = _read_csv(rd + "/experiment.csv")
    per = _read_csv(rd + "/perception.csv")
    summ = json.load(open(rd + "/run_summary.json"))
    manifest = json.load(open(rd + "/run_manifest.json"))
    t0 = float(summ["first_cmd_stamp"])
    is_c2 = cond == "C2"
    crash = None
    # trajectory + time series from experiment.csv
    tx, ty, traj_t, traj_x, traj_y = [], [], [], [], []
    belief_t, belief_x, belief_y, belief_cov, belief_r2s = [], [], [], [], []
    ep, r2s, rho, cam_t, cam_e = [], [], [], [], []
    for r in exp:
        x, y = _f(r, "truth_x"), _f(r, "truth_y")
        st = _f(r, "stamp")
        if math.isfinite(x) and math.isfinite(y):
            tx.append(x); ty.append(y)
            if math.isfinite(st) and (st - t0) >= -0.05:
                traj_t.append(st - t0); traj_x.append(x); traj_y.append(y)
        if not math.isfinite(st):
            continue
        tt = st - t0
        if tt < -0.05:   # only show time series after the first command
            continue
        e = _f(r, "truth_belief_error_m")
        bx, by = _f(r, "planner_belief_x"), _f(r, "planner_belief_y")
        cxx, cxy, cyy = _f(r, "planner_cov_x"), _f(r, "planner_cov_xy"), _f(r, "planner_cov_y")
        if math.isfinite(cxx) and math.isfinite(cyy):
            cov = np.array([[cxx, cxy], [cxy, cyy]], float)
            lam = float(np.linalg.eigvalsh(cov)[-1])
            r2 = 2.0 * math.sqrt(max(lam, 0.0))
            r2s.append((tt, r2))
            if math.isfinite(bx) and math.isfinite(by):
                belief_t.append(tt)
                belief_x.append(bx)
                belief_y.append(by)
                belief_cov.append(cov)
                belief_r2s.append(r2)
        if math.isfinite(e):
            ep.append((tt, e))
        # rho_plan as the planner assumes it ALONG THE EXECUTED PATH: C2 samples the GP,
        # C1 is constant-R (assumes perfect reliability everywhere).
        if math.isfinite(x) and math.isfinite(y):
            rho.append((tt, sample_rho(x, y) if is_c2 else 1.0))
        if _f(r, "measurement_available") >= 0.5 and math.isfinite(e):
            cam_t.append(tt); cam_e.append(e)
        if crash is None and math.isfinite(_f(r, "first_crash_stamp")) and _f(r, "first_crash_stamp") > 0:
            crash = _f(r, "first_crash_stamp") - t0
    # YOLO detections from perception.csv
    det = []  # (t, score, accepted)
    for r in per:
        # Place detection markers at the camera frame stamp.  Using log_stamp
        # shifts them along the trajectory by detector/logger latency.
        st = _f(r, "diag_stamp")
        if not math.isfinite(st):
            st = _f(r, "log_stamp")
        sc = _f(r, "yolo_score_selected")
        if not math.isfinite(st) or (st - t0) < -0.05:
            continue
        acc = _f(r, "yolo_detected_after_threshold") >= 0.5
        if math.isfinite(sc):
            det.append((st - t0, sc, acc))
    traj_t_arr = np.array(traj_t)
    traj_x_arr = np.array(traj_x)
    traj_y_arr = np.array(traj_y)
    det_acc_xy, det_miss_xy = [], []
    if len(traj_t_arr):
        for t, _score, accepted in det:
            idx = int(np.argmin(np.abs(traj_t_arr - t)))
            if abs(float(traj_t_arr[idx]) - t) <= 0.75:
                target = det_acc_xy if accepted else det_miss_xy
                target.append((float(traj_x_arr[idx]), float(traj_y_arr[idx])))
    def _thin(points, max_n):
        if len(points) <= max_n:
            return np.array(points, dtype=float)
        idx = np.linspace(0, len(points) - 1, max_n).round().astype(int)
        return np.array([points[i] for i in idx], dtype=float)
    det_acc_xy = _thin(det_acc_xy, 28)
    det_miss_xy = _thin(det_miss_xy, 28)
    # one-shot global PLANNED route (single plan_stamp in plan_samples.csv)
    px, py = [], []
    try:
        plan = _read_csv(rd + "/plan_samples.csv")
        stamps = sorted({_f(r, "plan_stamp") for r in plan if math.isfinite(_f(r, "plan_stamp"))})
        if stamps:
            pts = sorted((r for r in plan if _f(r, "plan_stamp") == stamps[0]), key=lambda r: _f(r, "point_idx"))
            for r in pts:
                x, y = _f(r, "x"), _f(r, "y")
                if math.isfinite(x) and math.isfinite(y):
                    px.append(x); py.append(y)
    except FileNotFoundError:
        pass
    reached = str(summ.get("completion_reason", "")) == "goal_reached"
    # Per-task start (first valid truth pose) and goal (logged goal_x/goal_y).
    start_xy = (float(tx[0]), float(ty[0])) if len(tx) else (math.nan, math.nan)
    gx = next((_f(r, "goal_x") for r in reversed(exp) if math.isfinite(_f(r, "goal_x"))), math.nan)
    gy = next((_f(r, "goal_y") for r in reversed(exp) if math.isfinite(_f(r, "goal_y"))), math.nan)
    return dict(start=start_xy, goal=(gx, gy),
                tx=np.array(tx), ty=np.array(ty),
                belief_t=np.array(belief_t), belief_x=np.array(belief_x),
                belief_y=np.array(belief_y), belief_cov=np.array(belief_cov),
                belief_r2s=np.array(belief_r2s),
                ep=np.array(ep), r2s=np.array(r2s),
                rho=np.array(rho), det=det, cam_t=np.array(cam_t), cam_e=np.array(cam_e),
                det_acc_xy=det_acc_xy, det_miss_xy=det_miss_xy,
                plan_x=np.array(px), plan_y=np.array(py),
                crash=crash, reached=reached,
                mean_e=float(np.nanmean([e for _, e in ep])) if ep else math.nan,
                run_dir=rd, summary=summ, manifest=manifest)


C1, C2 = load_cond("C1"), load_cond("C2")


def _pick_xy(*candidates):
    for a, b in candidates:
        if math.isfinite(a) and math.isfinite(b):
            return (float(a), float(b))
    return (math.nan, math.nan)


def _env_xy(name, default):
    v = os.environ.get(name)
    if v:
        try:
            a, b = (float(z) for z in v.split(","))
            return (a, b)
        except ValueError:
            pass
    return default


# Per-task geometry from the runs (C1/C2 agree; fall back across them just in case).
START = _env_xy("PAIRED_START", _pick_xy(C1["start"], C2["start"]))
GOAL = _env_xy("PAIRED_GOAL", _pick_xy(C1["goal"], C2["goal"]))

# Auto-fit the map window to the union of both trajectories / plans / belief means
# plus start and goal, padded and clamped to the GP field extent.
_xs_all, _ys_all = [], []
for _c in (C1, C2):
    for _kx, _ky in (("tx", "ty"), ("plan_x", "plan_y"), ("belief_x", "belief_y")):
        _ax = np.asarray(_c[_kx], float).ravel()
        _ay = np.asarray(_c[_ky], float).ravel()
        _m = np.isfinite(_ax) & np.isfinite(_ay)
        _xs_all += list(_ax[_m]); _ys_all += list(_ay[_m])
for _p in (START, GOAL):
    if math.isfinite(_p[0]) and math.isfinite(_p[1]):
        _xs_all.append(_p[0]); _ys_all.append(_p[1])
_pad = 0.5
if _xs_all:
    MAP_XLIM = (max(extent[0], min(_xs_all) - _pad), min(extent[1], max(_xs_all) + _pad))
    MAP_YLIM = (max(extent[2], min(_ys_all) - _pad), min(extent[3], max(_ys_all) + _pad))
else:
    MAP_XLIM, MAP_YLIM = (extent[0], extent[1]), (extent[2], extent[3])

plt.rcParams.update({"font.size": 10.0, "pdf.fonttype": 42, "ps.fonttype": 42})
fig = plt.figure(figsize=(11.4, 8.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.22, 0.78], hspace=0.28, wspace=0.12)
axa = fig.add_subplot(gs[0, 0])
axb = fig.add_subplot(gs[0, 1])
axd = fig.add_subplot(gs[1, :])


def draw_belief_band(ax, traj):
    """Draw a continuous-looking 2-sigma band around the belief mean."""
    x = np.asarray(traj["belief_x"], dtype=float)
    y = np.asarray(traj["belief_y"], dtype=float)
    r = np.asarray(traj["belief_r2s"], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(r)
    x, y, r = x[valid], y[valid], r[valid]
    if len(x) < 3:
        return

    step = max(1, len(x) // 120)
    idx = np.arange(0, len(x), step)
    if idx[-1] != len(x) - 1:
        idx = np.r_[idx, len(x) - 1]
    x, y, r = x[idx], y[idx], r[idx]

    if len(r) >= 7:
        kernel = np.ones(5, dtype=float) / 5.0
        r = np.convolve(np.r_[r[0], r[0], r, r[-1], r[-1]], kernel, mode="valid")

    for xi, yi, ri in zip(x, y, r):
        ax.add_patch(Circle((float(xi), float(yi)), float(ri),
                            facecolor=BELIEF_C, edgecolor="none",
                            alpha=0.055, zorder=4))


def draw_map(ax, traj, cmap, title, color, show_cbar=False):
    im = ax.imshow(P, origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=0.9, aspect="equal", zorder=0)
    for (x0, x1, y0, y1) in racks:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#7a7a7a", edgecolor="#1f1f1f", lw=0.5, alpha=0.9, zorder=3))
    if len(traj["belief_x"]):
        draw_belief_band(ax, traj)
        belief_line, = ax.plot(traj["belief_x"], traj["belief_y"], color=BELIEF_C, lw=1.45,
                               ls=(0, (4, 2)), alpha=0.92, zorder=5)
        belief_line.set_path_effects([pe.Stroke(linewidth=2.6, foreground="white", alpha=0.75), pe.Normal()])
    ax.plot(traj["tx"], traj["ty"], "-", color=color, lw=3.1, solid_capstyle="round", zorder=6)
    if len(traj["plan_x"]):  # one-shot global planned route
        plan_line, = ax.plot(traj["plan_x"], traj["plan_y"], "--", color="#111111",
                             lw=2.15, alpha=0.96, dashes=(5, 2.5), zorder=8)
        plan_line.set_path_effects([pe.Stroke(linewidth=3.8, foreground="white"), pe.Normal()])
    if len(traj["det_acc_xy"]):
        pts = traj["det_acc_xy"]
        ax.scatter(pts[:, 0], pts[:, 1], s=18, marker="o", facecolor="white",
                   edgecolor=color, lw=0.85, alpha=0.72, zorder=7)
    if len(traj["det_miss_xy"]):
        pts = traj["det_miss_xy"]
        ax.scatter(pts[:, 0], pts[:, 1], s=30, marker="x", color=color,
                   lw=1.05, alpha=0.62, zorder=7)
    ax.scatter([START[0]], [START[1]], s=90, c="#16a34a", edgecolor="k", zorder=8)
    ax.scatter([GOAL[0]], [GOAL[1]], s=150, marker="*", c="#e41a1c", edgecolor="k", zorder=8)
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.set_xlim(*MAP_XLIM); ax.set_ylim(*MAP_YLIM); ax.set_xlabel("$x$ (m)"); ax.set_ylabel("$y$ (m)")
    ax.set_aspect("equal")
    if show_cbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=r"$\rho_{\mathrm{plan}}$")


_TITLE_A = os.environ.get("PAIRED_TITLE_A", "(a) C1: short route, weaker camera support")
_TITLE_B = os.environ.get("PAIRED_TITLE_B", "(b) C2: longer route, stronger camera support")
draw_map(axa, C1, "Greys_r", _TITLE_A, C1C)
axa.legend(handles=[
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#16a34a", markeredgecolor="k", markersize=8, label="start"),
    Line2D([0], [0], marker="*", color="none", markerfacecolor="#e41a1c", markeredgecolor="k", markersize=11, label="goal"),
    Line2D([0], [0], color="#111", lw=2.15, ls="--", dashes=(5, 2.5), label="global plan"),
    Line2D([0], [0], color="#444", lw=3.1, label="executed trajectory"),
    Line2D([0], [0], color=BELIEF_C, lw=1.45, ls=(0, (4, 2)), label="belief mean"),
    Patch(facecolor=BELIEF_C, edgecolor="none", alpha=0.13, label=r"2$\sigma$ band"),
    Line2D([0], [0], marker="o", color="0.4", markerfacecolor="white", markeredgecolor="0.4", lw=0, markersize=6, label="YOLO update"),
    Line2D([0], [0], marker="x", color="0.4", lw=0, markersize=7, label="YOLO miss"),
], loc="lower left", fontsize=7.2, framealpha=0.9, ncol=2, columnspacing=0.9, handlelength=2.0)
draw_map(axb, C2, "viridis", _TITLE_B, C2C, show_cbar=True)

# (c) truth-belief error + 2-sigma radius
for cond, col, lab in ((C1, C1C, "C1"), (C2, C2C, "C2")):
    axd.plot(cond["ep"][:, 0], cond["ep"][:, 1], "-", color=col, lw=2.0,
             label=f"{lab} $e_p$ (mean {cond['mean_e']:.2f} m)")
    axd.plot(cond["r2s"][:, 0], cond["r2s"][:, 1], "--", color=col, lw=1.2, alpha=0.8)
    if len(cond["cam_t"]):
        axd.scatter(cond["cam_t"], cond["cam_e"], s=12, color=col, alpha=0.4, edgecolor="none", zorder=2)
if C1["crash"] is not None:
    axd.axvline(C1["crash"], color=C1C, ls=":", lw=1.4, alpha=0.8)
    axd.text(C1["crash"], 0.51, "C1 collision", color=C1C, fontsize=7.5, ha="center", va="bottom")
axd.axhline(0.5, color="0.4", ls=":", lw=1.0)
axd.set_title(r"(c) Truth--belief error with $2\sigma$ belief radius", fontsize=11.5, fontweight="bold")
axd.set_xlabel("time after first command (s)"); axd.set_ylabel("position error / radius (m)")
err_max = max(float(np.nanmax(C1["ep"][:, 1])), float(np.nanmax(C2["ep"][:, 1])), 0.5)
axd.set_ylim(0.0, min(0.75, max(0.55, 1.08 * err_max)))
axd.set_xlim(left=-0.3); axd.grid(alpha=0.25)
axd.legend(handles=[
    Line2D([0], [0], color=C1C, lw=2.0, label=f"C1 $e_p$ ({C1['mean_e']:.2f} m)"),
    Line2D([0], [0], color=C2C, lw=2.0, label=f"C2 $e_p$ ({C2['mean_e']:.2f} m)"),
    Line2D([0], [0], color="0.4", ls="--", lw=1.2, label=r"$2\sigma$ radius"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="0.4", markersize=6, label="camera update"),
], loc="upper left", fontsize=7.5, ncol=2, framealpha=0.85)

fig.subplots_adjust(left=0.07, right=0.94, top=0.94, bottom=0.08, hspace=0.34, wspace=0.16)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
selected_runs = {
    "C1": {
        "run_dir": C1["run_dir"],
        "completion_reason": C1["summary"].get("completion_reason"),
        "mean_truth_belief_error_after_first_cmd_m": C1["summary"].get("mean_truth_belief_error_after_first_cmd_m"),
        "mean_error_plotted_m": C1["mean_e"],
    },
    "C2": {
        "run_dir": C2["run_dir"],
        "completion_reason": C2["summary"].get("completion_reason"),
        "mean_truth_belief_error_after_first_cmd_m": C2["summary"].get("mean_truth_belief_error_after_first_cmd_m"),
        "mean_error_plotted_m": C2["mean_e"],
    },
}
settings = {
    "campaign": CAMP_NAME,
    "task": TASK,
    "seed": SEED,
    "world": C1["manifest"].get("world"),
    "nogo_mode": C1["manifest"].get("nogo_mode"),
    "nogo_penalty_type": C1["manifest"].get("nogo_penalty_type"),
    "nogo_weight": C1["manifest"].get("nogo_weight"),
    "nogo_warning_band": C1["manifest"].get("nogo_warning_band"),
    "use_belief_nogo_cost": C1["manifest"].get("use_belief_nogo_cost"),
    "nogo_belief_kappa": C1["manifest"].get("nogo_belief_kappa"),
    "global_horizon": C1["manifest"].get("global_horizon"),
    "dt": C1["manifest"].get("dt"),
    "v_max": C1["manifest"].get("v_max", _locked_config_value("v_max")),
    "optimizer_maxiter": C1["manifest"].get("optimizer_maxiter"),
    "optimizer_maxfun": C1["manifest"].get("optimizer_maxfun"),
    "optimizer_route_seed_mode": C1["manifest"].get("optimizer_route_seed_mode"),
    "yolo_use_masks": C1["manifest"].get("yolo_use_masks"),
    "yolo_conf_threshold": C1["manifest"].get("yolo_conf_threshold"),
    "yolo_iou_threshold": C1["manifest"].get("yolo_iou_threshold"),
    "gp_artifact": _rel(GPZ),
    "visibility_artifact_path_C2": C2["manifest"].get("visibility_artifact_path"),
    "map_xlim_m": MAP_XLIM,
    "map_ylim_m": MAP_YLIM,
    "git_sha_C1": C1["manifest"].get("git_sha"),
    "git_sha_C2": C2["manifest"].get("git_sha"),
}
provenance = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "script": _rel(Path(__file__)),
    "figure_layout": "Two large route maps plus one full-width truth-belief error panel. Map panels show global plan, executed trajectory, belief mean, continuous 2-sigma belief band, and lightly styled YOLO update/miss markers.",
    "output_pdf": _rel(OUT),
    "thesis_pdf": _rel(PAPER) if COPY_TO_THESIS else None,
    "source_campaign_log": _rel(CAMP / "campaign_log.json"),
    "settings": settings,
    "selected_runs": selected_runs,
}
provenance_path = _copy_source_data(selected_runs, provenance)
if COPY_TO_THESIS:
    PAPER.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT, PAPER)
    print(f"wrote {OUT}\ncopied -> {PAPER}")
else:
    print(f"wrote {OUT}")
print(f"wrote {provenance_path}")
print(f"copied source data -> {DATA_DIR}")
print(f"C1 reached={C1['reached']} crash@{C1['crash']}  C2 reached={C2['reached']}  "
      f"mean_e C1={C1['mean_e']:.3f} C2={C2['mean_e']:.3f}")
