#!/usr/bin/env python3
"""Presentation-grade figures (durable out-of-repo copy). Self-contained.

Reads /home/joostleliveld/Thesis/timing_presentation/runs/ and writes clean,
big-font figures to .../figures/.  F1 world+task | F2 multistart basin fix |
F3 solve scaling | F4 closed-loop failure | F5 pipeline timing budget.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

DURABLE = Path("/home/joostleliveld/Thesis/timing_presentation")
RUNS = DURABLE / "runs"
GAZ = RUNS / "gazebo"
OUT = DURABLE / "figures"
PROFILE = Path("/home/joostleliveld/Thesis/UnembodiedNavigation/src/experiments/config/world_profiles.yaml")
WORLD = "warehouse_aws.world.sdf"


def load_regions():
    import yaml
    w = yaml.safe_load(PROFILE.read_text())["worlds"][WORLD]
    regs = w.get("known_2d_regions", [])
    trav = [r for r in regs if r.get("type") == "traversable"]
    stag = [r for r in regs if r.get("type") == "non_driveable_staging"]
    return trav, stag


def draw_driveable(ax):
    trav, _ = load_regions()
    first = True
    for r in trav:
        ax.add_patch(Rectangle((r["xmin"], r["ymin"]), r["xmax"] - r["xmin"], r["ymax"] - r["ymin"],
                     facecolor="#b8e0b8", edgecolor="#7cc47c", lw=0.5, alpha=0.55, zorder=0,
                     label="driveable region" if first else None))
        first = False

START = (3.0, -1.0)
GOAL = (1.0, 2.0)
PLAN_PERIOD_S = 1.0

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 17, "axes.titleweight": "bold",
    "axes.labelsize": 14, "legend.fontsize": 11, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "figure.facecolor": "white", "axes.facecolor": "white"})

RUN_ORDER = ["H40_ms1", "H80_ms0", "H80_ms1", "H200_ms0", "H200_ms1"]
RUN_COLOR = {"H40_ms1": "tab:green", "H80_ms0": "tab:red", "H80_ms1": "tab:purple",
             "H200_ms0": "tab:blue", "H200_ms1": "tab:orange"}


def read_cols(path: Path):
    rows = list(csv.DictReader(path.open()))
    cols = {}
    if not rows:
        return cols, 0
    for k in rows[0].keys():
        out = np.full(len(rows), np.nan)
        for i, r in enumerate(rows):
            try:
                out[i] = float(r[k])
            except (TypeError, ValueError):
                pass
        cols[k] = out
    return cols, len(rows)


def find_run_dir(label):
    ld = GAZ / label
    if not ld.is_dir():
        return None
    c = [d for d in ld.iterdir() if d.is_dir() and (d / "experiment.csv").is_file()]
    return sorted(c)[-1] if c else None


def load_prisms(run_dir):
    cg = json.loads((run_dir / "run_manifest.json").read_text())["collision_geometry_json"]
    if isinstance(cg, str):
        cg = json.loads(cg)
    return cg.get("prisms", [])


def first_plan(run_dir):
    cols, n = read_cols(run_dir / "plan_samples.csv")
    if n == 0:
        return None
    ps = cols["plan_stamp"]; m = ps == np.nanmin(ps)
    return cols["x"][m], cols["y"][m]


def draw_layout(ax, prisms, legend=True):
    draw_driveable(ax)
    rl = cl = False
    for p in prisms:
        if "wall_" in p["name"]:
            continue
        w, h = p["xmax"] - p["xmin"], p["ymax"] - p["ymin"]
        crate = "low_crate" in p["name"]; r4 = "R4" in p["name"]
        face = "#e8a0a0" if crate else "#c8c8c8"; edge = "tab:red" if r4 else "0.5"
        lab = None
        if crate and not cl:
            lab, cl = "low crate", True
        elif (not crate) and not rl:
            lab, rl = "rack body", True
        ax.add_patch(Rectangle((p["xmin"], p["ymin"]), w, h, facecolor=face, edgecolor=edge,
                               lw=1.6 if r4 else 0.8, alpha=0.75, zorder=1, label=lab))
    ax.plot(*START, "o", color="k", ms=15, zorder=6, label="start (3,-1)")
    ax.plot(*GOAL, "*", color="k", ms=24, zorder=6, label="goal (1,2)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
    ax.grid(alpha=0.25); ax.set_xlim(-0.5, 4.4); ax.set_ylim(-3.3, 3.0)
    if legend:
        ax.legend(loc="lower left", framealpha=0.9, fontsize=9)


def load_offline():
    return list(csv.DictReader((RUNS / "offline_solve_scaling.csv").open()))


def fig1(prisms):
    fig, ax = plt.subplots(figsize=(10, 8))
    draw_layout(ax, prisms)
    ax.annotate("rack R4 body\n(blocks x<=2.27, y in [-0.8, 1.25])", xy=(2.0, 0.2),
                xytext=(2.6, 1.6), fontsize=12, color="tab:red", ha="left",
                arrowprops=dict(arrowstyle="->", color="tab:red"))
    ax.annotate("~0.31 m gap\n(robot dia. 0.25 m)", xy=(2.0, -0.95), xytext=(3.0, -1.7),
                fontsize=12, color="tab:red", ha="left",
                arrowprops=dict(arrowstyle="->", color="tab:red"))
    ax.set_title("B1 task: apron (3,-1) -> upper A3 (1,2) — R4 corner is the pinch point")
    fig.tight_layout(); fig.savefig(OUT / "F1_world_and_task.png", dpi=140); plt.close(fig)
    print("F1")


def fig2(prisms):
    rows = load_offline()
    H = sorted({int(r["horizon"]) for r in rows})
    td = {0: {}, 1: {}}
    for r in rows:
        td[int(r["multistart"])][int(r["horizon"])] = float(r["terminal_goal_distance_pred"])
    fig, (axb, axt) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.0, 1.15]})
    x = np.arange(len(H)); w = 0.38
    axb.bar(x - w / 2, [td[0][h] for h in H], w, color="tab:blue", label="single-start")
    axb.bar(x + w / 2, [td[1][h] for h in H], w, color="tab:red", label="multistart (4 seeds)")
    axb.axhline(0.5, color="k", ls=":", lw=1.5, label="goal-reach (0.5 m)")
    axb.set_xticks(x); axb.set_xticklabels([f"H={h}" for h in H])
    axb.set_ylabel("terminal distance to goal (m)")
    axb.set_title("Offline: multistart escapes the bad basin"); axb.legend(); axb.grid(alpha=0.25, axis="y")
    for xi, h in zip(x, H):
        axb.text(xi + w / 2, td[1][h] + 0.05, f"{td[1][h]:.2f}", ha="center", fontsize=10, color="tab:red")
    draw_layout(axt, prisms, legend=False)
    rd = find_run_dir("H200_ms1")
    if rd:
        fp = first_plan(rd)
        if fp is not None:
            axt.plot(fp[0], fp[1], "-", color="tab:orange", lw=3,
                     label="planned detour (H200, multistart)", zorder=5)
    axt.set_title("The visible detour: up aisle A4, across the top, into A3")
    axt.legend(loc="upper left", framealpha=0.9)
    fig.suptitle("Multistart fixes the half-circle problem — the route basin", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "F2_multistart_fix.png", dpi=140); plt.close(fig)
    print("F2")


def fig3():
    rows = load_offline()
    d = {0: [], 1: []}
    for r in rows:
        d[int(r["multistart"])].append((int(r["horizon"]), float(r["solve_s_mean"])))
    fig, ax = plt.subplots(figsize=(11, 7))
    for ms, color, lbl in [(0, "tab:blue", "single-start (1 seed)"), (1, "tab:red", "multistart (4 seeds)")]:
        pts = sorted(d[ms]); ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=color, lw=2.5, ms=9, label=lbl)
    ax.axhline(PLAN_PERIOD_S, color="k", ls="--", lw=2, label="real-time budget (1 s plan period)")
    mx = max(p[1] for p in d[1])
    ax.annotate(f"{mx:.0f} s at H=200", xy=(200, mx), xytext=(120, mx * 0.8), fontsize=13,
                color="tab:red", arrowprops=dict(arrowstyle="->", color="tab:red"))
    ax.set_xlabel("planning horizon (steps)"); ax.set_ylabel("solve time per plan (s)")
    ax.set_title("The detour needs a long horizon — but long horizon is too slow")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "F3_solve_scaling.png", dpi=140); plt.close(fig)
    print("F3")


def fig4(prisms):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.1, 1.0]})
    draw_layout(axL, prisms, legend=False)
    for label in RUN_ORDER:
        rd = find_run_dir(label)
        if not rd:
            continue
        cols, n = read_cols(rd / "experiment.csv")
        tx, ty = cols.get("truth_x"), cols.get("truth_y")
        if tx is None:
            continue
        m = np.isfinite(tx) & np.isfinite(ty); c = RUN_COLOR[label]
        axL.plot(tx[m], ty[m], "-", color=c, lw=2.5, label=label, zorder=4)
        if m.any():
            axL.plot(tx[m][-1], ty[m][-1], "X", color=c, ms=15, mec="k", zorder=7)
    axL.set_title("Every run collides at the R4 corner (X) after ~0.8 m")
    axL.legend(loc="upper left", framealpha=0.9, fontsize=10); axL.set_xlim(1.0, 4.0); axL.set_ylim(-2.0, 0.5)
    rd = find_run_dir("H200_ms1")
    if rd:
        cols, n = read_cols(rd / "experiment.csv")
        t = cols["stamp"]; t = t - t[np.isfinite(t)][0]
        nz = np.where(np.abs(cols["cmd_v"]) + np.abs(cols["cmd_w"]) > 1e-3)[0]
        idle = t[nz[0]] if nz.size else 0.0
        axR.plot(t, cols["cmd_v"], "-", color="tab:blue", lw=2, label="cmd_v (m/s)")
        axR.plot(t, cols["cmd_w"], "-", color="tab:orange", lw=2, label="cmd_w (rad/s)")
        axR.axhline(0, color="k", lw=0.6); axR.axvspan(0, idle, color="0.85", alpha=0.6)
        axR.annotate(f"idle ~{idle:.0f} s\n(planner solving)", xy=(idle / 2, 0.5), fontsize=12, ha="center", color="0.3")
        axR.set_xlabel("time (s)"); axR.set_ylabel("command")
        axR.set_title("H200 multistart: long idle, then open-loop pivot")
        axR.legend(loc="lower left"); axR.grid(alpha=0.3)
    fig.suptitle("Closed-loop breaks: solve-latency starvation + corner clip", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "F4_closedloop_fail.png", dpi=140); plt.close(fig)
    print("F4")


def fig5():
    dets, pixs, bels = [], [], []
    for label in RUN_ORDER:
        rd = find_run_dir(label)
        if not rd:
            continue
        pc, pn = read_cols(rd / "perception.csv") if (rd / "perception.csv").is_file() else ({}, 0)
        if pn:
            ds = np.sort(pc.get("diag_stamp", np.array([]))); ds = ds[np.isfinite(ds)]
            dd = np.diff(ds); dd = dd[(dd > 0) & (dd < 5)]
            if dd.size:
                dets.append(1.0 / np.median(dd))
            pa = pc.get("pixel_pose_age_s", np.array([])); pa = pa[np.isfinite(pa)]
            if pa.size:
                pixs.append(float(np.mean(pa)))
        ec, en = read_cols(rd / "experiment.csv")
        ba = ec.get("planner_belief_age_s", np.array([])); ba = ba[np.isfinite(ba)]
        if ba.size:
            bels.append(float(np.mean(ba)))
    det = np.mean(dets) if dets else 5.0
    pix = np.mean(pixs) if pixs else 0.6
    bel = np.mean(bels) if bels else 0.1
    off = {(int(r["multistart"]), int(r["horizon"])): float(r["solve_s_mean"]) for r in load_offline()}
    s_single = off.get((0, 200), 2.68); s_multi = off.get((1, 200), 27.9)
    labels = ["camera\nframe", "belief\nupdate", "command\npublish", "YOLO detect\n(rate)",
              "YOLO pose\nstaleness", "solve\nsingle H200", "solve\nmultistart H200"]
    vals = [1 / 30.0, bel, 0.1, 1 / det, pix, s_single, s_multi]
    colors = ["tab:green", "tab:green", "tab:green", "tab:orange", "tab:orange", "tab:red", "tab:red"]
    fig, ax = plt.subplots(figsize=(12, 7)); y = np.arange(len(labels))
    ax.barh(y, vals, color=colors)
    ax.axvline(PLAN_PERIOD_S, color="k", ls="--", lw=2, label="1 s plan period")
    for yi, v in zip(y, vals):
        ax.text(v * 1.1, yi, f"{v:.2f}s ({1/v:.0f} Hz)" if v < 0.2 else f"{v:.1f} s", va="center", fontsize=11)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xscale("log")
    ax.set_xlabel("latency / period  (s, log scale)")
    ax.set_title("Pipeline timing budget — solver (red) is the bottleneck")
    ax.invert_yaxis(); ax.legend(loc="lower right"); ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(OUT / "F5_pipeline_timing.png", dpi=140); plt.close(fig)
    print("F5")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rd = find_run_dir("H200_ms1") or find_run_dir("H80_ms1") or find_run_dir("H200_ms0")
    prisms = load_prisms(rd) if rd else []
    fig1(prisms); fig2(prisms); fig3(); fig4(prisms); fig5()
    print("figures in", OUT)


if __name__ == "__main__":
    main()
