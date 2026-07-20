#!/usr/bin/env python3
"""Per-task route animation: robot truth + belief moving over the reliability
map, with a truth-belief-error-vs-time panel shaded by occlusion. Generalises
the midterm `enhanced_06_paired_route_animation.gif` to any run(s).

Truth-belief error should GROW in occluded (red-shaded, rho<0.5) intervals and
RECOVER in visible intervals -- the deliverable the supervisor asked for.

Usage:
    # single C2 run:
    python diag_route_animation.py C2=<run_dir> [--out file.gif] [--frames 80]
    # paired C1 vs C2:
    python diag_route_animation.py C1=<run_dir> C2=<run_dir> [--out file.gif]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
import imageio

import diag_common as dc

COLORS = {"C1": "#e8000b", "C2": "#1a5fd6"}


def load_series(run_dir):
    perc, exp, summary = dc.load_run(run_dir)
    if {"gt_x", "gt_y"}.issubset(exp.columns) and exp["gt_x"].notna().any():
        xkey, ykey = "gt_x", "gt_y"
        err_key = "belief_error_gt_m" if "belief_error_gt_m" in exp.columns else "belief_error_odom_m"
        truth_label = "ground-truth"
    elif {"truth_x", "truth_y"}.issubset(exp.columns) and exp["truth_x"].notna().any():
        xkey, ykey = "truth_x", "truth_y"
        err_key = "truth_belief_error_m" if "truth_belief_error_m" in exp.columns else "belief_error_odom_m"
        truth_label = "legacy paper truth"
    else:
        xkey, ykey = "odom_map_x", "odom_map_y"
        err_key = "belief_error_odom_m"
        truth_label = "odom-as-truth"
    e = exp.dropna(subset=["stamp", xkey, ykey]).sort_values("stamp").reset_index(drop=True)
    t0 = dc.first_cmd_time(exp) or float(e["stamp"].iloc[0])
    e = e[e["stamp"] >= t0 - 2.0].copy()
    e["t"] = e["stamp"] - t0
    rho = dc.sample_rho(e[xkey].to_numpy(), e[ykey].to_numpy())
    return {
        "exp": e, "summary": summary, "rho": rho,
        "tmax": float(e["t"].max()),
        "goal": (float(e["goal_x"].iloc[-1]), float(e["goal_y"].iloc[-1])) if "goal_x" in e else None,
        "start": (float(e[xkey].iloc[0]), float(e[ykey].iloc[0])),
        "crash_t": (float(summary.get("first_crash_stamp", "nan") or "nan") - t0),
        "xkey": xkey,
        "ykey": ykey,
        "err_key": err_key,
        "truth_label": truth_label,
    }


def shade_occ(ax, t, rho, thr=0.5):
    occ = rho < thr
    start = None
    for i in range(len(t)):
        if occ[i] and start is None:
            start = t[i]
        elif not occ[i] and start is not None:
            ax.axvspan(start, t[i], color="#b00", alpha=0.10, lw=0); start = None
    if start is not None:
        ax.axvspan(start, t[-1], color="#b00", alpha=0.10, lw=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="COND=run_dir entries, e.g. C2=/path/run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    runs = {}
    for item in args.runs:
        cond, _, rd = item.partition("=")
        runs[cond] = load_series(Path(rd))
    conds = list(runs.keys())
    g = dc.gp()
    racks = []
    # racks from driveable prisms of first run (keep-in rects are the aisles; rack
    # blocks are the gaps between them -- just draw prisms as guide)
    prisms = dc.driveable_prisms(Path(args.runs[0].split("=", 1)[1]))

    tmax = max(r["tmax"] for r in runs.values())
    times = np.linspace(0, tmax, args.frames)

    out = Path(args.out) if args.out else (Path(args.runs[0].split("=", 1)[1]) / "diag" / "route_animation.gif")
    out.parent.mkdir(parents=True, exist_ok=True)

    vid = []
    ncol = len(conds)
    for tt in times:
        fig = plt.figure(figsize=(6.4 * ncol, 7.2), dpi=95)
        gs = fig.add_gridspec(2, ncol, height_ratios=[1.0, 0.42], hspace=0.28, wspace=0.16)
        err_ax = fig.add_subplot(gs[1, :])
        for ci, cond in enumerate(conds):
            r = runs[cond]; e = r["exp"]; col = COLORS.get(cond, "#444")
            xkey, ykey = r["xkey"], r["ykey"]
            ax = fig.add_subplot(gs[0, ci])
            ax.imshow(g["rho"], origin="lower", extent=g["extent"], cmap="viridis",
                      vmin=0, vmax=0.9, alpha=0.9, zorder=0)
            for (xm, xM, ym, yM) in prisms:
                ax.plot([xm, xM, xM, xm, xm], [ym, ym, yM, yM, ym], color="#cde", lw=0.7, alpha=0.5, zorder=1)
            past = e[e["t"] <= tt]
            ax.plot(past[xkey], past[ykey], "-", color="#111", lw=2.2, zorder=4)
            if "planner_belief_x" in past:
                pb = past.dropna(subset=["planner_belief_x"])
                ax.plot(pb["planner_belief_x"], pb["planner_belief_y"], "-", color=col, lw=1.8, alpha=0.9, zorder=5)
            if len(past):
                row = past.iloc[-1]
                ax.scatter([row[xkey]], [row[ykey]], s=70, c="#111", zorder=8, marker="o", edgecolor="w")
                if np.isfinite(row.get("planner_belief_x", np.nan)):
                    ax.scatter([row["planner_belief_x"]], [row["planner_belief_y"]], s=55, c=col, zorder=8, marker="o", edgecolor="w")
                    ell = covariance_ellipse(row, col)
                    if ell is not None:
                        ax.add_patch(ell)
            ax.scatter(*r["start"], s=90, c="lime", edgecolor="k", zorder=7)
            if r["goal"]:
                ax.scatter(*r["goal"], s=170, marker="*", c="red", edgecolor="k", zorder=7)
            if np.isfinite(r["crash_t"]) and tt >= r["crash_t"]:
                ci_row = e.iloc[(e["t"] - r["crash_t"]).abs().argmin()]
                ax.scatter([ci_row[xkey]], [ci_row[ykey]], s=220, marker="X", c="red", edgecolor="k", zorder=9)
            ax.set_title(f"{cond}: {r['summary'].get('completion_reason','?')}", fontsize=11, color=col)
            ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")

        # error panel
        for cond in conds:
            r = runs[cond]; e = r["exp"]; col = COLORS.get(cond, "#444")
            err_key = r["err_key"]
            err_ax.plot(e["t"], e[err_key], color=col, lw=1.0, alpha=0.25)
            past = e[e["t"] <= tt].dropna(subset=[err_key])
            if len(past):
                err_ax.plot(past["t"], past[err_key], color=col, lw=2.4, label=f"{cond} {r['truth_label']}-belief err")
                err_ax.scatter([past["t"].iloc[-1]], [past[err_key].iloc[-1]], s=40, c=col, zorder=6)
        # occlusion shading from first run's rho
        r0 = runs[conds[0]]
        shade_occ(err_ax, r0["exp"]["t"].to_numpy(), r0["rho"])
        err_ax.axvline(tt, color="#555", lw=0.8)
        err_ax.set_xlim(0, tmax); err_ax.set_ylim(0, None)
        err_ax.set_xlabel("time since first cmd (s)"); err_ax.set_ylabel("truth-belief error (m)")
        err_ax.legend(fontsize=9, loc="upper left")
        err_ax.set_title("Truth-belief error (red band = occluded, rho<0.5): grows occluded, recovers visible", fontsize=10, loc="left")
        err_ax.grid(alpha=0.25)

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        vid.append(buf.copy())
        plt.close(fig)

    imageio.mimsave(out, vid, duration=1.0 / max(args.fps, 1e-3), loop=0)
    print(f"wrote {out}  ({len(vid)} frames, {out.stat().st_size/1e6:.1f}MB)")


def covariance_ellipse(row, color):
    vals = [row.get("planner_cov_x", np.nan), row.get("planner_cov_xy", 0.0), row.get("planner_cov_y", np.nan)]
    if not all(np.isfinite(vals)):
        return None
    cov = np.array([[vals[0], vals[1]], [vals[1], vals[2]]], float)
    cov = (cov + cov.T) / 2.0
    w, v = np.linalg.eigh(cov)
    w = np.maximum(w, 0.0)
    order = w.argsort()[::-1]; w = w[order]; v = v[:, order]
    ang = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
    width, height = 2.0 * 2.0 * np.sqrt(w)
    e = Ellipse((float(row["planner_belief_x"]), float(row["planner_belief_y"])),
                width=max(width, 0.03), height=max(height, 0.03), angle=ang,
                fill=False, edgecolor=color, lw=1.0, alpha=0.6, zorder=6)
    return e


if __name__ == "__main__":
    main()
