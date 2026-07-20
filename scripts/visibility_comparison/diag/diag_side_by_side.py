#!/usr/bin/env python3
"""Side-by-side run video: LEFT = moving trajectory/belief plot over the
reliability map (+ truth-belief error vs time), RIGHT = the overhead camera
view of the driving robot with the YOLO box + selected pixel overlaid.

The two panels share a synchronized time cursor: as the robot drives on the
right, its truth (black) and belief (coloured) traces grow on the left, so you
can see how each camera detection moves the belief.

Requires camera frames captured for the run (yolo_debug_frame_dir set at launch,
saved under <frames_dir>/raw/frame_<pid>_<stamp>.png).

Usage:
    python diag_side_by_side.py --run-dir <run_dir> --frames-dir <frames_dir> \
        --cond C2 --out out.mp4 [--fps 12] [--nframes 240]
"""
import argparse
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import diag_common as dc

COLORS = {"C1": "#c0392b", "C2": "#e8820c"}


def load_series(run_dir):
    perc, exp, summary = dc.load_run(run_dir)
    e = exp.dropna(subset=["stamp", "odom_map_x", "odom_map_y"]).sort_values("stamp").reset_index(drop=True)
    t0 = dc.first_cmd_time(exp) or float(e["stamp"].iloc[0])
    e = e[e["stamp"] >= t0 - 2.0].copy()
    e["t"] = e["stamp"] - t0
    rho = dc.sample_rho(e["odom_map_x"].to_numpy(), e["odom_map_y"].to_numpy())
    return {
        "exp": e, "perc": perc, "summary": summary, "rho": rho, "t0": t0,
        "tmax": float(e["t"].max()),
        "goal": (float(e["goal_x"].iloc[-1]), float(e["goal_y"].iloc[-1])) if "goal_x" in e else None,
        "start": (float(e["odom_map_x"].iloc[0]), float(e["odom_map_y"].iloc[0])),
        "crash_t": (float(summary.get("first_crash_stamp", "nan") or "nan") - t0),
    }


def cov_ellipse(row, color):
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
    return Ellipse((float(row["planner_belief_x"]), float(row["planner_belief_y"])),
                   width=max(width, 0.03), height=max(height, 0.03), angle=ang,
                   fill=False, edgecolor=color, lw=1.2, alpha=0.7, zorder=6)


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


def render_plot_panel(r, prisms, g, cond, tt, dpi=100):
    col = COLORS.get(cond, "#444")
    e = r["exp"]
    fig = plt.figure(figsize=(6.6, 7.2), dpi=dpi)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.40], hspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(g["rho"], origin="lower", extent=g["extent"], cmap="viridis", vmin=0, vmax=0.9, alpha=0.9, zorder=0)
    for (xm, xM, ym, yM) in prisms:
        ax.plot([xm, xM, xM, xm, xm], [ym, ym, yM, yM, ym], color="#cde", lw=0.7, alpha=0.5, zorder=1)
    past = e[e["t"] <= tt]
    ax.plot(past["odom_map_x"], past["odom_map_y"], "-", color="#111", lw=2.4, zorder=4, label="truth")
    if "planner_belief_x" in past:
        pb = past.dropna(subset=["planner_belief_x"])
        ax.plot(pb["planner_belief_x"], pb["planner_belief_y"], "-", color=col, lw=1.9, alpha=0.9, zorder=5, label="belief")
    if len(past):
        row = past.iloc[-1]
        ax.scatter([row["odom_map_x"]], [row["odom_map_y"]], s=80, c="#111", zorder=8, marker="o", edgecolor="w")
        if np.isfinite(row.get("planner_belief_x", np.nan)):
            ax.scatter([row["planner_belief_x"]], [row["planner_belief_y"]], s=60, c=col, zorder=8, marker="o", edgecolor="w")
            ell = cov_ellipse(row, col)
            if ell is not None:
                ax.add_patch(ell)
    ax.scatter(*r["start"], s=95, c="lime", edgecolor="k", zorder=7)
    if r["goal"]:
        ax.scatter(*r["goal"], s=180, marker="*", c="red", edgecolor="k", zorder=7)
    if np.isfinite(r["crash_t"]) and tt >= r["crash_t"]:
        cr = e.iloc[(e["t"] - r["crash_t"]).abs().argmin()]
        ax.scatter([cr["odom_map_x"]], [cr["odom_map_y"]], s=240, marker="X", c="red", edgecolor="k", zorder=9)
    ax.set_title(f"{cond}: {r['summary'].get('completion_reason','?')}  (t={tt:5.1f}s)", fontsize=12, color=col)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="lower right")

    err_ax = fig.add_subplot(gs[1, 0])
    err_ax.plot(e["t"], e["belief_error_odom_m"], color=col, lw=1.0, alpha=0.25)
    pe = past.dropna(subset=["belief_error_odom_m"])
    if len(pe):
        err_ax.plot(pe["t"], pe["belief_error_odom_m"], color=col, lw=2.4)
        err_ax.scatter([pe["t"].iloc[-1]], [pe["belief_error_odom_m"].iloc[-1]], s=45, c=col, zorder=6)
    shade_occ(err_ax, e["t"].to_numpy(), r["rho"])
    err_ax.axvline(tt, color="#555", lw=0.9)
    err_ax.set_xlim(0, r["tmax"]); err_ax.set_ylim(0, None)
    err_ax.set_xlabel("time since first cmd (s)"); err_ax.set_ylabel("truth-belief err (m)")
    err_ax.set_title("red band = occluded (rho<0.5)", fontsize=9, loc="left")
    err_ax.grid(alpha=0.25)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
    out = cv2.cvtColor(buf.copy(), cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return out


def overlay_camera(img_bgr, prow):
    im = img_bgr.copy()
    if prow is not None:
        try:
            if all(np.isfinite([prow.get(k, np.nan) for k in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")])):
                x0, y0, x1, y1 = (int(prow["bbox_xmin"]), int(prow["bbox_ymin"]),
                                  int(prow["bbox_xmax"]), int(prow["bbox_ymax"]))
                cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)
            if np.isfinite(prow.get("obs_u", np.nan)) and np.isfinite(prow.get("obs_v", np.nan)):
                cv2.circle(im, (int(prow["obs_u"]), int(prow["obs_v"])), 7, (0, 220, 255), -1)
                cv2.circle(im, (int(prow["obs_u"]), int(prow["obs_v"])), 8, (0, 0, 0), 2)
        except Exception:
            pass
    return im


def nearest_perc_row(perc, stamp):
    if perc is None or "diag_stamp" not in perc:
        return None
    d = perc.dropna(subset=["diag_stamp"])
    if d.empty:
        return None
    idx = (d["diag_stamp"].astype(float) - stamp).abs().idxmin()
    return d.loc[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--frames-dir", required=True, help="dir containing raw/*.png camera frames")
    ap.add_argument("--cond", default="C2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--nframes", type=int, default=240)
    args = ap.parse_args()

    r = load_series(Path(args.run_dir))
    g = dc.gp()
    prisms = dc.driveable_prisms(Path(args.run_dir))
    frames_idx = dc.index_frames(Path(args.frames_dir))
    if not frames_idx:
        raise SystemExit(f"No camera frames found under {args.frames_dir}/raw/*.png")
    print(f"{len(frames_idx)} camera frames; run tmax={r['tmax']:.1f}s")

    times = np.linspace(0, r["tmax"], args.nframes)
    writer = None
    last_cam = None
    for tt in times:
        s_abs = r["t0"] + tt
        plot = render_plot_panel(r, prisms, g, args.cond, tt)
        fpath, _ = dc.frame_at(frames_idx, s_abs, tol=1.5)
        if fpath is not None:
            cam = cv2.imread(fpath)
            if cam is not None:
                prow = nearest_perc_row(r["perc"], s_abs)
                last_cam = overlay_camera(cam, prow)
        cam = last_cam
        if cam is None:
            cam = np.full((plot.shape[0], int(plot.shape[0] * 16 / 9), 3), 30, np.uint8)
        # resize camera to plot height
        scale = plot.shape[0] / cam.shape[0]
        cam_r = cv2.resize(cam, (int(cam.shape[1] * scale), plot.shape[0]))
        combo = cv2.hconcat([plot, cam_r])
        if writer is None:
            h, w = combo.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        writer.write(combo)
    if writer is not None:
        writer.release()
    sz = Path(args.out).stat().st_size / 1e6 if Path(args.out).exists() else 0
    print(f"wrote {args.out}  ({args.nframes} frames @ {args.fps}fps, {sz:.1f}MB)")


if __name__ == "__main__":
    main()
