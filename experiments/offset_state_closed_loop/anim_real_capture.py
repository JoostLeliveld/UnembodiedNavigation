#!/usr/bin/env python3
"""BEV animation of a REAL Gazebo capture: which camera is talking, and what the
offset-state filter makes of it.

Everything here is recorded data. No simulated trajectory, no injected fault.

  odometry      the real `odom_noisy` track from the run
  detections    the real detector pixels recorded per camera, re-projected with the
                CURRENT floor-plane IPM (`camera.pixel_to_world`) -- the stored pixel
                lets us bypass the retired v2 correction path entirely
  geometry      the real camera poses from `warehouse_full_4cam.world.sdf`
  ground truth  drawn only, and used only for the per-camera reference bias measured
                on THIS capture. It never enters the filter.

The filter is the offset-state form: state is position plus a 2-D offset per camera,
so a camera that reads consistently off to one side is something it can explain.

Panels
  left          the warehouse from above: racks, aisles, the four wall cameras, the
                robot's real track, the filter's belief with its +/-2 sigma ellipse,
                and a line from whichever camera reported to the point it reported.
  top right     close up, the only panel where centimetres are visible.
  middle right  each camera's estimated offset over time, against the bias actually
                measured for that camera on this capture.
  bottom right  where the blame for the latest sighting went.

  python3 anim_real_capture.py --list
  python3 anim_real_capture.py --capture fusion_handover_20260721 --still 400
  python3 anim_real_capture.py --capture fusion_handover_20260721

Outputs -> logs/studies/offset_state_closed_loop/real_capture_animation/
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse, Rectangle

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
for _rel in ("src/reliability", "src/unav_common",
             "experiments/operational_residual_rcond"):
    sys.path.insert(0, str(REPO / _rel))

import rcond_common as rc                                    # noqa: E402
from reliability.projection import camera_model_from_world    # noqa: E402

OUT = REPO / "logs/studies/offset_state_closed_loop/real_capture_animation"
WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"

CAMS = list(rc.CAMERAS)
SHORT = {c: c.replace("camera_", "") for c in CAMS}
CAM_XY = {"camera_A": (-6.0, -10.0), "camera_B": (-6.0, 10.0),
          "camera_C": (6.0, -10.0), "camera_D": (6.0, 10.0)}
CAM_COLOR = {"camera_A": "#5F6A73", "camera_B": "#56B4E9",
             "camera_C": "#D55E00", "camera_D": "#CC79A7"}

AISLES = [
    (-10.95, -9.42, -8.35, 8.35), (-8.23, -7.32, -8.35, 8.35),
    (-6.13, -5.22, -8.35, 8.35), (-4.03, -3.12, -8.35, 8.35),
    (-1.93, 1.93, -8.35, -1.47), (-1.93, -0.57, -1.47, -0.33),
    (0.57, 1.93, -1.47, -0.33), (-1.93, 1.93, -0.33, 8.35),
    (3.12, 4.03, -8.35, 8.35), (5.22, 6.13, -8.35, 8.35),
    (7.32, 8.23, -8.35, 8.35), (9.42, 11.25, -8.35, 8.35),
    (-10.95, 11.25, -8.35, -6.82), (-10.95, 11.25, -2.48, -1.52),
    (-10.95, 11.25, 1.92, 2.88), (-10.95, 11.25, 7.22, 8.35),
]
RACK_X = [(-9.42, -8.23), (-7.32, -6.13), (-5.22, -4.03), (-3.12, -1.93),
          (1.93, 3.12), (4.03, 5.22), (6.13, 7.32), (8.23, 9.42)]
RACK_Y = [(-6.82, -2.48), (-1.52, 1.92), (2.88, 7.22)]
PILLAR = (-0.25, 0.25, -1.15, -0.65)

# ---- filter settings, stated not tuned to the outcome
Q_POS_PER_M = 0.04 ** 2        # covariance added per metre of odometry travelled
R_SIGMA_AT_10M = 0.055         # camera scatter, grows with range
OFFSET_PRIOR = 0.20
OFFSET_WALK_PER_S = 0.002 ** 2
ZOOM_HALF = 0.30

INK = "#1A1A1A"
TRUTH_C = "#009E73"
BELIEF_C = "#0072B2"
FLOOR = "#EDE6D8"
DRIVE = "#BFE0F2"
RACK = "#C6CED5"


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#666666", "text.color": INK,
        "xtick.color": "#555555", "ytick.color": "#555555",
    })


def load_real(capture: str):
    """Recorded odometry + recorded pixels re-projected with the CURRENT IPM."""
    models = {c: camera_model_from_world(WORLD, include_name=rc.MODEL_INCLUDES[c])
              for c in CAMS}
    calib = rc.deployed_calibration()
    cap = rc.load_operational_capture(capture, models=models, calib=calib)

    dets = []
    for c in CAMS:
        for d in cap.detections[c]:
            # d.world came from the retired v2 path; re-project the stored pixel.
            world = models[c].pixel_to_world(d.u, d.v)
            if world is None:
                continue
            dets.append({"camera": c, "stamp": float(d.stamp),
                         "z": np.array([float(world[0]), float(world[1])]),
                         "range": float(d.range_m)})
    dets.sort(key=lambda r: r["stamp"])
    return cap, dets, models


def run_filter(cap, dets, truth_table):
    """Offset-state filter over the real detections. Truth used only for scoring."""
    dim = 2 + 2 * len(CAMS)
    slot = {c: 2 + 2 * i for i, c in enumerate(CAMS)}
    mean = np.zeros(dim)
    mean[0:2] = cap.odom[0]
    cov = np.zeros((dim, dim))
    cov[0, 0] = cov[1, 1] = 0.10 ** 2
    for i in range(2, dim):
        cov[i, i] = OFFSET_PRIOR ** 2

    stamps = cap.stamps
    idx_prev, stamp_prev = 0, float(stamps[0])
    frames = []
    for d in dets:
        idx = int(np.searchsorted(stamps, d["stamp"]))
        idx = min(max(idx, 0), len(stamps) - 1)
        if idx > idx_prev:
            step = cap.odom[idx] - cap.odom[idx_prev]
            dist = float(np.hypot(*step))
            mean[0:2] += step
            cov[0, 0] += Q_POS_PER_M * max(dist, 1e-6)
            cov[1, 1] += Q_POS_PER_M * max(dist, 1e-6)
            idx_prev = idx
        dt = max(d["stamp"] - stamp_prev, 0.0)
        stamp_prev = d["stamp"]
        for i in range(2, dim):
            cov[i, i] += OFFSET_WALK_PER_S * dt

        c = d["camera"]
        sig = R_SIGMA_AT_10M * max(d["range"], 1.0) / 10.0
        H = np.zeros((2, dim))
        H[:, 0:2] = np.eye(2)
        H[:, slot[c]:slot[c] + 2] = np.eye(2)
        R = np.eye(2) * sig ** 2
        pht = cov @ H.T
        S = H @ pht + R
        innov = d["z"] - H @ mean

        tr_s = float(np.trace(S))
        share = {
            "pos": float(np.trace(cov[0:2, 0:2] + cov[0:2, slot[c]:slot[c] + 2])) / tr_s,
            "off": float(np.trace(cov[slot[c]:slot[c] + 2, slot[c]:slot[c] + 2]
                                  + cov[slot[c]:slot[c] + 2, 0:2])) / tr_s,
            "noise": float(np.trace(R)) / tr_s,
        }
        gain = pht @ np.linalg.inv(S)
        mean = mean + gain @ innov
        cov = cov - gain @ pht.T
        cov = 0.5 * (cov + cov.T)

        hit = rc.truth_at(truth_table, d["stamp"])
        frames.append({
            "stamp": d["stamp"], "camera": c, "z": d["z"].copy(),
            "range": d["range"], "share": share,
            "mean": mean[0:2].copy(), "cov": cov[0:2, 0:2].copy(),
            "truth": None if hit is None else np.array(hit[:2], dtype=float),
            "offsets": {cc: mean[slot[cc]:slot[cc] + 2].copy() for cc in CAMS},
            "offset_sig": {cc: float(np.sqrt(max(
                np.trace(cov[slot[cc]:slot[cc] + 2, slot[cc]:slot[cc] + 2]) / 2, 0.0)))
                for cc in CAMS},
        })
    return frames


def measured_bias(frames):
    """Per-camera mean (projected - truth) ON THIS CAPTURE. Evaluation only."""
    acc = {c: [] for c in CAMS}
    for f in frames:
        if f["truth"] is not None:
            acc[f["camera"]].append(f["z"] - f["truth"])
    return {c: (np.mean(v, axis=0) if v else None) for c, v in acc.items()}, \
           {c: len(v) for c, v in acc.items()}


def draw_bev_static(ax, track) -> None:
    ax.add_patch(Rectangle((-12, -10), 24, 20, facecolor=FLOOR, edgecolor="none", zorder=0))
    for x0, x1, y0, y1 in AISLES:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=DRIVE,
                               edgecolor="none", zorder=1))
    for x0, x1 in RACK_X:
        for y0, y1 in RACK_Y:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=RACK,
                                   edgecolor="white", lw=0.8, zorder=3))
    ax.add_patch(Rectangle((PILLAR[0], PILLAR[2]), PILLAR[1] - PILLAR[0],
                           PILLAR[3] - PILLAR[2], facecolor=INK, zorder=5))
    ax.plot(track[:, 0], track[:, 1], color="#7C8B97", lw=1.0, ls=(0, (3, 3)), zorder=4)
    ax.plot([track[0, 0]], [track[0, 1]], marker="s", ms=7, color="#4C5661", zorder=6)
    ax.text(track[0, 0], track[0, 1] - 0.9, "run starts here", ha="center", va="top",
            fontsize=8.5, fontweight="bold", color="#4C5661", zorder=8)
    for c, (cx, cy) in CAM_XY.items():
        ax.add_patch(Rectangle((cx - 0.75, cy - 0.5), 1.5, 1.0, facecolor=CAM_COLOR[c],
                               edgecolor=INK, lw=1.0, zorder=7))
        ax.text(cx, cy + (-1.5 if cy > 0 else 1.5), f"camera {SHORT[c]}", ha="center",
                va="center", fontsize=10.5, fontweight="bold", color=CAM_COLOR[c], zorder=8)
        dy = -2.4 if cy > 0 else 2.4
        ax.annotate("", xy=(cx, cy + dy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color=CAM_COLOR[c], lw=1.7), zorder=7)
    ax.set_xlim(-12.2, 12.2)
    ax.set_ylim(-11.6, 12.4)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)


def build(capture, frames, bias, counts):
    _style()
    t0 = frames[0]["stamp"]
    ts = np.array([f["stamp"] - t0 for f in frames])
    track = np.array([f["truth"] if f["truth"] is not None else f["mean"]
                      for f in frames])

    fig = plt.figure(figsize=(15.6, 8.8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.28],
                          height_ratios=[0.88, 1.0, 0.58],
                          wspace=0.16, hspace=0.82, left=0.015, right=0.965,
                          top=0.862, bottom=0.075)
    ax_bev = fig.add_subplot(gs[:, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_off = fig.add_subplot(gs[1, 1])
    ax_bl = fig.add_subplot(gs[2, 1])

    draw_bev_static(ax_bev, track)
    trail_t, = ax_bev.plot([], [], color=TRUTH_C, lw=2.2, alpha=0.6, zorder=8)
    trail_b, = ax_bev.plot([], [], color=BELIEF_C, lw=1.5, alpha=0.6, ls="--", zorder=8)
    dot_t, = ax_bev.plot([], [], marker="o", ms=9, color=TRUTH_C, zorder=11,
                         markeredgecolor="white", markeredgewidth=1.2)
    dot_b, = ax_bev.plot([], [], marker="D", ms=7, color=BELIEF_C, zorder=11,
                         markeredgecolor="white", markeredgewidth=1.0)
    ray, = ax_bev.plot([], [], lw=1.7, alpha=0.95, zorder=9)
    dot_z, = ax_bev.plot([], [], marker="o", ms=6, zorder=10,
                         markeredgecolor="white", markeredgewidth=0.8)
    ell = Ellipse((0, 0), 0, 0, facecolor=BELIEF_C, alpha=0.26, edgecolor=BELIEF_C,
                  lw=1.2, zorder=6)
    ax_bev.add_patch(ell)
    zoom_box = Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=INK, lw=1.2,
                         ls=(0, (3, 2)), zorder=12)
    ax_bev.add_patch(zoom_box)
    banner = ax_bev.text(0.0, 11.7, "", ha="center", va="center", fontsize=11.5,
                         fontweight="bold", color=INK, zorder=12)

    # ---- close up
    ax_zoom.set_xticks([]); ax_zoom.set_yticks([])
    ax_zoom.set_facecolor(DRIVE)
    ax_zoom.set_aspect("equal")
    for s in ax_zoom.spines.values():
        s.set_edgecolor(INK); s.set_linewidth(1.2)
    ax_zoom.set_title(f"Close up — ±{ZOOM_HALF * 100:.0f} cm across\n"
                      "the only panel where centimetres are visible",
                      fontsize=10.5, fontweight="bold", loc="left")
    zt, = ax_zoom.plot([], [], marker="o", ms=13, color=TRUTH_C,
                       markeredgecolor="white", markeredgewidth=1.4, zorder=6)
    zb, = ax_zoom.plot([], [], marker="D", ms=11, color=BELIEF_C,
                       markeredgecolor="white", markeredgewidth=1.2, zorder=6)
    zz, = ax_zoom.plot([], [], marker="o", ms=9, markeredgecolor="white", zorder=5)
    zell = Ellipse((0, 0), 0, 0, facecolor=BELIEF_C, alpha=0.28, edgecolor=BELIEF_C, lw=1.4)
    ax_zoom.add_patch(zell)
    zgap, = ax_zoom.plot([], [], color=INK, lw=1.4, zorder=7)
    zlab = ax_zoom.text(0, 0, "", fontsize=9.5, fontweight="bold", color=INK, zorder=8)
    for m, col, lab in ((("o", 9), TRUTH_C, "really here (ground truth)"),
                        (("D", 8), BELIEF_C, "filter's belief (shaded = ±2σ)"),
                        (("o", 7), "#5F6A73", "what the camera reported")):
        ax_zoom.plot([], [], marker=m[0], ms=m[1], color=col, ls="none", label=lab)
    ax_zoom.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                   frameon=False, handletextpad=0.4)

    # ---- per-camera offset estimates
    # The north-south component, signed. Almost all of the real per-camera bias lives
    # there -- plotting the magnitude instead hid the sign, and the sign is the whole
    # point: the south pair reads short, the north pair reads long.
    lines = {}
    for c in CAMS:
        lines[c], = ax_off.plot([], [], color=CAM_COLOR[c], lw=2.2,
                                label=f"{SHORT[c]} ({counts[c]} sightings)")
        if bias[c] is not None:
            ax_off.axhline(100 * float(bias[c][1]), color=CAM_COLOR[c], lw=1.4,
                           ls=":", alpha=0.9)
    ax_off.axhline(0.0, color="#AAAAAA", lw=1.0)
    ax_off.set_xlim(0, ts[-1])
    ax_off.grid(color="#E4E4E4", lw=0.6)
    ax_off.set_ylabel("offset it has worked out,\nnorth–south component (cm)")
    ax_off.set_xlabel("seconds into the run")
    ax_off.set_title("What the filter concludes about each camera\n"
                     "solid = its own estimate · dotted = measured against truth "
                     "on this run",
                     fontsize=10.5, fontweight="bold", loc="left")
    now_off = ax_off.axvline(0, color=INK, lw=1.0, ls=":")
    ax_off.legend(fontsize=8.5, ncol=4, loc="upper center", frameon=False,
                  bbox_to_anchor=(0.5, -0.34))

    # ---- blame split
    ax_bl.set_xlim(0, 100)
    ax_bl.set_ylim(-0.7, 2.7)
    ax_bl.set_yticks([2, 1, 0])
    ax_bl.set_yticklabels(["my position\nwas wrong", "that camera\nreads off",
                           "just noise in\nthis sighting"], fontsize=9)
    ax_bl.set_xlabel("share of the latest surprise (%)")
    ax_bl.grid(axis="x", color="#E4E4E4", lw=0.6)
    ax_bl.set_title("Every sighting is split three ways, by how unsure it currently is",
                    fontsize=10.5, fontweight="bold", loc="left")
    bars = ax_bl.barh([2, 1, 0], [0, 0, 0], height=0.52,
                      color=[BELIEF_C, CAM_COLOR["camera_C"], "#BBBBBB"])
    bl_txt = ax_bl.text(99, -0.55, "", ha="right", va="center", fontsize=9.5,
                        color="#55606A")

    fig.suptitle(f"Real Gazebo run {capture}: which camera is talking, "
                 "and what the filter makes of it", fontsize=13.5, fontweight="bold")
    fig.text(0.5, 0.922,
             "Recorded odometry and recorded detector pixels, re-projected with the "
             "current floor-plane IPM. Nothing simulated, no injected fault. Ground "
             "truth is drawn and used for the dotted reference lines only.",
             ha="center", va="top", fontsize=8.8, color="#444444")

    off_hist = {c: [] for c in CAMS}

    def render(i):
        f = frames[i]
        mx, my = f["mean"]
        tr = f["truth"]
        trail_b.set_data([g["mean"][0] for g in frames[:i + 1]],
                         [g["mean"][1] for g in frames[:i + 1]])
        tt = [g["truth"] for g in frames[:i + 1] if g["truth"] is not None]
        if tt:
            trail_t.set_data([p[0] for p in tt], [p[1] for p in tt])
        dot_b.set_data([mx], [my])
        if tr is not None:
            dot_t.set_data([tr[0]], [tr[1]])

        vals, vecs = np.linalg.eigh(f["cov"])
        vals = np.maximum(vals, 1e-9)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        for e in (ell, zell):
            e.set_center((mx, my))
            e.width = 4 * np.sqrt(vals[0])
            e.height = 4 * np.sqrt(vals[1])
            e.angle = ang

        c = f["camera"]
        col = CAM_COLOR[c]
        cam = np.array(CAM_XY[c])
        ray.set_data([cam[0], f["z"][0]], [cam[1], f["z"][1]])
        ray.set_color(col)
        dot_z.set_data([f["z"][0]], [f["z"][1]])
        dot_z.set_color(col)
        zz.set_data([f["z"][0]], [f["z"][1]])
        zz.set_color(col)
        banner.set_text(f"camera {SHORT[c]} reporting, from {f['range']:.1f} m away"
                        f"   ·   {f['stamp'] - t0:.0f} s in")
        banner.set_color(col)

        sh = f["share"]
        for bar, v in zip(bars, [sh["pos"] * 100, sh["off"] * 100, sh["noise"] * 100]):
            bar.set_width(v)
        bars[1].set_color(col)
        bl_txt.set_text(f"from camera {SHORT[c]}")

        anchor = tr if tr is not None else np.array([mx, my])
        ax_zoom.set_xlim(anchor[0] - ZOOM_HALF, anchor[0] + ZOOM_HALF)
        ax_zoom.set_ylim(anchor[1] - ZOOM_HALF, anchor[1] + ZOOM_HALF)
        zoom_box.set_bounds(anchor[0] - ZOOM_HALF, anchor[1] - ZOOM_HALF,
                            2 * ZOOM_HALF, 2 * ZOOM_HALF)
        if tr is not None:
            zt.set_data([tr[0]], [tr[1]])
            zgap.set_data([tr[0], mx], [tr[1], my])
            zlab.set_position((anchor[0] - ZOOM_HALF * 0.92,
                               anchor[1] + ZOOM_HALF * 0.82))
            zlab.set_text(f"{100 * float(np.hypot(mx - tr[0], my - tr[1])):.0f} cm apart")
        zb.set_data([mx], [my])

        for cc in CAMS:
            off_hist[cc] = [100 * float(g["offsets"][cc][1]) for g in frames[:i + 1]]
            lines[cc].set_data(ts[:i + 1], off_hist[cc])
        seen = [v for v in off_hist.values() if v]
        lo = min([min(v) for v in seen] + [100 * float(b[1]) for b in bias.values()
                                           if b is not None] + [0.0])
        hi = max([max(v) for v in seen] + [100 * float(b[1]) for b in bias.values()
                                          if b is not None] + [0.0])
        pad = max(2.0, 0.22 * (hi - lo))
        ax_off.set_ylim(lo - pad, hi + pad)
        now_off.set_xdata([ts[i], ts[i]])
        return ()

    return fig, render


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", default="fusion_handover_20260721")
    parser.add_argument("--still", type=int, default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for name in rc.CAPTURES:
            print(f"  {name}")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    cap, dets, _ = load_real(args.capture)
    truth_table = rc.load_truth_table(args.capture)
    frames = run_filter(cap, dets, truth_table)
    bias, counts = measured_bias(frames)

    print(f"  {args.capture}: {cap.duration_s:.0f} s, {len(frames)} real detections")
    for c in CAMS:
        b = bias[c]
        est = frames[-1]["offsets"][c]
        sig = frames[-1]["offset_sig"][c]
        if b is None:
            print(f"    {SHORT[c]}: never seen in this run — offset stays at the prior")
            continue
        print(f"    {SHORT[c]}: {counts[c]:4d} sightings | measured bias "
              f"{100 * float(np.linalg.norm(b)):5.1f} cm | filter says "
              f"{100 * float(np.linalg.norm(est)):5.1f} cm (±{100 * sig:.1f})")

    fig, render = build(args.capture, frames, bias, counts)
    if args.still is not None:
        i = max(0, min(args.still, len(frames) - 1))
        render(i)
        path = OUT / f"{args.capture}_still_{i:04d}.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote -> {path}")
        return 0

    anim = FuncAnimation(fig, render, frames=len(frames), interval=110, blit=False)
    gif = OUT / f"{args.capture}.gif"
    anim.save(gif, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"wrote -> {gif}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
