#!/usr/bin/env python3
"""Every figure, animation and printed table both notebooks use.

The notebooks call these and contain no drawing code of their own, so a change to how a
result is presented never touches the code that produced it. Naming: `report_*` prints a
table, `animate_*` returns an HTML animation, everything else draws a figure.

Nothing here computes a result. If a function needs a number it takes it as an argument;
the arguments come from `notebook_model.py`.
"""

from __future__ import annotations

import math
import warnings

# This machine has matplotlib installed twice, which makes it warn about the 3D
# projection on import. Nothing here plots in 3D.
warnings.filterwarnings("ignore", message="Unable to import Axes3D")

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse, Rectangle
import numpy as np

import cv2
from pathlib import Path
from IPython.display import HTML, display
from scipy import stats

import notebook_data as nd
import notebook_model as nm

# Animations are embedded frame by frame, so the format matters. The run animation has a
# photograph in it and was 8 MB as PNG; JPEG frames at a modest resolution cut that to a
# third with no loss that shows at a thousand pixels wide. The line-art fitting animation
# measured smaller as JPEG too -- 1.36 MB against 2.11 MB -- so both use JPEG.
plt.rcParams["animation.embed_limit"] = 120.0

_BOX_CACHE: dict = {}
HALF_PX, ZOOM_M = 150, 0.30

# One palette for both notebooks, colour-blind safe.
C_TRUTH = "#111111"
C_OBS = "#8A8A8A"
C_FILTER = "#D55E00"
C_SMOOTH = "#0072B2"
C_ODOM = "#7B4EA8"
C_ACCENT = "#009E73"
CAMERA_COLOUR = {
    "camera_A": "#0072B2", "camera_B": "#D55E00",
    "camera_C": "#009E73", "camera_D": "#CC79A7",
}
CAMERA_SHORT = {c: c.replace("camera_", "") for c in nd.CAMERAS}


def style():
    """The rcParams both notebooks open with."""
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 120,
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.constrained_layout.use": True,
    })


def floor_footprint(model, inset_px: float = 2.0):
    """The image's four corners, back-projected onto the floor."""
    w, h = model.img_width, model.img_height
    corners = [(inset_px, inset_px), (w - inset_px, inset_px),
               (w - inset_px, h - inset_px), (inset_px, h - inset_px)]
    points = []
    for u, v in corners:
        p = model.pixel_to_world(u, v)
        if p is not None and all(math.isfinite(c) for c in p) and abs(p[0]) < 60 and abs(p[1]) < 60:
            points.append(p)
    return np.asarray(points) if len(points) >= 3 else None


def ellipse_from(mean, cov, n_sigma=2.0, **kwargs):
    """A patch showing the n-sigma contour of a 2-D Gaussian."""
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))
    width, height = 2 * n_sigma * np.sqrt(np.maximum(values, 0.0))
    return Ellipse(tuple(mean), width, height, angle=angle, **kwargs)


def two_sigma(P, axis):
    return 2.0 * np.sqrt(np.maximum(P[:, axis, axis], 0.0))


def draw_generative_model():
    """The joint above, as a picture: what generates what."""
    from matplotlib.patches import Circle, FancyBboxPatch

    _, ax = plt.subplots(figsize=(9.8, 5.9))
    ax.set_xlim(0.0, 10.0); ax.set_ylim(-4.25, 3.5)
    ax.axis("off")

    xs = [2.3, 5.0, 7.7]
    labels = [r"$\mathbf{x}_{k-1}$", r"$\mathbf{x}_{k}$", r"$\mathbf{x}_{k+1}$"]
    observed = [True, False, True]
    row_x, row_y, row_u, row_R = 1.55, -0.35, 2.85, -2.15

    def node(x, y, text, *, shaded, radius=0.44, colour=C_TRUTH, dotted=False,
             fontsize=11.0):
        ax.add_patch(Circle((x, y), radius, facecolor=("#DCDCDC" if shaded else "white"),
                            edgecolor=colour, lw=1.6, ls=(":" if dotted else "-"),
                            zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                color=(C_OBS if dotted else "black"), zorder=4)

    def arrow(p, q, *, colour=C_TRUTH, ls="-", lw=1.5, gap_start=19, gap_end=19):
        ax.annotate("", xy=q, xytext=p, zorder=2, arrowprops=dict(
            arrowstyle="-|>,head_length=0.7,head_width=0.28", color=colour, lw=lw,
            linestyle=ls, shrinkA=gap_start, shrinkB=gap_end, mutation_scale=18))

    # the state chain, with the run continuing off both ends
    for x, label in zip(xs, labels):
        node(x, row_x, label, shaded=False)
    for a, b in zip(xs, xs[1:]):
        arrow((a, row_x), (b, row_x))
    arrow((xs[0] - 0.95, row_x), (xs[0], row_x), gap_start=2)
    arrow((xs[-1], row_x), (xs[-1] + 0.95, row_x), gap_end=2)
    ax.text(xs[0] - 1.35, row_x, "...", fontsize=15, va="center", ha="center")
    ax.text(xs[-1] + 1.35, row_x, "...", fontsize=15, va="center", ha="center")

    # the odometry increments: known, so square and filled
    for x, label in zip(xs, labels):
        ax.add_patch(FancyBboxPatch((x - 0.19, row_u - 0.19), 0.38, 0.38,
                                    boxstyle="square,pad=0.02", facecolor=C_ODOM,
                                    edgecolor="none", zorder=3))
        ax.text(x, row_u, r"$\mathbf{u}$", ha="center", va="center", color="white",
                fontsize=9, zorder=4)
        arrow((x, row_u), (x, row_x), colour=C_ODOM, gap_start=11)

    # the observations: present only where a camera fired
    for x, label, seen in zip(xs, labels, observed):
        if seen:
            node(x, row_y, label.replace("x", "y"), shaded=True)
            arrow((x, row_x), (x, row_y))
        else:
            node(x, row_y, "no camera\ncould see", shaded=False, dotted=True,
                 fontsize=7.5)

    # the plate over cameras, holding the covariances
    ax.add_patch(FancyBboxPatch((3.30, row_R - 0.62), 3.55, 1.30,
                                boxstyle="round,pad=0.06", facecolor="none",
                                edgecolor=C_ACCENT, lw=1.4, ls="--", zorder=1))
    ax.text(6.72, row_R - 0.52, r"one per camera,  $c = 1 \ldots 4$", ha="right",
            va="bottom", fontsize=8.5, color=C_ACCENT)
    node(4.30, row_R, r"$\mathbf{R}_c$", shaded=False, colour=C_ACCENT)
    ax.add_patch(Circle((5.75, row_R), 0.33, facecolor=C_ACCENT, edgecolor="none",
                        zorder=3))
    ax.text(5.75, row_R, r"$\Psi,\nu$", ha="center", va="center", fontsize=8,
            color="white", zorder=4)
    arrow((5.75, row_R), (4.30, row_R), colour=C_ACCENT, gap_start=12, gap_end=20)
    for x, seen in zip(xs, observed):
        if seen:
            arrow((4.30, row_R), (x, row_y), colour=C_ACCENT, ls=":", lw=1.2,
                  gap_start=20, gap_end=20)

    # a key, along the bottom, so nothing overlaps the graph itself
    key_y = -3.85
    for x0, marker, text, colour in (
        (0.15, "circle_open", "inferred", C_TRUTH),
        (2.05, "circle_shaded", "observed", "#555555"),
        (3.95, "square", "known and fixed", C_ODOM),
        (6.60, "dotted", "no observation at this step", C_OBS),
    ):
        if marker == "square":
            ax.add_patch(FancyBboxPatch((x0 - 0.09, key_y - 0.09), 0.18, 0.18,
                                        boxstyle="square,pad=0.02", facecolor=colour,
                                        edgecolor="none"))
        else:
            ax.add_patch(Circle((x0, key_y), 0.14,
                                facecolor=("#DCDCDC" if marker == "circle_shaded" else "white"),
                                edgecolor=colour, lw=1.4,
                                ls=(":" if marker == "dotted" else "-")))
        ax.text(x0 + 0.26, key_y, text, va="center", fontsize=9, color=colour)

    ax.set_title("The generative model: how a run gets made", fontsize=11.5)
    plt.show()


def robot_crops(capture, camera, models, truth_table, messages, *, n=6, half=95,
                window=None, tol_s=0.6):
    """Crops centred on where the robot really was, with the runtime's verdict attached."""
    frames = [(s, p) for s, p in capture.frames(camera)
              if window is None or window[0] <= s <= window[1]]
    if not frames:
        return []
    picks = [frames[int(round(i * (len(frames) - 1) / max(n - 1, 1)))] for i in range(n)]
    model = models[camera]
    cam_xy = np.asarray(model.cam_pos[:2], dtype=float)
    out = []
    for stamp, path in picks:
        hit = nd.truth_at(truth_table, float(stamp), tol_s=0.2)
        if hit is None:
            continue
        u, v, in_frame = model.world_to_pixel(hit[0], hit[1], 0.0)
        # the runtime's verdict for the message nearest this frame
        nearby = [(abs(s - stamp), s, ok) for s, ok in messages[camera] if abs(s - stamp) <= tol_s]
        verdict = min(nearby)[2] if nearby else None
        out.append({
            "stamp": float(stamp), "path": path, "uv": (u, v),
            "range_m": float(np.linalg.norm(np.asarray(hit[:2]) - cam_xy)),
            "detected": verdict, "in_frame": bool(in_frame), "half": half,
        })
    return out


def draw_offset_geometry(cam_h=6.1, d=11.0, lift=0.05, body_w=0.34, body_h=0.19):
    """Side view: why the back-projected pixel is not the contact point.

    Two panels, because the effect and its cause live at different scales: the shallow
    arrival angle is a property of a 6 m by 11 m triangle, and the error it produces is a
    few centimetres. Drawing both on one axis makes the centimetres invisible.
    """
    from matplotlib.patches import Rectangle as Rect

    fig, (ax_wide, ax_zoom) = plt.subplots(1, 2, figsize=(11.6, 4.4),
                                           gridspec_kw={"width_ratios": [1.05, 1.0]})
    ground = 0.0
    slope_ref = (ground - cam_h) / d                      # ray to the contact point
    arrival_deg = math.degrees(math.atan2(cam_h, d))

    # ---- left: the triangle that sets the sensitivity
    ax_wide.fill_between([-0.8, 14.0], -0.9, ground, color="#EFEFEF", zorder=0)
    ax_wide.plot([-0.8, 14.0], [ground, ground], color=C_TRUTH, lw=2.0, zorder=1)
    ax_wide.plot([0], [cam_h], marker="s", ms=12, color=C_SMOOTH, zorder=4)
    ax_wide.annotate("camera", (0, cam_h), textcoords="offset points", xytext=(10, 2),
                     fontsize=9.5, color=C_SMOOTH, fontweight="bold")
    ax_wide.plot([0, d], [cam_h, ground], color=C_FILTER, lw=1.6, zorder=2)
    ax_wide.plot([0, 0], [ground, cam_h], color=C_SMOOTH, lw=1.2, ls=":", zorder=2)
    ax_wide.annotate("", xy=(0, cam_h), xytext=(0, ground),
                     arrowprops=dict(arrowstyle="<|-|>", color=C_SMOOTH, lw=1.3))
    ax_wide.annotate(f"{cam_h:.1f} m", (0.15, cam_h / 2), fontsize=9, color=C_SMOOTH,
                     rotation=90, va="center")
    ax_wide.annotate("", xy=(d, -0.55), xytext=(0, -0.55),
                     arrowprops=dict(arrowstyle="<|-|>", color=C_TRUTH, lw=1.3))
    ax_wide.annotate(f"{d:.0f} m", (d / 2, -0.62), fontsize=9, ha="center", va="top")
    ax_wide.plot([d], [ground], marker="v", ms=10, color=C_TRUTH, zorder=5)
    ax_wide.annotate(f"the ray arrives at only {arrival_deg:.0f}$\\degree$\nto the floor",
                     (d * 0.52, cam_h * 0.42), fontsize=9.5, color=C_FILTER)
    box = Rect((d - 0.55, ground), 1.1, 0.55, facecolor="none", edgecolor=C_TRUTH,
               lw=1.2, ls="--", zorder=5)
    ax_wide.add_patch(box)
    ax_wide.annotate("magnified\nright", (d - 0.6, 0.62), fontsize=8.5, ha="center",
                     color=C_TRUTH)
    ax_wide.set_xlim(-0.8, 14.0); ax_wide.set_ylim(-1.0, cam_h + 0.9)
    ax_wide.set_xlabel("distance along the floor, metres")
    ax_wide.set_ylabel("height, metres")
    ax_wide.set_title("A shallow arrival angle is what makes it sensitive", fontsize=10.5)
    ax_wide.grid(False)

    # ---- right: the same thing, to scale, where the centimetres are.
    # Two effects act here and they pull opposite ways, so keep them separate.
    lo, hi = d - 0.46, d + 0.30
    near_x = d - body_w / 2                     # the body edge facing the camera
    ax_zoom.fill_between([lo, hi], -0.10, ground, color="#EFEFEF", zorder=0)
    ax_zoom.plot([lo, hi], [ground, ground], color=C_TRUTH, lw=2.0, zorder=1)
    ax_zoom.add_patch(Rect((near_x, ground), body_w, body_h, facecolor="#F0A26B",
                           edgecolor=C_TRUTH, lw=1.4, alpha=0.85, zorder=3))
    ax_zoom.annotate("the robot", (d + 0.07, body_h * 0.5), ha="center", fontsize=9,
                     zorder=4)
    ax_zoom.plot([d], [ground], marker="v", ms=11, color=C_TRUTH, zorder=6)
    ax_zoom.annotate("the point we\nwant to measure", xy=(d, ground + 0.004),
                     xytext=(d + 0.10, 0.145), ha="left", fontsize=8.5,
                     fontweight="bold", zorder=8,
                     arrowprops=dict(arrowstyle="-", color=C_TRUTH, lw=1.0))

    hits = {}
    for lift_m, colour, ls in ((0.0, C_ACCENT, "-"), (lift, C_FILTER, "--")):
        seen = (near_x, ground + lift_m)
        ray_slope = (seen[1] - cam_h) / seen[0]
        hit = -cam_h / ray_slope
        hits[lift_m] = hit
        ax_zoom.plot([lo, hi], [cam_h + ray_slope * lo, cam_h + ray_slope * hi],
                     color=colour, lw=1.6, ls=ls, zorder=2)
        ax_zoom.plot(*seen, marker="o", ms=9, color=colour, zorder=7,
                     markeredgecolor="white", markeredgewidth=1.2)
        ax_zoom.plot([hit], [ground], marker="D", ms=9, color=colour, zorder=7,
                     markeredgecolor="white", markeredgewidth=1.1)
    ax_zoom.annotate("lowest visible point,\nif it is on the floor",
                     (near_x, ground), textcoords="offset points", xytext=(-9, 4),
                     ha="right", va="bottom", fontsize=8.5, color=C_ACCENT)
    ax_zoom.annotate(f"...and if it sits {100 * lift:.0f} cm up",
                     (near_x, ground + lift), textcoords="offset points",
                     xytext=(-9, 24), ha="right", va="bottom", fontsize=8.5,
                     color=C_FILTER)

    # effect 1: the visible edge is not the point we want (pulls TOWARDS the camera)
    ax_zoom.annotate("", xy=(hits[0.0], -0.030), xytext=(d, -0.030),
                     arrowprops=dict(arrowstyle="<|-|>", color=C_ACCENT, lw=1.5))
    ax_zoom.annotate(f"{100 * (hits[0.0] - d):+.0f} cm: we see the near\nedge, not the centre",
                     ((hits[0.0] + d) / 2, -0.035), ha="center", va="top", fontsize=8.5,
                     color=C_ACCENT, fontweight="bold")
    # effect 2: the lift pushes AWAY from the camera
    ax_zoom.annotate("", xy=(hits[lift], -0.077), xytext=(hits[0.0], -0.077),
                     arrowprops=dict(arrowstyle="<|-|>", color=C_FILTER, lw=1.5))
    ax_zoom.annotate(f"{100 * (hits[lift] - hits[0.0]):+.0f} cm from the lift",
                     ((hits[lift] + hits[0.0]) / 2, -0.082), ha="center", va="top",
                     fontsize=8.5, color=C_FILTER, fontweight="bold")

    ax_zoom.plot([], [], marker="D", ls="none", color=C_TRUTH,
                 label="where a ray meets the floor = the observation")
    ax_zoom.set_xlim(lo, hi); ax_zoom.set_ylim(-0.125, 0.30)
    ax_zoom.set_xlabel("distance along the floor, metres")
    ax_zoom.set_ylabel("height, metres")
    ax_zoom.set_title("Magnified, to scale: two effects, pulling\nopposite ways",
                      fontsize=10.5)
    ax_zoom.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8)
    ax_zoom.grid(False)
    plt.show()
    print(f"Camera {cam_h:.1f} m up, robot {d:.0f} m away, so the ray arrives at "
          f"{arrival_deg:.0f} degrees and the multiplier is range/height = {d / cam_h:.1f}.")
    print(f"  seeing the near body edge instead of the centre: "
          f"{100 * (hits[0.0] - d):+.0f} cm (towards the camera)")
    print(f"  that edge appearing {100 * lift:.0f} cm off the floor: "
          f"{100 * (hits[lift] - hits[0.0]):+.0f} cm (away from the camera)")
    print(f"  net, for this pose: {100 * (hits[lift] - d):+.0f} cm")
    print("Both depend on the bearing and on the robot's heading, and they partly cancel,")
    print("which is why the net offset is not something you can work out once and reuse.")


def boxes_for(path):
    """The detector's boxes on one saved frame, computed once and reused."""
    key = str(path)
    if key not in _BOX_CACHE:
        image = cv2.imread(key)
        _BOX_CACHE[key] = [] if image is None else nm.detect_on_frame(image, nm.detector_path())
    return _BOX_CACHE[key]


def build_run_frames(seq, forward, capture, *, every=8):
    """What each animation step needs, worked out once so drawing stays cheap."""
    last_frame = None
    last_camera = None
    out = []
    for k in range(0, seq.n_steps, every):
        if seq.camera[k] is not None:
            last_camera = seq.camera[k]
        if last_camera is not None:
            hit = capture.frame_at(last_camera, float(seq.stamps[k]), tol_s=1.2)
            if hit is not None:
                last_frame = (last_camera, hit[1])
        out.append({
            "k": k, "t": float(seq.stamps[k] - seq.stamps[0]), "display": last_frame,
            "used": bool(forward["used"][k]), "rejected": bool(forward["rejected"][k]),
            "nis": float(forward["nis"][k]),
        })
    return out




def replay_fitting(seq, *, passes=8, prior_nu=6.0, prior_sigma_m=0.05,
                   sigma_p=nm.PROCESS_SIGMA_PER_SQRT_M):
    """Re-run the loop, keeping both halves of every pass so it can be animated."""
    d = 2
    Psi = np.eye(d) * (prior_sigma_m**2) * prior_nu
    R_bar = {c: Psi / prior_nu for c in nd.CAMERAS}
    steps = []
    for iteration in range(passes):
        # --- the x step: refit the path with the R we currently believe
        forward_i = nm.kalman_filter(seq, R_bar, sigma_p=sigma_p)
        smooth_i = nm.rts_smoother(seq, forward_i)
        residuals = {c: [] for c in nd.CAMERAS}
        for k in range(seq.n_steps):
            if seq.camera[k] is not None and forward_i["used"][k]:
                residuals[seq.camera[k]].append(seq.y[k] - smooth_i["m"][k])
        residuals = {c: np.asarray(v) for c, v in residuals.items()}
        track = 100 * (forward_i["m"][:, 1] - seq.truth[:, 1])
        nees = float(np.median(nm.honesty(forward_i, seq, "it")["nees"]))
        steps.append({"pass": iteration, "half": "x", "R": {c: R_bar[c].copy() for c in nd.CAMERAS},
                      "residuals": residuals, "track": track, "nees": nees})

        # --- the R step: refit each ellipse to its own cloud
        new_R = {}
        for cam in nd.CAMERAS:
            rows = [k for k in range(seq.n_steps)
                    if seq.camera[k] == cam and forward_i["used"][k]]
            scatter = np.zeros((d, d))
            for k in rows:
                v = (seq.y[k] - smooth_i["m"][k]).reshape(d, 1)
                scatter += v @ v.T + smooth_i["P"][k]
            new_R[cam] = (Psi + scatter) / (prior_nu + len(rows))
        R_bar = new_R
        steps.append({"pass": iteration, "half": "R", "R": {c: R_bar[c].copy() for c in nd.CAMERAS},
                      "residuals": residuals, "track": track, "nees": nees})
    return steps










def report_recording(capture, truth_table):
    print(f"capture:   {capture.name}")
    print(f"odometry:  {capture.n_steps} steps, {capture.duration_s:.0f} s of simulated time "
          f"({capture.n_steps / capture.duration_s:.0f} Hz)")
    print(f"truth:     {truth_table[0].size} poses (scoring only)")
    print()
    print("whole recording, including the tail after the robot parked:")
    print(f"  {'camera':8s}{'detections':>12s}{'frames kept':>13s}")
    for cam in nd.CAMERAS:
        print(f"  {CAMERA_SHORT[cam]:8s}{len(capture.detections[cam]):>12d}"
              f"{len(capture.frames(cam)):>13d}")
    print(f"\n  total detections: {capture.n_detections}")
    print("\nThe next cell trims to the driven route; the per-camera counts there are the")
    print("ones to read, because the parked tail sits in one camera's view only.")


def camera_footprints(models, capture, truth_table):
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    for cam in nd.CAMERAS:
        model = models[cam]
        colour = CAMERA_COLOUR[cam]
        footprint = floor_footprint(model)
        if footprint is not None:
            ax.fill(footprint[:, 0], footprint[:, 1], color=colour, alpha=0.10, lw=0)
            ax.plot(np.append(footprint[:, 0], footprint[0, 0]),
                    np.append(footprint[:, 1], footprint[0, 1]),
                    color=colour, lw=1.0, alpha=0.6)
        ax.plot(*model.cam_pos[:2], marker="s", ms=9, color=colour,
                markeredgecolor="white", markeredgewidth=1.2, zorder=5)
        ax.annotate(f"camera {CAMERA_SHORT[cam]}", model.cam_pos[:2],
                    textcoords="offset points", xytext=(0, -16),
                    ha="center", color=colour, fontsize=9, fontweight="bold")

    t_stamps, t_xy, _ = truth_table
    ax.plot(t_xy[:, 0], t_xy[:, 1], color=C_TRUTH, lw=2.2, label="the drive (ground truth)")
    ax.plot(t_xy[0, 0], t_xy[0, 1], marker="o", ms=8, color=C_TRUTH, label="start")
    ax.plot(t_xy[-1, 0], t_xy[-1, 1], marker="X", ms=10, color=C_TRUTH, label="end")

    for cam in nd.CAMERAS:
        pts = np.asarray([d.world for d in capture.detections[cam]]) if capture.detections[cam] else None
        if pts is not None and len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=7, color=CAMERA_COLOUR[cam], alpha=0.55,
                       lw=0, label=f"seen by {CAMERA_SHORT[cam]}")

    ax.set_xlabel("x, metres"); ax.set_ylabel("y, metres")
    ax.set_title("Four cameras, one drive: what each camera sees on the floor")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=8.5)
    plt.show()


def generative_model():
    draw_generative_model()


def report_sequence(seq, n_obs, window, capture, truth_table):
    print(f"driven route: simulated seconds {window[0]:.1f} to {window[1]:.1f} "
          f"({window[1] - window[0]:.0f} s)" if window else "no route record; using everything")
    seq = nm.Sequence(capture, truth_table, window=window)
    n_obs = int(seq.observed.sum())
    print(f"grid:          {seq.n_steps} steps at {nm.GRID_HZ:.0f} Hz "
          f"({seq.stamps[-1] - seq.stamps[0]:.0f} s)")
    print(f"observed:      {n_obs} steps ({100 * n_obs / seq.n_steps:.0f}%)")
    print(f"unobserved:    {seq.n_steps - n_obs} steps "
          f"({100 * (1 - n_obs / seq.n_steps):.0f}%) -- these are pure prediction")
    print(f"truth present: {int(np.isfinite(seq.truth[:, 0]).sum())} steps")
    print()
    for cam in nd.CAMERAS:
        k = sum(1 for c in seq.camera if c == cam)
        print(f"  {CAMERA_SHORT[cam]}: {k:4d} of the observed steps")


def camera_relay(seq):
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9.0, 4.2), sharex=True, height_ratios=[2.0, 1.0])

    t0 = seq.stamps[0]
    for cam in nd.CAMERAS:
        idx = np.asarray([i for i, c in enumerate(seq.camera) if c == cam], dtype=int)
        if idx.size:
            ax_top.scatter(seq.stamps[idx] - t0, seq.y[idx, 1], s=12,
                           color=CAMERA_COLOUR[cam], lw=0, alpha=0.8,
                           label=f"camera {CAMERA_SHORT[cam]}")
    ax_top.plot(seq.stamps - t0, seq.truth[:, 1], color=C_TRUTH, lw=1.6,
                label="where the robot really was")
    ax_top.set_ylabel("y, metres")
    ax_top.set_title("Each camera sees the robot only while it is in that camera's patch of floor")
    ax_top.legend(loc="upper left", ncol=3, fontsize=8.5)

    for row, cam in enumerate(nd.CAMERAS):
        idx = np.asarray([i for i, c in enumerate(seq.camera) if c == cam], dtype=int)
        if idx.size:
            ax_bot.scatter(seq.stamps[idx] - t0, np.full(idx.size, row), s=14,
                           marker="|", color=CAMERA_COLOUR[cam])
    ax_bot.set_yticks(range(len(nd.CAMERAS)))
    ax_bot.set_yticklabels([CAMERA_SHORT[c] for c in nd.CAMERAS])
    ax_bot.set_ylabel("camera")
    ax_bot.set_xlabel("time since the start of the drive, seconds")
    ax_bot.set_ylim(-0.6, len(nd.CAMERAS) - 0.4)
    plt.show()


def report_example_choice(example, seq):
    print("example step:", {k: (v if not isinstance(v, Path) else v.name)
                            for k, v in (example or {}).items()})


def one_frame(example, seq, models, commissioned_sigma):
    import cv2

    image = None if example is None else cv2.imread(str(example["frame_path"]))
    if example is None or image is None:
        print("no recorded frame coincides with an observation; skipping the picture")
    else:
        cam = example["camera"]
        model = models[cam]
        boxes = nm.detect_on_frame(image, nm.detector_path())
        truth_xy = seq.truth[example["step"]]
        recorded_pixel = seq.pixel[example["step"]]

        # Re-running the detector here can surface boxes the runtime filtered out, so keep
        # the one whose bottom-centre matches the pixel the runtime actually recorded.
        # Otherwise the picture could show a box that never became this observation.
        if boxes and recorded_pixel is not None:
            boxes.sort(key=lambda b: (((b[0] + b[2]) / 2 - recorded_pixel[0]) ** 2
                                     + (b[3] - recorded_pixel[1]) ** 2))

        fig = plt.figure(figsize=(12.6, 4.2))
        ax_img = fig.add_subplot(1, 3, 1)
        ax_zoom = fig.add_subplot(1, 3, 2)
        ax_floor = fig.add_subplot(1, 3, 3)

        tu, tv, _ = model.world_to_pixel(truth_xy[0], truth_xy[1], 0.0)

        for ax, is_zoom in ((ax_img, False), (ax_zoom, True)):
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax.grid(False)
            ax.plot([tu], [tv], marker="+", ms=15, mew=2.2, color=C_TRUTH,
                    label="where the robot really is")
            if boxes:
                x1, y1, x2, y2, confidence = boxes[0]
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       ec=CAMERA_COLOUR[cam], lw=2.0))
                if is_zoom:
                    ax.annotate(f"robot, {confidence:.2f}", (x1, y1 - 4),
                                color=CAMERA_COLOUR[cam], fontsize=9, fontweight="bold")
                ax.plot([(x1 + x2) / 2], [y2], marker="o", ms=8, color=C_FILTER,
                        label="bottom of the box\n(taken as the contact point)")

        ax_img.set_title(f"the whole of what camera {CAMERA_SHORT[cam]} sees")
        ax_img.set_xlabel("pixels across"); ax_img.set_ylabel("pixels down")

        # A 1280x720 warehouse frame makes the robot about twenty pixels wide, so the
        # detail that the section is about is invisible without a magnified view.
        if boxes:
            cx, cy = (boxes[0][0] + boxes[0][2]) / 2, (boxes[0][1] + boxes[0][3]) / 2
        else:
            cx, cy = tu, tv
        half = 90
        ax_zoom.set_xlim(cx - half, cx + half)
        ax_zoom.set_ylim(cy + half, cy - half)          # image rows run downwards
        ax_zoom.set_title("the same frame, magnified on the robot")
        ax_zoom.set_xlabel("pixels across")
        ax_zoom.legend(loc="lower left", fontsize=7.5, facecolor="white", framealpha=0.8,
                       frameon=True)
        # mark the magnified region on the wide view
        ax_img.add_patch(Rectangle((cx - half, cy - half), 2 * half, 2 * half, fill=False,
                                   ec=C_TRUTH, lw=1.2, ls="--"))

        # the same thing on the floor
        ax_floor.plot(truth_xy[0], truth_xy[1], marker="+", ms=15, mew=2.2, color=C_TRUTH,
                      label="where the robot really is")
        obs = seq.y[example["step"]]
        ax_floor.plot(obs[0], obs[1], marker="o", ms=8, color=C_FILTER,
                      label="the observation, after back-projection")
        ax_floor.annotate("", xy=(obs[0], obs[1]), xytext=(truth_xy[0], truth_xy[1]),
                          arrowprops=dict(arrowstyle="->", color=C_OBS, lw=1.4))
        error_cm = 100 * float(np.hypot(*(obs - truth_xy)))
        ax_floor.annotate(f"{error_cm:.1f} cm",
                          ((obs[0] + truth_xy[0]) / 2, (obs[1] + truth_xy[1]) / 2),
                          textcoords="offset points", xytext=(8, 8), color=C_OBS, fontsize=9)
        sigma = commissioned_sigma[cam]
        # Drawn at TWO standard deviations, so the radius is 2 sigma and the circle is
        # 4 sigma across. Label the radius that is actually on the page: quoting the
        # one-sd figure beside the words "2 sd" made this circle look twice too big.
        ax_floor.add_patch(Ellipse(tuple(obs), 2 * 2 * sigma, 2 * 2 * sigma, fill=False,
                                   ec=C_FILTER, ls="--", lw=1.2,
                                   label=f"what the model expects: 2 sd, so a radius of "
                                         f"{100 * 2 * sigma:.1f} cm (1 sd = {100 * sigma:.1f} cm)"))
        pad = max(0.25, 1.6 * error_cm / 100)
        ax_floor.set_xlim(truth_xy[0] - pad, truth_xy[0] + pad)
        ax_floor.set_ylim(truth_xy[1] - pad, truth_xy[1] + pad)
        ax_floor.set_aspect("equal", adjustable="box")
        ax_floor.set_xlabel("x, metres"); ax_floor.set_ylabel("y, metres")
        ax_floor.set_title("the same moment, on the floor")
        ax_floor.legend(loc="upper left", fontsize=8)
        plt.show()

        print(f"observation error at this step: {error_cm:.1f} cm, "
              f"against an assumed one-standard-deviation of {100 * sigma:.1f} cm")


def detector_gallery(capture, models, truth_table, messages, window):


    messages = nd.load_messages(capture.name)
    sheet = {cam: robot_crops(capture, cam, models, truth_table, messages, window=window)
             for cam in nd.CAMERAS}

    n_cols = max((len(v) for v in sheet.values()), default=0)
    fig, axes = plt.subplots(len(nd.CAMERAS), n_cols,
                             figsize=(1.75 * n_cols, 1.95 * len(nd.CAMERAS)))
    for row, cam in enumerate(nd.CAMERAS):
        for col in range(n_cols):
            ax = axes[row, col]
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for side in ax.spines.values():
                side.set_visible(True); side.set_color(CAMERA_COLOUR[cam]); side.set_linewidth(1.6)
            if col >= len(sheet[cam]):
                ax.axis("off")
                continue
            item = sheet[cam][col]
            image = cv2.imread(str(item["path"]))
            if image is None:
                ax.axis("off"); continue
            u, v = item["uv"]
            half = item["half"]
            x0, y0 = int(round(u - half)), int(round(v - half))
            x0 = max(0, min(x0, image.shape[1] - 2 * half))
            y0 = max(0, min(y0, image.shape[0] - 2 * half))
            crop = image[y0:y0 + 2 * half, x0:x0 + 2 * half]
            ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            boxes = nm.detect_on_frame(image, nm.detector_path())
            drawn = None
            for x1, y1, x2, y2, conf in boxes:
                if abs((x1 + x2) / 2 - u) < half and abs(y2 - v) < half:
                    drawn = (x1, y1, x2, y2, conf)
                    break
            if drawn is not None:
                x1, y1, x2, y2, conf = drawn
                ax.add_patch(Rectangle((x1 - x0, y1 - y0), x2 - x1, y2 - y1, fill=False,
                                       ec=C_FILTER, lw=1.6))
            ax.plot([u - x0], [v - y0], marker="+", ms=9, mew=1.6, color=C_TRUTH)
            used = item["detected"]
            if used:
                tag, colour = "used", C_TRUTH
            elif not item["in_frame"]:
                tag, colour = "out of frame", "#8A8A8A"
            elif used is None:
                tag, colour = "no message", "#8A8A8A"
            else:
                tag, colour = "in frame, missed", "#B00020"
            ax.set_xlabel(f"{item['range_m']:.0f} m · {tag}", fontsize=7.5,
                          color=colour, labelpad=1.5)
            if col == 0:
                ax.set_ylabel(f"camera {CAMERA_SHORT[cam]}", fontsize=9.5,
                              color=CAMERA_COLOUR[cam], fontweight="bold")
    fig.suptitle("The robot as each camera saw it, spread across the drive\n"
                 "black cross = where it really was · orange box = what the detector found · "
                 "caption = range and the runtime's verdict", fontsize=10)
    plt.show()


def detector_outcomes(outcomes, north):
    fig, (ax_cover, ax_rate, ax_bar) = plt.subplots(
        1, 3, figsize=(14.4, 4.3), gridspec_kw={"width_ratios": [1.15, 1.15, 1.0]})
    y_edges = np.arange(-7.5, 7.6, 1.25)
    for cam in nd.CAMERAS:
        d = outcomes[cam]
        inside = d["in_frame"]
        xs_c, ys_c, xs_d, ys_d, ns_d = [], [], [], [], []
        for lo, hi in zip(y_edges[:-1], y_edges[1:]):
            band = (north[cam] >= lo) & (north[cam] < hi)
            if band.sum() >= 5:
                xs_c.append(0.5 * (lo + hi)); ys_c.append(inside[band].mean())
            both = band & inside
            if both.sum() >= 5:
                xs_d.append(0.5 * (lo + hi)); ys_d.append(d["ok"][both].mean())
                ns_d.append(int(both.sum()))
        if xs_c:
            ax_cover.plot(xs_c, ys_c, marker="o", ms=4, lw=1.9, color=CAMERA_COLOUR[cam],
                          label=f"camera {CAMERA_SHORT[cam]}")
        if xs_d:
            ax_rate.plot(xs_d, ys_d, marker="o", ms=4, lw=1.9, color=CAMERA_COLOUR[cam],
                         label=f"camera {CAMERA_SHORT[cam]}")

    for ax, title, ylabel in (
        (ax_cover, "Geometry: was the robot inside the image?",
         "fraction of frames with the robot in frame"),
        (ax_rate, "Detector: given that it was, was it found?",
         "detections, as a fraction of in-frame frames"),
    ):
        ax.set_xlabel("how far up the aisle the robot was, metres north")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(-0.04, 1.08)
        ax.set_title(title, fontsize=10.5)
        ax.legend(fontsize=8, ncol=2, loc="lower center")

    names = [f"camera {CAMERA_SHORT[c]}" for c in nd.CAMERAS]
    positions = np.arange(len(names))
    found = np.array([int((outcomes[c]["ok"] & outcomes[c]["in_frame"]).sum()) for c in nd.CAMERAS])
    in_miss = np.array([int((~outcomes[c]["ok"] & outcomes[c]["in_frame"]).sum()) for c in nd.CAMERAS])
    outside = np.array([int((~outcomes[c]["in_frame"]).sum()) for c in nd.CAMERAS])
    stray = np.array([int((outcomes[c]["ok"] & ~outcomes[c]["in_frame"]).sum()) for c in nd.CAMERAS])
    ax_bar.barh(positions, found, color=C_ACCENT, height=0.62, label="in frame, detected")
    ax_bar.barh(positions, in_miss, left=found, color="#E8A33D", height=0.62,
                label="in frame, nothing found")
    ax_bar.barh(positions, outside, left=found + in_miss, color="#D9D9D9", height=0.62,
                label="robot outside the image")
    for i, cam in enumerate(nd.CAMERAS):
        total = found[i] + in_miss[i] + outside[i]
        share = found[i] / max(found[i] + in_miss[i], 1)
        ax_bar.annotate(f"{share:.0%} of in-frame", (total, i), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=8.5)
    ax_bar.set_yticks(positions); ax_bar.set_yticklabels(names)
    ax_bar.set_xlim(0, 1.32 * max(found + in_miss + outside))
    ax_bar.set_xlabel("observation messages over the drive")
    ax_bar.set_title("Where the missing observations go")
    ax_bar.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=8)
    plt.show()

    print(f"  {'camera':8s}{'messages':>9s}{'in frame':>10s}{'detected':>10s}"
          f"{'of in-frame':>13s}{'range detected':>18s}")
    for cam in nd.CAMERAS:
        d = outcomes[cam]
        inside = d["in_frame"]
        hit_r = d["range"][d["ok"] & inside]
        span = f"{hit_r.min():.1f} to {hit_r.max():.1f} m" if hit_r.size else "never"
        print(f"  {CAMERA_SHORT[cam]:8s}{d['ok'].size:>9d}{int(inside.sum()):>10d}"
              f"{int((d['ok'] & inside).sum()):>10d}"
              f"{(d['ok'] & inside).sum() / max(inside.sum(), 1):>13.2f}{span:>18s}")
    if stray.sum():
        print(f"\n  {stray.sum()} detections arrived while the robot was outside the image "
              f"({dict(zip((CAMERA_SHORT[c] for c in nd.CAMERAS), stray.tolist()))}).")
        print("  Those are the detector boxing something that is not the robot -- the gate's job.")


def where_in_the_frame(outcomes, models, capture, seq):
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.2))
    for ax, cam in zip(axes.ravel(), nd.CAMERAS):
        d = outcomes[cam]
        frames = capture.frames(cam)
        if frames:
            backdrop = cv2.imread(str(frames[len(frames) // 2][1]))
            if backdrop is not None:
                ax.imshow(cv2.cvtColor(backdrop, cv2.COLOR_BGR2RGB), alpha=0.30)
        miss = ~d["ok"]
        ax.scatter(d["u"][miss], d["v"][miss], s=16, marker="x", lw=1.0, color="#B00020",
                   alpha=0.8, label="nothing found")
        # vmin/vmax pinned to the whole drive: without them each panel normalises to its own
        # time range and the shared colour bar would be lying about three of the four.
        sc = ax.scatter(d["u"][d["ok"]], d["v"][d["ok"]], s=18, lw=0,
                        c=d["stamp"][d["ok"]] - seq.stamps[0], cmap="viridis",
                        vmin=0.0, vmax=float(seq.stamps[-1] - seq.stamps[0]),
                        label="detected")
        ax.set_xlim(0, models[cam].img_width); ax.set_ylim(models[cam].img_height, 0)
        ax.set_title(f"camera {CAMERA_SHORT[cam]}  —  {d['ok'].mean():.0%} of frames detected",
                     fontsize=10, color=CAMERA_COLOUR[cam])
        ax.set_xlabel("pixels across"); ax.set_ylabel("pixels down")
        ax.grid(False)
    axes[0, 0].legend(loc="upper left", fontsize=8, facecolor="white", framealpha=0.8,
                      frameon=True)
    bar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    bar.set_label("seconds since the start of the drive")
    fig.suptitle("Where the robot was in each image, and whether the detector found it",
                 fontsize=11)
    plt.show()


def offset_geometry(models):
    draw_offset_geometry()


def radial_and_tangential(split, capture, models):
    fig, (ax_box, ax_rng) = plt.subplots(1, 2, figsize=(11.0, 4.3))

    positions, tick_labels = [], []
    for i, cam in enumerate(nd.CAMERAS):
        if cam not in split:
            continue
        for j, (key, colour) in enumerate((("radial", C_FILTER), ("tangential", C_SMOOTH))):
            data = 100 * split[cam][key]
            pos = i * 1.0 + (j - 0.5) * 0.34
            parts = ax_box.boxplot([data], positions=[pos], widths=0.28, vert=True,
                                   patch_artist=True, showfliers=False,
                                   medianprops=dict(color="white", lw=1.4))
            parts["boxes"][0].set_facecolor(colour)
            parts["boxes"][0].set_edgecolor(colour)
        positions.append(i * 1.0)
        tick_labels.append(f"camera {CAMERA_SHORT[cam]}")
    ax_box.axhline(0, color=C_TRUTH, lw=1.2)
    ax_box.set_xticks(positions); ax_box.set_xticklabels(tick_labels)
    ax_box.set_ylabel("error, centimetres")
    ax_box.set_title("Along the line of sight, and across it\n(all four runs pooled)")
    ax_box.plot([], [], color=C_FILTER, lw=7, label="along the line of sight")
    ax_box.plot([], [], color=C_SMOOTH, lw=7, label="across the line of sight")
    ax_box.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=8.5)

    for cam in nd.CAMERAS:
        if cam not in split:
            continue
        ax_rng.scatter(split[cam]["range"], 100 * split[cam]["radial"], s=6, lw=0, alpha=0.35,
                       color=CAMERA_COLOUR[cam], label=f"camera {CAMERA_SHORT[cam]}")
    ax_rng.axhline(0, color=C_TRUTH, lw=1.2)
    for lift_cm in (2.5, 5.0):
        rng = np.linspace(5, 16, 20)
        ax_rng.plot(rng, lift_cm * rng / 6.1, color=C_TRUTH, ls="--", lw=1.1)
        ax_rng.annotate(f"{lift_cm:.1f} cm up", (rng[-1], lift_cm * rng[-1] / 6.1),
                        fontsize=8, color=C_TRUTH, va="bottom", ha="right")
    ax_rng.set_xlabel("how far the robot is from that camera, metres")
    ax_rng.set_ylabel("error along the line of sight, cm")
    ax_rng.set_title("If a raised contact point were the cause,\nthe points would follow the dashed lines")
    ax_rng.legend(fontsize=8, ncol=2)
    plt.show()

    everything = {k: np.concatenate([split[c][k] for c in split])
                  for k in ("radial", "tangential", "range")}
    share = (everything["radial"] ** 2).mean() / (
        (everything["radial"] ** 2).mean() + (everything["tangential"] ** 2).mean())
    print(f"pooled over {len(everything['radial'])} detections from four runs:")
    print(f"  along the line of sight: mean {100 * everything['radial'].mean():+5.2f} cm, "
          f"spread {100 * everything['radial'].std():4.2f} cm")
    print(f"  across it:               mean {100 * everything['tangential'].mean():+5.2f} cm, "
          f"spread {100 * everything['tangential'].std():4.2f} cm")
    print(f"  share of the squared error lying along the line of sight: {100 * share:.0f}%")
    print()
    print(f"  {'camera':9s}{'across-track mean, per run (cm)':>34s}")
    for cam in nd.CAMERAS:
        per_run = []
        for name in list(nd.COMMISSIONING_CAPTURES) + [capture.name]:
            one = nm.decompose_errors([name], models)
            per_run.append(f"{100 * one[cam]['tangential'].mean():+6.1f}" if cam in one else "     -")
        print(f"  {CAMERA_SHORT[cam]:9s}{'  '.join(per_run):>34s}")


def per_camera_residuals(residuals):
    fig, (ax_scatter, ax_bar) = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for cam, res in residuals.items():
        ax_scatter.scatter(100 * res[:, 0], 100 * res[:, 1], s=10, alpha=0.45, lw=0,
                           color=CAMERA_COLOUR[cam], label=f"camera {CAMERA_SHORT[cam]}")
        mean = 100 * res.mean(axis=0)
        ax_scatter.plot(*mean, marker="X", ms=13, color=CAMERA_COLOUR[cam],
                        markeredgecolor="white", markeredgewidth=1.4, zorder=5)
    ax_scatter.axhline(0, color=C_TRUTH, lw=0.9)
    ax_scatter.axvline(0, color=C_TRUTH, lw=0.9)
    ax_scatter.set_xlabel("error east, cm"); ax_scatter.set_ylabel("error north, cm")
    ax_scatter.set_title("Observation errors are offset, not centred\n(X marks each camera's average)")
    ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.legend(fontsize=8.5)

    names = [f"camera {CAMERA_SHORT[c]}" for c in residuals]
    offsets = [100 * float(np.hypot(*residuals[c].mean(axis=0))) for c in residuals]
    spreads = [100 * float(np.sqrt(residuals[c].var(axis=0).mean())) for c in residuals]
    positions = np.arange(len(names))
    ax_bar.barh(positions + 0.19, offsets, height=0.36, color=C_FILTER,
                label="size of the average offset")
    ax_bar.barh(positions - 0.19, spreads, height=0.36, color=C_SMOOTH,
                label="spread around that offset")
    ax_bar.set_yticks(positions); ax_bar.set_yticklabels(names)
    ax_bar.set_xlabel("centimetres")
    ax_bar.set_title("For most cameras the offset is as large as the spread")
    ax_bar.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8.5)
    plt.show()

    for cam, res in residuals.items():
        mean = res.mean(axis=0)
        print(f"  camera {CAMERA_SHORT[cam]}: average offset "
              f"({100 * mean[0]:+6.1f}, {100 * mean[1]:+6.1f}) cm, "
              f"magnitude {100 * np.hypot(*mean):5.1f} cm, "
              f"spread {100 * np.sqrt(res.var(axis=0).mean()):5.1f} cm, n={len(res)}")


def offsets_do_not_transfer(residuals, commissioned_offset):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    scale = 100.0
    for cam in nd.CAMERAS:
        if cam not in residuals:
            continue
        here = tuple(scale * residuals[cam].mean(axis=0))
        there = tuple(scale * commissioned_offset[cam])
        colour = CAMERA_COLOUR[cam]
        ax.annotate("", xy=here, xytext=there,
                    arrowprops=dict(arrowstyle="->", color=colour, lw=1.6, alpha=0.75))
        ax.plot(*there, marker="s", ms=9, color=colour, markeredgecolor="white", mew=1.3)
        ax.plot(*here, marker="o", ms=9, color=colour, markeredgecolor="white", mew=1.3)
        ax.annotate(f" {CAMERA_SHORT[cam]}", here, fontsize=9, color=colour, fontweight="bold")
    ax.plot([], [], marker="s", ls="none", color=C_TRUTH, label="commissioned on earlier runs")
    ax.plot([], [], marker="o", ls="none", color=C_TRUTH, label="measured on this run")
    ax.axhline(0, color=C_TRUTH, lw=0.9); ax.axvline(0, color=C_TRUTH, lw=0.9)
    ax.set_xlabel("offset east, cm"); ax.set_ylabel("offset north, cm")
    ax.set_title("The offsets do not transfer between runs\n(arrow = how far each camera moved)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8.5)
    plt.show()

    print(f"  {'camera':8s}{'commissioned':>26s}{'this run':>20s}{'moved by':>11s}")
    for cam in nd.CAMERAS:
        if cam not in residuals:
            continue
        there = 100 * commissioned_offset[cam]
        here = 100 * residuals[cam].mean(axis=0)
        moved = float(np.hypot(*(here - there)))
        print(f"  {CAMERA_SHORT[cam]:8s}"
              f"({there[0]:+7.1f},{there[1]:+7.1f}) cm"
              f"({here[0]:+7.1f},{here[1]:+7.1f}) cm{moved:>9.1f} cm")


def report_filter(forward, n_obs):
    print(f"observations offered:  {n_obs}")
    print(f"observations used:     {int(forward['used'].sum())}")
    print(f"observations rejected: {int(forward['rejected'].sum())} by the gate")
    print(f"log evidence:          {forward['log_evidence']:.1f} nats")


def one_update(seq, forward, step, commissioned):
    cam = seq.camera[step]
    m_pred, P_pred = forward["m_pred"][step], forward["P_pred"][step]
    y, R = seq.y[step], commissioned["R_total"][cam]
    m_post, P_post = forward["m"][step], forward["P"][step]

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.add_patch(ellipse_from(m_pred, P_pred, fill=False, ec=C_ODOM, lw=2.0,
                              label="prediction: odometry carried forward"))
    ax.plot(*m_pred, marker="o", ms=8, color=C_ODOM)
    ax.add_patch(ellipse_from(y, R, fill=False, ec=C_OBS, lw=2.0, ls="--",
                              label=f"observation from camera {CAMERA_SHORT[cam]}, with its R"))
    ax.plot(*y, marker="o", ms=8, color=C_OBS)
    ax.add_patch(ellipse_from(m_post, P_post, facecolor=C_FILTER, alpha=0.18,
                              ec=C_FILTER, lw=2.2, label="posterior: the two combined"))
    ax.plot(*m_post, marker="o", ms=9, color=C_FILTER)
    ax.plot(*seq.truth[step], marker="+", ms=16, mew=2.4, color=C_TRUTH,
            label="where the robot really was")
    ax.annotate("", xy=tuple(m_post), xytext=tuple(m_pred),
                arrowprops=dict(arrowstyle="-|>", color=C_FILTER, lw=1.6,
                                shrinkA=8, shrinkB=8))
    ax.annotate("the correction", (0.5 * (m_pred[0] + m_post[0]), 0.5 * (m_pred[1] + m_post[1])),
                textcoords="offset points", xytext=(10, -14), fontsize=9, color=C_FILTER)
    ax.set_xlabel("x, metres"); ax.set_ylabel("y, metres")
    ax.set_title("One step of the filter, at the moment it had drifted furthest\n"
                 "(ellipses are 2 standard deviations)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), fontsize=8.5)
    plt.show()

    print(f"step {step} at {seq.stamps[step] - seq.stamps[0]:.1f} s, camera {CAMERA_SHORT[cam]}")
    print(f"  prediction  said +-{100 * math.sqrt(P_pred[0, 0]):5.1f} cm east, "
          f"+-{100 * math.sqrt(P_pred[1, 1]):5.1f} cm north")
    print(f"  observation said +-{100 * math.sqrt(R[0, 0]):5.1f} cm east, "
          f"+-{100 * math.sqrt(R[1, 1]):5.1f} cm north")
    print(f"  posterior   says +-{100 * math.sqrt(P_post[0, 0]):5.1f} cm east, "
          f"+-{100 * math.sqrt(P_post[1, 1]):5.1f} cm north")
    print(f"  the correction moved the estimate "
          f"{100 * float(np.linalg.norm(m_post - m_pred)):.1f} cm")
    print(f"  distance from truth: prediction {100 * float(np.linalg.norm(m_pred - seq.truth[step])):.1f} cm"
          f"  ->  posterior {100 * float(np.linalg.norm(m_post - seq.truth[step])):.1f} cm")


def filtered_track(seq, forward, time):
    used, rejected = forward["used"], forward["rejected"]
    fig, axes = plt.subplots(3, 1, figsize=(9.4, 7.4), sharex=True,
                             height_ratios=[1.0, 1.25, 1.25])

    ax = axes[0]
    ax.plot(time, seq.truth[:, 1], color=C_TRUTH, lw=1.8, label="where the robot really was")
    ax.plot(time, forward["m"][:, 1], color=C_FILTER, lw=1.3, label="the filter's estimate")
    ax.set_ylabel("north position,\nmetres")
    ax.set_title("The drive: 14 m up the central aisle in 100 seconds")
    ax.legend(loc="lower right", fontsize=8.5)

    for axis, ax in zip((0, 1), axes[1:]):
        name = "east" if axis == 0 else "north"
        band = 100 * two_sigma(forward["P"], axis)
        error = 100 * (forward["m"][:, axis] - seq.truth[:, axis])
        ax.fill_between(time, -band, band, color=C_FILTER, alpha=0.18,
                        label="what the filter claims (2 sd)")
        ax.axhline(0, color=C_TRUTH, lw=1.4, label="where the robot really was")
        ax.plot(time, error, color=C_FILTER, lw=1.6, label="the filter's error")
        odom_error = 100 * (seq.odom[:, axis] - seq.odom[0, axis]
                            + seq.truth[0, axis] - seq.truth[:, axis])
        ax.plot(time, odom_error, color=C_ODOM, lw=1.1, ls=":", label="wheel odometry alone")
        ax.scatter(time[used], 100 * (seq.y[used, axis] - seq.truth[used, axis]),
                   s=13, color=C_OBS, lw=0, alpha=0.8, label="observations used")
        if rejected.any():
            ax.scatter(time[rejected], 100 * (seq.y[rejected, axis] - seq.truth[rejected, axis]),
                       s=30, marker="x", color="#B00020", lw=1.1,
                       label="observations the gate threw away")
        ax.set_ylabel(f"{name} error,\ncentimetres")

    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.40), ncol=3, fontsize=8.5)
    axes[-1].set_xlabel("time since the start of the drive, seconds")

    # Mark where the southern pair stops contributing, and measure the error either side of
    # it, so the claim in the text is computed rather than eyeballed off the plot.
    SOUTH, NORTH = {"camera_A", "camera_C"}, {"camera_B", "camera_D"}
    south_steps = [k for k in range(seq.n_steps) if seq.camera[k] in SOUTH and forward["used"][k]]
    handover = int(max(south_steps)) if south_steps else None
    if handover is not None:
        for ax in axes:
            ax.axvline(time[handover], color=C_ACCENT, lw=1.4, ls="--", zorder=1)
        axes[0].annotate("last time a southern\ncamera saw the robot", (time[handover], 0.0),
                         textcoords="offset points", xytext=(-8, 8), ha="right",
                         fontsize=8.5, color=C_ACCENT)
    plt.show()

    if handover is not None:
        before = 100 * (forward["m"][:handover, 1] - seq.truth[:handover, 1])
        after = 100 * (forward["m"][handover:, 1] - seq.truth[handover:, 1])
        first_north = min((k for k in range(seq.n_steps)
                           if seq.camera[k] in NORTH and forward["used"][k]), default=None)
        print(f"the northern pair first contributes at {time[first_north]:.0f} s; "
              f"the southern pair last contributes at {time[handover]:.0f} s")
        print(f"mean north error while the southern cameras were still being used: "
              f"{before.mean():+5.1f} cm")
        print(f"mean north error after they stopped:                              "
              f"{after.mean():+5.1f} cm")


def animate_the_run(seq, forward, capture, models, commissioned):
    # Animations are embedded frame by frame, so the format matters. The run animation has a
    # photograph in it and was 8 MB as PNG; JPEG frames at a modest resolution cut that to a
    # third with no loss that shows at a thousand pixels wide. I assumed the line-art fitting
    # animation would be smaller as PNG and measured the opposite -- 2.11 MB against 1.36 MB --
    # so both use JPEG.







    run_frames = build_run_frames(seq, forward, capture, every=14)
    print(f"{len(run_frames)} steps spanning {run_frames[-1]['t']:.0f} s of driving")

    fig = plt.figure(figsize=(11.8, 4.2), dpi=74)
    mosaic = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.60, 1.0], wspace=0.30)
    ax_cam = fig.add_subplot(mosaic[0, 0])
    ax_map = fig.add_subplot(mosaic[0, 1])
    ax_zoom = fig.add_subplot(mosaic[0, 2])
    # supxlabel, not fig.text: placed with fig.text below the axes the arithmetic fell outside
    # the canvas and was silently clipped out of every frame. supxlabel is laid out with the
    # figure, so it survives.
    caption = fig.supxlabel("", fontsize=9, family="monospace")

    def draw_step(index):
        item = run_frames[index]
        k = item["k"]
        for ax in (ax_cam, ax_map, ax_zoom):
            ax.clear()
            ax.grid(False)

        title = "no camera has spoken yet"
        if item["display"] is not None:
            cam, path = item["display"]
            image = cv2.imread(str(path))
            if image is not None:
                model = models[cam]
                u, v, _ = model.world_to_pixel(seq.truth[k][0], seq.truth[k][1], 0.0)
                x0 = int(np.clip(u - HALF_PX, 0, image.shape[1] - 2 * HALF_PX))
                y0 = int(np.clip(v - HALF_PX, 0, image.shape[0] - 2 * HALF_PX))
                ax_cam.imshow(cv2.cvtColor(
                    image[y0:y0 + 2 * HALF_PX, x0:x0 + 2 * HALF_PX], cv2.COLOR_BGR2RGB))
                for x1, y1, x2, y2, conf in boxes_for(path):
                    if abs((x1 + x2) / 2 - u) < HALF_PX and abs(y2 - v) < HALF_PX:
                        ax_cam.add_patch(Rectangle((x1 - x0, y1 - y0), x2 - x1, y2 - y1,
                                                   fill=False, ec=C_FILTER, lw=2.0))
                        ax_cam.plot([(x1 + x2) / 2 - x0], [y2 - y0], marker="o", ms=7,
                                    color=C_FILTER)
                        break
                ax_cam.plot([u - x0], [v - y0], marker="+", ms=14, mew=2.0, color=C_TRUTH)
                state = ("detection used" if item["used"] else
                         "detection REJECTED" if item["rejected"] else "nothing found")
                title = f"camera {CAMERA_SHORT[cam]} — {state}"
                ax_cam.set_xlim(0, 2 * HALF_PX); ax_cam.set_ylim(2 * HALF_PX, 0)
        ax_cam.set_title(title, fontsize=10)
        ax_cam.set_xticks([]); ax_cam.set_yticks([])

        for cam in nd.CAMERAS:
            ax_map.plot(*models[cam].cam_pos[:2], marker="s", ms=7, color=CAMERA_COLOUR[cam])
        ax_map.plot(seq.truth[:, 0], seq.truth[:, 1], color="#DDDDDD", lw=1.4)
        ax_map.plot(seq.truth[:k + 1, 0], seq.truth[:k + 1, 1], color=C_TRUTH, lw=1.8)
        ax_map.plot(*seq.truth[k], marker="o", ms=7, color=C_TRUTH)
        ax_map.set_xlim(-8.0, 8.0); ax_map.set_ylim(-11.5, 11.5)
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.set_title(f"t = {item['t']:5.1f} s", fontsize=10)
        ax_map.set_xlabel("x, m"); ax_map.set_ylabel("y, m")

        centre = seq.truth[k]
        ax_zoom.add_patch(ellipse_from(forward["m_pred"][k], forward["P_pred"][k], fill=False,
                                       ec=C_ODOM, lw=1.8, ls="--", label="prediction, 2 sd"))
        if seq.camera[k] is not None:
            ax_zoom.add_patch(ellipse_from(seq.y[k], commissioned['R_total'][seq.camera[k]],
                                           fill=False, ec=C_OBS, lw=1.6,
                                           label="observation, 2 sd"))
            ax_zoom.plot(*seq.y[k], marker="o", ms=7, color=C_OBS)
        ax_zoom.add_patch(ellipse_from(forward["m"][k], forward["P"][k], facecolor=C_FILTER,
                                       alpha=0.20, ec=C_FILTER, lw=2.0,
                                       label="belief after this step, 2 sd"))
        ax_zoom.plot(*forward["m"][k], marker="o", ms=7, color=C_FILTER)
        ax_zoom.plot(*seq.truth[k], marker="+", ms=16, mew=2.4, color=C_TRUTH,
                     label="where it really was")
        ax_zoom.set_xlim(centre[0] - ZOOM_M, centre[0] + ZOOM_M)
        ax_zoom.set_ylim(centre[1] - ZOOM_M, centre[1] + ZOOM_M)
        ax_zoom.set_aspect("equal", adjustable="box")
        ax_zoom.set_title("the belief, magnified to ±30 cm", fontsize=10)
        ax_zoom.set_xlabel("x, m"); ax_zoom.set_ylabel("y, m")
        ax_zoom.legend(loc="upper left", fontsize=7.5)

        err = float(np.linalg.norm(forward["m"][k] - seq.truth[k]))
        sd = float(np.sqrt(np.trace(forward["P"][k]) / 2))
        if seq.camera[k] is None:
            caption.set_text(f"no observation   |   belief 2sd {200 * sd:5.1f} cm"
                             f"   |   off truth by {100 * err:5.1f} cm")
        else:
            innovation = seq.y[k] - forward["m_pred"][k]
            verdict = "USED" if item["used"] else "REJECTED"
            caption.set_text(
                f"innovation ({100 * innovation[0]:+6.1f},{100 * innovation[1]:+6.1f}) cm"
                f"   |   v'S-1v {item['nis']:6.2f} vs gate {nm.GATE_CHI2_2DOF:.2f} -> {verdict}"
                f"   |   belief 2sd {200 * sd:5.1f} cm   |   off truth by {100 * err:5.1f} cm")
        return []



    run_animation = animation.FuncAnimation(fig, draw_step, frames=len(run_frames),
                                            interval=420, blit=False)
    with plt.rc_context({"animation.frame_format": "jpeg"}):
        display(HTML(run_animation.to_jshtml(default_mode="once")))
    plt.close(fig)


def report_smoothing(seq, forward, smooth):
    print("distance from the truth:")
    err_filter = nm.error_summary(forward["m"], seq, "filtered")
    err_smooth = nm.error_summary(smooth["m"], seq, "smoothed")
    _ = nm.error_summary(seq.odom - seq.odom[0] + seq.truth[0], seq, "wheel odometry alone")


def smoothing_bands(seq, forward, smooth, time):
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 6.2), sharex=True)
    for axis, ax in enumerate(axes):
        name = "east" if axis == 0 else "north"
        for result, colour, label in ((forward, C_FILTER, "filtered (uses the past only)"),
                                      (smooth, C_SMOOTH, "smoothed (uses the whole drive)")):
            band = 100 * two_sigma(result["P"], axis)
            ax.fill_between(time, -band, band, color=colour, alpha=0.16, label=f"{label}, 2 sd")
            ax.plot(time, 100 * (result["m"][:, axis] - seq.truth[:, axis]), color=colour, lw=1.5)
        ax.axhline(0, color=C_TRUTH, lw=1.4, label="where the robot really was")
        ax.scatter(time[forward["used"]], 100 * (seq.y[forward["used"], axis]
                                                 - seq.truth[forward["used"], axis]),
                   s=10, color=C_OBS, lw=0, alpha=0.6, label="observations used")
        ax.set_ylabel(f"{name} error,\ncentimetres")
    axes[0].set_title("Smoothing narrows the belief, but the error stays where it is")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.34), ncol=2, fontsize=8.5)
    axes[-1].set_xlabel("time since the start of the drive, seconds")
    plt.show()


def report_learning(history, cameras=None):
    shown = list(cameras or nd.CAMERAS)
    keep = [h for h in history if h["iteration"] < 4 or h is history[-1]]
    # The ungated figure, deliberately: the gate admits a different subset of readings for
    # every R, so the gated total sums over different data from pass to pass.
    print(f"  {'pass':>6s}{'how well it fits':>20s}   {'learned noise, cm':s}")
    for record in keep:
        sigmas = "  ".join(f"{CAMERA_SHORT[c]}={100 * record['sigma_m'][c]:5.2f}"
                           for c in shown)
        print(f"  {record['iteration']:>6d}{record['log_evidence_all']:>20.1f}   {sigmas}")
    if len(keep) < len(history):
        print(f"  (passes 4 to {history[-2]['iteration']} omitted; nothing moves after 3)")


def learned_R_summary(seq, history, R_learned, commissioned_sigma, commissioned):
    fig, (ax_obj, ax_sigma) = plt.subplots(1, 2, figsize=(10.4, 4.0))

    forward_all = nm.kalman_filter(seq, commissioned['R_total'], gate=float("inf"))
    ax_obj.plot([h["iteration"] for h in history], [h["log_evidence_all"] for h in history],
                marker="o", ms=4, color=C_ACCENT, lw=1.6)
    ax_obj.axhline(forward_all["log_evidence"], color=C_TRUTH, ls="--", lw=1.2,
                   label="the commissioned R")
    ax_obj.set_xlabel("outer pass"); ax_obj.set_ylabel("plug-in log fit, nats")
    ax_obj.set_title("The fitted R explains these observations better\n"
                     "(not the ELBO and not a held-out score)")
    ax_obj.legend(fontsize=8.5)

    names = [f"camera {CAMERA_SHORT[c]}" for c in nd.CAMERAS]
    positions = np.arange(len(names))
    commissioned_bars = [100 * commissioned_sigma[c] for c in nd.CAMERAS]
    learned = [100 * float(np.sqrt(np.trace(R_learned[c]) / 2)) for c in nd.CAMERAS]
    ax_sigma.barh(positions + 0.19, commissioned_bars, height=0.36, color=C_TRUTH,
                  label="commissioned on other runs (scatter + offset)")
    ax_sigma.barh(positions - 0.19, learned, height=0.36, color=C_ACCENT,
                  label="learned on this run")
    ax_sigma.set_yticks(positions); ax_sigma.set_yticklabels(names)
    ax_sigma.set_xlabel("assumed observation noise, one standard deviation, cm")
    ax_sigma.set_title("by asking to trust the cameras MORE, not less")
    ax_sigma.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5)
    plt.show()

    for cam in nd.CAMERAS:
        print(f"  camera {CAMERA_SHORT[cam]}: commissioned {100 * commissioned_sigma[cam]:5.2f} cm"
              f"  ->  learned {100 * float(np.sqrt(np.trace(R_learned[cam]) / 2)):5.2f} cm"
              f"   ({history[-1]['counts'][cam]:4d} observations)")


def posterior_over_R(vb, history, commissioned_sigma):
    fig, (ax_pdf, ax_kl) = plt.subplots(1, 2, figsize=(11.0, 4.2),
                                        gridspec_kw={"width_ratios": [1.55, 1.0]})

    grid = np.linspace(0.002, 0.10, 400)
    prior_pdf = nm.sigma_density(vb["Psi_prior"], vb["nu_prior"], grid)
    ax_pdf.plot(100 * grid, prior_pdf / prior_pdf.max(), color=C_OBS, lw=1.6, ls="--",
                label="prior, before any data")
    for cam in nd.CAMERAS:
        post = vb["posterior"][cam]
        pdf = nm.sigma_density(post["Psi"], post["nu"], grid)
        ax_pdf.plot(100 * grid, pdf / pdf.max(), color=CAMERA_COLOUR[cam], lw=1.9,
                    label=f"camera {CAMERA_SHORT[cam]}  ({int(post['nu'] - vb['nu_prior'])} obs)")
        ax_pdf.axvline(100 * commissioned_sigma[cam], color=CAMERA_COLOUR[cam],
                       lw=1.0, ls=":", alpha=0.8)
    ax_pdf.set_xlabel("observation noise, one standard deviation, cm")
    ax_pdf.set_ylabel("posterior density (each scaled to its own peak)")
    ax_pdf.set_title("What the data believes about each camera's noise\n"
                     "(dotted verticals: what commissioning measured on other runs)")
    ax_pdf.set_xlim(0, 10)
    ax_pdf.legend(fontsize=8.5)

    names = [f"camera {CAMERA_SHORT[c]}" for c in nd.CAMERAS]
    positions = np.arange(len(names))
    kl = [history[-1]["kl_from_prior"][c] for c in nd.CAMERAS]
    ax_kl.barh(positions, kl, color=[CAMERA_COLOUR[c] for c in nd.CAMERAS], height=0.62)
    ax_kl.set_yticks(positions); ax_kl.set_yticklabels(names)
    ax_kl.set_xlabel("distance from the prior, nats")
    ax_kl.set_title("How far the data moved each one\n(the ELBO's penalty term)")
    plt.show()

    print("the east axis of each camera's noise, as a posterior rather than a number:")
    print(f"  {'camera':9s}{'obs':>6s}{'posterior mean':>16s}{'68% interval':>20s}"
          f"{'from the prior':>17s}")
    print(f"  {'':9s}{'':>6s}{'cm':>16s}{'cm':>20s}{'nats':>17s}")
    for cam in nd.CAMERAS:
        post = vb["posterior"][cam]
        n = int(post["nu"] - vb["nu_prior"])
        shape = (post["nu"] - 2.0 + 1.0) / 2.0
        scale = post["Psi"][0, 0] / 2.0
        lo, hi = np.sqrt(stats.invgamma.ppf([0.16, 0.84], a=shape, scale=scale))
        # E[var] of the same marginal, so the mean and the interval describe one quantity
        mean_sigma = float(np.sqrt(post["Psi"][0, 0] / (post["nu"] - 3.0)))
        print(f"  {CAMERA_SHORT[cam]:9s}{n:>6d}{100 * mean_sigma:>16.2f}"
              f"{f'{100 * lo:.2f} to {100 * hi:.2f}':>20s}"
              f"{history[-1]['kl_from_prior'][cam]:>17.1f}")
    print("\nMore observations means a narrower posterior and a larger distance from the prior:")
    print("camera C has the most of both, camera B the fewest.")


def report_fit_versus_honesty(scores, seq, forward, forward_all, forward_learned, history, commissioned):
    print(f"  {'':38s}{'median NEES':>13s}{'score (lower better)':>22s}{'distance':>11s}")
    print(f"  {'':38s}{'(1.39 = honest)':>13s}{'':>22s}{'RMSE, cm':>11s}")
    for s in scores:
        print(f"  {s['label']:38s}{s['median_nees']:>13.2f}{s['mean_nlpd']:>22.2f}"
              f"{s['rmse_cm']:>11.1f}")

    # The gate admits a different subset of observations for each R, so the gated evidence
    # sums over different data. For a comparison BETWEEN models it has to be recomputed with
    # the gate off, over every observation. Doing it the sloppy way overstated the gap by
    # about a hundred nats here.
    print("\nplug-in log fit, all 349 observations, gate off, so the rows use the same data:")
    evidence_arms = [
        ("commissioned scatter only", nm.kalman_filter(seq, commissioned['R_spread'],
                                                    gate=float("inf"))["log_evidence"],
         scores[0]["median_nees"]),
        ("commissioned scatter + offset", forward_all["log_evidence"], scores[1]["median_nees"]),
        ("learned on this run", history[-1]["log_evidence_all"], scores[2]["median_nees"]),
    ]
    print(f"  {'arm':32s}{'log fit, nats':>16s}{'median NEES':>14s}")
    for name, ev, nees in evidence_arms:
        print(f"  {name:32s}{ev:>16.1f}{nees:>14.2f}")
    by_evidence = [a[0] for a in sorted(evidence_arms, key=lambda a: -a[1])]
    by_honesty = [a[0] for a in sorted(evidence_arms, key=lambda a: abs(math.log(a[2] / nm.CALIBRATED_MEDIAN_NEES)))]
    print(f"\n  best fit first:      {'  >  '.join(by_evidence)}")
    print(f"  most honest first:   {'  >  '.join(by_honesty)}")
    print(f"  the two orderings are exact opposites: {by_evidence == by_honesty[::-1]}")


def fit_versus_honesty(scores):
    fig, (ax_nees, ax_score) = plt.subplots(1, 2, figsize=(10.6, 4.2))

    labels = [s["label"] for s in scores]
    positions = np.arange(len(labels))
    colours = [C_TRUTH, C_ACCENT, C_SMOOTH, "#8FD3C1"]
    ax_nees.barh(positions, [s["median_nees"] for s in scores], color=colours, height=0.6)
    ax_nees.axvline(nm.CALIBRATED_MEDIAN_NEES, color="#B00020", lw=1.6, ls="--",
                    label="an honest belief sits here")
    ax_nees.set_yticks(positions); ax_nees.set_yticklabels(labels, fontsize=8.5)
    ax_nees.set_xlabel("median normalised squared error")
    ax_nees.set_title("How honest the belief is\n(further right = overconfident)")
    ax_nees.legend(loc="lower right", fontsize=8.5)

    ax_score.barh(positions, [s["mean_nlpd"] for s in scores], color=colours, height=0.6)
    ax_score.set_yticks(positions); ax_score.set_yticklabels([])
    ax_score.set_xlabel("proper score, lower is better")
    ax_score.set_title("The score that cannot be gamed\nin either direction")
    plt.show()


def animate_the_fitting(seq, forward, time):
    fitting_steps = replay_fitting(seq)
    print(f"{len(fitting_steps)} half-steps ({len(fitting_steps) // 2} passes)")

    fig = plt.figure(figsize=(13.0, 4.3), dpi=82)
    mosaic = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.25, 0.95], wspace=0.30)
    ax_cloud = fig.add_subplot(mosaic[0, 0])
    ax_path = fig.add_subplot(mosaic[0, 1])
    ax_cost = fig.add_subplot(mosaic[0, 2])

    def draw_fitting(index):
        step = fitting_steps[index]
        for ax in (ax_cloud, ax_path, ax_cost):
            ax.clear()

        # --- left: the cloud being fitted, and the ellipse fitting it
        for cam in nd.CAMERAS:
            rows = step["residuals"][cam]
            if rows.size:
                ax_cloud.scatter(100 * rows[:, 0], 100 * rows[:, 1], s=9, lw=0, alpha=0.40,
                                 color=CAMERA_COLOUR[cam])
            ax_cloud.add_patch(ellipse_from(np.zeros(2), step["R"][cam] * 1e4, n_sigma=2.0,
                                            fill=False, ec=CAMERA_COLOUR[cam], lw=2.0))
            sd = 100 * float(np.sqrt(np.trace(step["R"][cam]) / 2))
            ax_cloud.plot([], [], color=CAMERA_COLOUR[cam], lw=2.0,
                          label=f"{CAMERA_SHORT[cam]}: {sd:.2f} cm")
        ax_cloud.axhline(0, color=C_TRUTH, lw=0.8); ax_cloud.axvline(0, color=C_TRUTH, lw=0.8)
        ax_cloud.set_xlim(-14, 14); ax_cloud.set_ylim(-14, 14)
        ax_cloud.set_aspect("equal", adjustable="box")
        ax_cloud.set_xlabel("residual east, cm"); ax_cloud.set_ylabel("residual north, cm")
        ax_cloud.set_title("the clouds, and the R fitted to them", fontsize=10)
        ax_cloud.legend(fontsize=7.5, loc="upper left", title="assumed noise",
                        title_fontsize=7.5)

        # --- middle: the path, which is what makes the clouds move
        ax_path.axhline(0, color=C_TRUTH, lw=1.5, label="where it really was")
        ax_path.plot(time, fitting_steps[0]["track"], color=C_OBS, lw=1.1, label="pass 0")
        ax_path.plot(time, step["track"], color=C_FILTER, lw=1.8,
                     label=f"pass {step['pass']}")
        ax_path.scatter(time[forward["used"]],
                        100 * (seq.y[forward["used"], 1] - seq.truth[forward["used"], 1]),
                        s=7, color=C_OBS, lw=0, alpha=0.45, label="observations")
        ax_path.set_ylim(-22, 12)
        ax_path.set_xlabel("time since the start of the drive, s")
        ax_path.set_ylabel("north error, cm")
        ax_path.set_title("the path, bending onto the observations", fontsize=10)
        ax_path.legend(fontsize=7.5, loc="lower left", ncol=2)

        # --- right: what it costs
        shown = [s for s in fitting_steps[:index + 1] if s["half"] == "x"]
        ax_cost.plot([s["pass"] for s in shown], [s["nees"] for s in shown], marker="o", ms=4,
                     color="#B00020", lw=1.8)
        ax_cost.axhline(nm.CALIBRATED_MEDIAN_NEES, color=C_TRUTH, ls="--", lw=1.4,
                        label="an honest belief")
        ax_cost.set_xlim(-0.4, fitting_steps[-1]["pass"] + 0.4)
        ax_cost.set_ylim(0, max(s["nees"] for s in fitting_steps) * 1.12)
        ax_cost.set_xlabel("pass"); ax_cost.set_ylabel("median normalised squared error")
        ax_cost.set_title("the belief, getting less honest", fontsize=10)
        ax_cost.legend(fontsize=8, loc="lower right")

        if step["half"] == "x":
            headline = (f"pass {step['pass']}  ·  x step: refit the path using the current R"
                        f"   →   watch the path (middle) move onto the observations")
        else:
            headline = (f"pass {step['pass']}  ·  R step: refit R to the residual clouds"
                        f"   →   watch the ellipses (left) collapse")
        fig.suptitle(headline, fontsize=12)
        return []



    fitting_animation = animation.FuncAnimation(fig, draw_fitting, frames=len(fitting_steps),
                                                interval=1100, blit=False)
    with plt.rc_context({"animation.frame_format": "jpeg"}):
        display(HTML(fitting_animation.to_jshtml(default_mode="once")))
    plt.close(fig)

    print(f"  {'pass':>5s}{'assumed noise, cm (A/B/C/D)':>34s}{'median NEES':>13s}")
    for step in fitting_steps:
        if step["half"] != "x":
            continue
        sigmas = " ".join(f"{100 * float(np.sqrt(np.trace(step['R'][c]) / 2)):4.2f}"
                          for c in nd.CAMERAS)
        print(f"  {step['pass']:>5d}{sigmas:>34s}{step['nees']:>13.2f}")


def animate_the_distributions(seq, forward, history, vb, forward_all, time,
                             commissioned_sigma):
    fit_history = [h for h in history if "posterior" in h]
    fit_nees = [float(np.median(nm.honesty(nm.kalman_filter(seq, h["R_bar"]), seq, "it")["nees"]))
                for h in fit_history]
    fit_evidence = [h["log_evidence_all"] for h in fit_history]

    # Also keep each iterate's filtered track, so the animation can show the trajectory being
    # bent to absorb the bias -- the mechanism claimed in the prose, which was never shown.
    fit_tracks = [nm.kalman_filter(seq, h["R_bar"])["m"] for h in fit_history]

    fig = plt.figure(figsize=(13.6, 3.9), dpi=82)
    mosaic = fig.add_gridspec(1, 4, width_ratios=[1.35, 1.0, 1.0, 1.25], wspace=0.34)
    ax_pdf = fig.add_subplot(mosaic[0, 0])
    ax_fit = fig.add_subplot(mosaic[0, 1])
    ax_hon = fig.add_subplot(mosaic[0, 2])
    ax_track = fig.add_subplot(mosaic[0, 3])

    sigma_grid = np.linspace(0.002, 0.075, 400)
    prior_curve = nm.sigma_density(vb["Psi_prior"], vb["nu_prior"], sigma_grid)

    def draw_fit(step):
        for ax in (ax_pdf, ax_fit, ax_hon, ax_track):
            ax.clear()

        ax_pdf.plot(100 * sigma_grid, prior_curve / prior_curve.max(), color=C_OBS, lw=1.5,
                    ls="--", label="prior, before any data")
        for cam in nd.CAMERAS:
            post = fit_history[step]["posterior"][cam]
            curve = nm.sigma_density(post["Psi"], post["nu"], sigma_grid)
            ax_pdf.plot(100 * sigma_grid, curve / curve.max(), color=CAMERA_COLOUR[cam],
                        lw=2.0, label=f"camera {CAMERA_SHORT[cam]}")
            ax_pdf.axvline(100 * commissioned_sigma[cam], color=CAMERA_COLOUR[cam],
                           lw=1.0, ls=":", alpha=0.7)
        ax_pdf.set_xlim(0, 7.5); ax_pdf.set_ylim(0, 1.12)
        ax_pdf.set_xlabel("observation noise, one standard deviation, cm")
        ax_pdf.set_ylabel("posterior density, scaled")
        ax_pdf.set_title(f"pass {fit_history[step]['iteration']}  —  what the data believes\n"
                         "(dotted: commissioned on other runs)", fontsize=10)
        ax_pdf.legend(fontsize=8, loc="upper right")

        xs = [h["iteration"] for h in fit_history[:step + 1]]
        ax_fit.plot(xs, fit_evidence[:step + 1], marker="o", ms=4, color=C_ACCENT, lw=1.8)
        ax_fit.axhline(forward_all["log_evidence"], color=C_TRUTH, ls="--", lw=1.3,
                       label="commissioned R")
        ax_fit.set_xlim(-0.4, fit_history[-1]["iteration"] + 0.4)
        # include the commissioned baseline in the view, or the comparison the dashed line is
        # there to make sits off the top of the axis and the legend describes nothing.
        low = min(min(fit_evidence), forward_all["log_evidence"])
        high = max(max(fit_evidence), forward_all["log_evidence"])
        ax_fit.set_ylim(low - 0.08 * (high - low), high + 0.08 * (high - low))
        ax_fit.set_xlabel("outer pass"); ax_fit.set_ylabel("plug-in log fit, nats")
        ax_fit.set_title("Fit to the observations\n(higher is better; not the objective)",
                         fontsize=10)
        ax_fit.legend(fontsize=8, loc="lower right")

        ax_hon.plot(xs, fit_nees[:step + 1], marker="o", ms=4, color="#B00020", lw=1.8)
        ax_hon.axhline(nm.CALIBRATED_MEDIAN_NEES, color=C_TRUTH, ls="--", lw=1.4,
                       label="an honest belief")
        ax_hon.set_xlim(-0.4, fit_history[-1]["iteration"] + 0.4)
        ax_hon.set_ylim(0, max(fit_nees) * 1.12)
        ax_hon.set_xlabel("pass"); ax_hon.set_ylabel("median normalised squared error")
        ax_hon.set_title("The belief gets less honest\n(lower is better)", fontsize=10)
        ax_hon.legend(fontsize=8, loc="lower right")

        # where the trajectory goes as R shrinks: the bias being absorbed into the state
        ax_track.axhline(0, color=C_TRUTH, lw=1.4, label="where it really was")
        ax_track.plot(time, 100 * (fit_tracks[0][:, 1] - seq.truth[:, 1]), color=C_OBS,
                      lw=1.1, label="pass 0")
        ax_track.plot(time, 100 * (fit_tracks[step][:, 1] - seq.truth[:, 1]), color=C_FILTER,
                      lw=1.7, label=f"pass {fit_history[step]['iteration']}")
        ax_track.scatter(time[forward["used"]],
                         100 * (seq.y[forward["used"], 1] - seq.truth[forward["used"], 1]),
                         s=7, color=C_OBS, lw=0, alpha=0.5, label="observations")
        ax_track.set_ylim(-22, 12)
        ax_track.set_xlabel("time since the start of the drive, s")
        ax_track.set_ylabel("north error, cm")
        ax_track.set_title("...because the track is bending\ntowards the biased observations",
                           fontsize=10)
        ax_track.legend(fontsize=7.5, loc="lower left", ncol=2)
        return []



    fit_animation = animation.FuncAnimation(fig, draw_fit, frames=len(fit_history),
                                            interval=700, blit=False)
    with plt.rc_context({"animation.frame_format": "jpeg"}):
        display(HTML(fit_animation.to_jshtml(default_mode="once")))
    plt.close(fig)

    print(f"  {'pass':>5s}{'log fit':>12s}{'median NEES':>14s}   per-camera one-sd, cm")
    for h, ev, ns in zip(fit_history, fit_evidence, fit_nees):
        sigmas = "  ".join(f"{CAMERA_SHORT[c]}={100 * float(np.sqrt(np.trace(h['R_bar'][c]) / 2)):4.2f}"
                           for c in nd.CAMERAS)
        print(f"  {h['iteration']:>5d}{ev:>12.1f}{ns:>14.2f}   {sigmas}")


def report_offsets_removed(final, R_learned, R_corrected):
    print(f"  {'':34s}{'median NEES':>13s}{'score':>10s}{'RMSE, cm':>11s}")
    for s in final:
        print(f"  {s['label']:34s}{s['median_nees']:>13.2f}{s['mean_nlpd']:>10.2f}"
              f"{s['rmse_cm']:>11.1f}")

    print()
    print("  learned observation noise, one standard deviation, cm:")
    for cam in nd.CAMERAS:
        before = 100 * float(np.sqrt(np.trace(R_learned[cam]) / 2))
        after = 100 * float(np.sqrt(np.trace(R_corrected[cam]) / 2))
        print(f"    camera {CAMERA_SHORT[cam]}: {before:5.2f}  ->  {after:5.2f} once the offset is removed")


def offsets_removed(final):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    labels = [s["label"] for s in final]
    positions = np.arange(len(labels))
    bar_colours = [C_TRUTH, C_ACCENT, C_SMOOTH, "#8FD3C1", "#7B4EA8", "#C9A0DC"]
    ax.barh(positions, [s["median_nees"] for s in final], color=bar_colours, height=0.62)
    ax.axvline(nm.CALIBRATED_MEDIAN_NEES, color="#B00020", lw=1.7, ls="--",
               label="an honest belief sits here")
    ax.set_yticks(positions); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("median normalised squared error (further right = more overconfident)")
    ax.set_title("Learning the noise does not buy honesty; removing the offset does")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.invert_yaxis()
    plt.show()


def report_heading(seq, odom_heading, models):
    print(f"\nheading available from odometry at {int(np.isfinite(odom_heading).sum())} "
          f"of {seq.n_steps} steps")


def does_it_predict(residual_by_heading, capture):
    print(f"  {'':24s}{'n':>7s}{'median':>9s}{'p90':>8s}{'mean':>8s}{'removed':>10s}")
    baseline = float(np.median(residual_by_heading["no correction"]))
    for label, values in residual_by_heading.items():
        if values.size == 0:
            continue
        share = "" if label == "no correction" else f"{1 - np.median(values) / baseline:>9.0%}"
        print(f"  {label:24s}{values.size:>7d}{100 * np.median(values):>8.2f}c"
              f"{100 * np.quantile(values, 0.9):>7.2f}c{100 * values.mean():>7.2f}c{share:>10s}")

    fig, (ax_hist, ax_bar) = plt.subplots(1, 2, figsize=(11.0, 4.1))
    bins = np.linspace(0, 20, 45)
    for label, colour in (("no correction", C_TRUTH), ("odometry heading", C_ACCENT),
                          ("heading assumed zero", "#B00020")):
        ax_hist.hist(100 * residual_by_heading[label], bins=bins, histtype="step", lw=1.9,
                     color=colour, density=True, label=label)
    ax_hist.set_xlabel("distance from what the observation actually was, cm")
    ax_hist.set_ylabel("density")
    ax_hist.set_title("Predicting the observation from the robot's shape")
    ax_hist.legend(fontsize=8.5)

    labels = [k for k in residual_by_heading if residual_by_heading[k].size]
    positions = np.arange(len(labels))
    ax_bar.barh(positions, [100 * np.median(residual_by_heading[k]) for k in labels],
                color=[C_TRUTH, C_SMOOTH, C_ACCENT, "#B00020"][:len(labels)], height=0.6)
    ax_bar.set_yticks(positions); ax_bar.set_yticklabels(labels, fontsize=9)
    ax_bar.set_xlabel("median distance from the observation, cm")
    ax_bar.set_title("A wrong heading is worse than no correction:\nthe heading is doing the work")
    ax_bar.invert_yaxis()
    plt.show()


def geometry_honesty(seq, seq_geometry, forward_corrected, seq_corrected, commissioned):
    print(f"corrected {seq_geometry.n_corrected} of {int(seq.observed.sum())} observations "
          f"with no ground truth\n")

    geometry_scores = [
        nm.honesty(nm.kalman_filter(seq, commissioned['R_total']), seq,
                "commissioned R, as before"),
        nm.honesty(nm.kalman_filter(seq_geometry, commissioned['R_total']), seq_geometry,
                "observation function corrected"),
        nm.honesty(nm.kalman_filter(seq_geometry, commissioned['R_spread']), seq_geometry,
                "corrected, and R back to scatter only"),
        nm.honesty(forward_corrected, seq_corrected,
                "offsets removed using ground truth (the ceiling)"),
    ]
    print(f"  {'':46s}{'median NEES':>13s}{'score':>9s}{'RMSE cm':>9s}")
    for s in geometry_scores:
        print(f"  {s['label']:46s}{s['median_nees']:>13.2f}{s['mean_nlpd']:>9.2f}"
              f"{s['rmse_cm']:>9.2f}")
    print(f"\n  an honest belief scores {nm.CALIBRATED_MEDIAN_NEES:.2f}")

    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    labels = [s["label"] for s in geometry_scores]
    positions = np.arange(len(labels))
    ax.barh(positions, [s["median_nees"] for s in geometry_scores],
            color=[C_TRUTH, C_ACCENT, C_SMOOTH, "#8FD3C1"], height=0.62)
    ax.axvline(nm.CALIBRATED_MEDIAN_NEES, color="#B00020", lw=1.7, ls="--",
               label="an honest belief sits here")
    ax.set_yticks(positions); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("median normalised squared error (further right = more overconfident)")
    ax.set_title("Correcting the observation function, with nothing the robot lacks")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.invert_yaxis()
    plt.show()


def report_offset_filter(offset_runs, forward_spread, seq, measured_offset):
    print(f"  {'offset model':32s}{'median NEES':>13s}{'RMSE cm':>9s}"
          f"{'offsets recovered to':>22s}")
    print(f"  {'no offset in the state':32s}"
          f"{nm.honesty(forward_spread, seq, 'x')['median_nees']:>13.2f}"
          f"{nm.honesty(forward_spread, seq, 'x')['rmse_cm']:>9.2f}{'-':>22s}")
    for label, run in offset_runs.items():
        scored = nm.score_offset_filter(run, seq, label)
        errors = [float(np.linalg.norm(run["m"][-1, 2 + 2 * i:4 + 2 * i] - measured_offset[cam]))
                  for i, cam in enumerate(nm.offset_cameras())]
        print(f"  {label:32s}{scored['median_nees']:>13.2f}{scored['rmse_cm']:>9.2f}"
              f"{100 * float(np.median(errors)):>21.1f}c")
    print(f"\n  an honest belief scores {nm.CALIBRATED_MEDIAN_NEES:.2f}")


def report_drift_sweep(seq, offered_per_camera, measured_offset, commissioned):
    print(f"  {'offset may move':>16s}{'NEES':>8s}{'RMSE':>8s}{'offset err':>12s}"
          f"   observations used")
    for walk in (0.0, 0.001, 0.002, 0.004, 0.008, 0.015):
        run = nm.offset_state_filter(seq, commissioned['R_spread'], sigma_b_walk=walk)
        used = {cam: sum(1 for k in range(seq.n_steps)
                         if seq.camera[k] == cam and run["used"][k]) for cam in nm.offset_cameras()}
        scored = nm.score_offset_filter(run, seq, "sweep")
        errors = [float(np.linalg.norm(run["m"][-1, 2 + 2 * i:4 + 2 * i] - measured_offset[cam]))
                  for i, cam in enumerate(nm.offset_cameras())]
        tally = " ".join(f"{CAMERA_SHORT[c]}={used[c]}/{offered_per_camera[c]}"
                         for c in nm.offset_cameras())
        print(f"  {1000 * walk:>13.0f} mm{scored['median_nees']:>8.2f}{scored['rmse_cm']:>8.2f}"
              f"{100 * float(np.median(errors)):>11.1f}c   {tally}")

    no_gate = nm.offset_state_filter(seq, commissioned['R_spread'], sigma_b_walk=0.002,
                                  gate=float("inf"))
    scored_no_gate = nm.score_offset_filter(no_gate, seq, "gate off")
    print(f"\n  with the gate switched off entirely, every observation used: "
          f"NEES {scored_no_gate['median_nees']:.2f}, RMSE {scored_no_gate['rmse_cm']:.2f} cm")
    print(f"  against NEES 0.74 and RMSE 5.29 cm with it on, so the conclusion does not rest")
    print(f"  on those rejections.")


def offset_traces(seq, best, offset_runs, measured_offset, offered_per_camera, time):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.0), sharex=True)
    for ax, (i, cam) in zip(axes.ravel(), enumerate(nm.offset_cameras())):
        for axis, style, name in ((0, "-", "east"), (1, "--", "north")):
            column = 2 + 2 * i + axis
            estimate = 100 * best["m"][:, column]
            band = 100 * 2 * best["sd"][:, column]
            ax.fill_between(time, estimate - band, estimate + band,
                            color=CAMERA_COLOUR[cam], alpha=0.13)
            ax.plot(time, estimate, color=CAMERA_COLOUR[cam], lw=1.8, ls=style,
                    label=f"{name}, learned")
            ax.axhline(100 * measured_offset[cam][axis], color=C_TRUTH, lw=1.1, ls=style,
                       alpha=0.7, label=f"{name}, what it averaged")
        seen = [k for k in range(seq.n_steps) if seq.camera[k] == cam and best["used"][k]]
        if seen:
            ax.scatter(time[seen], np.full(len(seen), ax.get_ylim()[0] + 0.6), s=5,
                       marker="|", color=CAMERA_COLOUR[cam])
        ax.set_title(f"camera {CAMERA_SHORT[cam]}  ({len(seen)} observations)", fontsize=10,
                     color=CAMERA_COLOUR[cam])
        ax.set_ylabel("offset, cm")
    axes[0, 0].legend(fontsize=7.5, ncol=2, loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel("time since the start of the drive, seconds")
    fig.suptitle("Each camera's offset, learned online with a 2 sd band\n"
                 "(ticks along the bottom mark when that camera contributed)", fontsize=11)
    plt.show()

    print("Camera B is the instructive one: it sits at its prior with a wide band until the")
    print("first observation it contributes, and its band never closes, because it contributes")
    print(f"only {sum(1 for k in range(seq.n_steps) if seq.camera[k] == 'camera_B' and best['used'][k])}"
          f" of the {offered_per_camera['camera_B']} it offers. The filter is right to be unsure "
          "about it.")


def animate_the_offsets(seq, best, measured_offset, time):
    offset_frames = list(range(0, seq.n_steps, 20))




    # One panel per camera rather than four ellipses on one axis: overlaid, they sit on top of
    # each other at the shared 10 cm prior and nothing is readable.
    fig = plt.figure(figsize=(11.6, 5.6), dpi=74)
    mosaic = fig.add_gridspec(2, 3, width_ratios=[0.52, 0.52, 1.35], wspace=0.40, hspace=0.30)
    offset_axes = [fig.add_subplot(mosaic[r, c]) for r in (0, 1) for c in (0, 1)]
    ax_pos = fig.add_subplot(mosaic[:, 2])

    def draw_offset_fit(index):
        step = offset_frames[index]
        for ax in offset_axes + [ax_pos]:
            ax.clear()

        for ax, (i, cam) in zip(offset_axes, enumerate(nm.offset_cameras())):
            colour = CAMERA_COLOUR[cam]
            centre = 100 * best["m"][step, 2 + 2 * i:4 + 2 * i]
            block = best["P_offset"][step, i] * 1e4                 # m^2 -> cm^2
            live = seq.camera[step] == cam
            ax.add_patch(ellipse_from(centre, block, n_sigma=2.0, fill=True,
                                      facecolor=colour, alpha=(0.30 if live else 0.15),
                                      ec=colour, lw=(2.4 if live else 1.4)))
            ax.plot(*centre, marker="o", ms=6, color=colour)
            ax.plot(100 * measured_offset[cam][0], 100 * measured_offset[cam][1], marker="X",
                    ms=11, color=C_TRUTH, markeredgecolor="white", mew=1.2)
            ax.axhline(0, color=C_TRUTH, lw=0.7); ax.axvline(0, color=C_TRUTH, lw=0.7)
            ax.set_xlim(-19, 19); ax.set_ylim(-19, 19)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{CAMERA_SHORT[cam]} — {contributions_by(step, cam)} used"
                         + ("  ●" if live else ""), fontsize=9, color=colour)
            ax.tick_params(labelsize=7)
        offset_axes[2].set_xlabel("offset east, cm", fontsize=8)
        offset_axes[0].set_ylabel("offset north, cm", fontsize=8)
        offset_axes[2].set_ylabel("offset north, cm", fontsize=8)
        offset_axes[3].set_xlabel("offset east, cm", fontsize=8)

        band = 100 * 2 * np.sqrt(np.maximum(best["P_position"][:step + 1, 1, 1], 0.0))
        error = 100 * (best["m"][:step + 1, 1] - seq.truth[:step + 1, 1])
        ax_pos.fill_between(time[:step + 1], -band, band, color=C_FILTER, alpha=0.18,
                            label="what the filter claims (2 sd)")
        ax_pos.axhline(0, color=C_TRUTH, lw=1.4, label="where it really was")
        ax_pos.plot(time[:step + 1], error, color=C_FILTER, lw=1.7, label="its error")
        seen = [k for k in range(step + 1) if seq.camera[k] is not None and best["used"][k]]
        if seen:
            ax_pos.scatter(time[seen], 100 * (seq.y[seen, 1] - seq.truth[seen, 1]), s=7,
                           color=C_OBS, lw=0, alpha=0.45, label="observations")
        ax_pos.set_xlim(0, time[-1]); ax_pos.set_ylim(-22, 14)
        ax_pos.set_xlabel("time since the start of the drive, s")
        ax_pos.set_ylabel("north error, cm")
        ax_pos.set_title("the position belief, while that is happening", fontsize=10)
        ax_pos.legend(fontsize=7.5, loc="lower left", ncol=2)

        live = seq.camera[step]
        who = f"camera {CAMERA_SHORT[live]} speaking" if live else "no camera"
        fig.suptitle(f"t = {time[step]:5.1f} s   ·   {who}   ·   "
                     f"X marks what each offset actually averaged", fontsize=11)
        return []

    def contributions_by(step, camera):
        return sum(1 for k in range(step + 1)
                   if seq.camera[k] == camera and best["used"][k])



    offset_animation = animation.FuncAnimation(fig, draw_offset_fit, frames=len(offset_frames),
                                               interval=380, blit=False)
    with plt.rc_context({"animation.frame_format": "jpeg"}):
        display(HTML(offset_animation.to_jshtml(default_mode="once")))
    plt.close(fig)


def three_routes(seq, best, models, measured_offset, learned_offset, geometric_offset, commissioned_offset):
    print(f"  {'cam':5s}{'measured':>19s}{'predicted':>19s}{'learned':>19s}"
          f"{'pred err':>10s}{'learn err':>11s}")
    for cam in nm.offset_cameras():
        m_, p_, l_ = measured_offset[cam], geometric_offset[cam], learned_offset[cam]
        print(f"  {CAMERA_SHORT[cam]:5s}"
              f"({100*m_[0]:+6.1f},{100*m_[1]:+6.1f})"
              f"({100*p_[0]:+6.1f},{100*p_[1]:+6.1f})"
              f"({100*l_[0]:+6.1f},{100*l_[1]:+6.1f})"
              f"{100*float(np.linalg.norm(p_-m_)):>9.1f}c{100*float(np.linalg.norm(l_-m_)):>10.1f}c")

    common_learned = np.mean([learned_offset[c] for c in nm.offset_cameras()], axis=0)
    common_measured = np.mean([measured_offset[c] for c in nm.offset_cameras()], axis=0)
    spread_errors = [float(np.linalg.norm((learned_offset[c] - common_learned)
                                          - (measured_offset[c] - common_measured)))
                     for c in nm.offset_cameras()]
    print(f"\n  the part common to all four cameras: learned "
          f"({100*common_learned[0]:+5.1f},{100*common_learned[1]:+5.1f}) vs measured "
          f"({100*common_measured[0]:+5.1f},{100*common_measured[1]:+5.1f}) cm, "
          f"off by {100*float(np.linalg.norm(common_learned-common_measured)):.1f} cm")
    print(f"  the differences between cameras, once that part is removed: "
          f"median error {100*float(np.median(spread_errors)):.1f} cm")

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for cam in nm.offset_cameras():
        colour = CAMERA_COLOUR[cam]
        for value, marker, size in ((measured_offset[cam], "X", 13),
                                    (geometric_offset[cam], "s", 9),
                                    (learned_offset[cam], "o", 9)):
            ax.plot(100 * value[0], 100 * value[1], marker=marker, ms=size, color=colour,
                    markeredgecolor="white", markeredgewidth=1.2)
        ax.plot([100 * measured_offset[cam][0], 100 * geometric_offset[cam][0]],
                [100 * measured_offset[cam][1], 100 * geometric_offset[cam][1]],
                color=colour, lw=1.0, alpha=0.5)
        ax.plot([100 * measured_offset[cam][0], 100 * learned_offset[cam][0]],
                [100 * measured_offset[cam][1], 100 * learned_offset[cam][1]],
                color=colour, lw=1.0, alpha=0.5, ls=":")
        ax.annotate(f" {CAMERA_SHORT[cam]}", 100 * measured_offset[cam], fontsize=10,
                    color=colour, fontweight="bold")
    for marker, name in (("X", "measured against truth"), ("s", "predicted from the shape"),
                         ("o", "learned as a state")):
        ax.plot([], [], marker=marker, ls="none", color=C_TRUTH, label=name)
    ax.axhline(0, color=C_TRUTH, lw=0.9); ax.axvline(0, color=C_TRUTH, lw=0.9)
    ax.set_xlabel("offset east, cm"); ax.set_ylabel("offset north, cm")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("The same offset, three independent ways")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), fontsize=8.5)
    plt.show()


def report_pixel_scale(scale_per_pixel, residual_after_correction):
    near = scale_per_pixel < np.median(scale_per_pixel)
    print(f"one pixel is worth between {100 * scale_per_pixel.min():.2f} and "
          f"{100 * scale_per_pixel.max():.2f} cm on the floor "
          f"({scale_per_pixel.max() / scale_per_pixel.min():.1f}x), depending on where in the "
          f"image the robot is.\n")
    print(f"  {'half of the detections':28s}{'cm per pixel':>14s}{'residual after the':>20s}")
    print(f"  {'':28s}{'':>14s}{'bias correction':>20s}")
    for label, mask in (("nearer, looking down", near), ("further, looking along", ~near)):
        print(f"  {label:28s}{100 * np.median(scale_per_pixel[mask]):>13.2f}c"
              f"{100 * np.median(residual_after_correction[mask]):>19.2f}c")
    print(f"\n  correlation between the two, per detection: "
          f"{float(np.corrcoef(scale_per_pixel, residual_after_correction)[0, 1]):+.2f}")
    print(f"  implied pixel-level noise: "
          f"{np.median(residual_after_correction) / np.median(scale_per_pixel):.2f} px")


def jacobian_R(scale_per_pixel, residual_after_correction, scale_camera, constant,
               jacobian_scores, seq_geometry, commissioned):
    print(f"  {'R model':40s}{'free numbers':>14s}{'median NEES':>13s}{'RMSE cm':>9s}")
    print(f"  {constant['label']:40s}{12:>14d}{constant['median_nees']:>13.2f}"
          f"{constant['rmse_cm']:>9.2f}")
    for _sigma_px, scored in jacobian_scores:
        print(f"  {scored['label']:40s}{1:>14d}{scored['median_nees']:>13.2f}"
              f"{scored['rmse_cm']:>9.2f}")
    print(f"\n  an honest belief scores {nm.CALIBRATED_MEDIAN_NEES:.2f}")

    fig, (ax_scatter, ax_bar) = plt.subplots(1, 2, figsize=(11.0, 4.1))
    ax_scatter.scatter(100 * scale_per_pixel, 100 * residual_after_correction, s=14, lw=0,
                       alpha=0.55, c=[CAMERA_COLOUR[c] for c in scale_camera])
    grid_px = np.linspace(0, 1.05 * scale_per_pixel.max(), 20)
    for px in (1.0, 2.0):
        ax_scatter.plot(100 * grid_px, 100 * px * grid_px, color=C_TRUTH, ls="--", lw=1.1)
        ax_scatter.annotate(f"{px:.0f} px", (100 * grid_px[-1], 100 * px * grid_px[-1]),
                            fontsize=8, ha="right", va="bottom", color=C_TRUTH)
    ax_scatter.set_xlabel("what one pixel is worth here, cm")
    ax_scatter.set_ylabel("residual after the bias correction, cm")
    ax_scatter.set_title("Noise grows where a pixel is worth more")
    for cam in nd.CAMERAS:
        ax_scatter.plot([], [], marker="o", ls="none", color=CAMERA_COLOUR[cam],
                        label=f"camera {CAMERA_SHORT[cam]}")
    ax_scatter.legend(fontsize=8, ncol=2)

    labels = [constant["label"]] + [s["label"] for _, s in jacobian_scores]
    values = [constant["median_nees"]] + [s["median_nees"] for _, s in jacobian_scores]
    positions = np.arange(len(labels))
    ax_bar.barh(positions, values, height=0.62,
                color=[C_TRUTH] + [C_ACCENT] * len(jacobian_scores))
    ax_bar.axvline(nm.CALIBRATED_MEDIAN_NEES, color="#B00020", lw=1.7, ls="--",
                   label="an honest belief")
    ax_bar.set_yticks(positions); ax_bar.set_yticklabels(labels, fontsize=8.5)
    ax_bar.set_xlabel("median normalised squared error")
    ax_bar.set_title("One pixel-noise number against twelve fitted ones")
    ax_bar.legend(fontsize=8.5, loc="lower right")
    ax_bar.invert_yaxis()
    plt.show()



# --------------------------------------- learning R, taken one camera at a time

def report_single_camera(seq_one, seq, camera, commissioned):
    """How much this one camera actually contributes, before anything is learned."""
    short = CAMERA_SHORT[camera]
    kept = sum(1 for c in seq_one.camera if c is not None)
    print(f"camera {short} alone: {kept} observations over {seq_one.n_steps} steps "
          f"({100 * kept / seq_one.n_steps:.0f}% of them)")
    print(f"  the other three cameras' {int(seq.observed.sum()) - kept} detections are "
          f"dropped; the filter dead-reckons through those stretches\n")
    print(f"  {'':40s}{'median':>9s}{'90th pct':>10s}{'worst':>8s}")
    for label, source in (("all four cameras", seq), (f"camera {short} only", seq_one)):
        result = nm.kalman_filter(source, commissioned["R_total"])
        ok = np.isfinite(source.truth[:, 0])
        err = 100 * np.linalg.norm(result["m"][ok] - source.truth[ok], axis=1)
        print(f"  {label + ', distance from truth':40s}{np.median(err):>9.1f}"
              f"{np.quantile(err, 0.9):>10.1f}{err.max():>8.1f}")
    print(f"\n  commissioned observation noise for {short}: "
          f"{100 * commissioned['sigma_total_m'][camera]:.2f} cm, one standard deviation.")
    print("  That number came from three other runs. The rest of this section asks what")
    print("  this run alone would say instead.")


def report_one_conjugate_update(update):
    """The R step, as arithmetic. The prose belongs in the markdown, not in here."""
    cm2 = 1e4

    def show(name, M):
        print(f"  {name:30s}[{M[0, 0] * cm2:7.1f} {M[0, 1] * cm2:7.1f}]")
        print(f"  {'':30s}[{M[1, 0] * cm2:7.1f} {M[1, 1] * cm2:7.1f}]")

    print(f"camera {CAMERA_SHORT[update['camera']]}, {update['n']} observations, "
          f"starting from sigma = {100 * update['sigma_prior_m']:.2f} cm\n")
    print("  cm^2")
    show("residuals about the path", update["residual_part"])
    show("+ the smoother's own P", update["smoother_part"])
    show("+ prior Psi", update["Psi_prior"])
    show("= Psi+", update["Psi_post"])
    print(f"\n  nu+ = {update['nu_prior']:.0f} + {update['n']} = {update['nu_post']:.0f}")
    show("Psi+ / nu+", update["R_post"])
    share = np.trace(update["smoother_part"]) / np.trace(update["scatter"])
    print(f"\n  sigma: {100 * update['sigma_prior_m']:.2f} cm -> "
          f"{100 * update['sigma_post_m']:.2f} cm     "
          f"({100 * share:.0f}% of the scatter was the smoother's own P)")


def single_camera_posterior(history, camera, reference_sigma, reference_label):
    """What the data did to the belief about one camera's noise, pass by pass."""
    short = CAMERA_SHORT[camera]
    grid = np.linspace(0.002, 0.12, 500)
    fig, (ax_pdf, ax_trace) = plt.subplots(1, 2, figsize=(11.0, 4.0),
                                           gridspec_kw={"width_ratios": [1.4, 1.0]})

    passes = [h for h in history if "posterior" in h]
    shown = [0, 1, 2, len(passes) - 1]
    shades = ["#CDE3F2", "#94C4E3", "#4E96C8", C_SMOOTH]
    prior_pdf = nm.sigma_density(passes[0]["posterior"][camera]["Psi"] * 0
                                 + np.eye(2) * 0.0025 * 6.0, 6.0, grid)
    ax_pdf.plot(100 * grid, prior_pdf / prior_pdf.max(), color=C_OBS, lw=1.6, ls="--",
                label="prior, before any data")
    for colour, i in zip(shades, shown):
        post = passes[i]["posterior"][camera]
        pdf = nm.sigma_density(post["Psi"], post["nu"], grid)
        ax_pdf.plot(100 * grid, pdf / pdf.max(), color=colour, lw=2.0,
                    label=f"after pass {i + 1}")
    ax_pdf.axvline(100 * reference_sigma, color=C_TRUTH, lw=1.6, ls=":",
                   label=reference_label)
    ax_pdf.set_xlim(0, 12)
    ax_pdf.set_xlabel("this camera's noise, one standard deviation, cm")
    ax_pdf.set_ylabel("how strongly the data prefers it")
    ax_pdf.set_title(f"Camera {short}'s noise: the belief narrows, and it lands\n"
                     f"well below what the errors really are")
    ax_pdf.legend(fontsize=8.2)

    sigmas = [100 * float(np.sqrt(np.trace(h["R_bar"][camera]) / 2)) for h in history]
    ax_trace.plot(range(len(sigmas)), sigmas, marker="o", ms=4, color=C_SMOOTH, lw=1.7,
                  label=f"learned on this run")
    ax_trace.axhline(100 * reference_sigma, color=C_TRUTH, lw=1.6, ls=":",
                     label=reference_label)
    ax_trace.set_xlabel("pass")
    ax_trace.set_ylabel("one standard deviation, cm")
    ax_trace.set_title("It settles after two or three passes")
    ax_trace.legend(fontsize=8.5)
    plt.show()


def where_R_is_large(models, seq, camera, field, constants=None):
    """R as a field over the floor: how big it is, which way it points, and what the
    drive actually met."""
    fig, (ax_field, ax_seen) = plt.subplots(1, 2, figsize=(12.4, 4.8))
    worth_cm = 100 * field["worth_m"]

    mesh = ax_field.pcolormesh(field["x"], field["y"], worth_cm, shading="auto",
                               cmap="viridis_r", vmin=np.nanmin(worth_cm),
                               vmax=np.nanmax(worth_cm))
    bar = fig.colorbar(mesh, ax=ax_field, pad=0.02)
    bar.set_label("cm of floor per pixel of detector error\n(bigger = R is bigger here)",
                  fontsize=8.5)

    # the covariance itself, at a coarse grid, so its ORIENTATION is visible too
    magnify = 26.0
    for x in np.linspace(-9.0, 9.0, 7):
        for y in np.linspace(-8.5, 8.5, 7):
            R = nm.R_at_floor_point(models, camera, x, y)
            if R is None:
                continue
            ax_field.add_patch(ellipse_from((x, y), R * magnify ** 2, n_sigma=1.0,
                                            fill=False, ec="white", lw=1.1, alpha=0.85))
    pos = np.asarray(models[camera].cam_pos[:2], dtype=float)
    ax_field.plot(*pos, marker="s", ms=10, color=CAMERA_COLOUR[camera],
                  markeredgecolor="white", markeredgewidth=1.4, zorder=6, clip_on=False)
    ax_field.annotate(f"camera {CAMERA_SHORT[camera]}", pos, textcoords="offset points",
                      xytext=(0, 12 if pos[1] < 0 else -20), ha="center", fontsize=9,
                      weight="bold", color=CAMERA_COLOUR[camera])
    ax_field.set_xlabel("x, metres")
    ax_field.set_ylabel("y, metres")
    ax_field.set_aspect("equal", adjustable="box")
    ax_field.set_title(f"One camera's R is a field, not a number\n"
                       f"white ellipses are 1 px of error, drawn {magnify:.0f}x oversize")
    ax_field.grid(False)

    ok = np.isfinite(seq.truth[:, 0])
    used = [k for k in range(seq.n_steps) if seq.camera[k] is not None and ok[k]]
    worth, north, who = [], [], []
    for k in used:
        R = nm.R_at_floor_point(models, seq.camera[k], seq.truth[k, 0], seq.truth[k, 1])
        if R is None:
            continue
        worth.append(100 * float(np.sqrt(np.trace(R) / 2)))
        north.append(seq.truth[k, 1])
        who.append(seq.camera[k])
    worth, north = np.asarray(worth), np.asarray(north)

    # The drive is one straight traverse, so a map of it is a stripe. Against position
    # along that traverse it is legible: each camera's own curve, and the handovers.
    for cam in nd.CAMERAS:
        pick = [i for i, c in enumerate(who) if c == cam]
        if not pick:
            continue
        order = np.argsort(north[pick])
        ax_seen.plot(north[pick][order], worth[pick][order], marker="o", ms=3.5, lw=1.2,
                     color=CAMERA_COLOUR[cam], label=f"camera {CAMERA_SHORT[cam]}")
    # The whole point of the comparison: a constant per camera is a FLAT LINE through a
    # quantity that is not flat. Draw the candidates on the same axis.
    for label, (sigmas, colour, style) in (constants or {}).items():
        value = 100 * sigmas[camera]
        ax_seen.axhline(value, color=colour, ls=style, lw=1.6)
        ax_seen.annotate(f"{label}: {value:.1f} cm", (ax_seen.get_xlim()[0], value),
                         textcoords="offset points", xytext=(4, 3), fontsize=8.2,
                         color=colour, va="bottom")
    ax_seen.set_xlabel("where the robot was along the aisle, y in metres")
    ax_seen.set_ylabel("cm of floor per pixel, at that detection")
    ax_seen.set_title("What the drive actually met, against what a constant claims\n"
                      f"the geometry says {worth.min():.1f} to {worth.max():.1f} cm; a "
                      f"single number per camera cannot say that")
    ax_seen.legend(fontsize=8.5, ncol=2)
    plt.show()

    print(f"  camera {CAMERA_SHORT[camera]} alone: a pixel is worth "
          f"{np.nanmin(worth_cm):.2f} cm of floor at best and {np.nanmax(worth_cm):.2f} cm "
          f"at worst, a factor of {np.nanmax(worth_cm) / np.nanmin(worth_cm):.1f}.")
    print(f"  over the drive, across all four cameras: {np.nanmin(worth):.2f} to "
          f"{np.nanmax(worth):.2f} cm per pixel.")
    print("  A single covariance per camera has to average over that.")


def report_fit_versus_honesty_one(arms, seq, camera):
    """Fit and calibration side by side, for one camera. They disagree; that is the point."""
    rows = []
    for label, R in arms:
        scored = nm.honesty(nm.kalman_filter(seq, R), seq, label)
        # Gate off, so every arm is scored over the SAME observations. Gated evidence sums
        # over a different subset for every R and cannot be compared across models.
        evidence = nm.kalman_filter(seq, R, gate=float("inf"))["log_evidence"]
        rows.append({"label": label, "sigma_cm": 100 * float(np.sqrt(np.trace(R[camera]) / 2)),
                     "evidence": evidence, "nees": scored["median_nees"],
                     "rmse_cm": scored["rmse_cm"], "nlpd": scored["mean_nlpd"]})

    print(f"  {'':32s}{'stated sd':>11s}{'evidence':>11s}{'median':>9s}{'RMSE':>8s}")
    print(f"  {'':32s}{'cm':>11s}{'nats':>11s}{'NEES':>9s}{'cm':>8s}")
    for r in rows:
        print(f"  {r['label']:32s}{r['sigma_cm']:>11.2f}{r['evidence']:>11.1f}"
              f"{r['nees']:>9.2f}{r['rmse_cm']:>8.2f}")
    print(f"\n  an honest belief scores {nm.CALIBRATED_MEDIAN_NEES:.2f}; above it the filter "
          f"is overconfident.")

    by_fit = [r["label"] for r in sorted(rows, key=lambda r: -r["evidence"])]
    by_honesty = [r["label"] for r in
                  sorted(rows, key=lambda r: abs(math.log(r["nees"] / nm.CALIBRATED_MEDIAN_NEES)))]
    print(f"\n  best fit first:    {'  >  '.join(by_fit)}")
    print(f"  most honest first: {'  >  '.join(by_honesty)}")
    print(f"  exact opposites:   {by_fit == by_honesty[::-1]}")

    fig, (ax_fit, ax_hon) = plt.subplots(1, 2, figsize=(11.0, 3.6))
    y = np.arange(len(rows))
    labels = [r["label"] for r in rows]
    ax_fit.barh(y, [r["evidence"] for r in rows], height=0.6, color=C_SMOOTH)
    ax_fit.set_yticks(y); ax_fit.set_yticklabels(labels, fontsize=8.5)
    ax_fit.invert_yaxis()
    ax_fit.set_xlabel("log evidence, nats  (further right = fits the data better)")
    ax_fit.set_title("How well it fits")
    ax_hon.barh(y, [r["nees"] for r in rows], height=0.6, color=C_FILTER)
    ax_hon.axvline(nm.CALIBRATED_MEDIAN_NEES, color="#B00020", lw=1.7, ls="--",
                   label="an honest belief")
    ax_hon.set_yticks(y); ax_hon.set_yticklabels([])
    ax_hon.invert_yaxis()
    ax_hon.set_xlabel("median normalised squared error  (further right = more overconfident)")
    ax_hon.set_title("How honest it is")
    ax_hon.legend(fontsize=8.5, loc="lower right")
    plt.show()


def report_offset_removed_one(seq, seq_corrected, camera, offset, R_learned, R_corrected,
                              commissioned):
    """The same loop, once the camera's mean error is subtracted by hand."""
    short = CAMERA_SHORT[camera]
    b = offset[camera]
    print(f"  camera {short}'s average error over this drive: "
          f"({100 * b[0]:+.1f}, {100 * b[1]:+.1f}) cm, magnitude "
          f"{100 * float(np.hypot(*b)):.1f} cm\n")
    rows = [
        ("as recorded, learned R", seq, R_learned),
        ("mean subtracted, learned R", seq_corrected, R_corrected),
        ("mean subtracted, commissioned R", seq_corrected, commissioned["R_spread"]),
    ]
    print(f"  {'':34s}{'learned sd cm':>15s}{'median NEES':>13s}{'RMSE cm':>9s}")
    for label, sequence, R in rows:
        scored = nm.honesty(nm.kalman_filter(sequence, R), sequence, label)
        print(f"  {label:34s}{100 * float(np.sqrt(np.trace(R[camera]) / 2)):>15.2f}"
              f"{scored['median_nees']:>13.2f}{scored['rmse_cm']:>9.2f}")
    print(f"\n  an honest belief scores {nm.CALIBRATED_MEDIAN_NEES:.2f}")


def report_offset_state_one(seq, camera, runs, measured, commissioned):
    """Can one camera learn its own offset? Reported against the 10 cm prior it started from."""
    short = CAMERA_SHORT[camera]
    i = nm.offset_cameras().index(camera)
    print(f"  what camera {short}'s offset actually averaged over this drive: "
          f"({100 * measured[0]:+.1f}, {100 * measured[1]:+.1f}) cm\n")
    print(f"  {'offset may move':>18s}{'estimate, cm':>22s}{'its own sd, cm':>18s}"
          f"{'off by':>9s}{'NEES':>7s}{'RMSE':>7s}")
    for label, run in runs.items():
        b = run["m"][-1, 2 + 2 * i:4 + 2 * i]
        sd = run["sd"][-1, 2 + 2 * i:4 + 2 * i]
        scored = nm.score_offset_filter(run, seq, label)
        print(f"  {label:>18s}{f'({100*b[0]:+.1f}, {100*b[1]:+.1f})':>22s}"
              f"{f'{100*sd[0]:.1f}, {100*sd[1]:.1f}':>18s}"
              f"{100 * float(np.linalg.norm(b - measured)):>9.1f}"
              f"{scored['median_nees']:>7.2f}{scored['rmse_cm']:>7.2f}")
    print("\n  The prior on the offset was 10 cm in each axis. An estimate whose own sd is")
    print("  still most of that has not been informed by the data — it is repeating the prior")
    print("  back. Note the NEES looks fine: the belief is honest because it is WIDE, which is")
    print("  the filter being correct about knowing very little, not the offset being found.")


# ================================================== notebook 1: learning R, one camera

def the_run(seq, camera):
    """What the robot knows on its own, and what the camera says. x = time, y = position."""
    t = seq.stamps - seq.stamps[0]
    dead = seq.odom - seq.odom[0] + seq.truth[0]
    seen = [k for k in range(seq.n_steps) if seq.camera[k] == camera]
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    ax.plot(t, seq.truth[:, 1], color=C_TRUTH, lw=2.0, label="where the robot really was")
    ax.plot(t, dead[:, 1], color=C_ODOM, lw=1.6, ls="--",
            label="wheel odometry alone, drifting")
    ax.scatter(t[seen], seq.y[seen, 1], s=14, lw=0, color=CAMERA_COLOUR[camera], alpha=0.75,
               label=f"camera {CAMERA_SHORT[camera]}'s readings")
    ax.set_xlabel("time since the start of the drive, seconds")
    ax.set_ylabel("position along the aisle, metres north")
    ax.set_title("One drive: odometry drifts away, the camera does not")
    ax.legend(fontsize=8.5)
    plt.show()


def filter_with_R(seq, forward, camera, label):
    """Error against the truth with the stated 2 sd band. x = time, y = error in cm."""
    t = seq.stamps - seq.stamps[0]
    err = 100 * (forward["m"][:, 1] - seq.truth[:, 1])
    band = 100 * two_sigma(forward["P"], 1)
    seen = [k for k in range(seq.n_steps) if forward["used"][k]]
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    ax.fill_between(t, -band, band, color=C_FILTER, alpha=0.18,
                    label="what the filter says it knows, 2 sd")
    ax.plot(t, err, color=C_FILTER, lw=1.6, label="how wrong it actually is")
    ax.axhline(0, color=C_TRUTH, lw=1.4)
    ax.scatter(t[seen], 100 * (seq.y[seen, 1] - seq.truth[seen, 1]), s=12, lw=0,
               color=CAMERA_COLOUR[camera], alpha=0.6, label="the camera's readings")
    ax.set_xlabel("time since the start of the drive, seconds")
    ax.set_ylabel("error along the aisle, centimetres\n(0 = exactly right)")
    ax.set_title(f"{label}\nthe truth should sit inside the band 95% of the time")
    ax.legend(fontsize=8.5, ncol=2)
    plt.show()


def conjugacy(trace, camera, reference_sigma, reference_label):
    """The posterior over sigma after n observations. x = sigma, y = probability density."""
    grid = np.linspace(0.002, 0.11, 500)
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    shades = plt.cm.viridis(np.linspace(0.85, 0.05, len(trace)))
    for colour, entry in zip(shades, trace):
        pdf = nm.sigma_density(entry["Psi"], entry["nu"], grid)
        ax.plot(100 * grid, pdf / 100.0, color=colour, lw=2.0,
                label=f"after {entry['n']} observations" if entry["n"] else "prior, no data")
    ax.axvline(100 * reference_sigma, color=C_TRUTH, ls=":", lw=1.8,
               label=reference_label)
    ax.set_xlabel(f"camera {CAMERA_SHORT[camera]}'s noise, one standard deviation, cm")
    ax.set_ylabel("probability density, per cm")
    ax.set_title("Every observation narrows the belief about R\n"
                 "and moves it away from what the errors really are")
    ax.set_xlim(0, 11)
    ax.legend(fontsize=8.2)
    plt.show()


def animate_conjugacy(seq, camera, reference_sigma, reference_label, *, every=4):
    """The same update, one observation at a time."""
    counts = set(range(0, 200, every))
    trace = nm.conjugate_trace(seq, camera, counts)
    grid = np.linspace(0.002, 0.11, 400)
    fig, ax = plt.subplots(figsize=(7.6, 3.9), dpi=78)

    def draw(i):
        ax.clear()
        entry = trace[i]
        pdf = nm.sigma_density(entry["Psi"], entry["nu"], grid) / 100.0
        ax.fill_between(100 * grid, 0, pdf, color=C_SMOOTH, alpha=0.30)
        ax.plot(100 * grid, pdf, color=C_SMOOTH, lw=2.0)
        ax.axvline(100 * reference_sigma, color=C_TRUTH, ls=":", lw=1.8,
                   label=reference_label)
        sigma = float(np.sqrt(np.trace(entry["Psi"] / entry["nu"]) / 2))
        ax.axvline(100 * sigma, color=C_FILTER, lw=1.6, label="what this data prefers")
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(f"camera {CAMERA_SHORT[camera]}'s noise, one standard deviation, cm")
        ax.set_ylabel("probability density, per cm")
        ax.set_title(f"{entry['n']:3d} observations   ->   {100 * sigma:.2f} cm")
        ax.legend(fontsize=8.5, loc="upper right")

    ani = animation.FuncAnimation(fig, draw, frames=len(trace), interval=180)
    plt.close(fig)
    return HTML(ani.to_jshtml(default_mode="once"))


def fixed_point(sigmas_in_m, sigmas_out_m, camera, reference_sigma, reference_label):
    """Feed sigma in, read sigma out. Where it crosses y = x, the loop stops moving."""
    a, b = 100 * np.asarray(sigmas_in_m), 100 * np.asarray(sigmas_out_m)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    lim = (0, max(a.max(), b.max()) * 1.06)
    ax.plot(lim, lim, color=C_TRUTH, ls="--", lw=1.4, label="unchanged (y = x)")
    ax.plot(a, b, marker="o", ms=4, lw=1.8, color=C_SMOOTH,
            label="one pass of the loop")
    crossing = float(np.interp(0.0, (b - a)[::-1], a[::-1])) if (b - a)[0] > 0 else None
    if crossing is not None:
        ax.plot([crossing], [crossing], marker="*", ms=16, color=C_FILTER, zorder=5,
                label=f"the only value it leaves alone: {crossing:.2f} cm")
    ax.axvline(100 * reference_sigma, color=C_OBS, ls=":", lw=1.6,
               label=f"{reference_label}: {100 * reference_sigma:.2f} cm")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("the noise level fed into the pass, cm")
    ax.set_ylabel("the noise level it hands back, cm")
    ax.set_title(f"Camera {CAMERA_SHORT[camera]}: wherever you start, one pass pulls it down\n"
                 "so the loop cannot be reasoned about one half at a time")
    ax.legend(fontsize=8.2, loc="upper left")
    plt.show()
    return crossing


def fit_versus_honesty_scatter(rows):
    """x = how well it fits, y = how honest it is. The two disagree; that is the result."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for i, r in enumerate(rows):
        ax.scatter(r["evidence"], r["nees"], s=90, color=r["colour"], zorder=4)
        side = -1 if i == len(rows) - 1 else 1
        ax.annotate(r["label"], (r["evidence"], r["nees"]), textcoords="offset points",
                    xytext=(10 * side, -4), ha="left" if side > 0 else "right",
                    va="center", fontsize=8.6, color=r["colour"])
    order = sorted(rows, key=lambda r: r["evidence"])
    ax.plot([r["evidence"] for r in order], [r["nees"] for r in order], lw=1.2,
            color="#BBBBBB", zorder=2)
    ax.axhline(nm.CALIBRATED_MEDIAN_NEES, color="#B00020", ls="--", lw=1.6)
    ax.annotate("an honest belief sits on this line", (ax.get_xlim()[0], nm.CALIBRATED_MEDIAN_NEES),
                textcoords="offset points", xytext=(6, -12), fontsize=8.4, color="#B00020")
    ax.set_yscale("log")
    ax.set_xlabel("plug-in log fit, nats   →   fits these observations better")
    ax.set_ylabel("median normalised squared error\n→   more overconfident")
    ax.set_title("Fitting better and being honest pull in opposite directions")
    plt.show()


def path_bending(seq, paths, camera):
    """The trajectory each pass believed. x = time, y = error along the aisle."""
    t = seq.stamps - seq.stamps[0]
    seen = [k for k in range(seq.n_steps) if seq.camera[k] == camera]
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    shades = plt.cm.viridis(np.linspace(0.85, 0.05, len(paths)))
    for i, (colour, m) in enumerate(zip(shades, paths)):
        ax.plot(t, 100 * (m[:, 1] - seq.truth[:, 1]), color=colour, lw=1.5,
                label="pass 1" if i == 0 else (f"pass {len(paths)}" if i == len(paths) - 1 else None))
    ax.scatter(t[seen], 100 * (seq.y[seen, 1] - seq.truth[seen, 1]), s=14, lw=0,
               color=CAMERA_COLOUR[camera], alpha=0.55, label="the camera's readings")
    ax.axhline(0, color=C_TRUTH, lw=1.4, label="the truth")
    ax.set_xlabel("time since the start of the drive, seconds")
    ax.set_ylabel("estimated position minus truth, cm")
    ax.set_title("Each pass bends the estimate further onto the readings\n"
                 "so the residuals shrink without the error shrinking")
    ax.legend(fontsize=8.5, ncol=2)
    plt.show()


# ==================================================================== notebook 1: forecasts
#
# Everything below answers one question: a covariance fitted on a drive fits that drive
# better -- but does it PREDICT better, and predict what?


def report_forecast_scores(summaries, *, heading=""):
    """The forecast scoreboard. Not one number in here comes from ground truth."""
    if heading:
        print(f"{heading}\n")
    width = max(30, max(len(s["label"]) for s in summaries) + 2)
    print(f"  {'candidate R':{width}s}{'forecast':>10s}{'reading':>9s}{'inside':>8s}"
          f"{'typical':>9s}{'score':>8s}")
    print(f"  {'':{width}s}{'width, cm':>10s}{'missed':>9s}{'the 95%':>8s}"
          f"{'surprise':>9s}{'per':>8s}")
    print(f"  {'':{width}s}{'':>10s}{'by, cm':>9s}{'ellipse':>8s}{'':>9s}{'reading':>8s}")
    for s in summaries:
        print(f"  {s['label']:{width}s}{s['sigma_cm']:10.2f}{s['distance_cm']:9.2f}"
              f"{100 * s['inside_95']:7.0f}%{s['median_nis']:9.2f}{s['log_p_mean']:8.2f}")
    print(f"\n  'inside the 95% ellipse' should be 95%; 'typical surprise' should be "
          f"{nm.CALIBRATED_MEDIAN_NEES:.2f},")
    print("  so every row here is UNDER-confident about the reading it is about to get.")
    print("  'score' is the average log-probability the filter gave the reading that")
    print("  actually arrived, before it arrived. Higher is a better forecast.")


def one_forecast(seq, forecasts, k, camera, *, span_cm=None):
    """One moment: what each candidate R said the next reading would be, and what came.

    The panels share axes deliberately -- what separates the arms is the size of the
    ellipse, and that only reads if the scale is the same on both.
    """
    n = len(forecasts)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.9), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    row0 = next(r for r in forecasts[0]["rows"] if r["k"] == k)
    picked = [next(r for r in e["rows"] if r["k"] == k) for e in forecasts]
    centre = row0["arrived"]
    if span_cm is None:
        reach = max(max(2.6 * r["sigma_m"] + r["distance_m"] for r in picked),
                    1.35 * max(r["distance_m"] for r in picked))
        span_cm = 100 * 2.3 * reach
    half = span_cm / 200.0

    for ax, entry in zip(axes, forecasts):
        row = next(r for r in entry["rows"] if r["k"] == k)
        inside = row["nis"] <= nm.GATE_CHI2_2DOF
        ax.add_patch(ellipse_from(row["predicted"], row["S"], n_sigma=math.sqrt(nm.GATE_CHI2_2DOF),
                                  facecolor=C_FILTER, alpha=0.16, edgecolor=C_FILTER, lw=1.7))
        ax.plot(*row["predicted"], marker="+", ms=13, mew=2.2, color=C_FILTER, zorder=5)
        ax.annotate("where the filter expected\nthe reading to land", row["predicted"],
                    textcoords="offset points", xytext=(0, -30), ha="center",
                    fontsize=8.0, color=C_FILTER)
        ax.plot(*row["arrived"], marker="o", ms=10, color=CAMERA_COLOUR[camera],
                markeredgecolor="white", markeredgewidth=1.2, zorder=6)
        ax.annotate("the reading that arrived", row["arrived"], textcoords="offset points",
                    xytext=(10, 8), fontsize=8.2, color=CAMERA_COLOUR[camera])
        if np.isfinite(row["truth"][0]):
            ax.plot(*row["truth"], marker="x", ms=11, mew=2.2, color=C_TRUTH, zorder=6)
            ax.annotate("where the robot\nreally was", row["truth"],
                        textcoords="offset points", xytext=(10, -22), fontsize=8.2,
                        color=C_TRUTH)
        ax.plot([row["predicted"][0], row["arrived"][0]],
                [row["predicted"][1], row["arrived"][1]], color="#666666", lw=1.0, ls="--",
                zorder=4)
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x, metres")
        verdict = "the reading lands inside it" if inside else "the reading lands OUTSIDE it"
        ax.set_title(f"{entry['label']}\nsays $\\pm${100 * row['sigma_m']:.1f} cm — "
                     f"{verdict}\nmiss {100 * row['distance_m']:.1f} cm, "
                     f"surprise {row['nis']:.2f}", fontsize=9.6)
    axes[0].set_ylabel("y, metres")
    fig.suptitle(f"A forecast, made before the reading arrived: t = {row0['t']:.1f} s\n"
                 "shaded = the 95% region each candidate R predicts for the next reading",
                 fontsize=11.5)
    plt.show()


def forecasts_over_the_run(forecasts, camera):
    """Every forecast in the drive, scored. x = time, y = how surprising the reading was."""
    fig, (ax_nis, ax_score) = plt.subplots(1, 2, figsize=(12.2, 4.1))
    for entry in forecasts:
        t = [r["t"] for r in entry["rows"]]
        # sqrt, so the axis is "how many forecast-widths away", not a squared quantity
        ax_nis.plot(t, [math.sqrt(r["nis"]) for r in entry["rows"]], marker="o", ms=3.0,
                    lw=0.9, alpha=0.85, color=entry["colour"], label=entry["label"])
        ax_score.plot(t, np.cumsum([r["log_p"] for r in entry["rows"]]), lw=2.0,
                      color=entry["colour"], label=entry["label"])
    honest = math.sqrt(nm.CALIBRATED_MEDIAN_NEES)
    reject = math.sqrt(nm.GATE_CHI2_2DOF)
    ax_nis.axhline(honest, color="#B00020", ls="--", lw=1.5, zorder=1)
    ax_nis.axhline(reject, color="#B00020", ls=":", lw=1.4, zorder=1)
    ax_nis.set_yscale("log")
    high = math.sqrt(max(r["nis"] for e in forecasts for r in e["rows"]))
    ax_nis.set_ylim(top=high * 3.0)
    # The two reference levels are labelled on a right-hand axis: every spot inside this
    # panel has data in it.
    ax_ref = ax_nis.twinx()
    ax_ref.set_yscale("log")
    ax_ref.set_ylim(ax_nis.get_ylim())
    ax_ref.set_yticks([honest, reject])
    ax_ref.set_yticklabels([f"{honest:.2f}\nwhat an honest\nforecast typically does",
                            f"{reject:.2f}\nbeyond here the filter\ncalls it an outlier"],
                           fontsize=7.6, color="#B00020")
    ax_ref.minorticks_off()
    ax_ref.grid(False)
    ax_nis.set_xlabel("time since the start of the drive, seconds")
    ax_nis.set_ylabel("how many forecast-widths away\nthe reading landed")
    ax_nis.set_title(f"Camera {CAMERA_SHORT[camera]}: each reading's distance from where it\n"
                     "was predicted, measured in widths of that very forecast")
    ax_nis.legend(fontsize=8.2, loc="upper left", ncol=2)
    ax_score.set_xlabel("time since the start of the drive, seconds")
    ax_score.set_ylabel("running total of how well the next reading\nwas predicted "
                        "(higher = better)")
    ax_score.set_title("The gap opens steadily, not in one burst:\nit is a property of every "
                       "reading, not of a few")
    ax_score.legend(fontsize=8.2, loc="upper left")
    plt.show()


def why_it_fits_better(summaries):
    """Split the forecast score into the two things it rewards. x = arm, y = nats."""
    labels = [s["label"] for s in summaries]
    miss = np.array([s["miss_mean"] for s in summaries])
    conf = np.array([s["confidence_mean"] for s in summaries])
    pos = np.arange(len(summaries))
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.bar(pos, conf, width=0.58, color=C_SMOOTH, alpha=0.85,
           label="credit for a narrow forecast")
    ax.bar(pos, miss, width=0.58, bottom=0, color=C_FILTER, alpha=0.9,
           label="penalty for the reading landing off it")
    ax.plot(pos, conf + miss, marker="D", ms=8, lw=0, color=C_TRUTH, zorder=5,
            label="the score that results")
    for i, (c, m) in enumerate(zip(conf, miss)):
        ax.annotate(f"{c + m:+.2f}", (i, max(c, c + m)), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=9.5, weight="bold")
        ax.annotate(f"{c:+.2f}", (i, c / 2), ha="center", va="center", fontsize=8.6,
                    color="white")
        ax.annotate(f"{m:+.2f}", (i, m), textcoords="offset points", xytext=(0, -13),
                    ha="center", fontsize=8.6, color=C_FILTER)
    ax.axhline(0, color="#444444", lw=1.0)
    ax.set_ylim(min(miss.min() * 2.6, -0.9), conf.max() * 1.30)
    ax.set_xticks(pos)
    ax.set_xticklabels([l.replace(": ", ":\n") for l in labels], fontsize=8.8)
    ax.set_ylabel("how well the next reading was predicted\n"
                  "(log-probability per reading, higher = better)")
    ax.set_title("Why the fitted covariance scores higher: almost all of it is credit for\n"
                 "claiming a narrow forecast, not for the readings landing closer")
    ax.legend(fontsize=8.6, loc="upper left", ncol=3)
    plt.show()


def report_held_out(table, *, heading=""):
    """Forecast score and belief honesty side by side, on data R was not fitted on."""
    if heading:
        print(f"{heading}\n")
    wide_d = max(18, max(len(r["drive"]) for r in table) + 2)
    wide_a = max(24, max(len(r["label"]) for r in table) + 2)
    print(f"  {'drive':{wide_d}s}{'candidate R':{wide_a}s}{'predicts the':>14s}"
          f"{'how much too':>14s}{'RMSE':>9s}")
    print(f"  {'':{wide_d}s}{'':{wide_a}s}{'next reading':>14s}{'sure it is':>14s}{'':>9s}")
    last = None
    for row in table:
        name = row["drive"] if row["drive"] != last else ""
        if name and last is not None:
            print()
        last = row["drive"]
        print(f"  {name:{wide_d}s}{row['label']:{wide_a}s}{row['log_p_mean']:14.2f}"
              f"{nm.times_too_confident(row['median_nees']):13.1f}x"
              f"{row['rmse_cm']:7.1f} cm")
    print("\n  'predicts the next reading' is the average log-probability it gave the")
    print("  reading that actually arrived, before it arrived. Higher is better, and it")
    print("  uses NO ground truth -- a robot could compute this column while driving.")
    print("  'how much too sure' is how many times further the truth sits than the")
    print("  filter's own uncertainty allows. 1.0x is honest. This column needs truth.")


def camera_versus_robot(table):
    """The result: the same R that predicts the camera best predicts the robot worst."""
    drives = list(dict.fromkeys(r["drive"] for r in table))
    arms = list(dict.fromkeys(r["label"] for r in table))
    colours = {arms[0]: C_TRUTH, arms[-1]: C_FILTER}
    for a in arms:
        colours.setdefault(a, C_SMOOTH)
    width = 0.8 / len(arms)
    pos = np.arange(len(drives))

    fig, (ax_cam, ax_rob) = plt.subplots(1, 2, figsize=(12.6, 4.6))
    for j, arm in enumerate(arms):
        vals_cam, vals_rob = [], []
        for drive in drives:
            row = next(r for r in table if r["drive"] == drive and r["label"] == arm)
            vals_cam.append(row["log_p_mean"])
            vals_rob.append(nm.times_too_confident(row["median_nees"]))
        offset = (j - (len(arms) - 1) / 2) * width
        ax_cam.bar(pos + offset, vals_cam, width=width * 0.92, color=colours[arm],
                   alpha=0.88, label=arm)
        ax_rob.bar(pos + offset, vals_rob, width=width * 0.92, color=colours[arm], alpha=0.88)

    # The titles state what the bars actually did, counted here, so they cannot go stale
    # when the data changes underneath them.
    fitted = arms[-1]
    cam_wins = sum(1 for d in drives
                   if next(r for r in table if r["drive"] == d and r["label"] == fitted)["log_p_mean"]
                   >= max(next(r for r in table if r["drive"] == d and r["label"] == a)["log_p_mean"]
                          for a in arms[:-1]))
    rob_wins = sum(1 for d in drives
                   if next(r for r in table if r["drive"] == d and r["label"] == fitted)["median_nees"]
                   <= min(next(r for r in table if r["drive"] == d and r["label"] == a)["median_nees"]
                          for a in arms[:-1]))
    n = len(drives)
    cam_word = "on every drive here" if cam_wins == n else f"on {cam_wins} of {n} drives"
    rob_word = ("on every drive here" if rob_wins == 0
                else f"on {n - rob_wins} of {n} drives")
    for ax, ylab, title in (
        (ax_cam, "how well it predicted the CAMERA's\nnext reading (higher = better)",
         f"Predicting the camera's next reading:\nthe covariance fitted on one drive wins {cam_word}"),
        (ax_rob, "how many times further the truth is\nthan the filter says it should be",
         f"Knowing where the robot is:\nthe same covariance loses {rob_word}"),
    ):
        ax.set_xticks(pos)
        ax.set_xticklabels([d.replace(", ", ",\n") for d in drives], fontsize=8.4)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=10.2)
    ax_cam.set_ylim(0, max(r["log_p_mean"] for r in table) * 1.12)
    worst = max(nm.times_too_confident(r["median_nees"]) for r in table)
    ax_rob.set_ylim(0, worst * 1.16)
    ax_rob.axhline(1.0, color="#B00020", ls="--", lw=1.8, zorder=6)
    # Bars fill the axes from the bottom, so there is nowhere inside to put this label
    # without landing on one. It goes on the axis instead.
    ticks = [1.0] + [v for v in (2, 4, 6, 8, 10, 12) if v < worst * 1.16]
    ax_rob.set_yticks(ticks)
    ax_rob.set_yticklabels(["1x\nhonest"] + [f"{v:.0f}x" for v in ticks[1:]])
    ax_rob.get_yticklabels()[0].set_color("#B00020")
    fig.legend(*ax_cam.get_legend_handles_labels(), fontsize=8.6, ncol=3,
               loc="outside lower center")
    plt.show()


def report_bias_invisible(views):
    """Three errors at the same instants; only the third is measurable while driving."""
    width = max(24, max(len(label) for label, _ in views) + 2)
    print(f"  {'candidate R':{width}s}{'the camera is':>20s}{'the belief is':>20s}"
          f"{'the filter is':>20s}")
    print(f"  {'':{width}s}{'wrong by, cm':>20s}{'wrong by, cm':>20s}"
          f"{'surprised by, cm':>20s}")
    for label, v in views:
        cells = [f"({100 * v[k].mean(axis=0)[0]:+.2f}, {100 * v[k].mean(axis=0)[1]:+.2f})"
                 for k in ("camera_error_m", "belief_error_m", "innovation_m")]
        print(f"  {label:{width}s}" + "".join(f"{c:>20s}" for c in cells))
    print("\n  The first two columns need ground truth and are not available on a robot.")
    print("  The third is what both the fit and the forecast are actually scored on.")


def bias_is_invisible(views, camera):
    """The lean is in the camera and in the belief, and cancels out of the surprise."""
    fig, axes = plt.subplots(1, len(views), figsize=(5.2 * len(views), 4.6),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (label, v) in zip(axes, views):
        for key, colour, name, marker in (
            ("camera_error_m", CAMERA_COLOUR[camera], "camera reading minus truth", "o"),
            ("innovation_m", C_FILTER, "reading minus what was predicted", "^"),
        ):
            pts = 100 * v[key]
            ax.scatter(pts[:, 0], pts[:, 1], s=13, lw=0, alpha=0.35, color=colour)
            m = pts.mean(axis=0)
            ax.plot(*m, marker=marker, ms=13, color=colour, markeredgecolor="white",
                    markeredgewidth=1.4, zorder=6,
                    label=f"{name}\naverage ({m[0]:+.1f}, {m[1]:+.1f}) cm")
        ax.axhline(0, color=C_TRUTH, lw=1.2)
        ax.axvline(0, color=C_TRUTH, lw=1.2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("error in x, cm")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8.0, loc="upper left")
    axes[0].set_ylabel("error along the aisle, cm")
    fig.suptitle(f"Camera {CAMERA_SHORT[camera]} leans the same way every time — and that lean "
                 "is missing\nfrom the only quantity the filter can measure while driving",
                 fontsize=11.5)
    plt.show()


def report_recovery(rows, *, heading=""):
    """Does the estimator return the covariance that generated the data?"""
    if heading:
        print(f"{heading}\n")
    print(f"  {'the noise put in':>18s}{'plus a lean of':>16s}{'what it learned':>20s}"
          f"{'recovered':>16s}")
    for r in rows:
        print(f"  {100 * r['true_sigma_m']:15.2f} cm{100 * r['bias_m']:13.2f} cm"
              f"{100 * r['learned_sigma_m']:15.2f} cm"
              f"{100 * r['ratio']:11.0f}% +-{100 * r['ratio_spread']:2.0f}%")
    print(f"\n  averaged over {rows[0]['repeats']} independent draws of "
          f"{rows[0]['n']} readings each")


def recovery(rows):
    """x = the noise the data really had, y = the noise the loop reported."""
    true_cm = 100 * np.array([r["true_sigma_m"] for r in rows])
    got_cm = 100 * np.array([r["learned_sigma_m"] for r in rows])
    err_cm = 100 * np.array([r["spread_sigma_m"] for r in rows])
    lean = np.array([r["bias_m"] for r in rows]) > 0
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    lim = (0, max(true_cm.max(), got_cm.max()) * 1.14)
    ax.plot(lim, lim, color=C_TRUTH, ls="--", lw=1.4, label="a correct answer sits here")
    ax.errorbar(true_cm[~lean], got_cm[~lean], yerr=err_cm[~lean], fmt="o", ms=9,
                color=C_SMOOTH, capsize=4, lw=1.2, zorder=5,
                label="readings with no lean")
    if lean.any():
        ax.errorbar(true_cm[lean], got_cm[lean], yerr=err_cm[lean], fmt="D", ms=8,
                    color=C_FILTER, capsize=4, lw=1.2, zorder=5,
                    label="the same, with a 7 cm lean added")
    for x, y, has_lean in zip(true_cm, got_cm, lean):
        ax.annotate(f"{100 * y / x:.0f}%", (x, y), textcoords="offset points",
                    xytext=(-10, 9) if has_lean else (11, -5),
                    ha="right" if has_lean else "left", fontsize=8.6,
                    color=C_FILTER if has_lean else C_SMOOTH)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("the noise the readings really had, cm")
    ax.set_ylabel("the noise the loop reported, cm")
    ax.set_title("On readings its own model describes exactly, the loop is right for a quiet\n"
                 "camera and steadily too optimistic for a noisy one\n"
                 "(bars are the spread over repeated draws)")
    ax.legend(fontsize=8.4, loc="upper left")
    plt.show()


def report_gate_ablation(rows):
    """Does throwing out large residuals during learning cause the shrinkage?"""
    for r in rows:
        trail = "  ".join(f"{100 * s:.2f}" for s in r["sigma_per_pass_m"][:6])
        print(f"  {r['label']:22s}kept {r['kept']:3d} of {r['offered']:3d} readings   "
              f"->  {100 * r['learned_sigma_m']:.2f} cm")
        print(f"  {'':22s}first six passes, cm:  {trail}")
    spread = [100 * r["learned_sigma_m"] for r in rows]
    print(f"\n  Turning the gate off moves the answer by {abs(spread[0] - spread[1]):.2f} cm. "
          f"That is not where the shrinkage comes from.")


def identifiability(rows, reference_sigma, reference_label, camera):
    """What the loop calls camera noise depends on what was assumed about the wheels."""
    sigma_p = np.array([r["sigma_p"] for r in rows])
    learned = 100 * np.array([r["learned_sigma_m"] for r in rows])
    too_sure = nm.times_too_confident(np.array([r["median_nees"] for r in rows]))
    fig, (ax_r, ax_h) = plt.subplots(1, 2, figsize=(11.6, 4.2))
    ax_r.plot(sigma_p, learned, marker="o", ms=5, lw=1.9, color=C_SMOOTH)
    ax_r.axhline(100 * reference_sigma, color=C_TRUTH, ls=":", lw=1.7,
                 label=reference_label)
    ax_r.set_xscale("log")
    ax_r.set_xlabel("assumed wheel-odometry noise, m per $\\sqrt{\\mathrm{m}}$ driven")
    ax_r.set_ylabel(f"learned noise for camera {CAMERA_SHORT[camera]}, cm")
    ax_r.set_title("The same readings, the same code:\nthe answer moves with an assumption "
                   "about the wheels")
    ax_r.legend(fontsize=8.4)

    ax_h.plot(sigma_p, too_sure, marker="o", ms=5, lw=1.9, color=C_FILTER)
    ax_h.axhline(1.0, color="#B00020", ls="--", lw=1.6)
    ax_h.annotate("an honest belief sits here", (sigma_p[0], 1.0),
                  textcoords="offset points", xytext=(4, 6), fontsize=8.4, color="#B00020")
    ax_h.set_xscale("log"); ax_h.set_yscale("log")
    ax_h.set_xlabel("assumed wheel-odometry noise, m per $\\sqrt{\\mathrm{m}}$ driven")
    ax_h.set_ylabel("how many times further the truth is\nthan the filter says it should be")
    ax_h.set_title("And the honesty of the result is set almost entirely\nby that assumption, "
                   "not by the learning")
    plt.show()


def report_identifiability(rows):
    print(f"  {'assumed wheel noise':>22s}{'learned camera noise':>24s}"
          f"{'belief honesty':>17s}{'RMSE':>11s}")
    for r in rows:
        print(f"  {r['sigma_p']:22.3f}{100 * r['learned_sigma_m']:21.2f} cm"
              f"{r['median_nees']:17.2f}{r['rmse_cm']:9.1f} cm")


# ============================================== what the camera and the detector see
#
# The observation is a pixel before it is a position. These draw that: the frame the
# camera recorded, the box the detector put on it, the pixel the pipeline took as the
# contact point, and where the robot actually was -- in the image and on the floor.


def _frame_and_boxes(capture, camera, stamp, recorded_pixel=None):
    """The recorded frame nearest a stamp, and the detector's boxes on it.

    The detector is re-run here rather than trusted from the log, so the picture shows a
    box rather than a claim about one. When the runtime's own pixel is known, the box
    whose bottom-centre is nearest to it is put first, because re-running can surface
    boxes the pipeline filtered out and the figure must show the one that actually became
    this observation.
    """
    import cv2

    hit = capture.frame_at(camera, float(stamp), tol_s=0.5)
    if hit is None:
        return None, None, []
    image = cv2.imread(str(hit[1]))
    if image is None:
        return None, None, []
    boxes = nm.detect_on_frame(image, nm.detector_path())
    if boxes and recorded_pixel is not None:
        boxes.sort(key=lambda b: (((b[0] + b[2]) / 2 - recorded_pixel[0]) ** 2
                                  + (b[3] - recorded_pixel[1]) ** 2))
    return hit[1], image, boxes


def the_camera_view(seq, capture, models, camera, *, step=None):
    """One instant, three ways: the whole frame, the robot magnified, and the floor.

    The magnified panel is the point of the figure. At 1280x720 across an eleven-metre
    warehouse the robot is a few dozen pixels wide, and the entire subject of this
    notebook -- the gap between the bottom of the detector's box and the robot's actual
    contact with the floor -- is a handful of those pixels.
    """
    import cv2

    if step is None:
        seen = np.flatnonzero(seq.observed & np.isfinite(seq.truth[:, 0]))
        step = int(seen[len(seen) // 2])
    model = models[camera]
    path, image, boxes = _frame_and_boxes(capture, camera, seq.stamps[step], seq.pixel[step])
    if image is None:
        print("no recorded frame coincides with this observation; skipping the picture")
        return

    truth_xy = seq.truth[step]
    observed = seq.y[step]
    tu, tv, _ = model.world_to_pixel(float(truth_xy[0]), float(truth_xy[1]), 0.0)
    ou, ov = seq.pixel[step]

    fig = plt.figure(figsize=(13.2, 4.5))
    ax_wide = fig.add_subplot(1, 3, 1)
    ax_zoom = fig.add_subplot(1, 3, 2)
    ax_floor = fig.add_subplot(1, 3, 3)

    for ax, zoomed in ((ax_wide, False), (ax_zoom, True)):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.grid(False)
        if boxes:
            x1, y1, x2, y2, confidence = boxes[0]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   ec=CAMERA_COLOUR[camera], lw=2.0))
            if zoomed:
                ax.annotate(f"the detector's box, {confidence:.2f} confident",
                            (x1, y1 - 6), color=CAMERA_COLOUR[camera], fontsize=8.4,
                            fontweight="bold")
        ax.plot([ou], [ov], marker="o", ms=9, color=C_FILTER, zorder=5,
                label="bottom-centre of the box\n= what the pipeline calls the contact point")
        ax.plot([tu], [tv], marker="+", ms=15, mew=2.4, color=C_TRUTH, zorder=6,
                label="where the robot's wheels really touch the floor")

    # The gap this figure exists to show is a few pixels wide, so the window is sized
    # from that gap and not from a fixed guess: at 78 px the two marks landed on top of
    # each other and the picture said nothing.
    apart = float(np.hypot(ou - tu, ov - tv))
    half = int(max(16.0, 2.4 * apart + 12.0))
    cx, cy = (ou + tu) / 2, (ov + tv) / 2
    ax_zoom.set_xlim(cx - half, cx + half)
    ax_zoom.set_ylim(cy + half, cy - half)              # image rows run downwards
    ax_wide.add_patch(Rectangle((cx - half, cy - half), 2 * half, 2 * half, fill=False,
                                ec=C_TRUTH, lw=1.2, ls="--"))
    ax_wide.set_title(f"everything camera {CAMERA_SHORT[camera]} sees, "
                      f"t = {seq.stamps[step] - seq.stamps[0]:.1f} s", fontsize=10)
    ax_wide.set_xlabel("pixels across"); ax_wide.set_ylabel("pixels down")
    ax_zoom.set_title(f"magnified {model.img_width / (2 * half):.0f}x: the two marks "
                      f"are only {apart:.0f} px apart", fontsize=10)
    ax_zoom.set_xlabel("pixels across")
    ax_zoom.legend(loc="lower left", fontsize=7.2, facecolor="white", framealpha=0.85,
                   frameon=True)

    ax_floor.plot(*truth_xy, marker="+", ms=15, mew=2.4, color=C_TRUTH,
                  label="where the robot really is")
    ax_floor.plot(*observed, marker="o", ms=9, color=C_FILTER,
                  label="that pixel, back-projected to the floor")
    ax_floor.annotate("", xy=tuple(observed), xytext=tuple(truth_xy),
                      arrowprops=dict(arrowstyle="->", color=C_OBS, lw=1.5))
    error_cm = 100 * float(np.hypot(*(observed - truth_xy)))
    ax_floor.annotate(f"{error_cm:.1f} cm", (observed + truth_xy) / 2,
                      textcoords="offset points", xytext=(10, 8), color=C_OBS, fontsize=9.5,
                      fontweight="bold")
    pad = max(0.16, 1.5 * error_cm / 100)
    ax_floor.set_xlim(truth_xy[0] - pad, truth_xy[0] + pad)
    ax_floor.set_ylim(truth_xy[1] - pad, truth_xy[1] + pad)
    ax_floor.set_aspect("equal", adjustable="box")
    ax_floor.set_xlabel("x, metres"); ax_floor.set_ylabel("y, metres")
    ax_floor.set_title("the same moment, on the floor", fontsize=10)
    ax_floor.legend(loc="upper left", fontsize=8)

    fig.suptitle("A few pixels at the bottom of a box become centimetres on the floor",
                 fontsize=11.5)
    plt.show()


def the_camera_view_along_the_drive(seq, capture, models, camera, *, n=5):
    """The same magnified crop at n points down the drive, near to far.

    One frame shows that the error exists. This shows that it is not an accident of one
    frame: the box sits the same way relative to the robot at every range, which is what
    "a lean" means and what no zero-mean noise model can represent.
    """
    import cv2

    seen = np.flatnonzero(seq.observed & np.isfinite(seq.truth[:, 0]))
    model = models[camera]
    ranges = np.array([np.hypot(seq.truth[k][0] - model.cam_pos[0],
                                seq.truth[k][1] - model.cam_pos[1]) for k in seen])
    order = seen[np.argsort(ranges)]
    picks = [int(order[int(round(f * (len(order) - 1)))])
             for f in np.linspace(0.0, 1.0, n)]

    fig, axes = plt.subplots(1, n, figsize=(2.55 * n, 3.5))
    for ax, step in zip(np.atleast_1d(axes), picks):
        path, image, boxes = _frame_and_boxes(capture, camera, seq.stamps[step],
                                              seq.pixel[step])
        truth_xy = seq.truth[step]
        rng = float(np.hypot(truth_xy[0] - model.cam_pos[0], truth_xy[1] - model.cam_pos[1]))
        error_cm = 100 * float(np.hypot(*(seq.y[step] - truth_xy)))
        if image is None:
            ax.text(0.5, 0.5, "no frame\nrecorded here", ha="center", va="center",
                    fontsize=8, transform=ax.transAxes)
            ax.axis("off")
            continue
        tu, tv, _ = model.world_to_pixel(float(truth_xy[0]), float(truth_xy[1]), 0.0)
        ou, ov = seq.pixel[step]
        half = max(34, int(1.9 * abs(ov - tv)) + 26)
        x0 = int(np.clip(round((ou + tu) / 2 - half), 0, image.shape[1] - 2 * half))
        y0 = int(np.clip(round((ov + tv) / 2 - half), 0, image.shape[0] - 2 * half))
        ax.imshow(cv2.cvtColor(image[y0:y0 + 2 * half, x0:x0 + 2 * half], cv2.COLOR_BGR2RGB))
        if boxes:
            x1, y1, x2, y2, _ = boxes[0]
            ax.add_patch(Rectangle((x1 - x0, y1 - y0), x2 - x1, y2 - y1, fill=False,
                                   ec=CAMERA_COLOUR[camera], lw=1.6))
        ax.plot([ou - x0], [ov - y0], marker="o", ms=7, color=C_FILTER, zorder=5)
        ax.plot([tu - x0], [tv - y0], marker="+", ms=12, mew=2.0, color=C_TRUTH, zorder=6)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(f"{rng:.1f} m away\n{np.hypot(ou - tu, ov - tv):.0f} px  ->  "
                     f"{error_cm:.1f} cm", fontsize=9)
    fig.suptitle(f"Camera {CAMERA_SHORT[camera]}, the same robot at five ranges: the box "
                 f"bottom (circle) sits the same side of\nthe true contact point (cross) "
                 f"every time — that repeatability is what makes it a lean, not noise",
                 fontsize=10.5)
    plt.show()


def the_drive_in_the_image(seq, capture, models, camera):
    """Where the whole drive lives in the image, and what a pixel is worth along it.

    Left: one frame with every detection of the drive drawn on it, and the true path
    projected beside them. Right: the same drive as centimetres of floor per pixel of
    detector error, which is the reason a single number for R cannot be right.
    """
    import cv2

    model = models[camera]
    seen = np.flatnonzero(seq.observed & np.isfinite(seq.truth[:, 0]))
    mid = int(seen[len(seen) // 2])
    _, image, _ = _frame_and_boxes(capture, camera, seq.stamps[mid])

    fig, (ax_img, ax_worth) = plt.subplots(1, 2, figsize=(12.8, 4.4),
                                           gridspec_kw={"width_ratios": [1.5, 1.0]})
    if image is not None:
        ax_img.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax_img.grid(False)

    truth_px = []
    for k in np.flatnonzero(np.isfinite(seq.truth[:, 0])):
        u, v, in_frame = model.world_to_pixel(float(seq.truth[k][0]), float(seq.truth[k][1]), 0.0)
        if in_frame:
            truth_px.append((u, v))
    truth_px = np.asarray(truth_px)
    if len(truth_px):
        ax_img.plot(truth_px[:, 0], truth_px[:, 1], color=C_TRUTH, lw=2.0,
                    label="where the robot really went")
    pix = np.asarray([seq.pixel[k] for k in seen])
    ax_img.scatter(pix[:, 0], pix[:, 1], s=11, lw=0, color=C_FILTER, alpha=0.8,
                   label="every contact pixel the detector reported")
    ax_img.set_xlim(0, model.img_width); ax_img.set_ylim(model.img_height, 0)
    ax_img.set_xlabel("pixels across"); ax_img.set_ylabel("pixels down")
    ax_img.set_title(f"The whole drive, inside camera {CAMERA_SHORT[camera]}'s frame",
                     fontsize=10.5)
    ax_img.legend(fontsize=8.2, loc="lower left", facecolor="white", framealpha=0.85,
                  frameon=True)

    worth, rng = [], []
    for k in seen:
        J = nm.projection_jacobian(model, *seq.pixel[k])
        if J is None:
            continue
        worth.append(100 * float(np.sqrt(np.trace(J @ J.T) / 2)))
        rng.append(float(np.hypot(seq.truth[k][0] - model.cam_pos[0],
                                  seq.truth[k][1] - model.cam_pos[1])))
    ax_worth.scatter(rng, worth, s=13, lw=0, color=CAMERA_COLOUR[camera], alpha=0.75)
    ax_worth.set_xlabel("how far the robot is from the camera, metres")
    ax_worth.set_ylabel("cm of floor per pixel of detector error")
    ax_worth.set_title(f"One pixel is worth {min(worth):.2f} cm near the camera and "
                       f"{max(worth):.2f} cm far away\n"
                       f"— a factor of {max(worth) / min(worth):.1f}, within this one drive",
                       fontsize=10.5)
    plt.show()
    return {"worth_cm_per_px": np.asarray(worth), "range_m": np.asarray(rng)}


def report_the_capture(seq, capture, camera, messages=None):
    """What this drive is, in numbers, before anything is estimated from it."""
    seen = int(seq.observed.sum())
    print(f"drive {capture.name}: {seq.n_steps} steps on a 10 Hz grid "
          f"({seq.n_steps / 10:.0f} s of driving)")
    print(f"  camera {CAMERA_SHORT[camera]} spoke on {seen} of them "
          f"({100 * seen / seq.n_steps:.0f}%)")
    if messages is not None:
        rows = messages.get(camera, [])
        window = (seq.stamps[0], seq.stamps[-1])
        inside = [d for t, d in rows if window[0] <= t <= window[1]]
        if inside:
            print(f"  the detector was handed {len(inside)} frames and found the robot in "
                  f"{sum(inside)} of them ({100 * sum(inside) / len(inside):.0f}%)")
    ok = np.isfinite(seq.truth[:, 0]) & seq.observed
    if ok.any():
        residual = seq.y[ok] - seq.truth[ok]
        mean = residual.mean(axis=0)
        spread = np.sqrt(((residual - mean) ** 2).sum(axis=1).mean() / 2)
        total = np.sqrt((residual ** 2).sum(axis=1).mean() / 2)
        print(f"\n  EVALUATION ONLY, to say what the data is:")
        print(f"    the camera's average error   ({100 * mean[0]:+.2f}, {100 * mean[1]:+.2f}) cm")
        print(f"    scatter about that average    {100 * spread:.2f} cm")
        print(f"    total, about zero             {100 * total:.2f} cm  "
              f"({total / spread:.1f}x the scatter)")


# ==================================================== R is a matrix, not a number

def what_R_is(seq, camera, oracle, R_learned=None):
    """Show that R is a 2x2 matrix: a size, a shape AND an orientation.

    Two panels because there are two jobs and one pair of axes cannot do both. This
    camera's errors are about twenty times longer along the aisle than across it, so an
    honest equal-aspect view of the big covariance squashes the small one to a line. The
    left panel therefore zooms on the cloud; the right draws the full covariance against
    the circle that the single number `sigma = sqrt(trace(R)/2)` stands for.
    """
    errors = 100 * np.asarray([seq.y[k] - seq.truth[k] for k in range(seq.n_steps)
                               if seq.camera[k] == camera
                               and np.isfinite(seq.truth[k, 0])])
    mean = errors.mean(axis=0)
    R_total = 1e4 * oracle["R_total"][camera]
    R_spread = 1e4 * oracle["R_spread"][camera]

    fig, (ax_cloud, ax_shape) = plt.subplots(1, 2, figsize=(12.6, 5.0))

    # ---- left: what the errors are, close up
    ax_cloud.scatter(errors[:, 0], errors[:, 1], s=16, lw=0, alpha=0.45, color=C_OBS,
                     label=f"the {len(errors)} errors this camera made")
    ax_cloud.axhline(0, color=C_TRUTH, lw=1.1)
    ax_cloud.axvline(0, color=C_TRUTH, lw=1.1)
    ax_cloud.plot(0, 0, marker="+", ms=17, mew=2.6, color=C_TRUTH, zorder=7)
    ax_cloud.annotate("no error at all", (0, 0), textcoords="offset points",
                      xytext=(10, 6), fontsize=9.5, color=C_TRUTH, fontweight="bold")
    ax_cloud.add_patch(ellipse_from(mean, R_spread, n_sigma=2.0, fill=False, ls="--",
                                    ec=C_SMOOTH, lw=2.2,
                                    label="their scatter, about their own centre"))
    if R_learned is not None:
        ax_cloud.add_patch(ellipse_from((0, 0), 1e4 * R_learned[camera], n_sigma=2.0,
                                        fill=False, ec=C_FILTER, lw=2.2,
                                        label="what the loop learned (no ground truth)"))
    ax_cloud.annotate("", xy=tuple(mean), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=2.4))
    ax_cloud.annotate(f"the lean: {np.hypot(*mean):.1f} cm,\nthe same way every reading",
                      tuple(mean * 0.5), textcoords="offset points", xytext=(12, 0),
                      fontsize=9.5, color=C_ACCENT, fontweight="bold", va="center")
    span = 1.35 * max(np.abs(errors).max(axis=0).max(), np.hypot(*mean))
    ax_cloud.set_xlim(-span, span); ax_cloud.set_ylim(-span, span)
    ax_cloud.set_aspect("equal", adjustable="box")
    ax_cloud.set_xlabel("error across the aisle, cm")
    ax_cloud.set_ylabel("error along the aisle, cm")
    ax_cloud.set_title("The errors are tight — and they are not centred on zero.\n"
                       "A covariance can describe the tightness. Nothing here holds the arrow.",
                       fontsize=10.2)
    ax_cloud.legend(fontsize=8.0, loc="upper left")

    # ---- right: the same covariance, honestly and as one number
    one_number = float(np.sqrt(np.trace(R_total) / 2))
    ax_shape.add_patch(ellipse_from((0, 0), R_total, n_sigma=2.0, facecolor=C_TRUTH,
                                    alpha=0.22, ec=C_TRUTH, lw=2.0,
                                    label="R as it really is"))
    ax_shape.add_patch(ellipse_from((0, 0), np.eye(2) * one_number ** 2, n_sigma=2.0,
                                    fill=False, ec="#B00020", lw=2.0, ls="--",
                                    label=f"R as the one number quoted elsewhere\n"
                                          f"(sigma = {one_number:.2f} cm, a circle)"))
    ax_shape.axhline(0, color="#888888", lw=0.9)
    ax_shape.axvline(0, color="#888888", lw=0.9)
    values, vectors = np.linalg.eigh(R_total)
    lim = 2.5 * max(one_number, float(np.sqrt(values.max())))
    ax_shape.set_xlim(-lim, lim); ax_shape.set_ylim(-lim, lim)
    ax_shape.set_aspect("equal", adjustable="box")
    ax_shape.set_xlabel("error across the aisle, cm")
    ax_shape.set_ylabel("error along the aisle, cm")
    tilt = math.degrees(math.atan2(vectors[1, -1], vectors[0, -1]))
    ax_shape.set_title(f"The R a zero-mean model needs: {np.sqrt(values.min()):.2f} cm one way,\n"
                       f"{np.sqrt(values.max()):.2f} cm the other, tilted {tilt:.0f}° — "
                       f"one number says none of that", fontsize=10.2)
    ax_shape.legend(fontsize=8.4, loc="upper left")
    plt.show()


def report_what_R_is(oracle, R_learned, camera):
    """R_learned may be None before the loop has run."""
    """The same three covariances as numbers, so the matrix entries are on the record."""
    def show(name, M, note=""):
        M = 1e4 * M
        sx, sy = math.sqrt(M[0, 0]), math.sqrt(M[1, 1])
        rho = M[0, 1] / (sx * sy) if sx * sy else 0.0
        print(f"  {name:34s}[{M[0, 0]:7.2f} {M[0, 1]:7.2f}]   across {sx:5.2f} cm")
        print(f"  {'':34s}[{M[1, 0]:7.2f} {M[1, 1]:7.2f}]   along  {sy:5.2f} cm, "
              f"correlation {rho:+.2f}")
        if note:
            print(f"  {'':34s}{note}")
        print()

    mean = 100 * oracle["mean_m"][camera]
    print(f"  R is a 2x2 matrix, in cm^2. All three of these are R.\n")
    show("what the errors actually are", oracle["R_total"][camera],
         "centred on zero, so it has to be big enough to reach a cloud that is not")
    show("their scatter alone", oracle["R_spread"][camera],
         "centred on the cloud, so it only describes the jitter")
    if R_learned is not None:
        show("learned from this drive", R_learned[camera],
             "found with no ground truth at all")
    print(f"  and the part no covariance can hold: a mean error of "
          f"({mean[0]:+.2f}, {mean[1]:+.2f}) cm,")
    print(f"  the same direction on every one of "
          f"{oracle['n'][camera]} readings.")


# ============================================ what changes as the loop iterates

def passes_of_the_loop(seq, camera, history, oracle, step, passes=(0, 1, 10)):
    """R, the forecast it implies, and the objective being climbed, at chosen passes.

    Three rows and one question each: what the loop believes R is, what that belief
    predicts for one particular reading, and whether the quantity it is maximising has
    actually gone up. The bottom row is the ELBO -- the thing coordinate ascent provably
    climbs -- with the plug-in evidence beside it, which is a different quantity and is
    NOT what the loop optimises.
    """
    # `p` counts R steps taken, so 0 is the prior. Each record pairs `R_in` -- the
    # covariance the bound was measured at -- with `elbo`; `R_bar` is one step later.
    picks = [p for p in passes if p < len(history)]
    errors = 100 * np.asarray([seq.y[k] - seq.truth[k] for k in range(seq.n_steps)
                               if seq.camera[k] == camera
                               and np.isfinite(seq.truth[k, 0])])
    R_true = 1e4 * oracle["R_total"][camera]

    fig = plt.figure(figsize=(4.3 * len(picks), 10.4))
    grid = fig.add_gridspec(3, len(picks), height_ratios=[1.25, 1.15, 1.0], hspace=0.34)
    span = 1.3 * max(np.abs(errors).max(), np.hypot(*errors.mean(axis=0)))
    # One scale for every innovation panel, or the shrinking ellipse is invisible.
    inno_span = 1.15 * max(
        math.sqrt(nm.GATE_CHI2_2DOF * np.max(np.linalg.eigvalsh(
            1e4 * np.median(np.asarray([r["S"] for r in nm.forecast(
                seq, history[p]["R_in"])["rows"]]), axis=0))))
        for p in picks)

    for column, p in enumerate(picks):
        record = history[p]
        R = record["R_in"][camera]

        # ---- row 1: what the loop thinks R is, against where the errors really fell
        ax = fig.add_subplot(grid[0, column])
        ax.scatter(errors[:, 0], errors[:, 1], s=11, lw=0, alpha=0.35, color=C_OBS,
                   label="where the errors fell" if column == 0 else None)
        ax.add_patch(ellipse_from((0, 0), 1e4 * R, n_sigma=2.0, facecolor=C_FILTER,
                                  alpha=0.25, ec=C_FILTER, lw=2.2,
                                  label="R, at 2 sd" if column == 0 else None))
        ax.plot(0, 0, marker="+", ms=13, mew=2.2, color=C_TRUTH, zorder=6)
        ax.set_xlim(-span, span); ax.set_ylim(-span, span)
        ax.set_aspect("equal", adjustable="box")
        along = math.sqrt(1e4 * R[1, 1])
        when = "before any learning" if p == 0 else f"after {p} pass{'' if p == 1 else 'es'}"
        ax.set_title(f"{when}\n{math.sqrt(1e4 * R[0, 0]):.2f} cm across the aisle, "
                     f"{along:.2f} cm along it", fontsize=10.4)
        if column == 0:
            ax.legend(fontsize=7.6, loc="upper left")
        ax.set_xlabel("error across the aisle, cm", fontsize=9)
        if column == 0:
            ax.set_ylabel("what the loop believes R is\n\nerror along the aisle, cm",
                          fontsize=9)

        # ---- row 2: the forecast, against EVERY reading rather than a chosen one
        #
        # This row used to draw a single observation, and it was the worst of the 276 --
        # picked elsewhere in the notebook precisely because it strains the fitted
        # covariance hardest. Shown without that label it read as a typical reading, and
        # invited exactly the wrong conclusion: that the prediction error grows with every
        # pass. It does not. Over all the readings the miss SHRINKS; what grows is the
        # miss measured against a forecast that is shrinking faster.
        ax = fig.add_subplot(grid[1, column])
        rows = nm.forecast(seq, record["R_in"])["rows"]
        innovations = 100 * np.asarray([r["innovation"] for r in rows])
        typical_S = 1e4 * np.median(np.asarray([r["S"] for r in rows]), axis=0)
        ax.scatter(innovations[:, 0], innovations[:, 1], s=13, lw=0, alpha=0.40,
                   color=CAMERA_COLOUR[camera],
                   label="where each reading landed,\nrelative to its prediction"
                         if column == 0 else None)
        ax.add_patch(ellipse_from((0, 0), typical_S, n_sigma=math.sqrt(nm.GATE_CHI2_2DOF),
                                  facecolor=C_FILTER, alpha=0.17, ec=C_FILTER, lw=1.9,
                                  label="the 95% region it forecasts" if column == 0 else None))
        ax.plot(0, 0, marker="+", ms=13, mew=2.2, color=C_FILTER, zorder=6)
        ax.set_xlim(-inno_span, inno_span); ax.set_ylim(-inno_span, inno_span)
        ax.set_aspect("equal", adjustable="box")
        missed = 100 * float(np.mean([r["distance_m"] for r in rows]))
        width = 100 * float(np.mean([r["sigma_m"] for r in rows]))
        ax.set_title(f"readings land {missed:.2f} cm from the prediction;\n"
                     f"it forecasts $\\pm${width:.2f} cm", fontsize=10.4)
        ax.set_xlabel("miss across the aisle, cm", fontsize=9)
        if column == 0:
            ax.set_ylabel("the prediction it makes\n\nmiss along the aisle, cm", fontsize=9)
            ax.legend(fontsize=7.4, loc="lower left")

    # ---- row 3: the objective, across every pass
    ax = fig.add_subplot(grid[2, :])
    bound = np.array([h["elbo"] for h in history])
    steps = np.arange(len(history))
    settled = int(np.argmax(bound >= bound.max() - 1e-3))
    # Everything happens in the first handful of passes; a linear axis over those shows
    # the shape, and the title says what the rest of the run did.
    shown = min(len(history), max(2 * settled + 4, max(picks) + 4))
    ax.plot(steps[:shown], bound[:shown], marker="o", ms=4, lw=2.2, color=C_SMOOTH,
            label="the ELBO — what the loop provably climbs")
    ax.plot(steps[:shown], [h["log_evidence_all"] for h in history[:shown]], marker="o",
            ms=3, lw=1.8, ls="--", color=C_OBS,
            label="the plug-in fit — a different quantity, not the objective")
    from matplotlib.transforms import blended_transform_factory
    at_bottom = blended_transform_factory(ax.transData, ax.transAxes)
    for p in picks:
        ax.axvline(p, color=C_FILTER, lw=1.0, ls=":")
        ax.annotate("the prior" if p == 0 else f"{p} pass{'' if p == 1 else 'es'}",
                    (p, 0.03), xycoords=at_bottom, textcoords="offset points",
                    xytext=(5, 0), fontsize=8.6, color=C_FILTER)
    ax.set_xlim(-0.4, shown - 0.6)
    ax.set_xlabel("passes of the loop completed")
    ax.set_ylabel("the score being maximised")
    ax.set_title(f"The ELBO rises {bound[0]:.0f} → {bound.max():.0f} and stops moving after "
                 f"{settled} passes.\nIt is still {bound[-1]:.3f} at pass "
                 f"{len(history) - 1}, so ten passes and a hundred are the same answer",
                 fontsize=10.6)
    ax.legend(fontsize=8.8, loc="center right")
    plt.show()


def report_passes_of_the_loop(history, camera, oracle, passes=(0, 1, 10), seq=None):
    """The same three passes as numbers, including what the forecast actually does.

    Pass `seq` to get the aggregate forecast columns. They exist because the figure's
    middle row invites one particular wrong reading -- that the prediction error grows --
    and the only way to settle it is over all the readings rather than one.
    """
    from notebook_model import forecast, forecast_summary
    print(f"  {'pass':>6s}{'R across, cm':>14s}{'R along, cm':>13s}{'the ELBO':>11s}"
          f"{'plug-in fit':>12s}")
    for p in passes:
        if p >= len(history):
            continue
        h = history[p]
        R = 1e4 * h["R_in"][camera]
        print(f"  {p:6d}{math.sqrt(R[0, 0]):14.3f}{math.sqrt(R[1, 1]):13.3f}"
              f"{h['elbo']:11.3f}{h['log_evidence_all']:12.2f}"
              + ("   <- the prior, before any learning" if p == 0 else ""))
    true_R = 1e4 * oracle["R_total"][camera]
    print(f"  {'truth':>6s}{math.sqrt(true_R[0, 0]):14.3f}{math.sqrt(true_R[1, 1]):13.3f}"
          f"{'':11s}{'':12s}  <- what the errors actually are")
    final = 1e4 * history[-1]["R_in"][camera]
    print(f"\n  across the aisle the loop is right: {math.sqrt(final[0, 0]):.2f} cm "
          f"against a true {math.sqrt(true_R[0, 0]):.2f} cm.")
    print(f"  along the aisle it is not: {math.sqrt(final[1, 1]):.2f} cm against a true "
          f"{math.sqrt(true_R[1, 1]):.2f} cm, "
          f"{math.sqrt(true_R[1, 1]) / math.sqrt(final[1, 1]):.0f}x too small --")
    print("  and along the aisle is exactly the direction the lean points.")
    if seq is not None:
        print(f"\n  and what that does to the forecast, over ALL of this camera's readings:\n")
        print(f"  {'pass':>6s}{'it forecasts':>16s}{'readings land':>18s}"
              f"{'as a fraction':>17s}")
        print(f"  {'':6s}{'+- this, cm':>16s}{'this far off, cm':>18s}"
              f"{'of the forecast':>17s}")
        for p in passes:
            if p >= len(history):
                continue
            f = forecast_summary(forecast(seq, history[p]["R_in"]), "")
            print(f"  {p:6d}{f['sigma_cm']:16.2f}{f['distance_cm']:18.2f}"
                  f"{math.sqrt(f['median_nis']):17.2f}")
        print("\n  THE MISS GOES DOWN, not up: the readings land CLOSER to the prediction")
        print("  with every pass. What goes up is the third column -- the miss measured")
        print("  against a forecast that is narrowing faster than the miss is.")
    bound = [h["elbo"] for h in history]
    rising = all(b >= a - 1e-8 for a, b in zip(bound, bound[1:]))
    print(f"\n  the ELBO increased on every one of {len(bound) - 1} steps: {rising}")
    print("  (it has to. A coordinate ascent that ever went down would be a bug, and this")
    print("   is the sharpest check on the implementation the notebook has.)")


# ==================================== R as a learned field, with its own uncertainty

def the_R_field(field, rows, models, *, magnify=26.0, every=9):
    """The learned R over the whole floor: how big, what shape, and how well known.

    Three panels because a field has three things worth seeing and one colour map can
    only carry one. Left is the size with the covariance drawn as ellipses, so shape and
    orientation are visible. Middle is what the drives actually supported -- the width of
    the 90% credible band on sigma_px -- which is the panel a two-endpoint blend has no
    way to produce at all. Right is the lean, which is not a covariance and is the part
    no R can hold.
    """
    camera = field["camera"]
    model = models[camera]
    fig, (ax_size, ax_known, ax_lean) = plt.subplots(1, 3, figsize=(16.4, 5.0))
    X, Y = np.meshgrid(field["x"], field["y"])

    size_cm = 100 * field["size_m"]
    mesh = ax_size.pcolormesh(field["x"], field["y"], size_cm, shading="auto",
                              cmap="viridis_r")
    fig.colorbar(mesh, ax=ax_size, pad=0.02).set_label(
        "R, one standard deviation, cm", fontsize=8.6)
    for j in range(0, len(field["y"]), every):
        for i in range(0, len(field["x"]), every):
            if not np.isfinite(field["size_m"][j, i]):
                continue
            got = nm.R_at(rows, models, camera, field["x"][i], field["y"][j])
            if got is None:
                continue
            ax_size.add_patch(ellipse_from((field["x"][i], field["y"][j]),
                                           got["R"] * magnify ** 2, n_sigma=1.0,
                                           fill=False, ec="white", lw=1.1, alpha=0.9))
    ax_size.set_title(f"The learned R, wherever camera {CAMERA_SHORT[camera]} can see\n"
                      f"ellipses {magnify:.0f}x oversize, so shape and tilt show",
                      fontsize=9.6)

    band = field["sigma_px_hi"] - field["sigma_px_lo"]
    mesh = ax_known.pcolormesh(field["x"], field["y"], band, shading="auto", cmap="magma_r")
    fig.colorbar(mesh, ax=ax_known, pad=0.02).set_label(
        "width of the 90% band on sigma_px\n(narrow = the drives pinned it down)",
        fontsize=8.6)
    for tag in dict.fromkeys(r["tag"] for r in rows):
        pts = np.asarray([r["at"] for r in rows if r["tag"] == tag])
        ax_known.plot(pts[:, 0], pts[:, 1], lw=2.2, alpha=0.85,
                      label=tag.replace("aws_", "").replace("_", " "))
    ax_known.legend(fontsize=6.8, loc="lower left", framealpha=0.85, frameon=True)
    ax_known.set_title("How much the field actually knows\n"
                       "a blend between two constants cannot say this at all",
                       fontsize=9.6)

    step = max(1, every // 2)
    lean = 100 * field["lean_m"]
    ax_lean.quiver(X[::step, ::step], Y[::step, ::step],
                   lean[::step, ::step, 0], lean[::step, ::step, 1],
                   np.hypot(lean[::step, ::step, 0], lean[::step, ::step, 1]),
                   cmap="cividis", scale=170, width=0.005)
    ax_lean.set_title("And the lean, which is not noise\nno covariance can hold this",
                      fontsize=9.6)

    for ax in (ax_size, ax_known, ax_lean):
        pos = np.asarray(model.cam_pos[:2], dtype=float)
        ax.plot(*pos, marker="s", ms=10, color=CAMERA_COLOUR[camera],
                markeredgecolor="white", markeredgewidth=1.3, zorder=7, clip_on=False)
        ax.set_xlabel("x, metres")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
    ax_size.set_ylabel("y, metres")
    plt.show()


def field_against_blend(field, rows, models, taus=(0.9, 0.5, 0.1),
                        r_visible_uv=2.5, r_miss_uv=40.0, gamma=1.0):
    """What the deployed blend produces at the same points, drawn beside the field.

    `reliability.planning_covariance.plan_covariance` interpolates a scalar trust between
    two endpoint covariances. Both endpoints are isotropic pixel constants, so whatever
    trust it is handed, the result is a circle: the blend can move R's size and nothing
    else. The learned field is drawn on the same axes at the same scale.
    """
    camera = field["camera"]
    model = models[camera]
    x, y = 1.075, 0.0                       # one point, so the true relative sizes show
    fig, (ax_shape, ax_size) = plt.subplots(1, 2, figsize=(12.8, 5.0))
    got = nm.R_at(rows, models, camera, x, y)
    u, v, _ = model.world_to_pixel(x, y, 0.0)
    J = nm.projection_jacobian(model, u, v)
    per_px = float(np.sqrt(np.trace(J @ J.T) / 2))

    ax_shape.add_patch(ellipse_from((0, 0), 1e4 * got["R"], n_sigma=2.0,
                                    facecolor=C_FILTER, alpha=0.30, ec=C_FILTER, lw=2.4,
                                    label=f"the learned field: "
                                          f"{100 * np.sqrt(np.linalg.eigvalsh(got['R'])[0]):.1f}"
                                          f" x {100 * np.sqrt(np.linalg.eigvalsh(got['R'])[1]):.1f} cm"))
    widest = 0.0
    for tau, style in zip(taus, (":", "--", "-")):
        var_px = (r_visible_uv ** 2) + (1 - tau) ** gamma * ((r_miss_uv ** 2) - (r_visible_uv ** 2))
        radius = 100 * math.sqrt(var_px) * per_px
        widest = max(widest, radius)
        ax_shape.add_patch(Ellipse((0, 0), 4 * radius, 4 * radius, fill=False,
                                   ec=C_TRUTH, lw=1.3, ls=style, alpha=0.8,
                                   label=f"the blend at trust {tau:.1f}: "
                                         f"{radius:.1f} cm, a circle"))
    ax_shape.plot(0, 0, marker="+", ms=12, mew=2.0, color=C_TRUTH)
    lim = 2.3 * widest
    ax_shape.set_xlim(-lim, lim); ax_shape.set_ylim(-lim, lim)
    ax_shape.set_aspect("equal", adjustable="box")
    ax_shape.set_xlabel("cm across the aisle"); ax_shape.set_ylabel("cm along the aisle")
    ax_shape.set_title(f"One floor point ({x:.1f}, {y:.1f}), everything at 2 sd and at the\n"
                       f"same scale. One pixel is worth {100 * per_px:.2f} cm here",
                       fontsize=10.2)
    ax_shape.legend(fontsize=7.8, loc="upper left")

    aspect = field["aspect"][np.isfinite(field["aspect"])]
    ax_size.hist(aspect, bins=40, color=C_FILTER, alpha=0.85)
    ax_size.axvline(1.0, color=C_TRUTH, lw=2.0, ls="--")
    ax_size.annotate("every covariance the blend can produce\nsits on this line",
                     (1.0, ax_size.get_ylim()[1] * 0.72), textcoords="offset points",
                     xytext=(10, 0), fontsize=8.8, color=C_TRUTH)
    ax_size.set_xlabel("how elongated R is (major axis / minor axis)")
    ax_size.set_ylabel("floor points")
    ax_size.set_title(f"The learned field is elongated by {np.median(aspect):.1f}x "
                      f"typically\nand up to {aspect.max():.1f}x; a scalar blend is "
                      f"always 1.0", fontsize=10.4)
    plt.show()


def report_the_R_field(field, rows, models, probes=None):
    """The field at a few named places, including one no drive ever visited."""
    camera = field["camera"]
    probes = probes or [("up the aisle, close in", 1.075, -4.0),
                        ("up the aisle, half way", 1.075, 0.0),
                        ("up the aisle, far end", 1.075, 4.0),
                        ("out on the west apron", -4.0, -4.3),
                        ("a corner no drive reached", -5.0, 3.5)]
    print(f"  learned from {field['learned']['n_readings']} readings across "
          f"{len(dict.fromkeys(r['tag'] for r in rows))} drives, "
          f"kernel {field['learned']['length_scale_m']:.1f} m\n")
    print(f"  {'place':28s}{'lean, cm':>16s}{'sigma_px':>10s}{'90% band':>16s}"
          f"{'data':>7s}{'R, cm':>16s}")
    for label, x, y in probes:
        got = nm.R_at(rows, models, camera, x, y)
        if got is None:
            print(f"  {label:28s}{'not visible from here':>65s}")
            continue
        sd = np.sqrt(np.linalg.eigvalsh(1e4 * got["R"]))
        lean = 100 * got["lean_m"]
        print(f"  {label:28s}({lean[0]:+5.1f},{lean[1]:+5.1f}) {got['sigma_px']:9.2f}"
              f"  [{got['sigma_px_lo']:.2f}, {got['sigma_px_hi']:.2f}]"
              f"{got['n_effective']:7.0f}{sd[0]:9.2f} x{sd[1]:5.2f}")
    print("\n  The last row is the point of the whole exercise: the field does not")
    print("  invent a number where nothing drove, it widens its band and says so.")


def blend_dominance(*, r_visible_uv=2.5, r_miss_uv=(40.0, 120.0), field=None, gamma=1.0):
    """How much of the blend's variance comes from its unregistered endpoint.

    With endpoints that far apart the interpolation is not really an interpolation: for
    every trust a planner would call 'good', the answer is the miss endpoint. Drawn for
    both candidate constants, because the offline and runtime code disagree about which
    one it is and neither is registered.
    """
    tau = 1.0 - np.logspace(-4.5, 0, 400)
    fig, (ax_share, ax_size) = plt.subplots(1, 2, figsize=(12.4, 4.4))
    for rm, style in zip(np.atleast_1d(r_miss_uv), ("-", "--")):
        miss = (1.0 - tau) ** gamma * (rm ** 2 - r_visible_uv ** 2)
        ax_share.plot(tau, 100 * miss / (r_visible_uv ** 2 + miss), lw=2.2, ls=style,
                      color=C_TRUTH if style == "-" else C_FILTER,
                      label=f"r_miss = {rm:.0f} px  (variance ratio "
                            f"{(rm / r_visible_uv) ** 2:.0f}x)")
        ax_size.plot(tau, np.sqrt(r_visible_uv ** 2 + miss), lw=2.2, ls=style,
                     color=C_TRUTH if style == "-" else C_FILTER,
                     label=f"r_miss = {rm:.0f} px")
    ax_share.axhline(50, color=C_OBS, lw=1.2, ls=":")
    ax_share.set_xscale("logit")
    ax_share.set_xlabel("trust the GP reports")
    ax_share.set_ylabel("share of the variance coming from\nthe MISS endpoint, %")
    ax_share.set_title("For any trust a planner would call good, the blend\n"
                       "returns its unregistered endpoint", fontsize=10.4)
    ax_share.legend(fontsize=8.4, loc="lower left")

    ax_size.axhline(r_visible_uv, color=C_OBS, lw=1.4, ls=":",
                    label=f"its 'perfectly visible' endpoint, {r_visible_uv} px")
    if field is not None:
        learned = field["sigma_px"][np.isfinite(field["sigma_px"])]
        ax_size.axhspan(np.percentile(learned, 5), np.percentile(learned, 95),
                        color=C_FILTER, alpha=0.22,
                        label="what the field actually measured")
    ax_size.set_xscale("logit"); ax_size.set_yscale("log")
    ax_size.set_xlabel("trust the GP reports")
    ax_size.set_ylabel("the pixel noise the blend implies")
    ax_size.set_title("And even at trust 1 it sits above the measured band:\n"
                      "no trust value reaches reality", fontsize=10.4)
    ax_size.legend(fontsize=8.4, loc="upper right")
    plt.show()


# ==================================== learning R once the lean is out of the way

def the_lean_leaves_R(seq, result, camera):
    """The same error cloud before and after the observation function is corrected.

    One picture for the whole argument. Left: the errors are a cloud sitting well away
    from zero, and the covariance a zero-mean model needs has to stretch from the origin
    out to it -- while the learned one describes only the cloud's own width. Right: the
    correction moves the cloud onto zero, so the covariance that is needed and the
    covariance that can be seen become the same object, and the learned ellipse is finally
    a description of the same thing.

    Both panels share limits, because the point is that the cloud MOVED.
    """
    corrected = result["corrected_sequence"]

    def errors_of(sequence):
        return 100 * np.asarray([sequence.y[k] - sequence.truth[k]
                                 for k in range(sequence.n_steps)
                                 if sequence.camera[k] == camera
                                 and np.isfinite(sequence.truth[k, 0])])

    raw_e, cor_e = errors_of(seq), errors_of(corrected)
    span = 1.15 * max(np.abs(np.concatenate([raw_e, cor_e])).max(), 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6), sharex=True, sharey=True)
    panels = [
        (axes[0], raw_e, result["raw"], "as the pipeline reports it",
         result["oracle_raw"], result["R_learned_raw"]),
        (axes[1], cor_e, result["corrected"], "with the mesh correction applied",
         result["oracle_corrected"], result["R_learned_corrected"]),
    ]
    for ax, errors, side, title, oracle, learned in panels:
        ax.scatter(errors[:, 0], errors[:, 1], s=14, lw=0, alpha=0.40, color=C_OBS,
                   label=f"{len(errors)} errors")
        ax.axhline(0, color=C_TRUTH, lw=1.0)
        ax.axvline(0, color=C_TRUTH, lw=1.0)
        ax.plot(0, 0, marker="+", ms=16, mew=2.4, color=C_TRUTH, zorder=8)
        ax.add_patch(ellipse_from((0, 0), 1e4 * oracle["R_total"][camera],
                                  facecolor="none", edgecolor=C_TRUTH, lw=2.0,
                                  label=f"what a zero-mean model NEEDS "
                                        f"({side['needed_cm']:.2f} cm)"))
        mean = 100 * oracle["mean_m"][camera]
        ax.add_patch(ellipse_from(mean, 1e4 * oracle["R_spread"][camera],
                                  facecolor="none", edgecolor=C_OBS, lw=1.8, ls="--",
                                  label=f"the scatter it can SEE "
                                        f"({side['visible_cm']:.2f} cm)"))
        if learned is not None:
            ax.add_patch(ellipse_from(mean, 1e4 * learned[camera], facecolor="none",
                                      edgecolor="0.25", lw=1.8, ls=":",
                                      label=f"what the loop LEARNED "
                                            f"({side['learned_cm']:.2f} cm)"))
        ax.set_title(f"{title}\nthe model has no term for "
                     f"{side['cannot_express']:.1f}x of what it needs"
                     if side["cannot_express"] > 1.2 else
                     f"{title}\nneeded and visible now agree to "
                     f"{side['cannot_express']:.2f}x", fontsize=10.5)
        ax.set_xlabel("error across the aisle (cm)")
        ax.set_aspect("equal")
        ax.set_xlim(-span, span); ax.set_ylim(-span, span)
        ax.legend(fontsize=8.4, loc="upper left", framealpha=0.92)
    axes[0].set_ylabel("error along the aisle (cm)")
    fig.suptitle("A covariance cannot hold a lean — so take the lean out of the "
                 "observation function instead")
    plt.show()


def report_the_right_order(result, *, honest=None):
    """The diagnostic ratio, the arms, and the pixel noise, as numbers."""
    honest = nm.CALIBRATED_MEDIAN_NEES if honest is None else honest
    raw, cor = result["raw"], result["corrected"]
    print(f"  {result['n_corrected']} observations were corrected by the mesh model, "
          f"with heading from odometry.\n")
    print(f"  {'':28s}{'as reported':>13s}{'corrected':>12s}")
    for key, label in [("lean_cm", "the lean"),
                       ("needed_cm", "sigma a zero-mean model NEEDS"),
                       ("visible_cm", "sigma the loop can SEE"),
                       ("learned_cm", "sigma the loop LEARNED")]:
        print(f"  {label:28s}{raw[key]:>12.2f}{cor[key]:>12.2f}   cm")
    print(f"  {'needed / visible':28s}{raw['cannot_express']:>11.1f}x"
          f"{cor['cannot_express']:>11.2f}x   <- lean the model cannot express\n")
    print(f"  So the question 'what is this camera's covariance' is not answerable while")
    print(f"  the lean is in the data, and becomes answerable once it is not.\n")

    print(f"  {'belief':32s}{'NEES':>9s}{'x too sure':>12s}{'RMSE cm':>9s}")
    for name, arm in result["arms"].items():
        ratio = math.sqrt(arm["nees"] / honest)
        print(f"  {name:32s}{arm['nees']:>9.2f}{ratio:>11.1f}x{arm['rmse_cm']:>9.2f}")
    print(f"\n  honest is NEES {honest:.3f}; below it the belief is conservative.")
    print(f"  The floor is {result['floor_cm']:.1f} cm added to R, isotropic. The loop "
          f"cannot find it:")
    print(f"  it is the lean the mesh correction leaves behind, and a lean is not a "
          f"scatter.\n")
    print(f"  implied detector noise: {result['sigma_px_raw']:.2f} px as reported, "
          f"{result['sigma_px_corrected']:.2f} px corrected")
    print(f"  -- nearly the same, because this estimator is fitted to innovations too and")
    print(f"     so is blind to the lean in exactly the same way. That makes it a robust")
    print(f"     reading of the detector, and it is far below the 2.5 px the runtime "
          f"assumes.")


# ============================ the bias, predicted rather than absorbed into R

def report_bias_and_noise(split, camera):
    """How much of this camera's error is a repeatable lean and how much is noise."""
    total, scatter, after = split["total_cm"], split["scatter_cm"], split["after_cm"]
    mean = 100 * split["mean_m"]
    print(f"  camera {CAMERA_SHORT[camera]}, {len(split['steps'])} readings\n")
    print(f"  the whole error, typically          {total:6.2f} cm")
    print(f"  of which repeats every reading      ({mean[0]:+.2f}, {mean[1]:+.2f}) cm")
    print(f"  and the scatter about it            {scatter:6.2f} cm")
    print(f"\n  so {100 * (1 - scatter / total):.0f}% of it is a LEAN, not noise —")
    print(f"  and after predicting that lean from geometry: {after:6.2f} cm")
    print(f"  ({100 * (1 - after / total):.0f}% of the error removed, zero fitted parameters)")


def report_prediction_transfer(rows):
    """The prediction on every drive, and the control that has to fail."""
    print(f"  {'drive':24s}{'n':>5s}{'no fix':>10s}{'truth yaw':>12s}"
          f"{'ODOM yaw':>11s}{'yaw = 0':>10s}")
    for row in rows:
        name = row["drive"].replace("aws_", "").replace("_", " ")
        rule = "  " + "-" * 70 if name == "POOLED" else None
        if rule:
            print(rule)
        print(f"  {name:24s}{row['n']:5d}{row['none']:8.2f} cm{row['truth']:10.2f} cm"
              f"{row['odometry']:9.2f} cm{row['heading zero']:8.2f} cm")
    pooled = rows[-1]
    print(f"\n  Odometry heading does as well as the true heading and uses NO ground truth,")
    print(f"  removing {100 * (1 - pooled['odometry'] / pooled['none']):.0f}% of the error "
          f"with zero fitted parameters.")
    print(f"  The last column is the control: assuming the robot always points the same way")
    print(f"  scores {pooled['heading zero']:.2f} cm against {pooled['odometry']:.2f} cm, so "
          f"the pose-dependence is doing real work.")


def report_calibration_arms(table):
    """Four treatments, under correct and deliberately incorrect calibration."""
    degrees = list(dict.fromkeys(row["degrees"] for row in table))
    arms = list(dict.fromkeys(row["arm"] for row in table))
    print(f"  {'':34s}" + "".join(f"{('correct' if d == 0 else f'{d} deg off'):>22s}"
                                  for d in degrees))
    print(f"  {'arm':34s}" + "".join(f"{'too sure':>11s}{'RMSE':>11s}" for _ in degrees))
    for arm in arms:
        line = f"  {arm:34s}"
        for d in degrees:
            row = next(r for r in table if r["arm"] == arm and r["degrees"] == d)
            line += (f"{nm.times_too_confident(row['median_nees']):10.1f}x"
                     f"{row['rmse_cm']:8.2f} cm")
        print(line)
    print("\n  Read the RMSE columns across: inflating R never improves accuracy in any")
    print("  column. It buys an admission of being wrong, and nothing else. Modelling the")
    print("  bias improves accuracy AND honesty, and degrades gracefully as calibration")
    print("  drifts, because a calibration error is another pose-dependent bias.")


def calibration_arms(table):
    """The same table drawn: accuracy against honesty, as calibration drifts."""
    degrees = list(dict.fromkeys(row["degrees"] for row in table))
    arms = list(dict.fromkeys(row["arm"] for row in table))
    colours = [C_TRUTH, C_OBS, C_SMOOTH, C_FILTER]
    fig, (ax_rmse, ax_sure) = plt.subplots(1, 2, figsize=(12.6, 4.6))
    width = 0.8 / len(arms)
    pos = np.arange(len(degrees))
    for j, (arm, colour) in enumerate(zip(arms, colours)):
        picked = [next(r for r in table if r["arm"] == arm and r["degrees"] == d)
                  for d in degrees]
        offset = (j - (len(arms) - 1) / 2) * width
        ax_rmse.bar(pos + offset, [r["rmse_cm"] for r in picked], width=width * 0.9,
                    color=colour, alpha=0.9, label=arm)
        ax_sure.bar(pos + offset, [nm.times_too_confident(r["median_nees"]) for r in picked],
                    width=width * 0.9, color=colour, alpha=0.9)
    for ax, label, title in (
        (ax_rmse, "distance from the truth, cm",
         "Accuracy: inflating R does not help in any column"),
        (ax_sure, "how many times further the truth is\nthan the filter says it should be",
         "Honesty: only modelling the bias reaches 1x"),
    ):
        ax.set_xticks(pos)
        ax.set_xticklabels(["calibration\ncorrect" if d == 0 else f"{d}° of pitch\nerror"
                            for d in degrees])
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(title, fontsize=10.4)
    ax_sure.axhline(1.0, color="#B00020", ls="--", lw=1.7)
    ax_sure.set_yscale("log")
    fig.legend(*ax_rmse.get_legend_handles_labels(), fontsize=8.4, ncol=4,
               loc="outside lower center")
    plt.show()


def report_calibration_sensitivity(sensitivity):
    """What a given pointing error costs on the floor, at several ranges."""
    print(f"  this camera is {sensitivity['focal_px']:.0f} px per radian, so "
          f"{sensitivity['focal_px'] / 1000:.2f} px per milliradian:\n")
    print(f"  {'calibration error':24s}" +
          "".join(f"{f'{r:.1f} m away':>14s}" for r in sensitivity["ranges_m"]))
    for row in sensitivity["rows"]:
        print(f"  {row['label']:24s}" +
              "".join(f"{100 * v:11.1f} cm" for v in row["moved_m"]))
    print("\n  Every row grows with range, which is the signature of a BIAS FIELD and not")
    print("  of noise: the same displacement every time the robot stands in the same place.")


def report_gate_after_correction(rows):
    """A gate tuned in the biased regime, applied to a corrected observation function."""
    for row in rows:
        print(f"  {row['label']:22s}kept {row['used']:3d} of {row['offered']:3d} readings"
              f"   {nm.times_too_confident(row['median_nees']):5.1f}x too sure"
              f"   RMSE {row['rmse_cm']:6.2f} cm")
    print("\n  Once the observation function is right the innovations are small, and a")
    print("  threshold chosen when they were large starts rejecting good readings. Each")
    print("  rejection lets the belief drift, which makes the next innovation larger,")
    print("  which rejects more. The gate has to be re-derived with the model, not kept.")
