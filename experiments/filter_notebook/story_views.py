"""Figures for `camera_localisation_from_scratch`.

House rules, from `CLAUDE.md`, applied to every figure here:

  * the TITLE states the finding, not the variable names
  * axis labels say what the number means and which way is good
  * no bare condition codes as tick labels
  * honesty and sharpness ALWAYS appear together -- a filter that just draws a huge
    ellipse passes any honesty test and is useless
  * say what the data is: how many detections, which drive, real or ablated

Separate from `notebook_views.py`, which is frozen evidence for the `pp4_*` notebooks.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

CLEAN, PARTIAL, HIDDEN = "#2a9d8f", "#e9c46a", "#c1121f"
LEAN, SCATTER, PLAIN = "#264653", "#8ab17d", "#9b2226"
INK = "#22333b"


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 110, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "legend.frameon": False, "figure.autolayout": False,
    })


def _note(fig, text: str) -> None:
    """The provenance line every figure carries, so it stands alone."""
    fig.text(0.005, 0.005, text, fontsize=7.5, color="#6b705c", va="bottom")


# ---------------------------------------------------------------- Part 0

def odometry_drifts(d):
    """Where the wheels say the robot is, against where it actually is."""
    seq = d["seq"]
    ok = np.isfinite(seq.truth[:, 0])
    truth, odom = seq.truth[ok], seq.odom[ok]
    gap = np.linalg.norm(odom - truth, axis=1)
    path = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(truth, axis=0), axis=1))])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.plot(truth[:, 0], truth[:, 1], color=INK, lw=2.4, label="where the robot really is")
    ax.plot(odom[:, 0], odom[:, 1], color=PLAIN, lw=2.0, ls="--",
            label="where wheel-counting says it is")
    for i in range(0, len(truth), max(len(truth) // 12, 1)):
        ax.plot([truth[i, 0], odom[i, 0]], [truth[i, 1], odom[i, 1]],
                color=PLAIN, alpha=0.35, lw=1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("east-west position in the warehouse (m)")
    ax.set_ylabel("north-south position (m)")
    ax.set_title("Counting wheel turns loses track of the robot")
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    ax.plot(path, 100 * gap, color=PLAIN, lw=2.4)
    ax.fill_between(path, 0, 100 * gap, color=PLAIN, alpha=0.12)
    ax.set_xlabel("distance driven (m)")
    ax.set_ylabel("how far wrong the wheel estimate is (cm)\nlower is better")
    ax.set_title(f"The error only grows: {100 * gap[-1]:.0f} cm adrift after {path[-1]:.0f} m")
    ax.annotate(f"{100 * gap[-1]:.0f} cm", xy=(path[-1], 100 * gap[-1]),
                xytext=(-45, -18), textcoords="offset points", fontsize=10,
                color=PLAIN, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _note(fig, f"One recorded Gazebo drive ({d['tag']}), {int(ok.sum())} steps. "
               f"Truth is simulator ground truth and is used only to score.")
    return fig


def what_the_camera_gives(d, models, *, step=None):
    """One real frame: the detector's box, the pixel taken from it, where it lands."""
    import cv2
    camera = models["camera_A"]
    rows = d["rows"]
    row = rows[len(rows) // 2] if step is None else rows[step]
    frame = d["capture"].frame_at("camera_A", row["stamp"], tol_s=0.6)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    if frame is not None:
        raw = cv2.imread(str(frame[1]))
        image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB) if raw is not None else None
        ax.imshow(image) if image is not None else None
        u, v = row["uv"]
        ax.plot([u], [v], "o", ms=9, mfc=PLAIN, mec="white", mew=1.6, zorder=5)
        ax.annotate("bottom of the detector's box\n= the pixel the system uses",
                    xy=(u, v), xytext=(60, -70), textcoords="offset points", fontsize=9,
                    color=PLAIN, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=PLAIN, lw=1.4))
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("What the camera actually produces: an image")

    ax = axes[1]
    ax.plot([0, 11], [0, 0], color="#adb5bd", lw=3, solid_capstyle="butt")
    ax.text(5.5, -0.42, "warehouse floor", ha="center", color="#6b705c", fontsize=9)
    cam_x, cam_h = 0.0, 4.8
    ax.plot([cam_x], [cam_h], "s", ms=11, color=INK)
    ax.text(cam_x + 0.25, cam_h, "camera on the wall\n4.8 m up", fontsize=9, va="center")
    land = row["observed"]
    reach = float(np.hypot(land[0] - camera.cam_pos[0], land[1] - camera.cam_pos[1]))
    ax.plot([cam_x, reach], [cam_h, 0], color=PLAIN, lw=1.8, ls="--")
    ax.plot([reach], [0], "o", ms=9, mfc=PLAIN, mec="white", mew=1.5)
    ax.text(reach, 0.35, "where that ray\nmeets the floor", ha="center",
            fontsize=9, color=PLAIN, fontweight="bold")
    ax.add_patch(Rectangle((reach - 0.55, 0), 0.35, 0.6, color=INK, alpha=0.75))
    ax.set_xlim(-0.6, 11); ax.set_ylim(-0.9, 5.6)
    ax.set_xlabel("distance from the camera along the floor (m)")
    ax.set_ylabel("height (m)")
    ax.set_title("Turning that pixel into a position")
    ax.set_aspect("equal")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _note(fig, f"Real recorded frame from {d['tag']}, real detector output. "
               f"No fitted parameters anywhere in the pixel-to-floor step.")
    return fig


# ---------------------------------------------------------------- Part 1

def the_lean(d, *, n_arrows=90):
    """The hook: the camera's error is a lean, not scatter."""
    rows = d["rows"]
    truth = np.array([r["truth"] for r in rows])
    error = np.array([r["error"] for r in rows])
    mean = error.mean(axis=0)
    scatter = error - mean

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    ax = axes[0]
    idx = np.linspace(0, len(rows) - 1, min(n_arrows, len(rows))).astype(int)
    ax.quiver(truth[idx, 0], truth[idx, 1], error[idx, 0], error[idx, 1],
              angles="xy", scale_units="xy", scale=0.25, color=PLAIN,
              width=0.005, alpha=0.75)
    ax.plot(truth[:, 0], truth[:, 1], color=INK, lw=1.2, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("east-west (m)"); ax.set_ylabel("north-south (m)")
    ax.set_title("Every camera error, drawn where it happened\n(arrows 4x exaggerated)")

    ax = axes[1]
    ax.scatter(100 * error[:, 0], 100 * error[:, 1], s=13, color=PLAIN, alpha=0.45,
               label="each sighting")
    ax.plot([0], [0], "+", ms=16, mew=2.4, color=INK)
    ax.annotate("where the robot\nreally is", xy=(0, 0), xytext=(14, 14),
                textcoords="offset points", fontsize=9, color=INK)
    ax.arrow(0, 0, 100 * mean[0], 100 * mean[1], color=LEAN, lw=2.6,
             head_width=0.8, length_includes_head=True, zorder=5)
    ax.annotate(f"the lean: {100 * np.linalg.norm(mean):.1f} cm,\nthe same every time",
                xy=(100 * mean[0], 100 * mean[1]), xytext=(12, -30),
                textcoords="offset points", fontsize=9.5, color=LEAN, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlabel("camera error, east-west (cm)")
    ax.set_ylabel("camera error, north-south (cm)")
    ax.set_title("It is not scattered around the truth.\nIt is offset from it.")
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[2]
    share_lean = float(np.linalg.norm(mean))
    share_scatter = float(np.median(np.linalg.norm(scatter, axis=1)))
    total = share_lean + share_scatter
    ax.barh([1], [100 * share_lean / total], color=LEAN, height=0.5,
            label=f"repeats every frame ({100 * share_lean:.1f} cm)")
    ax.barh([1], [100 * share_scatter / total], left=[100 * share_lean / total],
            color=SCATTER, height=0.5,
            label=f"genuine randomness ({100 * share_scatter:.1f} cm)")
    ax.set_yticks([]); ax.set_xlim(0, 100)
    ax.set_xlabel("share of the camera's total error (%)")
    ax.set_title(f"{100 * share_lean / total:.0f}% of the error is the same mistake,\n"
                 "made over and over")
    ax.legend(loc="lower center", fontsize=9, bbox_to_anchor=(0.5, -0.42))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _note(fig, f"{len(rows)} detections from one recorded Gazebo drive ({d['tag']}). "
               f"Ground truth used only to score.")
    return fig


def averaging_does_not_help(d, *, repeats=400, seed=0):
    """Averaging kills randomness and leaves the lean untouched."""
    rng = np.random.default_rng(seed)
    error = np.array([r["error"] for r in d["rows"]])
    mean = error.mean(axis=0)
    scatter = error - mean
    counts = np.unique(np.round(np.logspace(0, math.log10(len(error)), 22)).astype(int))

    lean_curve, scatter_curve = [], []
    for n in counts:
        picks = rng.integers(0, len(error), size=(repeats, n))
        lean_curve.append(100 * np.median(np.linalg.norm(error[picks].mean(axis=1), axis=1)))
        scatter_curve.append(100 * np.median(np.linalg.norm(scatter[picks].mean(axis=1), axis=1)))

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(counts, scatter_curve, color=SCATTER, lw=2.6, marker="o", ms=4,
            label="the random part — shrinks like 1/√N")
    ax.plot(counts, lean_curve, color=PLAIN, lw=2.6, marker="s", ms=4,
            label="the whole error — does not shrink")
    lean_cm = float(100 * np.linalg.norm(mean))
    ax.axhline(lean_cm, color=LEAN, ls=":", lw=1.8)
    ax.text(counts[-1], lean_cm * 1.06,
            f"the lean: {lean_cm:.1f} cm, and it stays there",
            ha="right", fontsize=9, color=LEAN, fontweight="bold")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of sightings averaged together")
    ax.set_ylabel("error left after averaging (cm)\nlower is better")
    ax.set_title("More looks cannot fix it:\naveraging removes randomness, not a lean")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _note(fig, f"{len(error)} detections from {d['tag']}, {repeats} random draws per point.")
    return fig


def lean_is_not_constant(summary, reversal):
    """No fixed frame describes the lean: the reversal pair settles it."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    ax = axes[0]
    for s in summary:
        m = s["mean_world_m"]
        ax.arrow(0, 0, 100 * m[0], 100 * m[1], color=LEAN, alpha=0.55, lw=1.8,
                 head_width=0.5, length_includes_head=True)
        ax.annotate(s["tag"].replace("aws_", ""), xy=(100 * m[0], 100 * m[1]),
                    xytext=(4, 2), textcoords="offset points", fontsize=7.6, color=INK)
    ax.plot([0], [0], "+", ms=15, mew=2.2, color=INK)
    ax.set_aspect("equal")
    ax.set_xlabel("lean, east-west (cm)"); ax.set_ylabel("lean, north-south (cm)")
    ax.set_title("The lean is different on every drive,\nso no single constant removes it")

    ax = axes[1]
    labels, worlds, bodies = [], [], []
    for key, colour in (("forward", CLEAN), ("backward", PARTIAL)):
        e = reversal[key]
        labels.append(f"{e['tag'].replace('aws_', '')}\npointing {e['yaw_deg']:+.0f}°")
        worlds.append(100 * np.linalg.norm(e["world_m"]))
        bodies.append(e["world_m"])
    x = np.arange(2)
    ax.bar(x - 0.18, [100 * b[0] for b in bodies], 0.34, color=LEAN, label="east-west part")
    ax.bar(x + 0.18, [100 * b[1] for b in bodies], 0.34, color=SCATTER, label="north-south part")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0, color=INK, lw=1)
    ax.set_ylabel("the camera's lean (cm)")
    ax.set_title(f"Same line, driven both ways: the lean moves {reversal['world_gap_cm']:.1f} cm\n"
                 "— so it is not a property of the place")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _note(fig, "Nine recorded Gazebo drives. The reversal pair is the SAME line driven "
               "in both directions: identical ranges and bearings, only heading differs.")
    return fig


def lean_against_angle(la):
    """What the lean actually is: a smooth swing driven by one angle."""
    binned = la["binned"]
    centre = np.array([b["centre_deg"] for b in binned])
    radial = np.array([b["radial_cm"] for b in binned])
    radial_sd = np.array([b["radial_sd_cm"] for b in binned])
    across = np.array([b["across_cm"] for b in binned])
    across_sd = np.array([b["across_sd_cm"] for b in binned])

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    ax = axes[0]
    ax.plot(centre, radial, color=LEAN, lw=2.6, marker="o", ms=5,
            label="towards/away from the camera")
    ax.fill_between(centre, radial - radial_sd, radial + radial_sd, color=LEAN, alpha=0.18)
    ax.plot(centre, across, color=SCATTER, lw=2.6, marker="s", ms=5,
            label="sideways across the view")
    ax.fill_between(centre, across - across_sd, across + across_sd, color=SCATTER, alpha=0.18)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xlabel("angle between where the robot points and where the camera sees it from "
                  "(degrees)\n0° = driving directly away from the camera")
    ax.set_ylabel("the camera's lean (cm)")
    ax.set_title("The lean is a smooth swing, driven by one angle")
    ax.legend(fontsize=9, loc="lower right")

    ax = axes[1]
    names = ["as a warehouse\nconstant", "as a constant\ncarried by the robot",
             "given the\nviewing angle"]
    values = [la["radial_sd_cm"], la["radial_sd_cm"], la["radial_sd_conditioned_cm"]]
    colours = [PLAIN, PLAIN, CLEAN]
    bars = ax.bar(names, values, color=colours, width=0.55)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1, f"{v:.2f} cm",
                ha="center", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("how much of the lean is left unexplained (cm)\nlower is better")
    ax.set_title("One angle explains most of it —\nno fixed frame comes close")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _note(fig, f"{len(la['points'])} detections pooled over nine recorded Gazebo drives, "
               f"binned by viewing angle. Shaded band is one standard deviation.")
    return fig


# ---------------------------------------------------------------- Part 2 / 3

def honesty_and_sharpness(scores):
    """Accuracy AND stated uncertainty AND coverage, always together."""
    labels = [s["label"] for s in scores]
    error = [s["median_error_cm"] for s in scores]
    stated = [s["stated_sigma_cm"] for s in scores]
    cover = [s["coverage_95"] for s in scores]
    x = np.arange(len(scores))

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.5))
    ax = axes[0]
    ax.bar(x - 0.19, error, 0.36, color=PLAIN, label="how wrong it actually is")
    ax.bar(x + 0.19, stated, 0.36, color=LEAN, label="how wrong it SAYS it might be")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.6)
    ax.set_ylabel("centimetres")
    ax.set_title("Accuracy against the claim made about it\n(bars should match)")
    ax.legend(fontsize=8.6)

    ax = axes[1]
    colours = [CLEAN if 80 <= c <= 99 else PLAIN for c in cover]
    ax.bar(x, cover, 0.5, color=colours)
    ax.axhline(95, color=INK, ls="--", lw=1.6)
    ax.text(len(scores) - 0.5, 96.5, "95% is honest", ha="right", fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.6)
    ax.set_ylim(0, 105)
    ax.set_ylabel("how often the truth is inside the stated 95% ellipse (%)")
    ax.set_title("Below the line = overconfident\nAt 100% = too vague to be useful")

    ax = axes[2]
    ratio = [s / e for s, e in zip(stated, error)]
    colours = [CLEAN if 0.7 <= r <= 1.6 else (PARTIAL if r > 1.6 else PLAIN) for r in ratio]
    ax.bar(x, ratio, 0.5, color=colours)
    ax.axhline(1.0, color=INK, ls="--", lw=1.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.6)
    ax.set_yscale("log")
    ax.set_ylabel("stated uncertainty ÷ actual error")
    ax.set_title("1 = knows how good it is\nbelow = overconfident, above = vague")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _note(fig, f"Averaged over nine recorded Gazebo drives, {scores[0]['n']} steps each. "
               "Ground truth scores only; no estimator reads it.")
    return fig


def lean_recovery(rows):
    """Does the estimated lean land on the real one, and where does it miss?"""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
    ax = axes[0]
    for r in rows:
        rec, true = r["recovered"], r["true"]
        ax.plot([100 * true[0], 100 * rec[0]], [100 * true[1], 100 * rec[1]],
                color="#adb5bd", lw=1.1, zorder=1)
        ax.plot(100 * true[0], 100 * true[1], "o", ms=8, color=LEAN, zorder=3)
        ax.plot(100 * rec[0], 100 * rec[1], "^", ms=8, color=CLEAN, zorder=3)
    ax.plot([], [], "o", color=LEAN, label="the lean that was really there")
    ax.plot([], [], "^", color=CLEAN, label="what the robot worked out on its own")
    ax.set_aspect("equal")
    ax.set_xlabel("lean, east-west (cm)"); ax.set_ylabel("lean, north-south (cm)")
    ax.set_title("The robot recovers the lean without ever\nbeing told the right answer")
    ax.legend(fontsize=9, loc="best")

    ax = axes[1]
    along = [r["along_cm"] for r in rows]
    across = [r["across_cm"] for r in rows]
    x = np.arange(len(rows))
    ax.barh(x - 0.19, along, 0.36, color=PLAIN, label="along the camera's line of sight")
    ax.barh(x + 0.19, across, 0.36, color=CLEAN, label="across it")
    ax.set_yticks(x)
    ax.set_yticklabels([r["tag"].replace("aws_", "") for r in rows], fontsize=8.4)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("how much of the lean it failed to recover (cm)\ncloser to zero is better")
    ax.set_title("It misses along the line of sight and\nnot across it — and that is structural")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _note(fig, "Nine recorded Gazebo drives. The estimator reads only the detected pixel, "
               "the camera's mounting and wheel odometry.")
    return fig


def why_it_cannot_split(camera_xy=(0.0, -5.5), robot_xy=(1.0, 2.5)):
    """The confounding, drawn: two different worlds, identical measurements."""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharex=True, sharey=True)
    cx, cy = camera_xy
    rx, ry = robot_xy
    direction = np.array([rx - cx, ry - cy]); direction /= np.linalg.norm(direction)

    for ax, title, shift_robot in (
            (axes[0], "The camera leans, and the robot is where it thinks", False),
            (axes[1], "The camera is fine, and the robot is somewhere else", True)):
        ax.plot([cx], [cy], "s", ms=12, color=INK)
        ax.text(cx + 0.25, cy - 0.1, "camera", fontsize=9)
        true = np.array([rx, ry]) + (0.9 * direction if shift_robot else 0)
        ax.plot(*true, "o", ms=13, color=LEAN)
        ax.text(true[0] + 0.3, true[1], "where the\nrobot really is", fontsize=8.8, color=LEAN)
        reported = np.array([rx, ry]) + 0.9 * direction
        ax.plot(*reported, "X", ms=13, color=PLAIN)
        ax.text(reported[0] + 0.3, reported[1] - 0.7, "what the camera\nreports", fontsize=8.8,
                color=PLAIN)
        ax.plot([cx, reported[0] + 1.2 * direction[0]], [cy, reported[1] + 1.2 * direction[1]],
                color="#adb5bd", ls="--", lw=1.3)
        if not shift_robot:
            ax.annotate("", xy=reported, xytext=true,
                        arrowprops=dict(arrowstyle="->", color=PLAIN, lw=2.2))
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal"); ax.set_xlim(-2.2, 4.4); ax.set_ylim(-6.4, 4.6)
        ax.set_xlabel("east-west (m)")
    axes[0].set_ylabel("north-south (m)")
    fig.suptitle("Along the camera's line of sight these two are the same measurement",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    _note(fig, "Diagram, not data. This is why the lean and the robot's position cannot be "
               "separated along the sightline — and why a second camera at a different "
               "bearing would break the tie.")
    return fig


# ---------------------------------------------------------------- Part 4

def speed_costs_looks(table):
    """Faster driving buys fewer looks per metre and a bigger step between them."""
    routes = sorted({r["tag"].split("_v")[0] for r in table})
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.4))
    marks = ["o", "s", "^"]
    for mark, route in zip(marks, routes):
        pts = sorted([r for r in table if r["tag"].split("_v")[0] == route],
                     key=lambda r: r["speed_mps"])
        label = route.replace("aws_", "").replace("_", " ")
        speeds = [r["speed_mps"] for r in pts]
        axes[0].plot(speeds, [r["per_metre"] for r in pts], marker=mark, lw=2.2,
                     ms=6, label=label)
        axes[1].plot(speeds, [r["gap_cm"] for r in pts], marker=mark, lw=2.2, ms=6,
                     label=label)
        axes[2].plot(speeds, [r["detection_rate"] for r in pts], marker=mark, lw=2.2,
                     ms=6, label=label)
    axes[0].set_ylabel("sightings per metre driven\nhigher is better")
    axes[0].set_title("Driving faster buys fewer looks")
    axes[1].set_ylabel("distance travelled between sightings (cm)\nlower is better")
    axes[1].axhline(19.1, color=PLAIN, ls="--", lw=1.5)
    axes[1].text(0.16, 20.5, "the robot's own body length", fontsize=8.6, color=PLAIN)
    axes[1].set_title("At warehouse speed the robot moves\nfurther than its own length between looks")
    axes[2].set_ylabel("sightings that found the robot (%)")
    axes[2].set_ylim(0, 105)
    axes[2].set_title("The detector itself keeps up fine")
    for ax in axes:
        ax.set_xlabel("driving speed (m/s)")
        ax.legend(fontsize=8.6)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _note(fig, "New Gazebo captures, same route and detector at each speed; only the "
               "commanded speed differs. Gazebo renders instantaneous frames, so there is "
               "no motion blur here — on real hardware there would be.")
    return fig


def occlusion_displaces(split):
    """Partial occlusion does not remove a measurement, it moves it."""
    summary = split["summary"]
    keys = [k for k in ("clean", "partial") if summary.get(k)]
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.5))

    ax = axes[0]
    names = {"clean": "camera sees where the\nrobot meets the floor",
             "partial": "camera sees only the\nrobot's top"}
    vals = [summary[k]["median_cm"] for k in keys]
    bars = ax.bar([names[k] for k in keys], vals, color=[CLEAN, PARTIAL][:len(keys)], width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.15, f"{v:.1f} cm",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("how far wrong the camera's answer is (cm)\nlower is better")
    ax.set_title("A half-hidden robot still produces a\nreading — a worse one")

    ax = axes[1]
    lean = [summary[k]["mean_cm"] for k in keys]
    scat = [summary[k]["scatter_cm"] for k in keys]
    x = np.arange(len(keys))
    ax.bar(x - 0.19, lean, 0.36, color=LEAN, label="repeats every frame")
    ax.bar(x + 0.19, scat, 0.36, color=SCATTER, label="genuine randomness")
    ax.set_xticks(x); ax.set_xticklabels([names[k] for k in keys], fontsize=8.6)
    ax.set_ylabel("centimetres")
    ax.set_title("Occlusion adds a NEW lean,\nnot extra noise")
    ax.legend(fontsize=9)

    ax = axes[2]
    per = [r for r in split["per_drive"] if r["n"] > 30]
    x = np.arange(len(per))
    bottom = np.zeros(len(per))
    for key, colour, label in (("clean", CLEAN, "sees the contact point"),
                               ("partial", PARTIAL, "sees only the top"),
                               ("hidden", HIDDEN, "sees nothing")):
        vals = np.array([r[f"{key}_pct"] for r in per])
        ax.bar(x, vals, 0.6, bottom=bottom, color=colour, label=label)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([r["tag"].replace("aws_", "") for r in per], rotation=35,
                       ha="right", fontsize=7.8)
    ax.set_ylabel("share of the drive (%)")
    ax.set_title("What the camera could see, per drive")
    ax.legend(fontsize=8.4, loc="lower right")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _note(fig, "Visibility computed by ray-testing the camera against every shelf in the "
               "warehouse file — the floorplan a warehouse already has. No robot model used.")
    return fig


def occlusion_mostly_removes(detection, split):
    """Partial occlusion mostly stops the detector rather than displacing it."""
    pooled = detection["pooled"]
    keys = [k for k in ("clean", "partial", "hidden") if pooled[k]["chances"] > 10]
    names = {"clean": "sees where the robot\nmeets the floor",
             "partial": "sees only the\nrobot's top",
             "hidden": "sees nothing\nof the robot"}
    colours = {"clean": CLEAN, "partial": PARTIAL, "hidden": HIDDEN}

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.5))
    ax = axes[0]
    rates = [pooled[k]["rate"] for k in keys]
    bars = ax.bar([names[k] for k in keys], rates, color=[colours[k] for k in keys], width=0.55)
    for bar, k in zip(bars, keys):
        ax.text(bar.get_x() + bar.get_width() / 2, pooled[k]["rate"] + 2.5,
                f"{pooled[k]['rate']:.0f}%\n({pooled[k]['found']} of {pooled[k]['chances']})",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 118)
    ax.set_ylabel("how often the detector found the robot (%)")
    ax.set_title("Hide where the robot meets the floor and\nthe detector mostly fails outright")

    ax = axes[1]
    summary = split["summary"]
    have = [k for k in ("clean", "partial") if summary.get(k) and summary[k]["n"] >= 5]
    vals = [summary[k]["median_cm"] for k in have]
    bars = ax.bar([names[k] for k in have], vals, color=[colours[k] for k in have], width=0.5)
    for bar, k, v in zip(bars, have, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2,
                f"{v:.1f} cm\n(n={summary[k]['n']})", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("how far wrong the camera's answer is (cm)\nlower is better")
    ax.set_title("The few readings that do get through\nare worse — but there are few of them")

    ax = axes[2]
    per = [r for r in detection["per_drive"]
           if sum(r[k]["chances"] for k in ("clean", "partial", "hidden")) > 30]
    x = np.arange(len(per))
    bottom = np.zeros(len(per))
    for key in ("clean", "partial", "hidden"):
        vals = np.array([100 * r[key]["chances"]
                         / sum(r[q]["chances"] for q in ("clean", "partial", "hidden"))
                         for r in per])
        ax.bar(x, vals, 0.6, bottom=bottom, color=colours[key],
               label=names[key].replace("\n", " "))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([r["tag"].replace("aws_", "") for r in per], rotation=30,
                       ha="right", fontsize=7.8)
    ax.set_ylabel("share of the drive (%)")
    ax.set_title("What the camera could see, per drive\n(computed from the floorplan alone)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _note(fig, "Every observation message, misses included. Visibility computed by "
               "ray-testing the camera against every shelf in the warehouse file — the "
               "floorplan a warehouse already has. No robot model used.")
    return fig
