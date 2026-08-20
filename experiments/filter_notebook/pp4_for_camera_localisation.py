#!/usr/bin/env python3
# %% [markdown]
# # Bayesian filtering and smoothing, for a robot watched by four cameras
#
# This notebook follows the same arc as the BMLIP course notebook *PP4 - Bayesian
# filtering and smoothing*, but every number in it comes from a real recorded run of
# the warehouse system: a TurtleBot3 driving the central aisle while four ceiling
# cameras try to see it.
#
# The course notebook estimates a position from noisy observations of a simulated
# object, then learns the process noise. Here the observations are produced by a
# detector looking at real rendered camera images, and the quantity we learn is the
# **observation** noise `R`, not the process noise `Q`. That choice is the point of
# the notebook: `Q` describes how the robot moves, which is known from the wheel
# odometry model; `R` describes how much to trust a camera, which is exactly what
# nobody can write down in advance.
#
# | *PP4* | here |
# |---|---|
# | simulated object on a plane | TurtleBot3 in a warehouse, real recorded drive |
# | `x_k` position and velocity | `x_k` position in warehouse metres |
# | `u_k` known control input | wheel-odometry increment over the step |
# | `y_k` noisy observation of position | a camera detection back-projected onto the floor |
# | observations at every step | observations at ~3 per second, from whichever camera can see |
# | learns the process noise | learns the **observation** noise, per camera |
#
# The three things worth watching for:
#
# 1. Most steps have **no observation at all**. Coverage is a relay: two cameras hand
#    the robot over to two others halfway along the aisle.
# 2. The observation is not a measurement of the robot. It is a measurement of *where
#    the bottom of a detected box back-projects to*, which is a different thing, and
#    the difference is not zero-mean.
# 3. Learning `R` makes the model fit the data better **and makes the belief less
#    honest**. Both statements are measured below, and the tension between them is
#    the real lesson.

# %%
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import warnings

# This machine has matplotlib installed twice, which makes it warn about the 3D
# projection on import. Nothing here plots in 3D.
warnings.filterwarnings("ignore", message="Unable to import Axes3D")

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import numpy as np

sys.path.insert(0, str(Path.cwd()))
import notebook_data as nd

np.set_printoptions(precision=4, suppress=True)

# One palette for the whole notebook, colour-blind safe.
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

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 120,
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})

# %% [markdown]
# ## 0. The recording
#
# One traverse of the central aisle, south to north, recorded from a single Gazebo
# session so that the camera images, the detections, the wheel odometry and the
# ground truth all share one clock.
#
# Ground truth is loaded here, but it is used **only to score** what the filter
# believes. Nothing the filter consumes is derived from it.

# %%
capture = nd.load_capture()
truth_table = nd.load_truth(capture.name)
models = nd.camera_models()

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

# %% [markdown]
# ### Where the cameras are, and what they can see
#
# Four cameras on the walls, 6.1 m up, looking down and inwards at the aisle. The
# robot enters at the bottom of this picture and drives to the top, so it starts in
# the footprints of the two southern cameras and finishes in the two northern ones.
# The shaded quadrilaterals are what each camera's image covers on the floor,
# obtained by back-projecting the four image corners.

# %%
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

# %% [markdown]
# ## 1. The model
#
# The state is the robot's position on the floor, in warehouse metres:
#
# $$\mathbf{x}_k = \begin{bmatrix} x_k \\ y_k \end{bmatrix}$$
#
# **How it moves.** The wheel odometry reports how far the robot travelled during
# step $k$. That increment is the control input $\mathbf{u}_k$, and it is wrong by an
# amount that grows with the distance driven:
#
# $$\mathbf{x}_k = \mathbf{x}_{k-1} + \mathbf{u}_k + \mathbf{q}_k,
#   \qquad \mathbf{q}_k \sim \mathcal{N}\!\left(\mathbf{0},\, \mathbf{Q}_k\right),
#   \qquad \mathbf{Q}_k = \sigma_p^2 \lVert \mathbf{u}_k \rVert \, \mathbf{I}$$
#
# This is PP4's $\mathbf{x}_k = A\mathbf{x}_{k-1} + B\mathbf{u}_k + \mathbf{q}_k$ with
# $A = B = \mathbf{I}$. The one difference worth noticing is that $\mathbf{Q}_k$ is
# not constant: standing still adds no uncertainty, and driving a metre adds
# $\sigma_p^2$. $\sigma_p = 0.04\ \mathrm{m}/\sqrt{\mathrm{m}}$ is the commissioned
# wheel-odometry figure and is **not** learned here.
#
# **How it is observed.** When a camera produces a detection, that detection is
# back-projected onto the floor and treated as a direct, noisy observation of the
# position:
#
# $$\mathbf{y}_k = \mathbf{x}_k + \mathbf{r}_k^{(c)},
#   \qquad \mathbf{r}_k^{(c)} \sim \mathcal{N}\!\left(\mathbf{0},\, \mathbf{R}_c\right)$$
#
# so $C = \mathbf{I}$, with one $\mathbf{R}_c$ per camera $c$. Most steps have no
# $\mathbf{y}_k$ at all, exactly as PP4 handles missing observations.
#
# **The assumption that will fail.** $\mathbf{r}^{(c)}$ is assumed zero-mean. Section
# 5 measures whether it is.
#
# ### The generative model, written out
#
# Everything above is a recipe for *generating* a run: draw an observation covariance for
# each camera, draw a starting position, then step forward and emit an observation
# whenever a camera happens to be looking. Written as a joint distribution over
# everything unobserved and everything observed,
#
# $$p\!\left(\mathbf{R}_{1:C},\, \mathbf{x}_{0:T},\, \mathbf{y}_{\mathcal{K}} \,\middle|\, \mathbf{u}_{1:T}\right)
# = \underbrace{\prod_{c=1}^{C} p(\mathbf{R}_c)}_{\text{prior on each camera}}
#   \; \underbrace{p(\mathbf{x}_0)}_{\text{start}}
#   \; \underbrace{\prod_{k=1}^{T} p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)}_{\text{driving}}
#   \; \underbrace{\prod_{k \in \mathcal{K}} p\!\left(\mathbf{y}_k \mid \mathbf{x}_k, \mathbf{R}_{c_k}\right)}_{\text{being seen}}$$
#
# with the factors
#
# $$p(\mathbf{R}_c) = \mathcal{IW}(\Psi, \nu), \qquad
#   p(\mathbf{x}_0) = \mathcal{N}(\mathbf{m}_0, \mathbf{S}_0), \qquad
#   p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k) = \mathcal{N}\!\left(\mathbf{x}_{k-1} + \mathbf{u}_k,\, \mathbf{Q}_k\right), \qquad
#   p(\mathbf{y}_k \mid \mathbf{x}_k, \mathbf{R}_c) = \mathcal{N}\!\left(\mathbf{x}_k,\, \mathbf{R}_c\right)$$
#
# Three things to be clear about before any inference happens:
#
# * **Latent:** the trajectory $\mathbf{x}_{0:T}$ and the four covariances
#   $\mathbf{R}_{1:C}$. Both are inferred; neither is observed.
# * **Observed:** $\mathbf{y}_k$ for $k \in \mathcal{K}$, the steps where some camera
#   produced a detection — 35% of them. $\mathcal{K}$ and the camera identity $c_k$ are
#   treated as given, not modelled: *which* camera sees the robot is a fact about the
#   geometry of the building, not a random variable we need a distribution over.
# * **Known and fixed:** the odometry increments $\mathbf{u}_k$, and $\mathbf{Q}_k$. The
#   whole point of learning $\mathbf{R}$ rather than $\mathbf{Q}$ is that $\mathbf{Q}$
#   comes from a commissioned wheel model and $\mathbf{R}$ does not come from anywhere.
#
# What we want is the posterior $p(\mathbf{x}_{0:T}, \mathbf{R} \mid \mathbf{y})$. It has
# no closed form, because the trajectory and the covariances are coupled — which is what
# section 5 is about. Fix $\mathbf{R}$ and the trajectory posterior *is* closed-form, and
# that is the Kalman filter and smoother of sections 3 and 4.

# %%
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


draw_generative_model()

# %%
PROCESS_SIGMA_PER_SQRT_M = 0.04     # commissioned wheel-odometry drift, m / sqrt(m)
INITIAL_SIGMA_M = 0.05              # how well the start pose is known
GRID_HZ = 10.0                      # the uniform time grid the state sequence lives on
ASSOC_TOL_S = 0.06                  # a detection belongs to the grid step it lands nearest
GATE_CHI2_2DOF = 5.991              # 95% of a chi-squared with 2 degrees of freedom

# The commissioned per-camera observation noise, measured by
# `commission_observation_noise.py` on the three captures that predate this one, with
# the same parameter-free homography this notebook uses. This capture is held out of
# that fit, so nothing below is scored against a covariance fitted on itself.
#
# Two covariances come out of commissioning, and the difference is not cosmetic:
#
#   R_spread  the covariance about each camera's mean residual -- pure scatter
#   R_total   the second moment about zero -- scatter AND offset together
#
# A model that says `y = x + zero-mean noise` has no term for a mean, so it can never
# subtract one. Commissioning honestly *for that model* therefore has to hand over
# R_total: the offset has to be paid for somewhere, and inflated noise is the only
# pocket the model has. R_spread describes a quantity the filter cannot use.
_commissioned = json.loads(
    (nd.STUDY_ROOT / "commissioned_observation_noise.json").read_text(encoding="utf-8"))

R_COMMISSIONED_SPREAD = {c: np.asarray(_commissioned["per_camera"][c]["R_spread"])
                         for c in nd.CAMERAS}
R_COMMISSIONED_TOTAL = {c: np.asarray(_commissioned["per_camera"][c]["R_total"])
                        for c in nd.CAMERAS}
COMMISSIONED_OFFSET_M = {c: np.asarray(_commissioned["per_camera"][c]["mean_offset_m"])
                         for c in nd.CAMERAS}
COMMISSIONED_R_SIGMA_M = {c: float(_commissioned["per_camera"][c]["sigma_total_m"])
                          for c in nd.CAMERAS}

print("commissioned on held-out captures, current projection path:")
print(f"  {'camera':8s}{'n':>6s}{'offset cm':>11s}{'spread cm':>11s}{'total cm':>10s}")
for cam in nd.CAMERAS:
    s = _commissioned["per_camera"][cam]
    print(f"  {CAMERA_SHORT[cam]:8s}{s['n']:>6d}{100 * s['offset_magnitude_m']:>11.2f}"
          f"{100 * s['sigma_spread_m']:>11.2f}{100 * s['sigma_total_m']:>10.2f}")


class Sequence:
    """A uniform-time state sequence: control in, observations where they exist.

    PP4's `y` is a dense array with gaps marked missing. This is the same object: on
    a 10 Hz grid most steps carry no detection, because camera coverage is a relay
    with holes in it.
    """

    def __init__(self, capture, truth_table, *, grid_hz=GRID_HZ, window=None):
        stamps = np.asarray(capture.stamps, dtype=float)
        odom = np.asarray(capture.odom, dtype=float)
        # Trim to the driven route. The recorders outlive the drive, and the robot sits
        # parked in one camera's view until shutdown; left in, that stationary tail
        # tripled camera B's detection count and would misrepresent every per-camera
        # statistic in this notebook.
        lo = stamps[0] if window is None else max(stamps[0], window[0])
        hi = stamps[-1] if window is None else min(stamps[-1], window[1])
        grid = np.arange(lo, hi, 1.0 / grid_hz)

        odom_on_grid = np.column_stack([np.interp(grid, stamps, odom[:, i]) for i in range(2)])
        self.stamps = grid
        self.dt = 1.0 / grid_hz
        self.odom = odom_on_grid
        self.u = np.vstack([np.zeros((1, 2)), np.diff(odom_on_grid, axis=0)])

        detections = sorted(
            ((cam, d) for cam in nd.CAMERAS for d in capture.detections[cam]),
            key=lambda item: item[1].stamp,
        )
        self.y = np.full((len(grid), 2), np.nan)
        self.camera: list[str | None] = [None] * len(grid)
        self.pixel: list[tuple[float, float] | None] = [None] * len(grid)
        for cam, detection in detections:
            index = int(np.argmin(np.abs(grid - detection.stamp)))
            if abs(grid[index] - detection.stamp) > ASSOC_TOL_S:
                continue
            if self.camera[index] is not None:
                continue                       # one observation per step, as the model assumes
            self.y[index] = detection.world
            self.camera[index] = cam
            self.pixel[index] = (detection.u, detection.v)

        # EVALUATION ONLY -- for scoring, never for filtering
        self.truth = np.full((len(grid), 2), np.nan)
        for index, stamp in enumerate(grid):
            hit = nd.truth_at(truth_table, float(stamp))
            if hit is not None:
                self.truth[index] = hit[:2]

    @property
    def n_steps(self) -> int:
        return len(self.stamps)

    @property
    def observed(self) -> np.ndarray:
        return ~np.isnan(self.y[:, 0])


window = nd.route_window(capture.name)
print(f"driven route: simulated seconds {window[0]:.1f} to {window[1]:.1f} "
      f"({window[1] - window[0]:.0f} s)" if window else "no route record; using everything")
seq = Sequence(capture, truth_table, window=window)
n_obs = int(seq.observed.sum())
print(f"grid:          {seq.n_steps} steps at {GRID_HZ:.0f} Hz "
      f"({seq.stamps[-1] - seq.stamps[0]:.0f} s)")
print(f"observed:      {n_obs} steps ({100 * n_obs / seq.n_steps:.0f}%)")
print(f"unobserved:    {seq.n_steps - n_obs} steps "
      f"({100 * (1 - n_obs / seq.n_steps):.0f}%) -- these are pure prediction")
print(f"truth present: {int(np.isfinite(seq.truth[:, 0]).sum())} steps")
print()
for cam in nd.CAMERAS:
    k = sum(1 for c in seq.camera if c == cam)
    print(f"  {CAMERA_SHORT[cam]}: {k:4d} of the observed steps")

# %% [markdown]
# ### Which camera is talking, and when
#
# This is the coverage relay in one picture. The southern pair sees the robot first,
# the northern pair takes over, and in between there is a stretch where the robot is
# tracked by dead reckoning alone.

# %%
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

# %% [markdown]
# ## 2. Where an observation actually comes from
#
# In PP4 an observation is `y = C x + noise`, and the noise is given. Here the
# observation has to be manufactured out of an image, in three steps:
#
# 1. the camera renders a frame;
# 2. a detector puts a box around the robot, and the **bottom-centre of that box** is
#    taken as the point where the robot meets the floor;
# 3. that pixel is back-projected onto the floor plane.
#
# Step 3 is a homography. With the camera's intrinsics $K$ and its rotation $R$ and
# translation $t$, the map from a floor point to a pixel is
#
# $$\tilde{\mathbf{p}} \;=\; \underbrace{K \begin{bmatrix} \mathbf{r}_1 & \mathbf{r}_2 & \mathbf{t}\end{bmatrix}}_{H}
#   \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
#
# and the observation is obtained by inverting it, $\begin{bmatrix} x & y & 1\end{bmatrix}^{\!\top} \propto H^{-1}\tilde{\mathbf{p}}$.
# There are **no fitted parameters** in this step: $H$ comes from where the camera is
# bolted and what lens it has. Every fitted correction that was tried made the result
# worse, so the deployed system uses the plain homography.
#
# Steps 2 and 3 are where the model's assumptions start to strain, and the next cell
# shows why on a real frame.

# %%
_YOLO_CACHE: dict = {}


def detect_on_frame(image_bgr, model_path, *, imgsz=960, conf=0.05, iou=0.45):
    """Run the deployed detector on one frame; return boxes as (x1, y1, x2, y2, conf)."""
    from ultralytics import YOLO

    if model_path not in _YOLO_CACHE:
        _YOLO_CACHE[model_path] = YOLO(str(model_path))
    result = _YOLO_CACHE[model_path].predict(
        source=[image_bgr], imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confidence = boxes.conf.cpu().numpy()
    return [(*xyxy[i], float(confidence[i])) for i in np.argsort(-confidence)]


MODEL_PATH = nd.REPO / "logs/perception_models/warehouse_yolo_detector_4cam_v3_960/model.pt"

# Pick the step with the strongest evidence: an observed step whose camera frame was
# also written to disk, as near the middle of that camera's visibility as possible.
example = None
for cam in nd.CAMERAS:
    idx = [i for i, c in enumerate(seq.camera) if c == cam and np.isfinite(seq.truth[i, 0])]
    if not idx:
        continue
    middle = idx[len(idx) // 2]
    frame = capture.frame_at(cam, float(seq.stamps[middle]), tol_s=0.6)
    if frame is not None:
        example = {"camera": cam, "step": middle, "stamp": float(seq.stamps[middle]),
                   "frame_stamp": frame[0], "frame_path": frame[1]}
        break

print("example step:", {k: (v if not isinstance(v, Path) else v.name)
                        for k, v in (example or {}).items()})

# %%
import cv2

image = None if example is None else cv2.imread(str(example["frame_path"]))
if example is None or image is None:
    print("no recorded frame coincides with an observation; skipping the picture")
else:
    cam = example["camera"]
    model = models[cam]
    boxes = detect_on_frame(image, MODEL_PATH)
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
    sigma = COMMISSIONED_R_SIGMA_M[cam]
    ax_floor.add_patch(Ellipse(tuple(obs), 2 * 2 * sigma, 2 * 2 * sigma, fill=False,
                               ec=C_FILTER, ls="--", lw=1.2,
                               label=f"what the model expects (2 sd, {100*sigma:.1f} cm)"))
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

# %% [markdown]
# ### What the detector saw across the whole drive
#
# One frame is an anecdote. The rest of this section is the population it came from.
#
# The first thing to know is that most of the time the detector returns **nothing**. It is
# handed a frame roughly three times a second per camera, and for between a half and four
# fifths of them it reports no robot. Those are not errors in the filter's sense — the
# filter simply has no observation at that step — but they are the reason two thirds of the
# grid is unobserved, and they are not distributed evenly.
#
# The contact sheet below is built two ways on purpose: the **picture** comes from the
# saved frame, and the **verdict** comes from the runtime's own log. So a panel can show a
# perfectly visible robot and still be labelled a miss, which is exactly the behaviour
# worth seeing.

# %%
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
        boxes = detect_on_frame(image, MODEL_PATH)
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

# %% [markdown]
# Look along each row and the reason for most of the misses is not subtle: where the cross
# sits, there is **shelving**. The robot is behind a shelf row from that camera's point of
# view. Camera A's misses at 7, 9, 12 and 15 m are all of that kind, and so are camera B's
# at 13, 16 and 18 m. A couple of the others are the robot leaving the frame entirely.
#
# So the missing observations are not mostly a detector that is weak at range. They are
# geometry. The next cell separates the two, because they call for completely different
# responses: a detector limit is something to retrain, and an occlusion is something to
# plan around.
#
# ### When does it work, and where in the frame?
#
# Every message the runtime logged, valid or not, placed against where the robot was. Range
# and image position come from ground truth, so this is a diagnostic view — the filter never
# had it. It is the only way to see the misses, because a miss carries no position of its
# own.
#
# Two questions have to be asked separately, or the answer is meaningless:
#
# 1. **Was the robot inside the image at all?** Pure geometry, decided by where the cameras
#    point.
# 2. **Given that it was, did the detector find it?**
#
# Lumping them together produces the nonsense I got on the first attempt: a detection rate
# against range that jumps up and down, because on one straight traverse range and
# in-frame-ness move together, and a rate of zero at 7 m meant "there was nothing to
# detect", not "the detector failed".
#
# One caveat on the second question. Inside the image is not the same as visible — the
# contact sheet just showed shelves standing in the way — and nothing here models occlusion.
# So "in frame and missed" is an **upper bound** on the detector's own failures, and a good
# part of it is really the building.

# %%
def message_outcomes(messages, truth_table, models, window=None):
    """Every message, with the robot's true range, image position, and whether it was
    inside that camera's image at all.

    The last one matters. On a single straight traverse the range to a camera and whether
    the robot is even in frame move together, so a raw "detection rate against range"
    mixes 'the detector failed' with 'there was nothing to detect' and comes out
    non-monotone and meaningless. Splitting them is the whole point of this cell.

    Inside the image is not the same as visible: a shelf can stand in the way, and nothing
    here accounts for occlusion. So 'in frame and missed' is an upper bound on the
    detector's own failures.
    """
    out = {}
    for cam in nd.CAMERAS:
        model = models[cam]
        cam_xy = np.asarray(model.cam_pos[:2], dtype=float)
        rows = {"range": [], "u": [], "v": [], "ok": [], "stamp": [], "in_frame": []}
        for stamp, ok in messages[cam]:
            if window is not None and not (window[0] <= stamp <= window[1]):
                continue
            hit = nd.truth_at(truth_table, stamp, tol_s=0.05)
            if hit is None:
                continue
            u, v, visible = model.world_to_pixel(hit[0], hit[1], 0.0)
            rows["range"].append(float(np.linalg.norm(np.asarray(hit[:2]) - cam_xy)))
            rows["u"].append(u); rows["v"].append(v)
            rows["ok"].append(bool(ok)); rows["stamp"].append(stamp)
            rows["in_frame"].append(bool(visible))
        out[cam] = {k: np.asarray(v) for k, v in rows.items()}
    return out


outcomes = message_outcomes(messages, truth_table, models, window=window)

# Range turns out not to be the controlling variable: conditioned on being in frame, the
# rate against range is non-monotone for every camera. Position along the aisle is the axis
# that explains the behaviour, because it is what moves the robot in and out of each
# camera's patch of floor and behind the shelf rows.
north = {}
for cam in nd.CAMERAS:
    d = outcomes[cam]
    ys = []
    for stamp in d["stamp"]:
        hit = nd.truth_at(truth_table, float(stamp), tol_s=0.05)
        ys.append(hit[1] if hit is not None else np.nan)
    north[cam] = np.asarray(ys)

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

# %% [markdown]
# Read as a pair, those two panels say something the single-frame example could not.
#
# The **geometry** panel is the coverage relay, measured: the southern cameras hold the
# robot for the first half of the aisle, the northern pair for the second, and each hands
# over as the robot crosses out of its image. That is the same handover that shows up later
# as a step in the filter's error.
#
# The **detector** panel is the surprise. Even restricted to frames where the robot was in
# the image, the rate is not a smooth function of anything — it collapses to zero over
# particular stretches of aisle and recovers afterwards. Camera A finds nothing between
# about −4 and −2 m north; camera B finds nothing between +2 and +6. Those are dead bands
# in *space*, not at a particular range, which is the signature of the robot passing behind
# a shelf row rather than of a detector running out of pixels.
#
# The summary bar puts a number on it: given the robot was in frame, cameras C and D found
# it about 85% of the time, while A and B managed a quarter to a third. It would be wrong to
# read that as "A and B have a worse detector" — it is the same detector, the same weights,
# on all four. What differs is what stands between each camera and the aisle.

# %%
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

# %% [markdown]
# ### Why the bottom of the box is not the robot
#
# So far this section has been about *whether* the detector fires. The rest of it is about
# what the detection means when it does, because a detection in the right place is still
# not a measurement of the robot.
#
# The detector draws a box around the robot's **silhouette**. The bottom edge of that
# silhouette is the lowest visible point of the body as seen *from that camera*, and
# the TurtleBot3's body overhangs its wheel contact patch. So the point that gets
# back-projected sits a little away from the true contact point, and which way it
# sits depends on the direction the camera is looking from.
#
# Look at the magnified panel again and note the scale. The true position and the
# bottom of the box are a handful of pixels apart, and that handful of pixels is the
# 8.8 cm on the floor. The camera is six metres up and ten metres away, so its view of
# the ground is steeply foreshortened and one pixel is worth a centimetre or more. The
# detector is not sloppy — it is working at the resolution available to it, and the
# geometry amplifies whatever is left.
#
# The consequence for the model is precise and damaging: the observation error has a
# **non-zero mean** that differs per camera, and $\mathbf{R}_c$ — a covariance — has no
# way to represent a mean.
#
# The next figure is the geometry of it. Two things have to line up for the observation
# to be correct: the lowest visible point of the robot has to *be* the point where the
# robot touches the floor, and the ray through that pixel has to meet the floor where the
# model thinks the floor is. Neither is exactly true, and the error each introduces is
# multiplied by the range-to-height ratio — about 2:1 for these cameras.

# %%
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


draw_offset_geometry()

# %% [markdown]
# ### So what *is* the offset, exactly?
#
# It is worth being precise, because the word has been doing a lot of work.
#
# Two different points are being compared:
#
# * **What the pipeline reports** — the floor point that the bottom-centre pixel of the
#   detector's box back-projects to.
# * **What ground truth reports** — the robot's own origin, `base_footprint`, which sits on
#   the floor at the midpoint of the wheel axis.
#
# Those are not the same point, and they were never going to be. The offset is the
# systematic part of the gap between them. It is not an error in the ordinary sense; it is
# two different definitions of "where the robot is", differenced.
#
# **In this dataset, that is nearly the whole story.** It is worth stating what is *not*
# contributing, because a real installation would have more:
#
# | possible cause | active here? |
# |---|---|
# | the box bottom is not the contact point | **yes — this is the offset** |
# | camera pose wrong | no: the model is built from the same world file the renderer uses |
# | lens distortion | no: the camera definitions carry no distortion term |
# | floor not at $z=0$ | no: in this world it is exactly $z = 0$ |
# | shadow read as part of the robot | possibly, at the margin |
# | the silhouette clipped by a shelf | **yes — and this is the residual, see below** |
#
# So the honest description is narrower and more useful than "cameras have biases": the
# detector reports the bottom of a silhouette, the silhouette belongs to a 19 cm tall body
# whose plan centroid sits 37 mm behind the robot's origin, and the pipeline treats that
# report as if it were the origin itself. In a real building the middle three rows would
# also be non-zero, and would be indistinguishable from this one in a single residual.
#
# The next cell splits each error into the part along the camera's line of sight and the
# part across it, and section 6 goes further and predicts it.

# %%
def decompose_errors(names, models):
    """Split every error into radial (along camera->robot) and across-track parts."""
    rows = {cam: {"radial": [], "tangential": [], "range": []} for cam in nd.CAMERAS}
    for name in names:
        cap = nd.load_capture(name, models=models)
        table = nd.load_truth(name)
        for cam in nd.CAMERAS:
            cx, cy = float(models[cam].cam_pos[0]), float(models[cam].cam_pos[1])
            for det in cap.detections[cam]:
                hit = nd.truth_at(table, det.stamp, tol_s=0.05)
                if hit is None:
                    continue
                err = np.array([det.world[0] - hit[0], det.world[1] - hit[1]])
                sight = np.array([hit[0] - cx, hit[1] - cy])
                rng = float(np.linalg.norm(sight))
                if rng < 1e-6:
                    continue
                along = sight / rng
                across = np.array([-along[1], along[0]])
                rows[cam]["radial"].append(float(err @ along))
                rows[cam]["tangential"].append(float(err @ across))
                rows[cam]["range"].append(rng)
    return {cam: {k: np.asarray(v) for k, v in d.items()}
            for cam, d in rows.items() if len(d["radial"]) > 5}


split = decompose_errors(list(nd.COMMISSIONING_CAPTURES) + [capture.name], models)

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
        one = decompose_errors([name], models)
        per_run.append(f"{100 * one[cam]['tangential'].mean():+6.1f}" if cam in one else "     -")
    print(f"  {CAMERA_SHORT[cam]:9s}{'  '.join(per_run):>34s}")

# %% [markdown]
# Neither geometric story survives on its own, and that is the useful result.
#
# The error divides almost evenly: about half of it lies along the line of sight and half
# across it. A raised contact point can only produce the along-sight half, and it would
# have to grow in proportion to range — the dashed guides in the right-hand panel. The
# points do not follow them.
#
# The across-track half is the more interesting one, because for cameras B and C it is
# stable to about a centimetre across all four runs, and it is the component a small error
# in where the camera is *aimed* would produce. But it is not stable for A and D, so that
# does not close the case either.
#
# What this rules out is the comfortable conclusion. The offset is not one geometric
# constant waiting to be measured — not a contact height, not an aiming error, not the
# robot's body width. It is several effects at once, with the mix depending on where the
# robot is and which way it is pointing. That is why the next section learns `R` from the
# data rather than deriving it, and why even that turns out not to be enough.

# %%
residuals = {}
for cam in nd.CAMERAS:
    rows = [(seq.y[i] - seq.truth[i]) for i, c in enumerate(seq.camera)
            if c == cam and np.isfinite(seq.truth[i, 0])]
    if rows:
        residuals[cam] = np.asarray(rows)

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

# %% [markdown]
# ### Could the offset just be calibrated away?
#
# The obvious response to a known per-camera offset is to measure it once and subtract
# it forever. Commissioning measured exactly that, on three earlier runs. This capture
# measures it again. If the offset were a fixed property of each camera, the two would
# agree.

# %%
fig, ax = plt.subplots(figsize=(6.2, 5.4))
scale = 100.0
for cam in nd.CAMERAS:
    if cam not in residuals:
        continue
    here = tuple(scale * residuals[cam].mean(axis=0))
    there = tuple(scale * COMMISSIONED_OFFSET_M[cam])
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
    there = 100 * COMMISSIONED_OFFSET_M[cam]
    here = 100 * residuals[cam].mean(axis=0)
    moved = float(np.hypot(*(here - there)))
    print(f"  {CAMERA_SHORT[cam]:8s}"
          f"({there[0]:+7.1f},{there[1]:+7.1f}) cm"
          f"({here[0]:+7.1f},{here[1]:+7.1f}) cm{moved:>9.1f} cm")

# %% [markdown]
# They do not agree, and they do not disagree by a little. A correction frozen from the
# earlier runs would be pointing the wrong way for some of these cameras — worse than no
# correction at all.
#
# This is the same conclusion the offset-state work in this repository reached from the
# other direction: the offset is not a constant of the camera, in any frame. It depends
# on the bearing from which the silhouette is seen, which depends on where the robot is
# and which way it is facing, and those differ between runs. So the offset is real,
# systematic, and worth several centimetres — and still not something a one-off
# calibration can remove. It has to be estimated while driving, which is why the rest of
# this notebook treats `R` as the thing to learn and then shows why `R` alone is not
# enough.
#
# ## 3. The filter
#
# Standard Kalman recursions, written out rather than imported so the mechanism is
# visible. Prediction, for every step:
#
# $$\mathbf{m}_k^- = \mathbf{m}_{k-1} + \mathbf{u}_k, \qquad
#   \mathbf{P}_k^- = \mathbf{P}_{k-1} + \mathbf{Q}_k$$
#
# and where an observation exists, the update, with innovation
# $\mathbf{v}_k = \mathbf{y}_k - \mathbf{m}_k^-$ and innovation covariance
# $\mathbf{S}_k = \mathbf{P}_k^- + \mathbf{R}_c$:
#
# $$\mathbf{K}_k = \mathbf{P}_k^- \mathbf{S}_k^{-1}, \qquad
#   \mathbf{m}_k = \mathbf{m}_k^- + \mathbf{K}_k \mathbf{v}_k, \qquad
#   \mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k)\mathbf{P}_k^-(\mathbf{I} - \mathbf{K}_k)^{\!\top} + \mathbf{K}_k \mathbf{R}_c \mathbf{K}_k^{\!\top}$$
#
# The last line is the Joseph form. Algebraically it equals the more familiar
# $\mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k)\mathbf{P}_k^-$, but it stays symmetric
# and positive semi-definite under floating point, which the short form does not —
# and a covariance that quietly loses positive-definiteness has caused real damage in
# this codebase before.
#
# Two practical additions to the textbook recursion:
#
# * **A gate.** If $\mathbf{v}_k^{\!\top}\mathbf{S}_k^{-1}\mathbf{v}_k$ exceeds the
#   95% point of a $\chi^2$ with two degrees of freedom, the observation is rejected.
#   A detector that boxes a shelf leg instead of the robot produces an observation
#   that is not merely noisy but wrong, and the model has no term for that.
# * **The running log evidence** $\log p(\mathbf{y}_{1:T})$, accumulated from the
#   innovations. Section 5 needs it to compare models.

# %%
def kalman_filter(seq, R_per_camera, *, sigma_p=PROCESS_SIGMA_PER_SQRT_M,
                  initial_sigma=INITIAL_SIGMA_M, gate=GATE_CHI2_2DOF, m0=None):
    """Forward pass. Returns per-step means, covariances and diagnostics."""
    identity = np.eye(2)
    m = np.asarray(m0 if m0 is not None else seq.odom[0], dtype=float).copy()
    P = identity * initial_sigma**2

    out = {
        "m": np.zeros((seq.n_steps, 2)), "P": np.zeros((seq.n_steps, 2, 2)),
        "m_pred": np.zeros((seq.n_steps, 2)), "P_pred": np.zeros((seq.n_steps, 2, 2)),
        "innovation": np.full((seq.n_steps, 2), np.nan),
        "nis": np.full(seq.n_steps, np.nan),
        "used": np.zeros(seq.n_steps, dtype=bool),
        "rejected": np.zeros(seq.n_steps, dtype=bool),
        "log_evidence": 0.0,
    }

    for k in range(seq.n_steps):
        u = seq.u[k]
        Q = identity * (sigma_p**2 * float(np.linalg.norm(u)))
        m = m + u
        P = P + Q
        out["m_pred"][k], out["P_pred"][k] = m, P

        camera = seq.camera[k]
        if camera is not None:
            R = R_per_camera[camera]
            v = seq.y[k] - m                      # innovation
            S = P + R                             # innovation covariance
            S_inv = np.linalg.inv(S)
            nis = float(v @ S_inv @ v)
            out["innovation"][k], out["nis"][k] = v, nis
            if nis <= gate:
                K = P @ S_inv
                m = m + K @ v
                IKH = identity - K
                P = IKH @ P @ IKH.T + K @ R @ K.T      # Joseph form
                P = 0.5 * (P + P.T)
                out["used"][k] = True
                out["log_evidence"] += -0.5 * (
                    nis + math.log(max(np.linalg.det(2 * math.pi * S), 1e-300)))
            else:
                out["rejected"][k] = True

        out["m"][k], out["P"][k] = m, P
    return out


forward = kalman_filter(seq, R_COMMISSIONED_TOTAL)

print(f"observations offered:  {n_obs}")
print(f"observations used:     {int(forward['used'].sum())}")
print(f"observations rejected: {int(forward['rejected'].sum())} by the gate")
print(f"log evidence:          {forward['log_evidence']:.1f} nats")

# %% [markdown]
# ### One update, drawn
#
# Before the whole run, here is a single step, which is the entire filter in one picture:
# a prediction that has drifted and widened, one observation with its own uncertainty, and
# the posterior that combines them. The posterior is smaller than either input — that is
# what combining independent information buys — and it sits between them, nearer whichever
# was more confident.

# %%
def ellipse_from(mean, cov, n_sigma=2.0, **kwargs):
    """A patch showing the n-sigma contour of a 2-D Gaussian."""
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))
    width, height = 2 * n_sigma * np.sqrt(np.maximum(values, 0.0))
    return Ellipse(tuple(mean), width, height, angle=angle, **kwargs)


# The step where the prediction was widest among those that actually used an observation
# -- the moment the cameras had most to contribute. Skip the first few seconds: there the
# prediction is just the initial covariance, which would show the filter being started
# rather than the filter recovering from drift.
settled = int(5.0 * GRID_HZ)
candidates = [k for k in range(settled, seq.n_steps) if forward["used"][k]]
step = max(candidates, key=lambda k: np.trace(forward["P_pred"][k]))
cam = seq.camera[step]

m_pred, P_pred = forward["m_pred"][step], forward["P_pred"][step]
y, R = seq.y[step], R_COMMISSIONED_TOTAL[cam]
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

# %% [markdown]
# Note how many observations the gate threw away. They are not bad detections — the one
# pictured in section 2 was among the rejects when the noise was commissioned from
# scatter alone. They get rejected because the model expects them within a few
# centimetres and they land consistently further out. The gate is doing its job on the
# model it was given; the model is what is wrong. Commissioning $\mathbf{R}$ to include
# the offset already buys some of them back, which is the first sign that where the
# offset gets paid for matters more than how big $\mathbf{R}$ is.
#
# ### The picture PP4 draws
#
# Observations, the estimate, and the band the filter says the truth should lie in.
# The band is $\pm 2$ standard deviations, so the truth should be inside it about 95%
# of the time. Where no camera can see, the band widens — that is the odometry drift
# term doing its work.

# %%
def two_sigma(P, axis):
    return 2.0 * np.sqrt(np.maximum(P[:, axis, axis], 0.0))


time = seq.stamps - seq.stamps[0]
used, rejected = forward["used"], forward["rejected"]

# The drive covers 14 m north but the errors are centimetres, so plotting both on one
# axis hides the thing the notebook is about. Top panel: the traverse, so the reader
# knows what happened. Lower panels: everything measured against the truth, in
# centimetres, which is also the scale the 2 sd band is a claim about.
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

# %% [markdown]
# Two things in that picture are worth more than the summary statistics.
#
# **The error is not noise, it is a level.** For most of the drive the north error sits
# near $-8$ cm and stays there — and stays *outside* the band the filter draws for itself.
# A filter fed zero-mean observations cannot end up parked eight centimetres off; it ends
# up there because the observations themselves are parked eight centimetres off.
#
# **The level changes when the cameras change.** The dashed green line marks the last
# moment a southern camera contributed. Before it, both pairs are in play and the north
# error sits at the level printed above; after it, only the northern pair remains and the
# error returns to near zero. Nothing about the robot changed as it crossed that line —
# it was driving at a constant speed down a straight aisle. What changed is *whose* offset
# was being applied. That is the clearest evidence available that the offset belongs to
# the viewing geometry, and not to the robot, the odometry, or the clock.
#
# ### Watching it run
#
# Everything above is a summary. This is the filter itself, step by step, on the real frames.
#
# * **Left** — the camera that last produced a detection, cropped around the robot, with the
#   detector's actual box on it. When the caption says nothing was found, look at what is in
#   the way.
# * **Middle** — the whole aisle. The belief ellipse is far too small to see at this scale,
#   which is exactly why the third panel exists.
# * **Right** — the same moment magnified to a few tens of centimetres, where the belief *is*
#   visible: the prediction it arrived with, the observation and its assumed noise, and the
#   posterior it leaves with. Watch the ellipse swell through the blind stretches and snap
#   down when a camera speaks.
# * **Below** — the arithmetic of that step: the innovation, its normalised size, and what
#   the gate did with it.

# %%
from matplotlib import animation  # noqa: E402
from IPython.display import HTML, display  # noqa: E402

# Animations are embedded frame by frame, so the format matters. The run animation has a
# photograph in it and was 8 MB as PNG; JPEG frames at a modest resolution cut that to a
# third with no loss that shows at a thousand pixels wide. I assumed the line-art fitting
# animation would be smaller as PNG and measured the opposite -- 2.11 MB against 1.36 MB --
# so both use JPEG.
plt.rcParams["animation.embed_limit"] = 120.0

_BOX_CACHE: dict = {}


def boxes_for(path):
    """The detector's boxes on one saved frame, computed once and reused."""
    key = str(path)
    if key not in _BOX_CACHE:
        image = cv2.imread(key)
        _BOX_CACHE[key] = [] if image is None else detect_on_frame(image, MODEL_PATH)
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
HALF_PX, ZOOM_M = 150, 0.30


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
        ax_zoom.add_patch(ellipse_from(seq.y[k], R_COMMISSIONED_TOTAL[seq.camera[k]],
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
            f"   |   v'S-1v {item['nis']:6.2f} vs gate {GATE_CHI2_2DOF:.2f} -> {verdict}"
            f"   |   belief 2sd {200 * sd:5.1f} cm   |   off truth by {100 * err:5.1f} cm")
    return []


run_animation = animation.FuncAnimation(fig, draw_step, frames=len(run_frames),
                                        interval=420, blit=False)
with plt.rc_context({"animation.frame_format": "jpeg"}):
    display(HTML(run_animation.to_jshtml(default_mode="once")))
plt.close(fig)

# %% [markdown]
# ## 4. Smoothing
#
# The filter at step $k$ has only used observations up to $k$. A smoother goes
# backwards and lets later observations improve earlier estimates — which matters
# most exactly in the coverage gaps, where the filter was drifting blind and the next
# camera has not spoken yet.
#
# The Rauch-Tung-Striebel backward pass, with
# $\mathbf{G}_k = \mathbf{P}_k (\mathbf{P}_{k+1}^-)^{-1}$:
#
# $$\mathbf{m}_k^s = \mathbf{m}_k + \mathbf{G}_k\left(\mathbf{m}_{k+1}^s - \mathbf{m}_{k+1}^-\right),
#   \qquad
#   \mathbf{P}_k^s = \mathbf{P}_k + \mathbf{G}_k\left(\mathbf{P}_{k+1}^s - \mathbf{P}_{k+1}^-\right)\mathbf{G}_k^{\!\top}$$

# %%
def rts_smoother(seq, forward):
    """Backward pass. Returns smoothed means and covariances."""
    n = seq.n_steps
    ms = forward["m"].copy()
    Ps = forward["P"].copy()
    for k in range(n - 2, -1, -1):
        P_next_pred = forward["P_pred"][k + 1]
        G = forward["P"][k] @ np.linalg.inv(P_next_pred)
        ms[k] = forward["m"][k] + G @ (ms[k + 1] - forward["m_pred"][k + 1])
        Ps[k] = forward["P"][k] + G @ (Ps[k + 1] - P_next_pred) @ G.T
        Ps[k] = 0.5 * (Ps[k] + Ps[k].T)
    return {"m": ms, "P": Ps}


smooth = rts_smoother(seq, forward)


def error_summary(means, seq, label):
    """Distance from truth, where truth exists."""
    ok = np.isfinite(seq.truth[:, 0])
    err = np.linalg.norm(means[ok] - seq.truth[ok], axis=1)
    print(f"  {label:22s} median {100 * np.median(err):5.1f} cm | "
          f"90th percentile {100 * np.quantile(err, 0.9):5.1f} cm | "
          f"worst {100 * err.max():5.1f} cm")
    return err


print("distance from the truth:")
err_filter = error_summary(forward["m"], seq, "filtered")
err_smooth = error_summary(smooth["m"], seq, "smoothed")
_ = error_summary(seq.odom - seq.odom[0] + seq.truth[0], seq, "wheel odometry alone")

# %%
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

# %% [markdown]
# ## 5. Learning `R`
#
# So far $\mathbf{R}_c$ was asserted. Now infer it from the data.
#
# The exact posterior $p(\mathbf{x}, \mathbf{R} \mid \mathbf{y})$ is intractable because
# the trajectory and the covariances are coupled: what you believe about one changes what
# you believe about the other. Variational inference replaces it with the closest
# *factorised* distribution, and the factorisation is the approximation:
#
# $$p(\mathbf{x}_{0:T}, \mathbf{R} \mid \mathbf{y}) \;\approx\; q(\mathbf{x}_{0:T})\, \prod_c q(\mathbf{R}_c)$$
#
# "Closest" means largest evidence lower bound. Writing $\mathcal{L}$ for that bound,
#
# $$\log p(\mathbf{y}) \;=\; \underbrace{\mathbb{E}_q\!\left[\log \frac{p(\mathbf{x}, \mathbf{R}, \mathbf{y})}{q(\mathbf{x})q(\mathbf{R})}\right]}_{\mathcal{L}(q)\ \text{— the ELBO}}
# \;+\; \underbrace{\mathrm{KL}\!\left(q(\mathbf{x})q(\mathbf{R}) \,\|\, p(\mathbf{x}, \mathbf{R}\mid\mathbf{y})\right)}_{\ge\, 0}$$
#
# The log evidence is fixed, so pushing $\mathcal{L}$ up pulls the approximation towards
# the true posterior. Maximising $\mathcal{L}$ one factor at a time gives a two-step loop
# that cannot go backwards, and for this model both steps are closed-form.
#
# **The $\mathbf{x}$ step.** Holding $q(\mathbf{R})$ fixed, the optimal $q(\mathbf{x})$ is
# the exact posterior of a linear-Gaussian model whose observation covariance is
# $\bar{\mathbf{R}}_c = \left(\mathbb{E}_q\!\left[\mathbf{R}_c^{-1}\right]\right)^{-1}$ —
# so it is the Kalman filter and smoother of sections 3 and 4, run with $\bar{\mathbf{R}}$.
# Note it is the expected **precision** that enters, not the expected covariance. Using
# the mode or the mean of $q(\mathbf{R}_c)$ here would be a different algorithm.
#
# **The $\mathbf{R}$ step.** Holding $q(\mathbf{x})$ fixed, and with an inverse-Wishart
# prior $\mathbf{R}_c \sim \mathcal{IW}(\Psi, \nu)$, the optimal $q(\mathbf{R}_c)$ is
# inverse-Wishart again — conjugacy — with
#
# $$\Psi_c^{+} = \Psi + \sum_{k \in \mathcal{K}_c}\left[
#     \left(\mathbf{y}_k - \mathbf{m}_k^s\right)\left(\mathbf{y}_k - \mathbf{m}_k^s\right)^{\!\top}
#     + \mathbf{P}_k^s \right],
#   \qquad \nu_c^{+} = \nu + \lvert \mathcal{K}_c \rvert$$
#
# from which, for $d = 2$ dimensions,
#
# $$\bar{\mathbf{R}}_c = \frac{\Psi_c^{+}}{\nu_c^{+}}, \qquad
#   \mathbb{E}_q[\mathbf{R}_c] = \frac{\Psi_c^{+}}{\nu_c^{+} - d - 1}, \qquad
#   \mathrm{mode}\left[\mathbf{R}_c\right] = \frac{\Psi_c^{+}}{\nu_c^{+} + d + 1}$$
#
# The $\mathbf{P}_k^s$ term is what makes this more than fitting to residuals: the
# trajectory is uncertain, and the part of a residual that the trajectory's own
# uncertainty already explains must not be charged to the camera.
#
# The output is a **distribution** over each camera's noise, not a number — so the
# notebook can show how sure the data has made us, not just what it prefers.
#
# **Only `R` is learned.** $\mathbf{Q}$ stays at its commissioned value. Learning both at
# once from one traverse would leave them trading off against each other with nothing to
# pin the split down.
#
# ### This is variational inference, not active inference
#
# Worth being blunt about, because the two get run together and only one of them is here.
#
# What this notebook does is **variational inference**: given observations, approximate a
# posterior over things that are hidden — the trajectory, and the observation noise. The free
# energy being minimised is over *beliefs*. Nothing in it chooses an action.
#
# **Active inference** would extend that to the choice of where to drive, by scoring candidate
# routes on their *expected* free energy — including the epistemic part, "how well will I be
# able to see myself if I go that way". This robot did none of that. It followed a fixed
# commissioned route from a study configuration, at a constant 0.15 m/s, and would have driven
# exactly the same line whatever the cameras reported.
#
# The connection is real but one-directional. An expected-free-energy planner needs to predict
# the precision of an observation it has not made yet, at a pose it has not visited yet — which
# is precisely the per-camera, pose-dependent quantity the rest of this notebook is measuring.
# Sections 6 and 7 turn out to matter for that more than section 5 does: what the planner needs
# is not a better `R` per camera, it is an observation function that knows the bias depends on
# where the robot will be standing and which way it will be facing. So this notebook supplies
# an ingredient for active inference and contains none of it.

# %%
def iw_kl_from_prior(Psi_q, nu_q, Psi_p, nu_p, d=2):
    """KL( IW(Psi_q, nu_q) || IW(Psi_p, nu_p) ) -- how far the data moved the belief.

    Taking expectations of the inverse-Wishart log density under q, and using
    E_q[log|R|] = log|Psi_q| - d log 2 - sum_i psi((nu_q - i + 1)/2) and
    E_q[R^-1] = nu_q Psi_q^-1 (so that tr(Psi_q E_q[R^-1]) = nu_q d):

        KL = (nu_q/2) log|Psi_q| - (nu_p/2) log|Psi_p|
             - (d/2)(nu_q - nu_p) log 2
             - logGamma_d(nu_q/2) + logGamma_d(nu_p/2)
             - ((nu_q - nu_p)/2) E_q[log|R|]
             - (1/2) tr((Psi_q - Psi_p) E_q[R^-1])

    Verified below against the two properties any KL must have.
    """
    from scipy.special import multigammaln, digamma

    sign_q, logdet_q = np.linalg.slogdet(Psi_q)
    sign_p, logdet_p = np.linalg.slogdet(Psi_p)
    if sign_q <= 0 or sign_p <= 0:
        return float("nan")
    e_logdet = logdet_q - d * math.log(2.0) - sum(
        digamma((nu_q - i + 1) / 2.0) for i in range(1, d + 1))
    e_inv = nu_q * np.linalg.inv(Psi_q)
    return float(
        0.5 * nu_q * logdet_q - 0.5 * nu_p * logdet_p
        - 0.5 * d * (nu_q - nu_p) * math.log(2.0)
        - multigammaln(nu_q / 2.0, d) + multigammaln(nu_p / 2.0, d)
        - 0.5 * (nu_q - nu_p) * e_logdet
        - 0.5 * np.trace((Psi_q - Psi_p) @ e_inv)
    )


# A divergence that comes out negative is a bug, and this one did on the first attempt.
# Two checks: zero against itself, and non-negative on random pairs.
_Psi0, _nu0 = np.eye(2) * 0.0025 * 6.0, 6.0
assert abs(iw_kl_from_prior(_Psi0, _nu0, _Psi0, _nu0)) < 1e-9, "KL of a distribution from itself must be 0"
_rng = np.random.default_rng(0)
for _ in range(200):
    _A = _rng.normal(size=(2, 2))
    _Psi1 = _A @ _A.T + np.eye(2) * 1e-3
    _nu1 = 5.0 + 40.0 * _rng.random()
    assert iw_kl_from_prior(_Psi1, _nu1, _Psi0, _nu0) >= -1e-9, "KL must be non-negative"
print("inverse-Wishart KL passes both checks (zero from itself, never negative)")


def learn_R(seq, *, iterations=12, prior_nu=6.0, prior_sigma_m=0.05,
            sigma_p=PROCESS_SIGMA_PER_SQRT_M):
    """Mean-field variational inference for the per-camera R. Q is never touched.

    Returns the posterior parameters, not just a point estimate, so the notebook can
    plot how sure the data has made us.
    """
    d = 2
    Psi = np.eye(d) * (prior_sigma_m**2) * prior_nu
    # start from the prior: R_bar = (E[R^-1])^-1 = Psi / nu
    R_bar = {c: Psi / prior_nu for c in nd.CAMERAS}
    posterior = {}
    history = []

    for iteration in range(iterations):
        forward_i = kalman_filter(seq, R_bar, sigma_p=sigma_p)          # the x step
        smooth_i = rts_smoother(seq, forward_i)

        new_R_bar, counts = {}, {}
        for cam in nd.CAMERAS:                                          # the R step
            steps = [k for k in range(seq.n_steps)
                     if seq.camera[k] == cam and forward_i["used"][k]]
            counts[cam] = len(steps)
            scatter = np.zeros((d, d))
            for k in steps:
                v = (seq.y[k] - smooth_i["m"][k]).reshape(d, 1)
                scatter += v @ v.T + smooth_i["P"][k]
            Psi_post = Psi + scatter
            nu_post = prior_nu + len(steps)
            posterior[cam] = {"Psi": Psi_post, "nu": nu_post}
            # what the x step needs is the expected PRECISION, inverted
            new_R_bar[cam] = Psi_post / nu_post
        R_bar = new_R_bar
        # The gate admits a different subset of observations for every R, so the gated
        # evidence sums over different data and cannot be compared across arms. Keep an
        # ungated figure, over all observations, for any comparison between models.
        # Snapshot the posterior every iteration, so the fitting can be replayed later --
        # `honesty` is not defined yet at this point in the notebook, so the calibration of
        # each iterate is scored where the animation is built rather than here.
        history.append({
            "iteration": iteration,
            "log_evidence": forward_i["log_evidence"],
            "log_evidence_all": kalman_filter(seq, R_bar, sigma_p=sigma_p,
                                              gate=float("inf"))["log_evidence"],
            "R_bar": {c: R_bar[c].copy() for c in nd.CAMERAS},
            "posterior": {c: {"Psi": posterior[c]["Psi"].copy(),
                              "nu": float(posterior[c]["nu"])} for c in nd.CAMERAS},
            "sigma_m": {c: float(np.sqrt(np.trace(R_bar[c]) / d)) for c in nd.CAMERAS},
            "kl_from_prior": {c: iw_kl_from_prior(posterior[c]["Psi"], posterior[c]["nu"],
                                                  Psi, prior_nu, d) for c in nd.CAMERAS},
            "counts": counts,
        })

    # Each record above was scored with the R that went INTO that iteration, so the last
    # one does not describe the R actually returned. Score that too.
    final_forward = kalman_filter(seq, R_bar, sigma_p=sigma_p)
    history.append(dict(
        history[-1], iteration=iterations,
        log_evidence=final_forward["log_evidence"],
        log_evidence_all=kalman_filter(seq, R_bar, sigma_p=sigma_p,
                                       gate=float("inf"))["log_evidence"],
        R_bar={c: R_bar[c].copy() for c in nd.CAMERAS},
        sigma_m={c: float(np.sqrt(np.trace(R_bar[c]) / d)) for c in nd.CAMERAS}))
    return R_bar, history, {"posterior": posterior, "Psi_prior": Psi, "nu_prior": prior_nu}


R_learned, history, vb = learn_R(seq)

print(f"  {'iteration':>10s}{'log evidence':>15s}   per-camera one-sd, cm")
for record in history:
    sigmas = "  ".join(f"{CAMERA_SHORT[c]}={100 * record['sigma_m'][c]:5.2f}"
                       for c in nd.CAMERAS)
    print(f"  {record['iteration']:>10d}{record['log_evidence']:>15.1f}   {sigmas}")

# %%
forward_learned = kalman_filter(seq, R_learned)
smooth_learned = rts_smoother(seq, forward_learned)

fig, (ax_obj, ax_sigma) = plt.subplots(1, 2, figsize=(10.4, 4.0))

forward_all = kalman_filter(seq, R_COMMISSIONED_TOTAL, gate=float("inf"))
ax_obj.plot([h["iteration"] for h in history], [h["log_evidence_all"] for h in history],
            marker="o", ms=4, color=C_ACCENT, lw=1.6)
ax_obj.axhline(forward_all["log_evidence"], color=C_TRUTH, ls="--", lw=1.2,
               label="the commissioned R")
ax_obj.set_xlabel("iteration"); ax_obj.set_ylabel("log evidence, nats")
ax_obj.set_title("Learning R fits the data better\n(higher is a better fit)")
ax_obj.legend(fontsize=8.5)

names = [f"camera {CAMERA_SHORT[c]}" for c in nd.CAMERAS]
positions = np.arange(len(names))
commissioned = [100 * COMMISSIONED_R_SIGMA_M[c] for c in nd.CAMERAS]
learned = [100 * float(np.sqrt(np.trace(R_learned[c]) / 2)) for c in nd.CAMERAS]
ax_sigma.barh(positions + 0.19, commissioned, height=0.36, color=C_TRUTH,
              label="commissioned on other runs (scatter + offset)")
ax_sigma.barh(positions - 0.19, learned, height=0.36, color=C_ACCENT,
              label="learned on this run")
ax_sigma.set_yticks(positions); ax_sigma.set_yticklabels(names)
ax_sigma.set_xlabel("assumed observation noise, one standard deviation, cm")
ax_sigma.set_title("by asking to trust the cameras MORE, not less")
ax_sigma.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5)
plt.show()

for cam in nd.CAMERAS:
    print(f"  camera {CAMERA_SHORT[cam]}: commissioned {100 * COMMISSIONED_R_SIGMA_M[cam]:5.2f} cm"
          f"  ->  learned {100 * float(np.sqrt(np.trace(R_learned[cam]) / 2)):5.2f} cm"
          f"   ({history[-1]['counts'][cam]:4d} observations)")

# %% [markdown]
# ### What the posterior over `R` actually looks like
#
# The output of the $\mathbf{R}$ step is a distribution, so it can say how firmly the data
# has spoken. For one axis of a two-dimensional inverse-Wishart the marginal on a variance
# is inverse-gamma, so the density over the standard deviation $\sigma$ follows by change
# of variables. Cameras with hundreds of observations get a narrow posterior; camera B,
# with 26 usable detections on this run, does not — and the prior is still visible in it.
#
# The right-hand panel is the same story in one number per camera: how far the posterior
# has moved from the prior, in nats. It is the term that appears in the ELBO with a minus
# sign — the price paid for departing from the prior, which the fit has to earn back.

# %%
from scipy import stats  # noqa: E402


def sigma_density(Psi, nu, sigma_grid, axis=0, d=2):
    """Marginal density over one axis' standard deviation under IW(Psi, nu)."""
    # marginal variance ~ InvGamma(shape=(nu - d + 1)/2, scale=Psi[i,i]/2)
    shape = (nu - d + 1.0) / 2.0
    scale = Psi[axis, axis] / 2.0
    var = sigma_grid**2
    pdf_var = stats.invgamma.pdf(var, a=shape, scale=scale)
    return pdf_var * 2.0 * sigma_grid            # Jacobian d(var)/d(sigma)


fig, (ax_pdf, ax_kl) = plt.subplots(1, 2, figsize=(11.0, 4.2),
                                    gridspec_kw={"width_ratios": [1.55, 1.0]})

grid = np.linspace(0.002, 0.10, 400)
prior_pdf = sigma_density(vb["Psi_prior"], vb["nu_prior"], grid)
ax_pdf.plot(100 * grid, prior_pdf / prior_pdf.max(), color=C_OBS, lw=1.6, ls="--",
            label="prior, before any data")
for cam in nd.CAMERAS:
    post = vb["posterior"][cam]
    pdf = sigma_density(post["Psi"], post["nu"], grid)
    ax_pdf.plot(100 * grid, pdf / pdf.max(), color=CAMERA_COLOUR[cam], lw=1.9,
                label=f"camera {CAMERA_SHORT[cam]}  ({int(post['nu'] - vb['nu_prior'])} obs)")
    ax_pdf.axvline(100 * COMMISSIONED_R_SIGMA_M[cam], color=CAMERA_COLOUR[cam],
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

# %% [markdown]
# ### Better fit, worse belief
#
# The evidence went up, so by the usual model-selection argument the learned
# $\mathbf{R}$ is the better model. That argument is incomplete, because evidence
# measures fit to the observations, and what a robot needs is for its *belief* to be
# honest: when it says "I am here, give or take 5 cm", it should be right about the
# give or take.
#
# One computational note before the numbers, because it changes them. The gate admits a
# different subset of observations for every $\mathbf{R}$, so the running log evidence of
# section 3 sums over *different data* for each arm and cannot be used to compare them.
# Everything below recomputes it with the gate off, over all 349 observations. Doing it
# the sloppy way overstated the gap between learned and commissioned by about a hundred
# nats.
#
# The measure of honesty is the normalised squared error,
#
# $$\text{NEES}_k = \left(\mathbf{x}^{\text{true}}_k - \mathbf{m}_k\right)^{\!\top}
#   \mathbf{P}_k^{-1}\left(\mathbf{x}^{\text{true}}_k - \mathbf{m}_k\right)$$
#
# which for an honest two-dimensional belief has median $1.386$. Above that, the
# filter is claiming more precision than it has.
#
# Watch the smoothed rows in the table below, because they are counter-intuitive.
# Smoothing gets *closer* to the truth and *further* from honest at the same time. It
# narrows $\mathbf{P}$, which is the whole point of using later observations — but the
# part of the error that comes from a per-camera offset does not shrink, because every
# observation carries the same offset and averaging them cannot remove it. A smaller
# covariance around an error that stays put is exactly what overconfidence is.
#
# The companion is a strictly proper score, the negative log predictive density
#
# $$\text{NLPD} = -\log \mathcal{N}\!\left(\mathbf{x}^{\text{true}}; \mathbf{m}, \mathbf{P}\right)
#   = \tfrac{1}{2}\left(\text{NEES} + \log\det 2\pi\mathbf{P}\right)$$
#
# which cannot be improved by lying in either direction: claiming too little
# precision is punished by the volume term, claiming too much by the error term.
# Distance from truth alone — RMSE — would not settle this, because it does not look
# at $\mathbf{P}$ at all.

# %%
CALIBRATED_MEDIAN_NEES = 1.386


def honesty(result, seq, label):
    ok = np.isfinite(seq.truth[:, 0])
    nees, nlpd = [], []
    for k in np.flatnonzero(ok):
        e = seq.truth[k] - result["m"][k]
        P = result["P"][k]
        P_inv = np.linalg.inv(P)
        value = float(e @ P_inv @ e)
        nees.append(value)
        nlpd.append(0.5 * (value + math.log(max(np.linalg.det(2 * math.pi * P), 1e-300))))
    nees = np.asarray(nees); nlpd = np.asarray(nlpd)
    err = np.linalg.norm(result["m"][ok] - seq.truth[ok], axis=1)
    return {"label": label, "median_nees": float(np.median(nees)),
            "mean_nlpd": float(np.mean(nlpd)), "rmse_cm": float(100 * np.sqrt((err**2).mean())),
            "nees": nees}


forward_spread = kalman_filter(seq, R_COMMISSIONED_SPREAD)

scores = [
    honesty(forward_spread, seq, "commissioned scatter only, filtered"),
    honesty(forward, seq, "commissioned scatter + offset, filtered"),
    honesty(forward_learned, seq, "learned R, filtered"),
    honesty(smooth, seq, "commissioned scatter + offset, smoothed"),
    honesty(smooth_learned, seq, "learned R, smoothed"),
]

print(f"  {'':38s}{'median NEES':>13s}{'score (lower better)':>22s}{'distance':>11s}")
print(f"  {'':38s}{'(1.39 = honest)':>13s}{'':>22s}{'RMSE, cm':>11s}")
for s in scores:
    print(f"  {s['label']:38s}{s['median_nees']:>13.2f}{s['mean_nlpd']:>22.2f}"
          f"{s['rmse_cm']:>11.1f}")

# The gate admits a different subset of observations for each R, so the gated evidence
# sums over different data. For a comparison BETWEEN models it has to be recomputed with
# the gate off, over every observation. Doing it the sloppy way overstated the gap by
# about a hundred nats here.
print("\nlog evidence, all 349 observations, gate off, so the arms are comparable:")
evidence_arms = [
    ("commissioned scatter only", kalman_filter(seq, R_COMMISSIONED_SPREAD,
                                                gate=float("inf"))["log_evidence"],
     scores[0]["median_nees"]),
    ("commissioned scatter + offset", forward_all["log_evidence"], scores[1]["median_nees"]),
    ("learned on this run", history[-1]["log_evidence_all"], scores[2]["median_nees"]),
]
print(f"  {'arm':32s}{'evidence, nats':>16s}{'median NEES':>14s}")
for name, ev, nees in evidence_arms:
    print(f"  {name:32s}{ev:>16.1f}{nees:>14.2f}")
by_evidence = [a[0] for a in sorted(evidence_arms, key=lambda a: -a[1])]
by_honesty = [a[0] for a in sorted(evidence_arms, key=lambda a: abs(math.log(a[2] / CALIBRATED_MEDIAN_NEES)))]
print(f"\n  best fit first:      {'  >  '.join(by_evidence)}")
print(f"  most honest first:   {'  >  '.join(by_honesty)}")
print(f"  the two orderings are exact opposites: {by_evidence == by_honesty[::-1]}")

# %%
fig, (ax_nees, ax_score) = plt.subplots(1, 2, figsize=(10.6, 4.2))

labels = [s["label"] for s in scores]
positions = np.arange(len(labels))
colours = [C_TRUTH, C_ACCENT, C_SMOOTH, "#8FD3C1"]
ax_nees.barh(positions, [s["median_nees"] for s in scores], color=colours, height=0.6)
ax_nees.axvline(CALIBRATED_MEDIAN_NEES, color="#B00020", lw=1.6, ls="--",
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

# %% [markdown]
# ### Watching it fit, one half-step at a time
#
# The loop is two moves repeated, and the animation below shows them **separately** — one
# frame per half-step — because the whole story is in the alternation.
#
# * **The $\mathbf{x}$ step** refits the path using the current $\bar{\mathbf{R}}$. Watch the
#   middle panel: the path moves towards the observations.
# * **The $\mathbf{R}$ step** refits each ellipse to its own residual cloud. Watch the
#   ellipses in the left panel collapse.
#
# The thing to notice is *where the clouds already are on pass 0*. They are tight — a couple of
# centimetres — before any learning has happened at all. That is not because the observations
# are good; it is because the smoother has already bent the path onto them. Two thirds of the
# steps have no observation and the process noise is permissive, so the path has the freedom to
# go where the observations point, and it takes it.
#
# So the $\mathbf{R}$ step opens with a cloud that looks excellent and duly concludes the
# cameras are precise. The next $\mathbf{x}$ step, now trusting them more, pulls the path even
# further onto them. Round and round: **each half-step is locally correct, and the pair of them
# walks the filter into overconfidence.** The right panel is the price.
#
# Nothing here is a re-enactment; each frame is an actual iterate.

# %%
def replay_fitting(seq, *, passes=8, prior_nu=6.0, prior_sigma_m=0.05,
                   sigma_p=PROCESS_SIGMA_PER_SQRT_M):
    """Re-run the loop, keeping both halves of every pass so it can be animated."""
    d = 2
    Psi = np.eye(d) * (prior_sigma_m**2) * prior_nu
    R_bar = {c: Psi / prior_nu for c in nd.CAMERAS}
    steps = []
    for iteration in range(passes):
        # --- the x step: refit the path with the R we currently believe
        forward_i = kalman_filter(seq, R_bar, sigma_p=sigma_p)
        smooth_i = rts_smoother(seq, forward_i)
        residuals = {c: [] for c in nd.CAMERAS}
        for k in range(seq.n_steps):
            if seq.camera[k] is not None and forward_i["used"][k]:
                residuals[seq.camera[k]].append(seq.y[k] - smooth_i["m"][k])
        residuals = {c: np.asarray(v) for c, v in residuals.items()}
        track = 100 * (forward_i["m"][:, 1] - seq.truth[:, 1])
        nees = float(np.median(honesty(forward_i, seq, "it")["nees"]))
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
    ax_cost.axhline(CALIBRATED_MEDIAN_NEES, color=C_TRUTH, ls="--", lw=1.4,
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

# %% [markdown]
# ### The same thing as distributions, and the objective
#
# The panel above draws $\bar{\mathbf{R}}$ as a single ellipse per camera. What the
# $\mathbf{R}$ step actually produces is a *distribution* over each camera's noise, so here it
# is, alongside the two scores.
#
# One thing to read carefully in the middle panel. The quantity plotted is
# $\log p(\mathbf{y} \mid \bar{\mathbf{R}})$, the fit of the observations under the current
# estimate. That is **not** the objective the algorithm ascends — the objective is the ELBO,
# which is monotone by construction and is not computed here. Plug-in evidence carries no
# such guarantee, and sure enough it overshoots on the first pass and settles back. What
# survives is the endpoint comparison: it finishes well above the commissioned value (the
# dashed line), which is the claim being made, and the honesty panel deteriorates the whole
# way regardless.

# %%
fit_history = [h for h in history if "posterior" in h]
fit_nees = [float(np.median(honesty(kalman_filter(seq, h["R_bar"]), seq, "it")["nees"]))
            for h in fit_history]
fit_evidence = [h["log_evidence_all"] for h in fit_history]

# Also keep each iterate's filtered track, so the animation can show the trajectory being
# bent to absorb the bias -- the mechanism claimed in the prose, which was never shown.
fit_tracks = [kalman_filter(seq, h["R_bar"])["m"] for h in fit_history]

fig = plt.figure(figsize=(13.6, 3.9), dpi=82)
mosaic = fig.add_gridspec(1, 4, width_ratios=[1.35, 1.0, 1.0, 1.25], wspace=0.34)
ax_pdf = fig.add_subplot(mosaic[0, 0])
ax_fit = fig.add_subplot(mosaic[0, 1])
ax_hon = fig.add_subplot(mosaic[0, 2])
ax_track = fig.add_subplot(mosaic[0, 3])

sigma_grid = np.linspace(0.002, 0.075, 400)
prior_curve = sigma_density(vb["Psi_prior"], vb["nu_prior"], sigma_grid)


def draw_fit(step):
    for ax in (ax_pdf, ax_fit, ax_hon, ax_track):
        ax.clear()

    ax_pdf.plot(100 * sigma_grid, prior_curve / prior_curve.max(), color=C_OBS, lw=1.5,
                ls="--", label="prior, before any data")
    for cam in nd.CAMERAS:
        post = fit_history[step]["posterior"][cam]
        curve = sigma_density(post["Psi"], post["nu"], sigma_grid)
        ax_pdf.plot(100 * sigma_grid, curve / curve.max(), color=CAMERA_COLOUR[cam],
                    lw=2.0, label=f"camera {CAMERA_SHORT[cam]}")
        ax_pdf.axvline(100 * COMMISSIONED_R_SIGMA_M[cam], color=CAMERA_COLOUR[cam],
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
    ax_fit.set_xlabel("pass"); ax_fit.set_ylabel("log evidence, nats")
    ax_fit.set_title("Fit to the observations\n(higher is better; not the objective)",
                     fontsize=10)
    ax_fit.legend(fontsize=8, loc="lower right")

    ax_hon.plot(xs, fit_nees[:step + 1], marker="o", ms=4, color="#B00020", lw=1.8)
    ax_hon.axhline(CALIBRATED_MEDIAN_NEES, color=C_TRUTH, ls="--", lw=1.4,
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

print(f"  {'pass':>5s}{'evidence':>12s}{'median NEES':>14s}   per-camera one-sd, cm")
for h, ev, ns in zip(fit_history, fit_evidence, fit_nees):
    sigmas = "  ".join(f"{CAMERA_SHORT[c]}={100 * float(np.sqrt(np.trace(h['R_bar'][c]) / 2)):4.2f}"
                       for c in nd.CAMERAS)
    print(f"  {h['iteration']:>5d}{ev:>12.1f}{ns:>14.2f}   {sigmas}")

# %% [markdown]
# ### Why learning it makes things worse
#
# This is the result worth taking away, and it is not the one that was expected.
#
# Learning $\mathbf{R}$ does not ask for more noise to cover the offset. It asks for
# **less** — every camera's learned standard deviation comes out below what
# commissioning measured. And the belief gets correspondingly less honest.
#
# The mechanism is in the E step. Two thirds of the steps have no observation, and the
# process noise is permissive, so the smoothed trajectory has real freedom to bend. Given
# observations that are all displaced the same way, the cheapest explanation available to
# the smoother is not "the camera is biased" — it has no term for that — but "the robot
# was over there". So it bends the path towards the displaced observations. The M step
# then measures the residuals *about that bent path*, finds them small, and duly returns
# a small $\hat{\mathbf{R}}$. The next E step trusts the cameras even more. The offset has
# been laundered into the state estimate.
#
# That is why the distance from truth improves slightly (the path is being pulled towards
# observations that are wrong in a consistent direction, which happens to reduce squared
# error here) while the honesty collapses: the filter ends up *confidently* wrong instead
# of *cautiously* wrong. It is the failure mode that matters most for a robot, and the
# log evidence rose by nearly three hundred nats while it happened.
#
# Note also what the well-commissioned baseline achieved. Simply inflating
# $\mathbf{R}$ to include the offset — paying for the bias out of the only pocket the
# model has — took the median normalised error from 25.9 to 9.2 and the proper score from
# 7.8 to 1.2, with no change to the estimate itself. That is the best a zero-mean model
# can do, and it is still six times too confident.
#
# And notice what the evidence made of that arm: it ranked it **last** of the three. The
# printout above puts the three arms in order by fit and in order by honesty, and the two
# orderings are exact reverses of one another. This is not a near-miss or a tie that a
# larger dataset would resolve — on this run, choosing by model evidence would have picked
# the least honest of the three every time, and rejected the most honest.
#
# The last cell makes the point directly: subtract each camera's measured average
# offset before filtering, and see what happens to the honesty of the belief. This
# uses ground truth to compute the offsets, so it is a diagnostic and not a
# deployable method — it establishes what the ceiling would be if the offsets could be
# estimated without truth, which is the subject of the offset-state work elsewhere in
# this repository.

# %%
class OffsetCorrected(Sequence):
    """The same sequence with each camera's average error removed. Diagnostic only."""

    def __init__(self, base, offsets):
        self.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray) else list(v)
                                  if isinstance(v, list) else v)
                              for k, v in base.__dict__.items()})
        for k in range(self.n_steps):
            cam = self.camera[k]
            if cam is not None and cam in offsets:
                self.y[k] = self.y[k] - offsets[cam]


offsets = {cam: res.mean(axis=0) for cam, res in residuals.items()}
seq_corrected = OffsetCorrected(seq, offsets)

forward_corrected = kalman_filter(seq_corrected, R_COMMISSIONED_SPREAD)
R_corrected, history_corrected, vb_corrected = learn_R(seq_corrected)
forward_corrected_learned = kalman_filter(seq_corrected, R_corrected)

final = scores + [
    honesty(forward_corrected, seq_corrected, "offsets removed, scatter-only R"),
    honesty(forward_corrected_learned, seq_corrected, "offsets removed, learned R"),
]

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

# %%
fig, ax = plt.subplots(figsize=(7.6, 4.4))
labels = [s["label"] for s in final]
positions = np.arange(len(labels))
bar_colours = [C_TRUTH, C_ACCENT, C_SMOOTH, "#8FD3C1", "#7B4EA8", "#C9A0DC"]
ax.barh(positions, [s["median_nees"] for s in final], color=bar_colours, height=0.62)
ax.axvline(CALIBRATED_MEDIAN_NEES, color="#B00020", lw=1.7, ls="--",
           label="an honest belief sits here")
ax.set_yticks(positions); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("median normalised squared error (further right = more overconfident)")
ax.set_title("Learning the noise does not buy honesty; removing the offset does")
ax.legend(loc="lower right", fontsize=8.5)
ax.invert_yaxis()
plt.show()

# %% [markdown]
# ## 6. Fixing the observation model instead
#
# Everything so far has tried to describe the offset — as noise, as a per-camera constant,
# as something to inflate `R` for. All of that treats it as a nuisance to be absorbed.
#
# But section 2 said what it actually is: the detector reports the bottom of the robot's
# silhouette, and the pipeline pretends that is the robot's origin. That is not a nuisance,
# it is a **wrong observation function**. The model says
#
# $$\mathbf{y}_k = \mathbf{x}_k + \mathbf{r}_k$$
#
# when the truth is closer to
#
# $$\mathbf{y}_k \;=\; \underbrace{g_c\!\left(\mathbf{x}_k, \theta_k\right)}_{\text{where the silhouette's bottom lands}} \;+\; \mathbf{r}_k$$
#
# where $g_c$ is computable: take the robot's shape, put it at position $\mathbf{x}_k$ with
# heading $\theta_k$, project it into camera $c$, take the bottom-centre of the bounding
# box, and back-project that. Every ingredient is known — the shape from the robot's own
# mesh files, the camera from the world file, the projection from section 2.
#
# Two things follow, and they are the point of this section:
#
# 1. If $g_c$ is right, it should **predict** the observations, not merely describe them.
# 2. $g_c$ needs the heading $\theta_k$ — and the state $\mathbf{x}$ does not contain one.
#    That is the real defect: not that `R` was wrong, but that the state is too small to
#    express the thing that biases the measurement.
#
# The heading is not lost, though. The odometry knows which way the robot is pointing.

# %%
ROBOT_POINTS = nd.robot_point_cloud()
print(f"robot surface from its own mesh files: {len(ROBOT_POINTS)} vertices")
print(f"  width  {ROBOT_POINTS[:, 1].ptp():.4f} m")
print(f"  height {ROBOT_POINTS[:, 2].max():.4f} m")
print(f"  plan centroid sits {1000 * ROBOT_POINTS[:, 0].mean():+.1f} mm from the robot's origin")


def silhouette_bottom(model, x, y, yaw, points=ROBOT_POINTS):
    """Where the bottom-centre of the robot's projected bounding box lands on the floor.

    This is the observation function the pipeline should have been using. Zero fitted
    parameters: the shape is the mesh, the camera is the world file.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    world = points @ rotation.T + np.array([x, y, 0.0])
    in_camera = (world - model.cam_pos) @ model.R.T
    ahead = in_camera[:, 2] > 1e-6
    if not ahead.any():
        return None
    projected = (model.K @ in_camera[ahead].T).T
    uv = projected[:, :2] / projected[:, 2:3]
    return model.pixel_to_world(0.5 * (uv[:, 0].min() + uv[:, 0].max()), uv[:, 1].max())


def predicted_offset(model, x, y, yaw):
    """What g_c says the observation will be displaced by, at this pose."""
    landing = silhouette_bottom(model, x, y, yaw)
    return None if landing is None else np.array([landing[0] - x, landing[1] - y])


def heading_from_odometry(seq, span=8):
    """Which way the robot is pointing, from the odometry track alone.

    No ground truth: this is the direction the dead-reckoned position is moving in. It is
    available to the filter at run time, which is the whole point.
    """
    out = np.full(seq.n_steps, np.nan)
    for k in range(seq.n_steps):
        a, b = max(0, k - span), min(seq.n_steps - 1, k + span)
        step = seq.odom[b] - seq.odom[a]
        if float(np.linalg.norm(step)) > 0.02:
            out[k] = math.atan2(step[1], step[0])
    return out


odom_heading = heading_from_odometry(seq)
print(f"\nheading available from odometry at {int(np.isfinite(odom_heading).sum())} "
      f"of {seq.n_steps} steps")

# %% [markdown]
# ### Does it predict?
#
# For every detection in all four runs: put the robot where truth says it was, project its
# shape, and see whether the predicted landing point is where the observation actually
# landed. The fixed-heading row is the control — if the improvement were just "subtract a
# constant per camera", a wrong heading would not matter.

# %%
def prediction_residuals(names, models, points=ROBOT_POINTS):
    """|observation - prediction| under several ways of supplying the heading."""
    out = {k: [] for k in ("no correction", "true heading", "odometry heading",
                           "heading assumed zero")}
    for name in names:
        cap = nd.load_capture(name, models=models)
        table = nd.load_truth(name)
        grid = Sequence(cap, table, window=nd.route_window(name))
        head = heading_from_odometry(grid)
        for cam in nd.CAMERAS:
            for det in cap.detections[cam]:
                hit = nd.truth_at(table, det.stamp, tol_s=0.05)
                if hit is None:
                    continue
                observed = np.asarray(det.world)
                truth_xy = np.asarray(hit[:2])
                out["no correction"].append(float(np.linalg.norm(observed - truth_xy)))
                index = int(np.argmin(np.abs(grid.stamps - det.stamp)))
                supplies = {"true heading": hit[2],
                            "odometry heading": (head[index] if index < grid.n_steps else np.nan),
                            "heading assumed zero": 0.0}
                for label, yaw in supplies.items():
                    if yaw is None or not np.isfinite(yaw):
                        continue
                    landing = silhouette_bottom(models[cam], truth_xy[0], truth_xy[1],
                                                float(yaw), points)
                    if landing is not None:
                        out[label].append(float(np.linalg.norm(observed - np.asarray(landing))))
    return {k: np.asarray(v) for k, v in out.items()}


residual_by_heading = prediction_residuals(
    list(nd.COMMISSIONING_CAPTURES) + [capture.name], models)

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

# %% [markdown]
# ### And does it make the belief honest?
#
# Prediction accuracy is not the goal — an honest belief is. So run the filter again with
# the observation function corrected, using **only** things the filter has: the robot's mesh,
# the camera's pose, the heading from odometry, and its own running estimate of position.
# No ground truth enters the correction at any point.
#
# For each observation, the predicted displacement at the filter's current estimate is
# subtracted before the update. It is the cheap version of the right thing — the right thing
# is to linearise $g_c$ and run an extended filter — but it is enough to answer the question.

# %%
class GeometryCorrected(Sequence):
    """The same sequence with the observation function corrected by the object model.

    Uses the filter's own running estimate for position and odometry for heading, so it is
    deployable: nothing here is unavailable at run time.
    """

    def __init__(self, base, models, R_per_camera, heading, *,
                 sigma_p=PROCESS_SIGMA_PER_SQRT_M, initial_sigma=INITIAL_SIGMA_M):
        self.__dict__.update({
            k: (v.copy() if isinstance(v, np.ndarray) else list(v) if isinstance(v, list) else v)
            for k, v in base.__dict__.items()})
        identity = np.eye(2)
        m = self.odom[0].copy()
        P = identity * initial_sigma**2
        self.n_corrected = 0
        for k in range(self.n_steps):
            u = self.u[k]
            m = m + u
            P = P + identity * (sigma_p**2 * float(np.linalg.norm(u)))
            camera = self.camera[k]
            if camera is not None and np.isfinite(heading[k]):
                offset = predicted_offset(models[camera], m[0], m[1], float(heading[k]))
                if offset is not None:
                    self.y[k] = self.y[k] - offset
                    self.n_corrected += 1
            if camera is not None:
                R = R_per_camera[camera]
                innovation = self.y[k] - m
                S = P + R
                if float(innovation @ np.linalg.inv(S) @ innovation) <= GATE_CHI2_2DOF:
                    K = P @ np.linalg.inv(S)
                    m = m + K @ innovation
                    P = (identity - K) @ P @ (identity - K).T + K @ R @ K.T
                    P = 0.5 * (P + P.T)


seq_geometry = GeometryCorrected(seq, models, R_COMMISSIONED_TOTAL, odom_heading)
print(f"corrected {seq_geometry.n_corrected} of {int(seq.observed.sum())} observations "
      f"with no ground truth\n")

geometry_scores = [
    honesty(kalman_filter(seq, R_COMMISSIONED_TOTAL), seq,
            "commissioned R, as before"),
    honesty(kalman_filter(seq_geometry, R_COMMISSIONED_TOTAL), seq_geometry,
            "observation function corrected"),
    honesty(kalman_filter(seq_geometry, R_COMMISSIONED_SPREAD), seq_geometry,
            "corrected, and R back to scatter only"),
    honesty(forward_corrected, seq_corrected,
            "offsets removed using ground truth (the ceiling)"),
]
print(f"  {'':46s}{'median NEES':>13s}{'score':>9s}{'RMSE cm':>9s}")
for s in geometry_scores:
    print(f"  {s['label']:46s}{s['median_nees']:>13.2f}{s['mean_nlpd']:>9.2f}"
          f"{s['rmse_cm']:>9.2f}")
print(f"\n  an honest belief scores {CALIBRATED_MEDIAN_NEES:.2f}")

fig, ax = plt.subplots(figsize=(7.8, 3.8))
labels = [s["label"] for s in geometry_scores]
positions = np.arange(len(labels))
ax.barh(positions, [s["median_nees"] for s in geometry_scores],
        color=[C_TRUTH, C_ACCENT, C_SMOOTH, "#8FD3C1"], height=0.62)
ax.axvline(CALIBRATED_MEDIAN_NEES, color="#B00020", lw=1.7, ls="--",
           label="an honest belief sits here")
ax.set_yticks(positions); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("median normalised squared error (further right = more overconfident)")
ax.set_title("Correcting the observation function, with nothing the robot lacks")
ax.legend(loc="lower right", fontsize=8.5)
ax.invert_yaxis()
plt.show()

# %% [markdown]
# That is the answer the notebook was looking for, and it was not `R`.
#
# Correcting the observation function takes the median normalised error from **9.21 to
# 1.11**, against 1.39 for a perfectly honest belief, and the distance from truth from 8.4
# to 3.7 cm. The ground-truth-derived correction — the ceiling, which no deployed system
# could compute — reaches 0.74, which is slightly *under*-confident. The deployable version
# lands closer to honest than the oracle does.
#
# What is left is about 2 cm, and the most likely explanation is in the contact sheet from
# section 2: when a shelf clips the robot's silhouette, the bottom of the *visible* box is
# not the bottom of the whole robot, and the predictor assumes it is. Modelling that needs
# occlusion, which needs a depth test against the building — a different piece of work.
#
# The general lesson is worth separating from this particular robot. A persistent
# measurement bias is a signal that the observation function is wrong, and covariance is the
# wrong place to fix it. Here the missing ingredient was the robot's heading: it determines
# the bias, it was absent from the state, and the odometry had it all along.

# %% [markdown]
# ## 7. Learning the offset, rather than predicting it
#
# Section 6 computed the offset from the robot's shape. That works because this robot's shape
# is known and its heading is available. Neither is guaranteed — a different payload, an
# unknown pallet, a camera whose survey has drifted, and the prediction is wrong.
#
# So the other route: put the offset **in the state** and let the filter learn it. The state
# grows from two numbers to ten,
#
# $$\mathbf{z}_k = \begin{bmatrix}\mathbf{x}_k \\ \mathbf{b}^{(A)} \\ \mathbf{b}^{(B)} \\ \mathbf{b}^{(C)} \\ \mathbf{b}^{(D)}\end{bmatrix},
# \qquad
# \mathbf{y}_k = \underbrace{\begin{bmatrix} \mathbf{I} & \cdots & \mathbf{I} & \cdots \end{bmatrix}}_{H_c:\ \text{position plus that camera's offset}} \mathbf{z}_k + \mathbf{r}_k$$
#
# where $H_c$ picks out the position and the offset of whichever camera fired. The offsets get
# a broad prior, $\mathbf{b}^{(c)} \sim \mathcal{N}(\mathbf{0}, (10\,\text{cm})^2\mathbf{I})$,
# and optionally a slow random walk so they can track a quantity that is not really constant.
#
# This is the same filter — the recursions of section 3 with a bigger $H$. Nothing new is
# needed except the honesty to admit the state was too small.
#
# **The identifiability question comes first.** Adding a constant to every $\mathbf{b}^{(c)}$
# and subtracting it from $\mathbf{x}$ leaves every prediction unchanged, so the *common* part
# of the offsets is invisible to the cameras. Only the odometry pins position down, and only
# in increments. Whether that common part is recoverable at all is a property of the route,
# not of the filter, so it has to be measured rather than assumed.

# %%
OFFSET_CAMERAS = list(nd.CAMERAS)
STATE_DIM = 2 + 2 * len(OFFSET_CAMERAS)


def offset_state_filter(seq, R_per_camera, *, sigma_b_prior=0.10, sigma_b_walk=0.0,
                        sigma_p=PROCESS_SIGMA_PER_SQRT_M, initial_sigma=INITIAL_SIGMA_M,
                        gate=GATE_CHI2_2DOF):
    """Position and one 2-D offset per camera, all estimated together."""
    identity = np.eye(STATE_DIM)
    m = np.zeros(STATE_DIM)
    m[:2] = seq.odom[0]
    P = np.zeros((STATE_DIM, STATE_DIM))
    P[:2, :2] = np.eye(2) * initial_sigma**2
    for i in range(len(OFFSET_CAMERAS)):
        P[2 + 2 * i:4 + 2 * i, 2 + 2 * i:4 + 2 * i] = np.eye(2) * sigma_b_prior**2

    out = {"m": np.zeros((seq.n_steps, STATE_DIM)),
           "sd": np.zeros((seq.n_steps, STATE_DIM)),
           "P_position": np.zeros((seq.n_steps, 2, 2)),
           # the 2x2 block for each camera's offset, so its uncertainty can be drawn
           "P_offset": np.zeros((seq.n_steps, len(OFFSET_CAMERAS), 2, 2)),
           "used": np.zeros(seq.n_steps, dtype=bool)}

    for k in range(seq.n_steps):
        u = seq.u[k]
        m[:2] = m[:2] + u
        Q = np.zeros((STATE_DIM, STATE_DIM))
        Q[:2, :2] = np.eye(2) * (sigma_p**2 * float(np.linalg.norm(u)))
        for i in range(len(OFFSET_CAMERAS)):
            Q[2 + 2 * i:4 + 2 * i, 2 + 2 * i:4 + 2 * i] = np.eye(2) * sigma_b_walk**2
        P = P + Q

        camera = seq.camera[k]
        if camera is not None:
            i = OFFSET_CAMERAS.index(camera)
            H = np.zeros((2, STATE_DIM))
            H[:, :2] = np.eye(2)
            H[:, 2 + 2 * i:4 + 2 * i] = np.eye(2)
            R = R_per_camera[camera]
            innovation = seq.y[k] - H @ m
            S = H @ P @ H.T + R
            if float(innovation @ np.linalg.inv(S) @ innovation) <= gate:
                K = P @ H.T @ np.linalg.inv(S)
                m = m + K @ innovation
                closed = identity - K @ H
                P = closed @ P @ closed.T + K @ R @ K.T
                P = 0.5 * (P + P.T)
                out["used"][k] = True
        out["m"][k] = m
        out["sd"][k] = np.sqrt(np.maximum(np.diag(P), 0.0))
        out["P_position"][k] = P[:2, :2]
        for i in range(len(OFFSET_CAMERAS)):
            out["P_offset"][k, i] = P[2 + 2 * i:4 + 2 * i, 2 + 2 * i:4 + 2 * i]
    return out


def score_offset_filter(result, seq, label):
    """Honesty of the POSITION marginal -- the offsets are means to an end."""
    ok = np.isfinite(seq.truth[:, 0])
    values = []
    for k in np.flatnonzero(ok):
        e = seq.truth[k] - result["m"][k, :2]
        values.append(float(e @ np.linalg.inv(result["P_position"][k]) @ e))
    err = np.linalg.norm(result["m"][ok, :2] - seq.truth[ok], axis=1)
    return {"label": label, "median_nees": float(np.median(values)),
            "rmse_cm": float(100 * np.sqrt((err**2).mean()))}


# EVALUATION ONLY: what the offsets actually averaged, for scoring the estimates
measured_offset = {
    cam: np.mean([seq.y[k] - seq.truth[k] for k in range(seq.n_steps)
                  if seq.camera[k] == cam and np.isfinite(seq.truth[k, 0])], axis=0)
    for cam in OFFSET_CAMERAS}

offset_runs = {
    "constant offset": offset_state_filter(seq, R_COMMISSIONED_SPREAD, sigma_b_walk=0.0),
    "offset drifting 2 mm/step": offset_state_filter(seq, R_COMMISSIONED_SPREAD,
                                                     sigma_b_walk=0.002),
}

print(f"  {'offset model':32s}{'median NEES':>13s}{'RMSE cm':>9s}"
      f"{'offsets recovered to':>22s}")
print(f"  {'no offset in the state':32s}"
      f"{honesty(forward_spread, seq, 'x')['median_nees']:>13.2f}"
      f"{honesty(forward_spread, seq, 'x')['rmse_cm']:>9.2f}{'-':>22s}")
for label, run in offset_runs.items():
    scored = score_offset_filter(run, seq, label)
    errors = [float(np.linalg.norm(run["m"][-1, 2 + 2 * i:4 + 2 * i] - measured_offset[cam]))
              for i, cam in enumerate(OFFSET_CAMERAS)]
    print(f"  {label:32s}{scored['median_nees']:>13.2f}{scored['rmse_cm']:>9.2f}"
          f"{100 * float(np.median(errors)):>21.1f}c")
print(f"\n  an honest belief scores {CALIBRATED_MEDIAN_NEES:.2f}")

# %% [markdown]
# ### One camera does not play along
#
# Before the traces, an awkward number. With the offsets in the state and `R` set to the
# scatter alone, the gate throws away most of camera **B**'s observations. That deserves
# checking rather than glossing, because a result that depends on discarding three quarters of
# a camera is not a result.
#
# Two questions: does letting the offset move faster keep them, and does the answer depend on
# keeping them at all?

# %%
offered_per_camera = {cam: sum(1 for k in range(seq.n_steps) if seq.camera[k] == cam)
                      for cam in OFFSET_CAMERAS}
print(f"  {'offset may move':>16s}{'NEES':>8s}{'RMSE':>8s}{'offset err':>12s}"
      f"   observations used")
for walk in (0.0, 0.001, 0.002, 0.004, 0.008, 0.015):
    run = offset_state_filter(seq, R_COMMISSIONED_SPREAD, sigma_b_walk=walk)
    used = {cam: sum(1 for k in range(seq.n_steps)
                     if seq.camera[k] == cam and run["used"][k]) for cam in OFFSET_CAMERAS}
    scored = score_offset_filter(run, seq, "sweep")
    errors = [float(np.linalg.norm(run["m"][-1, 2 + 2 * i:4 + 2 * i] - measured_offset[cam]))
              for i, cam in enumerate(OFFSET_CAMERAS)]
    tally = " ".join(f"{CAMERA_SHORT[c]}={used[c]}/{offered_per_camera[c]}"
                     for c in OFFSET_CAMERAS)
    print(f"  {1000 * walk:>13.0f} mm{scored['median_nees']:>8.2f}{scored['rmse_cm']:>8.2f}"
          f"{100 * float(np.median(errors)):>11.1f}c   {tally}")

no_gate = offset_state_filter(seq, R_COMMISSIONED_SPREAD, sigma_b_walk=0.002,
                              gate=float("inf"))
scored_no_gate = score_offset_filter(no_gate, seq, "gate off")
print(f"\n  with the gate switched off entirely, every observation used: "
      f"NEES {scored_no_gate['median_nees']:.2f}, RMSE {scored_no_gate['rmse_cm']:.2f} cm")
print(f"  against NEES 0.74 and RMSE 5.29 cm with it on, so the conclusion does not rest")
print(f"  on those rejections.")

# %% [markdown]
# So: letting the offset drift faster does keep camera B's observations — at 15 mm per step it
# keeps 35 of 36 — and ruins everything else, taking the distance from truth from 5.3 cm to
# 14.1 cm and the offset estimates from 3.3 cm to 17.4 cm out. There is no drift rate that both
# keeps B and stays accurate.
#
# And switching the gate off entirely changes almost nothing. So the rejections are not holding
# the result up.
#
# The reading that fits all of this is that **camera B's observations genuinely do not lie on
# any smooth offset**. It is the camera the geometric predictor also got most wrong, and the one
# whose silhouette shelving clips most often — so its residuals jump around as the robot passes
# behind successive racks, in a way no slowly-varying bias can track. The gate is doing exactly
# what it is for: isolating the sensor that does not fit the model, while the other three carry
# the estimate. That is an argument for reliability-aware weighting rather than against the
# gate.
#
# ### Watching the offsets being learned
#
# The offsets start at zero with a 10 cm prior and are pulled into place by the observations.
# Each camera's estimate only moves while that camera can see the robot — which is why they
# arrive at different times, and why two of them are still moving when the drive ends.

# %%
best = offset_runs["offset drifting 2 mm/step"]
fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.0), sharex=True)
for ax, (i, cam) in zip(axes.ravel(), enumerate(OFFSET_CAMERAS)):
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

# %% [markdown]
# ### Watching the offsets being fitted
#
# The traces show each component against time; this shows all four offsets in the plane they
# actually live in, converging as the drive proceeds. Each ellipse is a camera's belief about
# its own offset at that moment.
#
# What to watch for: an ellipse only moves while its camera is contributing, it shrinks in
# proportion to how much that camera has said, and it drifts outward again when the camera goes
# quiet. Camera B's barely closes at all.

# %%
offset_frames = list(range(0, seq.n_steps, 20))


def contributions_by(step, camera):
    return sum(1 for k in range(step + 1)
               if seq.camera[k] == camera and best["used"][k])


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

    for ax, (i, cam) in zip(offset_axes, enumerate(OFFSET_CAMERAS)):
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


offset_animation = animation.FuncAnimation(fig, draw_offset_fit, frames=len(offset_frames),
                                           interval=380, blit=False)
with plt.rc_context({"animation.frame_format": "jpeg"}):
    display(HTML(offset_animation.to_jshtml(default_mode="once")))
plt.close(fig)

# %% [markdown]
# ### Three routes to the same quantity
#
# The offset has now been arrived at three independent ways: measured against ground truth,
# predicted from the robot's shape, and learned as a state. Agreement between them is the
# real check, because no two of them share an assumption.

# %%
learned_offset = {cam: best["m"][-1, 2 + 2 * i:4 + 2 * i]
                  for i, cam in enumerate(OFFSET_CAMERAS)}
geometric_offset = {}
for cam in OFFSET_CAMERAS:
    rows = []
    for k in range(seq.n_steps):
        if seq.camera[k] != cam or not np.isfinite(odom_heading[k]):
            continue
        landing = silhouette_bottom(models[cam], seq.truth[k, 0], seq.truth[k, 1],
                                    float(odom_heading[k]))
        if landing is not None:
            rows.append(np.asarray(landing) - seq.truth[k])
    geometric_offset[cam] = np.mean(rows, axis=0) if rows else np.full(2, np.nan)

print(f"  {'cam':5s}{'measured':>19s}{'predicted':>19s}{'learned':>19s}"
      f"{'pred err':>10s}{'learn err':>11s}")
for cam in OFFSET_CAMERAS:
    m_, p_, l_ = measured_offset[cam], geometric_offset[cam], learned_offset[cam]
    print(f"  {CAMERA_SHORT[cam]:5s}"
          f"({100*m_[0]:+6.1f},{100*m_[1]:+6.1f})"
          f"({100*p_[0]:+6.1f},{100*p_[1]:+6.1f})"
          f"({100*l_[0]:+6.1f},{100*l_[1]:+6.1f})"
          f"{100*float(np.linalg.norm(p_-m_)):>9.1f}c{100*float(np.linalg.norm(l_-m_)):>10.1f}c")

common_learned = np.mean([learned_offset[c] for c in OFFSET_CAMERAS], axis=0)
common_measured = np.mean([measured_offset[c] for c in OFFSET_CAMERAS], axis=0)
spread_errors = [float(np.linalg.norm((learned_offset[c] - common_learned)
                                      - (measured_offset[c] - common_measured)))
                 for c in OFFSET_CAMERAS]
print(f"\n  the part common to all four cameras: learned "
      f"({100*common_learned[0]:+5.1f},{100*common_learned[1]:+5.1f}) vs measured "
      f"({100*common_measured[0]:+5.1f},{100*common_measured[1]:+5.1f}) cm, "
      f"off by {100*float(np.linalg.norm(common_learned-common_measured)):.1f} cm")
print(f"  the differences between cameras, once that part is removed: "
      f"median error {100*float(np.median(spread_errors)):.1f} cm")

fig, ax = plt.subplots(figsize=(6.4, 5.6))
for cam in OFFSET_CAMERAS:
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

# %% [markdown]
# Two things are worth taking from this.
#
# **The belief becomes honest even though the offsets do not become accurate.** The learned
# offsets land 1 to 5 cm from what they averaged, a median of 3.3 cm on quantities of about
# 5 cm — not a good estimate. And yet the position belief scores 0.74 against 1.39 for honest,
# up from 25.9 without the offset state. That is not a contradiction: the filter now carries
# the *uncertainty* in the offsets through into the position, so it stops claiming precision
# it does not have. Being right and knowing how wrong you are are different achievements, and
# for a robot the second is the one that keeps it out of the shelving.
#
# **What is identifiable is what the route makes identifiable.** The part common to all four
# offsets is invisible to the cameras by construction, and comes out 2.1 cm off; the
# differences between cameras recover to 2.2 cm. On this route those are comparable, but the
# reason differs — the differences are pinned by cameras seeing the robot from different
# bearings, so a route with more angular diversity would sharpen them and a route with less
# would not. That is a property of where you drive, not of the estimator.
#
# **Prediction and learning are not rivals, and they fail in different places.** Predicting
# from the shape gives the better position — 3.7 cm against 5.3 cm — because it knows the
# mechanism, including how the offset swings as the robot turns. But look at the per-camera
# errors: the predictor is excellent on A, C and D (1.8 to 3.4 cm) and poor on **B (9.4 cm)**,
# the camera whose silhouette is clipped by shelving most often, where its assumption that
# the whole robot is visible breaks. Learning has no such assumption and does better there
# (3.4 cm). So they cover for each other, and the sensible arrangement is the obvious one:
# predict what the geometry gives you, and learn the part it cannot.

# %% [markdown]
# ## 8. `R` should not be one number per camera either
#
# Everything so far has carried one $\mathbf{R}_c$ per camera, fixed for the whole drive. That
# is convenient and wrong, and section 2 already said why: the observation is a *pixel*, and
# the map from pixel error to metres-on-the-floor is the derivative of the back-projection,
#
# $$\mathbf{R}_k \;=\; J(u_k, v_k)\; \Sigma_{\text{px}}\; J(u_k, v_k)^{\!\top},
# \qquad J = \frac{\partial (x, y)}{\partial (u, v)}$$
#
# and $J$ depends on **where in the image the detection happened**. Near the bottom of the
# frame the camera is looking almost straight down and a pixel is worth little; near the top it
# is looking along the floor and a pixel is worth a lot. A single covariance per camera has to
# average over that.
#
# This is not a new proposal. It is what the deployed runtime already does —
# `reliability.projection.project_observation_to_world_with_covariance` differentiates the
# projection numerically and propagates $J \Sigma_{uv} J^{\!\top}$. The constant-per-camera
# $\mathbf{R}$ used up to here was the simplification, adopted so that sections 3 to 5 could be
# about one thing at a time.
#
# Two questions: how much does $J$ actually vary, and is one pixel-noise number enough to
# replace the twelve fitted ones?

# %%
def projection_jacobian(model, u, v, step_px=0.5):
    """d(world)/d(pixel) by the same central difference the runtime uses."""
    columns = []
    for axis in (0, 1):
        delta = [step_px if axis == 0 else 0.0, step_px if axis == 1 else 0.0]
        plus = model.pixel_to_world(u + delta[0], v + delta[1])
        minus = model.pixel_to_world(u - delta[0], v - delta[1])
        if plus is None or minus is None:
            return None
        columns.append((np.asarray(plus) - np.asarray(minus)) / (2.0 * step_px))
    return np.column_stack(columns)


scale_per_pixel, residual_after_correction, scale_camera = [], [], []
for k in range(seq.n_steps):
    camera = seq.camera[k]
    if camera is None or not np.isfinite(odom_heading[k]) or not np.isfinite(seq.truth[k, 0]):
        continue
    J = projection_jacobian(models[camera], *seq.pixel[k])
    landing = silhouette_bottom(models[camera], seq.truth[k, 0], seq.truth[k, 1],
                                float(odom_heading[k]))
    if J is None or landing is None:
        continue
    scale_per_pixel.append(float(np.sqrt(np.trace(J @ J.T) / 2)))
    residual_after_correction.append(
        float(np.linalg.norm(seq.y[k] - (np.asarray(landing) - seq.truth[k]) - seq.truth[k])))
    scale_camera.append(camera)
scale_per_pixel = np.asarray(scale_per_pixel)
residual_after_correction = np.asarray(residual_after_correction)

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

# %%
def kalman_filter_jacobian_R(seq, sigma_px, fallback, *, sigma_p=PROCESS_SIGMA_PER_SQRT_M,
                             initial_sigma=INITIAL_SIGMA_M, gate=GATE_CHI2_2DOF):
    """The same filter, with R rebuilt at every observation from that pixel's Jacobian."""
    identity = np.eye(2)
    m = seq.odom[0].copy()
    P = identity * initial_sigma**2
    out = {"m": np.zeros((seq.n_steps, 2)), "P": np.zeros((seq.n_steps, 2, 2)),
           "used": np.zeros(seq.n_steps, dtype=bool)}
    for k in range(seq.n_steps):
        u = seq.u[k]
        m = m + u
        P = P + identity * (sigma_p**2 * float(np.linalg.norm(u)))
        camera = seq.camera[k]
        if camera is not None:
            J = projection_jacobian(models[camera], *seq.pixel[k])
            R = fallback[camera] if J is None else (sigma_px**2) * (J @ J.T)
            innovation = seq.y[k] - m
            S = P + R
            if float(innovation @ np.linalg.inv(S) @ innovation) <= gate:
                K = P @ np.linalg.inv(S)
                m = m + K @ innovation
                closed = identity - K
                P = closed @ P @ closed.T + K @ R @ K.T
                P = 0.5 * (P + P.T)
                out["used"][k] = True
        out["m"][k] = m
        out["P"][k] = P
    return out


print(f"  {'R model':40s}{'free numbers':>14s}{'median NEES':>13s}{'RMSE cm':>9s}")
constant = honesty(kalman_filter(seq_geometry, R_COMMISSIONED_TOTAL), seq_geometry,
                   "one covariance per camera")
print(f"  {constant['label']:40s}{12:>14d}{constant['median_nees']:>13.2f}"
      f"{constant['rmse_cm']:>9.2f}")
jacobian_scores = []
for sigma_px in (0.5, 1.0, 1.5, 2.0, 3.0):
    scored = honesty(kalman_filter_jacobian_R(seq_geometry, sigma_px, R_COMMISSIONED_TOTAL),
                     seq_geometry, f"Jacobian, pixel noise {sigma_px:.1f} px")
    jacobian_scores.append((sigma_px, scored))
    print(f"  {scored['label']:40s}{1:>14d}{scored['median_nees']:>13.2f}"
          f"{scored['rmse_cm']:>9.2f}")
print(f"\n  an honest belief scores {CALIBRATED_MEDIAN_NEES:.2f}")

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
ax_bar.axvline(CALIBRATED_MEDIAN_NEES, color="#B00020", lw=1.7, ls="--",
               label="an honest belief")
ax_bar.set_yticks(positions); ax_bar.set_yticklabels(labels, fontsize=8.5)
ax_bar.set_xlabel("median normalised squared error")
ax_bar.set_title("One pixel-noise number against twelve fitted ones")
ax_bar.legend(fontsize=8.5, loc="lower right")
ax_bar.invert_yaxis()
plt.show()

# %% [markdown]
# So `R` is state-dependent, and it matters.
#
# A pixel is worth between about 1 and 4 cm on this floor depending on where in the image the
# detection lands — a factor of three that a single covariance per camera has to average over.
# The residual does grow with it, in the right direction and by roughly the right amount: the
# nearer half of the detections sits at 1.5 cm per pixel and leaves 2.0 cm of residual, the
# further half at 2.4 cm per pixel and leaves 2.9 cm.
#
# The correlation per detection is only about +0.2, which is worth being honest about: at any
# one pixel the residual is still dominated by whatever the detector did with that particular
# silhouette, not by the geometry. The geometry sets the *scale*, not the individual value.
#
# And the practical result: **one pixel-noise number, propagated through the Jacobian, does the
# work of twelve fitted ones.** At 2 px it scores 1.20 against 1.11 for the four hand-fitted
# covariances — nominally a hair worse, actually a hair *closer* to the honest 1.39 — while
# needing no commissioning campaign, generalising to a camera it has never seen, and correctly
# trusting a detection at the bottom of the frame more than one at the top.
#
# Which puts section 5 in its place. Learning `R` per camera was solving the wrong problem
# twice over: the bias belonged in the observation function, and the spread belonged to the
# projection geometry. Neither of them was ever a property of the camera.

# %% [markdown]
# ## What the notebook establishes
#
# 1. **The machinery transfers.** PP4's filter, smoother and conjugate covariance
#    learning apply unchanged to a real robot watched by real cameras. The only
#    structural difference is that most steps carry no observation, which the same
#    recursions already handle.
#
# 2. **Learning `R` is easy and improves the fit.** One conjugate update per camera,
#    a dozen iterations, and the log evidence rises by 293 nats. If model evidence were
#    the criterion, the job would be done here.
#
# 3. **It is nonetheless the worst option measured.** Learning `R` on the run being
#    filtered gave a median normalised error of 44, against 9 for an `R` commissioned on
#    three *other* runs. Fitting the noise to the data you are filtering is not a
#    refinement of commissioning — here it is strictly worse than it, and the score that
#    says so is a proper one.
#
#    Worse than that: rank the three candidate covariances by model evidence and by
#    calibration and **the two orderings come out exactly reversed**. Evidence does not
#    merely fail to notice the problem; on this run it points reliably at the wrong
#    answer. Any procedure that selects `R` by marginal likelihood alone — which is the
#    textbook thing to do — would have chosen the least honest of the three.
#
# 4. **Because the error is a displacement, and the model has no term for one.** With
#    most steps unobserved, the smoother can explain consistently displaced observations
#    by bending the trajectory instead, so the M step sees small residuals and shrinks
#    `R`, and the bias ends up inside the state estimate. The best a zero-mean model can
#    manage is to inflate `R` until it covers the offset — which is what commissioning on
#    the second moment does, and which still leaves the belief six times too confident.
#
# 5. **The displacement cannot be calibrated away as a constant.** It is the robot's own
#    silhouette seen from four bearings, so it changes with where the robot is and which way
#    it faces. Measured on earlier runs and again here, the per-camera offsets move by 2 to
#    14 cm, and camera B's reverses sign in both axes — a frozen correction would actively
#    push that camera the wrong way.
#
# 6. **But it can be predicted, and that fixes the belief.** The offset stops being a mystery
#    once it is named properly: the detector reports the bottom of a silhouette belonging to
#    a body 19 cm tall whose plan centroid sits 37 mm behind the robot's origin. Project that
#    shape and the observations become predictable to about 2 cm — and supplying a *wrong*
#    heading makes it worse than no correction at all, so it is the pose-dependence doing the
#    work and not a constant in disguise. Correcting the observation function with the robot's
#    mesh, the camera's pose, the heading from odometry and the filter's own estimate —
#    nothing a deployed robot lacks — takes the median normalised error from **9.21 to 1.11**,
#    where an honest belief is 1.39.
#
# So the useful conclusion is not "estimate the offset". It is that a persistent measurement
# bias means the **observation function** is wrong, and no covariance can stand in for it.
# The missing ingredient here was the robot's heading: it determines the bias, the state did
# not contain it, and the odometry had it the whole time.
#
# 7. **Or learn it, and get an honest belief without knowing the robot at all.** Putting a
#    per-camera offset in the state takes the median normalised error from 25.9 to **0.74**,
#    and it does so while recovering the offsets themselves only to about 3 cm. The belief
#    becomes honest not because the offsets become right but because their uncertainty is
#    finally being carried. Prediction wins on accuracy where the geometry holds; learning
#    wins where it does not, which here is the camera whose view is most often clipped.
#
# What remains genuinely open is the last 2 cm, and the section-2 contact sheet already points
# at it: when a shelf clips the silhouette, the bottom of the visible box is not the bottom of
# the robot, and predicting that needs a depth test against the building rather than a better
# `R`.
#
# And one thing this notebook deliberately does **not** contain: any active inference. It
# approximates posteriors given data; it never chooses an action. What it supplies to an
# expected-free-energy planner is the piece that turns out to matter — an observation model
# whose precision depends on where the robot will stand and which way it will face.

# %%
summary = {
    "capture": capture.name,
    "steps": seq.n_steps,
    "observations": int(n_obs),
    "observation_rate_hz": round(n_obs / (seq.stamps[-1] - seq.stamps[0]), 2),
    "rejected_by_gate": int(forward["rejected"].sum()),
    "per_camera_offset_cm": {CAMERA_SHORT[c]: round(100 * float(np.hypot(*offsets[c])), 2)
                             for c in offsets},
    "commissioned_sigma_cm": {CAMERA_SHORT[c]: round(100 * COMMISSIONED_R_SIGMA_M[c], 2)
                              for c in nd.CAMERAS},
    "commissioned_offset_cm": {CAMERA_SHORT[c]: round(100 * float(np.hypot(*COMMISSIONED_OFFSET_M[c])), 2)
                               for c in nd.CAMERAS},
    "offset_moved_since_commissioning_cm": {
        CAMERA_SHORT[c]: round(100 * float(np.hypot(
            *(residuals[c].mean(axis=0) - COMMISSIONED_OFFSET_M[c]))), 2)
        for c in residuals},
    "learned_sigma_cm": {CAMERA_SHORT[c]: round(100 * float(np.sqrt(np.trace(R_learned[c]) / 2)), 2)
                         for c in nd.CAMERAS},
    "scores": [{k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in s.items() if k != "nees"} for s in final],
    "geometry_correction": {
        "robot_plan_centroid_mm": round(1000 * float(ROBOT_POINTS[:, 0].mean()), 1),
        "robot_height_m": round(float(ROBOT_POINTS[:, 2].max()), 4),
        "observations_corrected": int(seq_geometry.n_corrected),
        "median_residual_cm": {
            label: round(100 * float(np.median(values)), 2)
            for label, values in residual_by_heading.items() if values.size},
        "scores": [{k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in s.items() if k != "nees"} for s in geometry_scores],
    },
}
out = nd.STUDY_ROOT / "notebook_summary.json"
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"\nwrote {out}")
