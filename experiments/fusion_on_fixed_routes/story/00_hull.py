"""The predicted bounding box, explained on one real four-camera moment.

Three figures into logs/studies/fusion_on_fixed_routes/00_hull_observation/:

  01_each_camera_hands_an_ellipse  what one camera's reading actually is: a position AND an
                                   ellipse -- and what the admission check throws away
  02_why_not_convert_the_pixel     the same pixel under three headings lands in three
                                   different places, which is why the box is predicted
                                   rather than converted
  03_three_rules_one_moment        what the four fusion rules do with the same three
                                   admitted readings. Mechanism, not evidence

Everything is read from the frozen commissioning capture. Nothing is fitted here.
"""
import sys, math
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "measurement_commissioning"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src/reliability"))
from _hull_common import (CAMS, OUT, SIGMA_PX, crop, ellipse, moment,   # noqa: E402
                          solve_position)
import style as D                                                       # noqa: E402
from admission import gate                                              # noqa: E402
from reliability.contracts import CameraQuality                         # noqa: E402
from reliability.fusion import MapObservation, joint_network_estimate_2d  # noqa: E402

(TX, TY, YAW), M = moment()
TRUTH = np.array([TX, TY])
ADMIT = {c: gate(d["pred_box"], d["det_box"]) for c, d in M.items()}
KEPT = [c for c in M if ADMIT[c][0]]
DROPPED = [c for c in M if not ADMIT[c][0]]


def cm(v):
    """Metres, relative to the true position, in centimetres."""
    return (np.asarray(v) - TRUTH) * 100.0


def sigmas(cov):
    w, _ = np.linalg.eigh(cov)
    return math.sqrt(max(w)) * 100, math.sqrt(min(w)) * 100


def draw_frame(ax, cam_id, d, admitted, reasons):
    im = cv2.imread(str(d["image"]))[:, :, ::-1]
    x0, y0, x1, y1 = crop(im, d["pred_box"])
    ax.imshow(im[y0:y1, x0:x1])
    p = d["pred_box"]
    ax.add_patch(plt.Rectangle((p[0] - x0, p[1] - y0), p[2] - p[0], p[3] - p[1],
                               fill=False, edgecolor=D.ROBOT, lw=2.4, ls=(0, (5, 3)), zorder=4))
    q = d["det_box"]
    col = D.GOOD if admitted else D.BAD
    ax.add_patch(plt.Rectangle((q[0] - x0, q[1] - y0), q[2] - q[0], q[3] - q[1],
                               fill=False, edgecolor=col, lw=2.8, zorder=5))
    ax.plot(0.5 * (q[0] + q[2]) - x0, q[3] - y0, "o", ms=9, color=col, mec="white", mew=1.6,
            zorder=6)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(D.CAM_COLOUR[cam_id[-1]]); s.set_linewidth(2.6)
    a, b = sigmas(d["cov"])
    ax.set_title(f"camera {cam_id[-1]} — {d['range_m']:.1f} m away",
                 loc="left", fontsize=14, color=D.CAM_COLOUR[cam_id[-1]], pad=5)
    if admitted:
        ax.set_xlabel(f"kept · its ellipse is {a:.1f} x {b:.1f} cm", fontsize=12,
                      color=D.INK2, labelpad=6)
    else:
        why = " and ".join(r.replace("_", " ") for r in reasons)
        ax.set_xlabel(f"THROWN AWAY · {why}", fontsize=12, color=D.BAD,
                      fontweight="bold", labelpad=6)


# =========================================================================
# 01 — what one camera's reading is
# =========================================================================
fig = plt.figure(figsize=(17.4, 9.8), constrained_layout=True)
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.22])
order = sorted(M)
for i, cam_id in enumerate(order):
    draw_frame(fig.add_subplot(gs[0, i]), cam_id, M[cam_id], *ADMIT[cam_id])

ax = fig.add_subplot(gs[1, :2])
ax.axhline(0, color="#eceae4", lw=1); ax.axvline(0, color="#eceae4", lw=1)
for cam_id in KEPT:
    d = M[cam_id]; col = D.CAM_COLOUR[cam_id[-1]]
    e = ellipse(d["cov"]) * 100.0 + cm(d["est"])
    ax.fill(e[:, 0], e[:, 1], color=col, alpha=0.13, lw=0, zorder=3)
    ax.plot(e[:, 0], e[:, 1], color=col, lw=2.4, zorder=4)
    ax.plot(*cm(d["est"]), "o", ms=11, color=col, mec="white", mew=1.8, zorder=6)
    a, b = sigmas(d["cov"])
    p_cm = cm(d["est"])
    out = p_cm / max(np.linalg.norm(p_cm), 1e-6)
    lab = p_cm + out * (a * 0.75 + 1.4)
    ax.text(lab[0], lab[1], cam_id[-1], ha="center", va="center",
            fontsize=15, fontweight="bold", color=col, zorder=7)
ax.plot(0, 0, "*", ms=26, color=D.INK, mec="white", mew=1.4, zorder=8)
ax.text(0.02, 0.03, "★  where the robot really was", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=12.5, color=D.INK)
ax.set_aspect("equal"); ax.grid(True, color="#f2f1ec", lw=0.7); ax.set_axisbelow(True)
lim = max(9.0, max(sigmas(M[c]["cov"])[0] for c in KEPT) * 2.4)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim * 0.72, lim * 0.72)
ax.set_xlabel("centimetres from the true position", fontsize=12.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("Every kept camera hands over a position and an ellipse",
             loc="left", fontsize=16, color=D.INK)

ax = fig.add_subplot(gs[1, 2:])
ax.axhline(0, color="#eceae4", lw=1); ax.axvline(0, color="#eceae4", lw=1)
for cam_id in KEPT:
    d = M[cam_id]; col = D.CAM_COLOUR[cam_id[-1]]
    e = ellipse(d["cov"]) * 100.0 + cm(d["est"])
    ax.fill(e[:, 0], e[:, 1], color=col, alpha=0.16, lw=0, zorder=3)
    ax.plot(*cm(d["est"]), "o", ms=7, color=col, zorder=5)
worst = None
for cam_id in DROPPED:
    d = M[cam_id]; col = D.CAM_COLOUR[cam_id[-1]]
    p = cm(d["est"]); worst = np.linalg.norm(p)
    e = ellipse(d["cov"]) * 100.0 + p
    ax.plot(e[:, 0], e[:, 1], color=col, lw=2.0, ls=(0, (4, 3)), zorder=4)
    ax.annotate("", xy=tuple(p), xytext=(0, 0), zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4, shrinkA=10, shrinkB=8))
    ax.plot(*p, "X", ms=15, color=col, mec="white", mew=1.6, zorder=6)
    ax.text(p[0], p[1] - 4.0, f"camera {cam_id[-1]} would have moved the robot\n"
                              f"{np.linalg.norm(p):.0f} cm — the check refused it",
            ha="center", va="top", fontsize=13, color=D.BAD, fontweight="bold")
ax.plot(0, 0, "*", ms=22, color=D.INK, mec="white", mew=1.2, zorder=8)
ax.set_aspect("equal"); ax.grid(True, color="#f2f1ec", lw=0.7); ax.set_axisbelow(True)
lim2 = max(20.0, (worst or 20.0) * 1.45)
ax.set_xlim(-lim2, lim2); ax.set_ylim(-lim2 * 0.72, lim2 * 0.72)
ax.set_xlabel("centimetres from the true position — same scene, zoomed out", fontsize=12.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("Zoomed out, to the scale of the refused reading",
             loc="left", fontsize=16, color=D.INK)

kept_txt = ", ".join(f"{c[-1]} {np.linalg.norm(cm(M[c]['est'])):.1f} cm"
                     for c in KEPT)
fig.suptitle("One moment, four cameras: what a camera reading actually is",
             x=0.005, ha="left", fontsize=21, color=D.INK)
fig.text(0.005, -0.035,
         f"One robot pose from the commissioning capture ({TX:.2f}, {TY:.2f}) m, facing "
         f"{math.degrees(YAW):.0f}°, seen by {len(M)} of the 5 cameras at once.  Dashed blue is the box "
         f"predicted from the robot's shape; solid is what the detector drew.\n"
         f"Every ellipse comes from the SAME one number — 0.76 px of detector noise — pushed through that "
         f"camera's own geometry, which is why they differ in size and direction without anything being fitted "
         f"per camera.\n"
         f"Errors of the kept readings: {kept_txt}.  Camera {DROPPED[0][-1]}'s robot was partly hidden at "
         f"{M[DROPPED[0]]['range_m']:.1f} m, so its box came out short: the check compares it against the "
         f"prediction and refuses it.\n"
         f"That is the difference between a measurement that is absent and one that is wrong — and it is "
         f"decided without ground truth.",
         fontsize=12, color=D.INK2, va="top", linespacing=1.55)
fig.savefig(OUT / "01_each_camera_hands_an_ellipse.png", dpi=170, bbox_inches="tight")
print(f"01: kept {KEPT} dropped {DROPPED}; worst refused reading {worst:.1f} cm")


# =========================================================================
# 02 — why the box is predicted rather than converted
# =========================================================================
CID = "camera_B"
d = M[CID]
cam = d["cam"]
u, v = d["det_uv"]
naive = np.array(cam.pixel_to_world(u, v)[:2])
HEADINGS = [YAW, YAW + math.pi / 3, YAW + 2 * math.pi / 3]
solved = [(th, solve_position(cam, (u, v), th, naive)) for th in HEADINGS]
solved = [(th, p) for th, p in solved if p is not None]
FOOT_L, FOOT_W = 0.80, 0.55

fig = plt.figure(figsize=(15.6, 7.4), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15])

ax = fig.add_subplot(gs[0, 0])
im = cv2.imread(str(d["image"]))[:, :, ::-1]
x0, y0, x1, y1 = crop(im, d["det_box"], pad_x=1.25, pad_y=1.35)
ax.imshow(im[y0:y1, x0:x1])
q = d["det_box"]
ax.add_patch(plt.Rectangle((q[0] - x0, q[1] - y0), q[2] - q[0], q[3] - q[1],
                           fill=False, edgecolor=D.GOOD, lw=3.0, zorder=4))
ax.plot(u - x0, v - y0, "o", ms=15, color=D.GOOD, mec="white", mew=2.2, zorder=6)
ax.annotate("this one pixel is everything\nthe detector tells us",
            xy=(u - x0, v - y0), xytext=(0.06, 0.12), textcoords="axes fraction",
            fontsize=13, color=D.INK, zorder=7,
            arrowprops=dict(arrowstyle="-|>", lw=2.2, color=D.INK, shrinkB=10))
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_edgecolor(D.CAM_COLOUR[CID[-1]]); s.set_linewidth(2.6)
ax.set_title(f"What camera {CID[-1]} reported", loc="left", fontsize=17, color=D.INK, pad=6)

ax = fig.add_subplot(gs[0, 1])
ax.axhline(0, color="#eceae4", lw=1); ax.axvline(0, color="#eceae4", lw=1)

# the robot itself, once, to scale, at the pose it was really in
c, s_ = math.cos(YAW), math.sin(YAW)
R0 = np.array([[c, -s_], [s_, c]])
foot = (R0 @ (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=float)
              * np.array([FOOT_L / 2, FOOT_W / 2])).T).T * 100.0
ax.fill(foot[:, 0], foot[:, 1], facecolor="#eeede8", edgecolor=D.MUTED, lw=1.2,
        alpha=0.9, zorder=2)
ax.text(0.98, 0.03, "the robot's own footprint, to scale", transform=ax.transAxes,
        fontsize=11.5, color=D.MUTED, ha="right", va="bottom", zorder=3)

# where the pixel puts the robot, once per candidate heading
for i, (th, p) in enumerate(solved):
    ax.plot(*cm(p), "o", ms=13, color=D.ROBOT, mec="white", mew=1.8, zorder=6)
sol = np.array([cm(p) for _t, p in solved])
ax.annotate(f"ask which position makes this pixel:\n"
            f"{len(solved)} candidate headings, {max(np.linalg.norm(sol - sol[0], axis=1)):.0f} cm apart",
            xy=tuple(sol.mean(axis=0)), xytext=(0.62, 0.80), textcoords="axes fraction",
            fontsize=12.5, color=D.ROBOT, fontweight="bold", ha="left", zorder=7,
            arrowprops=dict(arrowstyle="-|>", lw=2.0, color=D.ROBOT, shrinkB=12))

ax.annotate("", xy=tuple(cm(naive)), xytext=(0, 0), zorder=4,
            arrowprops=dict(arrowstyle="-|>", lw=2.6, color=D.BAD, shrinkA=8, shrinkB=8))
ax.plot(*cm(naive), "X", ms=17, color=D.BAD, mec="white", mew=1.8, zorder=6)
ax.text(*(cm(naive) + np.array([3.2, 0.0])),
        f"treat the pixel AS the robot:\n{np.linalg.norm(cm(naive)):.0f} cm out",
        ha="left", va="center", fontsize=13, color=D.BAD, fontweight="bold", zorder=7)
ax.plot(0, 0, "*", ms=26, color=D.INK, mec="white", mew=1.4, zorder=8)
ax.text(0.02, 0.03, "★  where the robot really was", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=12.5, color=D.INK)
ax.set_aspect("equal"); ax.grid(True, color="#f2f1ec", lw=0.7); ax.set_axisbelow(True)
span = max(62.0, np.linalg.norm(cm(naive)) * 1.7)
ax.set_xlim(-span, span); ax.set_ylim(-span * 0.75, span * 0.75)
ax.set_xlabel("centimetres from the true position", fontsize=12.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("What that one pixel can, and cannot, tell you",
             loc="left", fontsize=17, color=D.INK)

spread = max(np.linalg.norm(cm(p) - cm(solved[0][1])) for _t, p in solved)
fig.suptitle("A box cannot be turned into a position. A position can be turned into a box.",
             x=0.004, ha="left", fontsize=21, color=D.INK)
fig.text(0.004, -0.03,
         f"Camera {CID[-1]}, {d['range_m']:.1f} m away.  Taking the box's bottom-centre to BE the robot puts it "
         f"{np.linalg.norm(cm(naive)):.0f} cm out, because that pixel is the nearest point of the robot's "
         f"body to the camera, not its centre.\n"
         f"Asking instead which position would produce that pixel needs a heading — and the answers are "
         f"{spread:.0f} cm apart across headings 60° apart, several times the detector's own {2.2:.1f} cm of scatter.  "
         f"The box is also blind to a 180° flip: front and back give the same answer.\n"
         f"So the heading comes from the robot's own odometry, and the measurement is made the other way round: "
         f"predict the box from the believed pose, compare it with the box the detector drew, and convert only "
         f"the DISAGREEMENT into centimetres.",
         fontsize=12, color=D.INK2, va="top", linespacing=1.55)
fig.savefig(OUT / "02_why_not_convert_the_pixel.png", dpi=170, bbox_inches="tight")
print(f"02: naive {np.linalg.norm(cm(naive)):.1f} cm off; heading spread {spread:.1f} cm")


# =========================================================================
# 03 — what the four rules do with the same three readings
# =========================================================================
EPS = 1e-6
def _cos_to_floor(cam, xy):
    """cos of the angle between this camera's line of sight and the floor's normal.

    A steep look scores 1, a grazing look scores 0. Frozen as written: this is the F2
    heuristic's only ingredient besides range, and it is deliberately not tuned.
    """
    ray = np.array([xy[0], xy[1], 0.0]) - np.asarray(cam.cam_pos, dtype=float)
    return abs(ray[2]) / max(np.linalg.norm(ray), EPS)


def rules(kept):
    est = {c: M[c]["est"] for c in kept}
    cov = {c: M[c]["cov"] for c in kept}
    n = len(kept)
    out = {}

    best = min(kept, key=lambda c: np.trace(cov[c]))
    out["F1  the single best camera"] = (est[best], cov[best],
                                        f"smallest ellipse: camera {best[-1]}")

    q = {c: _cos_to_floor(M[c]["cam"], est[c]) / (M[c]["range_m"] ** 2 + EPS) for c in kept}
    tot = sum(q.values())
    w = {c: q[c] / tot for c in kept}
    mu = sum(w[c] * est[c] for c in kept)
    S = sum((w[c] ** 2) * cov[c] for c in kept)
    out["F2  weighted by distance and angle"] = (
        mu, S, " ".join(f"{c[-1]}:{w[c]*100:.0f}%" for c in kept))

    info = sum(np.linalg.inv(cov[c]) for c in kept)
    vec = sum(np.linalg.inv(cov[c]) @ est[c] for c in kept)
    S3 = np.linalg.inv(info)
    out["F3  precisions add"] = (S3 @ vec, S3, f"all {n} cameras, independent")
    batch = [MapObservation(
        camera_id=c,
        timestamp_s=0.0,
        xy_m=tuple(est[c]),
        covariance_m2=tuple(tuple(row) for row in cov[c]),
        quality=CameraQuality(camera_id=c),
        source="commissioning_figure",
    ) for c in kept]
    joint_mean, joint_covariance = joint_network_estimate_2d(batch)
    out["F4  joint network estimator"] = (
        np.asarray(joint_mean), np.asarray(joint_covariance),
        f"one robust estimate from all {n} cameras")
    return out


RULES = rules(KEPT)
fig, axes = plt.subplots(1, len(RULES), figsize=(19.2, 5.9), constrained_layout=True)
lim = 4.6
for ax, (name, (mu, S, note)) in zip(axes, RULES.items()):
    ax.axhline(0, color="#eceae4", lw=1); ax.axvline(0, color="#eceae4", lw=1)
    for cam_id in KEPT:
        d = M[cam_id]
        e = ellipse(d["cov"]) * 100.0 + cm(d["est"])
        ax.plot(e[:, 0], e[:, 1], color=D.CAM_COLOUR[cam_id[-1]], lw=1.3, alpha=0.55, zorder=3)
    e = ellipse(S) * 100.0 + cm(mu)
    ax.fill(e[:, 0], e[:, 1], color=D.ROBOT, alpha=0.16, lw=0, zorder=4)
    ax.plot(e[:, 0], e[:, 1], color=D.ROBOT, lw=2.8, zorder=5)
    ax.plot(*cm(mu), "o", ms=12, color=D.ROBOT, mec="white", mew=1.8, zorder=6)
    ax.plot(0, 0, "*", ms=22, color=D.INK, mec="white", mew=1.2, zorder=7)
    claim = math.sqrt(np.trace(S) / 2) * 100
    err = np.linalg.norm(cm(mu))
    ax.set_title(name, loc="left", fontsize=15, color=D.INK, pad=6)
    ax.set_xlabel(f"claims ± {claim:.1f} cm    ·    is {err:.1f} cm out\n{note}",
                  fontsize=12.5, color=D.INK2, labelpad=8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.grid(True, color="#f4f3ee", lw=0.6); ax.set_axisbelow(True)
    ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_yticks([-4, -2, 0, 2, 4])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("centimetres from the true position", fontsize=12.5)

c3 = math.sqrt(np.trace(RULES["F3  precisions add"][1]) / 2) * 100
c4 = math.sqrt(np.trace(RULES["F4  joint network estimator"][1]) / 2) * 100
fig.suptitle("The same three readings, four ways of combining them",
             x=0.004, ha="left", fontsize=21, color=D.INK)
fig.text(0.004, -0.055,
         f"The one moment from the previous figures: cameras "
         f"{', '.join(c[-1] for c in KEPT)} kept, camera {DROPPED[0][-1]} refused.  Thin ellipses are the "
         f"individual cameras; the thick one is what each rule hands the filter.\n"
         f"F3 assumes the views are independent. F4 solves one robust batch problem and then fits one "
         f"network ellipse from both within-view uncertainty and the disagreement visible in this batch "
         f"(± {c3:.1f} cm against ± {c4:.1f} cm here).\n"
         f"MECHANISM ONLY. One moment out of a 30 m route is not evidence about any rule: a single reading can "
         f"flatter a bad rule and embarrass a good one.  Nothing here is an experimental result.",
         fontsize=12, color=D.INK2, va="top", linespacing=1.55)
fig.savefig(OUT / "03_four_rules_one_moment.png", dpi=170, bbox_inches="tight")
for name, (mu, S, _n) in RULES.items():
    print(f"03: {name:38s} claims {math.sqrt(np.trace(S)/2)*100:4.1f} cm, "
          f"is {np.linalg.norm(cm(mu)):4.1f} cm out")


# =========================================================================
# the numbers this storyline quotes
# =========================================================================
import json  # noqa: E402
numbers = {
    "moment": {"x_m": TX, "y_m": TY, "yaw_deg": math.degrees(YAW),
               "cameras_that_saw_it": len(M), "kept": [c[-1] for c in KEPT],
               "refused": [c[-1] for c in DROPPED]},
    "sigma_px": SIGMA_PX,
    "per_camera": {c[-1]: {"range_m": round(M[c]["range_m"], 2),
                           "confidence": round(M[c]["conf"], 3),
                           "residual_px": [round(v, 2) for v in M[c]["residual_px"]],
                           "sigma_major_cm": round(sigmas(M[c]["cov"])[0], 2),
                           "sigma_minor_cm": round(sigmas(M[c]["cov"])[1], 2),
                           "error_cm": round(float(np.linalg.norm(cm(M[c]["est"]))), 2),
                           "admitted": ADMIT[c][0],
                           "refused_because": ADMIT[c][1]} for c in M},
    "pixel_conversion": {
        "camera": CID,
        "bottom_centre_as_robot_error_cm": round(float(np.linalg.norm(cm(naive))), 1),
        "spread_across_headings_cm": round(float(spread), 1),
        "headings_deg": [round(math.degrees(t) % 360, 1) for t, _p in solved]},
    "rules_at_this_moment": {
        name: {"claimed_sigma_cm": round(math.sqrt(np.trace(S) / 2) * 100, 2),
               "error_cm": round(float(np.linalg.norm(cm(mu))), 2)}
        for name, (mu, S, _n) in RULES.items()},
    "caveat": "one commissioned moment, mechanism only; no arm is evaluated here",
}
(OUT / "numbers.json").write_text(json.dumps(numbers, indent=2))
print(f"wrote {OUT/'numbers.json'}")
