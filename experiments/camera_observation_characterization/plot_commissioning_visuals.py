#!/usr/bin/env python3
"""Presentation-ready figures from the real 2026-09-02 repeated-state R capture.

The stationary figures use Camera B's fixed-offset interpretation of 40 fresh captures at
each of six predeclared states.  The driven comparison uses the truth-free camera-disagreement
estimate on the schema-5 O2 overlap-rich diagnostic drive.  The latter is deliberately not
called a Bayesian posterior: joint Bayesian R inference has not yet been implemented.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Wedge
import numpy as np


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "logs/studies/camera_observation_characterization_20260831/12_commissioning_R_visuals"
REPEAT_CAPTURE = REPO / "logs/perception_datasets/warehouse_v2_bbox_repeat_panel_20260902"
REPEAT_TABLE = REPEAT_CAPTURE / "observation_interpretations.csv"
REPEAT_MANIFEST = REPEAT_CAPTURE / "capture_manifest.json"
DRIVE_ESTIMATE = OUT / "drive_disagreement_estimate_O2.json"
CAMERA = "camera_B"
METHOD = "fixed"
STYLE_DIR = REPO / "experiments/deck_figures"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))
import style as D  # noqa: E402


BLUE = D.ROBOT
GREEN = D.GOOD
ORANGE = D.BAD
VIOLET = D.OLD
INK = D.INK
MUTED = D.MUTED
SURF = D.SURF
PALE_BLUE = "#e9f2fb"
PALE_GREEN = "#e8f6f0"
PALE_ORANGE = "#fff0e9"
PALE_GREY = "#f0f0ed"


def canvas(title: str, subtitle: str | None = None):
    fig = plt.figure(figsize=(16, 9), facecolor=SURF)
    fig.text(0.04, 0.94, title, fontsize=28, fontweight="bold", ha="left", va="top")
    if subtitle:
        fig.text(0.04, 0.888, subtitle, fontsize=14, color=D.INK2, ha="left", va="top")
    return fig


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, facecolor=SURF)
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repeat_groups():
    if not REPEAT_TABLE.is_file() or not REPEAT_MANIFEST.is_file():
        raise RuntimeError(f"Missing complete repeat capture under {REPEAT_CAPTURE}")
    manifest = json.loads(REPEAT_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or int(manifest.get("failed_batches", -1)) != 0:
        raise RuntimeError("Repeat capture is not complete and failure-free")
    rows = [row for row in csv.DictReader(REPEAT_TABLE.open(encoding="utf-8"))
            if row["camera_id"] == CAMERA]
    labels = list(manifest["plan"]["pose_labels"])
    groups = {}
    for label in labels:
        attempts = [row for row in rows if row["dataset_split"] == label]
        hits = [row for row in attempts if row[f"{METHOD}_valid"] == "1"]
        residuals = np.asarray([[float(row[f"{METHOD}_dx"]), float(row[f"{METHOD}_dy"])]
                                for row in hits], dtype=float)
        pixels = np.asarray([[float(row["u_bbox_bottom"]), float(row["v_bbox_bottom"])]
                             for row in hits], dtype=float)
        cov = (np.cov((residuals - residuals.mean(axis=0)).T, ddof=1)
               if len(residuals) > 1 else np.zeros((2, 2)))
        groups[label] = {
            "attempts": attempts,
            "hits": hits,
            "residuals": residuals,
            "pixels": pixels,
            "mean": residuals.mean(axis=0),
            "cov": cov,
            "bias_cm": float(np.linalg.norm(residuals.mean(axis=0)) * 100.0),
            "repeat_sigma_cm": float(math.sqrt(max(np.trace(cov), 0.0) / 2.0) * 100.0),
            "unique_images": len({row["image_sha1"] for row in attempts}),
            "truth": np.asarray([float(hits[0]["robot_x"]), float(hits[0]["robot_y"])]),
            "yaw": float(hits[0]["robot_yaw"]),
            "range_m": float(hits[0]["camera_range_m"]),
            "mean_xy": np.asarray([[float(row[f"{METHOD}_x"]), float(row[f"{METHOD}_y"])]
                                    for row in hits]).mean(axis=0),
        }
    return manifest, groups


def clean(ax, xlim=(0, 1), ylim=(0, 1)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(SURF)


def box(ax, xy, wh, text, face=PALE_GREY, edge="#c8c7c1", fontsize=16,
        weight="bold", radius=0.025, align="center", color=INK, zorder=3):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=2.0, zorder=zorder,
    )
    ax.add_patch(patch)
    ha = "center" if align == "center" else "left"
    tx = x + w / 2 if align == "center" else x + 0.035 * w
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fontsize,
            fontweight=weight, color=color, linespacing=1.2, zorder=zorder + 1)
    return patch


def arrow(ax, start, end, color=MUTED, lw=2.6, style="-|>", zorder=2,
          connectionstyle="arc3"):
    patch = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=18,
                            linewidth=lw, color=color, connectionstyle=connectionstyle,
                            shrinkA=3, shrinkB=3, zorder=zorder)
    ax.add_patch(patch)
    return patch


def covariance_ellipse(points: np.ndarray, center: np.ndarray, scale=2.0):
    cov = np.cov((points - points.mean(axis=0)).T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    return Ellipse(center, width=2 * scale * math.sqrt(vals[0]),
                   height=2 * scale * math.sqrt(vals[1]), angle=angle)


def fixed_state_repeat():
    _manifest, groups = repeat_groups()
    result = groups["image_centre"]
    residual_cm = result["residuals"] * 100.0
    mean_cm = result["mean"] * 100.0
    fig = canvas(
        "What did one camera measurement look like at a fixed robot state?",
        "Camera B · image-centre state · fixed 30.9 cm correction · 40 fresh Gazebo captures",
    )
    ax = fig.add_axes([0.04, 0.13, 0.58, 0.70])
    lo = np.minimum(np.array([0.0, 0.0]), mean_cm) - 8.0
    hi = np.maximum(np.array([0.0, 0.0]), mean_cm) + 8.0
    span = max(float(np.max(hi - lo)), 30.0)
    centre = 0.5 * (lo + hi)
    clean(ax, (centre[0] - span / 2, centre[0] + span / 2),
          (centre[1] - span / 2, centre[1] + span / 2))
    ax.set_aspect("equal")
    ax.axhline(0, color="#dddcd7", lw=1.2, zorder=0)
    ax.axvline(0, color="#dddcd7", lw=1.2, zorder=0)
    # Plot every real row without jitter: they coincide exactly in this deterministic scene.
    ax.scatter(residual_cm[:, 0], residual_cm[:, 1], s=105, color=GREEN, alpha=0.18,
               edgecolor="none", zorder=3, label="40 corrected measurements")
    ax.scatter([mean_cm[0]], [mean_cm[1]], s=155, facecolor=GREEN, edgecolor="white",
               linewidth=1.5, zorder=7)
    ax.scatter([0], [0], marker="+", s=340, linewidth=3.8, color=INK, zorder=8,
               label="reference position")
    arrow(ax, (0.2, -0.1), tuple(mean_cm * 0.94), color=ORANGE, lw=3.2, zorder=7)
    midpoint = mean_cm * 0.52
    ax.text(midpoint[0] - 1.0, midpoint[1] + 1.4, r"mean residual  $\bar{r}$",
            color=ORANGE, fontsize=14, fontweight="bold", rotation=-30)
    ax.annotate(
        "40/40 measurements coincide\n(no covariance ellipse to draw)",
        xy=mean_cm, xytext=(mean_cm[0] - 3.0, mean_cm[1] + 8.0),
        ha="center", fontsize=14, color=GREEN, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.0),
    )
    ax.set_xlabel("Camera-reading residual x (cm)", fontsize=13)
    ax.set_ylabel("Camera-reading residual y (cm)", fontsize=13)
    ax.set_xticks(np.arange(math.floor(ax.get_xlim()[0] / 10) * 10,
                            math.ceil(ax.get_xlim()[1] / 10) * 10 + 1, 10))
    ax.set_yticks(np.arange(math.floor(ax.get_ylim()[0] / 10) * 10,
                            math.ceil(ax.get_ylim()[1] / 10) * 10 + 1, 10))
    ax.tick_params(labelsize=11)
    ax.legend(loc="upper left", frameon=False, fontsize=12)

    right = fig.add_axes([0.66, 0.16, 0.30, 0.65])
    clean(right)
    box(right, (0.02, 0.68), (0.96, 0.21),
        r"$\bar{r}=\frac{1}{N}\sum_k r_k$" "\n"
        f"bias magnitude = {result['bias_cm']:.2f} cm",
        face=PALE_ORANGE, edge=ORANGE, fontsize=20)
    box(right, (0.02, 0.37), (0.96, 0.23),
        r"$R_{\mathrm{emp}}=\operatorname{Cov}(r_k-\bar{r})$" "\n"
        f"repeat sigma = {result['repeat_sigma_cm']:.2f} cm",
        face=PALE_GREEN, edge=GREEN, fontsize=18)
    right.text(0.50, 0.19, "Mean offset is measurable", ha="center", va="center",
               fontsize=18, fontweight="bold", color=ORANGE)
    right.text(0.50, 0.09, r"Stationary Gazebo spread is zero", ha="center", va="center",
               fontsize=18, fontweight="bold", color=GREEN)
    fig.text(0.50, 0.045,
             "Actual result: fresh timestamps, identical pixels, identical YOLO box, identical ground-plane reading.",
             ha="center", fontsize=13, color=D.INK2)
    save(fig, "30_fixed_state_repeat_design.png")


def geometry_repeat_panel():
    _manifest, groups = repeat_groups()
    fig = canvas(
        "Did Camera-B repeatability change with viewing geometry?",
        "Six predeclared states · 40 fresh captures per state · fixed 30.9 cm correction",
    )
    ax = fig.add_axes([0.035, 0.12, 0.55, 0.73])
    D.draw_warehouse(ax, D.layout(), show_cameras=True, camera_labels=True, rack_alpha=0.58)
    order = ["near", "medium", "far", "image_centre", "image_edge", "changed_heading"]
    labels = {
        "near": "near", "medium": "medium", "far": "far",
        "image_centre": "image centre", "image_edge": "image edge",
        "changed_heading": "same place,\nchanged heading",
    }
    arrow_gain = 3.0
    map_order = ["near", "medium", "far", "image_centre", "image_edge"]
    map_labels = {
        "near": "1  near", "medium": "2  medium", "far": "3  far",
        "image_centre": "4 / 6  image centre\ntwo headings", "image_edge": "5  image edge",
    }
    text_offsets = {
        "near": (-34, 26), "medium": (24, -30), "far": (22, 22),
        "image_centre": (55, 5), "image_edge": (-5, 30),
    }
    for name in map_order:
        result = groups[name]
        truth = result["truth"]
        mean = result["mean"]
        ax.scatter([truth[0]], [truth[1]], marker="+", s=180, linewidth=2.8,
                   color=INK, zorder=8)
        ax.annotate("", xy=truth + arrow_gain * mean, xytext=truth,
                    arrowprops=dict(arrowstyle="-|>", lw=2.6, color=ORANGE), zorder=7)
        ax.scatter([truth[0] + arrow_gain * mean[0]], [truth[1] + arrow_gain * mean[1]],
                   s=58, facecolor=GREEN, edgecolor="white", linewidth=0.8, zorder=9)
        ax.annotate(map_labels[name], xy=truth, xytext=text_offsets[name],
                    textcoords="offset points", fontsize=10.5, fontweight="bold",
                    ha="center", va="center", zorder=11,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=SURF,
                              edgecolor="none", alpha=0.86))
    # The changed-heading state shares the image-centre position; its mean point is almost
    # identical, so drawing it again would imply a sixth spatial site.
    heading = groups["changed_heading"]
    truth = heading["truth"]
    mean = heading["mean"]
    ax.scatter([truth[0] + arrow_gain * mean[0]], [truth[1] + arrow_gain * mean[1]],
               s=25, facecolor=VIOLET, edgecolor="white", linewidth=0.5, zorder=10)
    fig.text(0.31, 0.085,
             "Black + = reference   Orange arrow = mean residual (×3)   Green dot = all 40 readings",
             fontsize=10.5, color=D.INK2, ha="center")

    right = fig.add_axes([0.62, 0.17, 0.34, 0.62])
    names = order[::-1]
    y = np.arange(len(names))
    bias = [groups[name]["bias_cm"] for name in names]
    spread = [groups[name]["repeat_sigma_cm"] for name in names]
    right.barh(y + 0.16, bias, height=0.28, color=ORANGE, alpha=0.85,
               label="mean residual magnitude (bias)")
    right.barh(y - 0.16, spread, height=0.28, color=GREEN,
               label="within-state repeat sigma")
    right.set_yticks(y, [labels[name].replace("\n", " ") for name in names], fontsize=12)
    right.set_xlabel("Centimetres", fontsize=13)
    right.grid(axis="x", color="#dddcd7", lw=0.9)
    right.spines[["top", "right", "left"]].set_visible(False)
    right.legend(loc="lower right", frameon=False, fontsize=11)
    right.set_title("Bias changes; repeated spread does not", fontsize=17,
                    fontweight="bold", pad=12)
    for yi, value in zip(y, bias):
        right.text(value + 0.8, yi + 0.16, f"{value:.1f}", va="center", fontsize=11)
        right.text(0.6, yi - 0.16, "0.00", va="center", fontsize=10.5,
                   color=GREEN, fontweight="bold")
    fig.text(0.79, 0.070,
             "All six empirical covariance ellipses collapse to a point.\n"
             "This simulator test cannot justify a non-zero stationary R.",
             ha="center", fontsize=15, color=GREEN, fontweight="bold", linespacing=1.25)
    save(fig, "31_geometry_repeat_panel_design.png")


def bayesian_commissioning_flow():
    _manifest, groups = repeat_groups()
    total_hits = sum(len(result["hits"]) for result in groups.values())
    fig = canvas(
        "How driven Bayesian R commissioning would work",
        "The filter must separate uncertain robot state from uncertain camera measurements",
    )
    ax = fig.add_axes([0.04, 0.10, 0.58, 0.76])
    clean(ax)

    box(ax, (0.04, 0.76), (0.34, 0.13), "odometry / controls\n$u_{1:T}$",
        face=PALE_BLUE, edge=BLUE, fontsize=17)
    box(ax, (0.04, 0.50), (0.34, 0.14), "state prediction\n$\hat{s}^{-}_k,\ P^{-}_k$",
        face=PALE_BLUE, edge=BLUE, fontsize=17)
    arrow(ax, (0.21, 0.755), (0.21, 0.65), color=BLUE)

    box(ax, (0.60, 0.76), (0.34, 0.13), "camera observation\n$z_{i,k}$",
        face=PALE_GREEN, edge=GREEN, fontsize=17)
    box(ax, (0.33, 0.28), (0.38, 0.14),
        "innovation\n" + r"$\nu_{i,k}=z_{i,k}-h_i(\hat{s}^{-}_k)$",
        face="#faf8ee", edge="#c7af54", fontsize=16)
    arrow(ax, (0.21, 0.49), (0.40, 0.43), color=BLUE)
    arrow(ax, (0.77, 0.755), (0.63, 0.43), color=GREEN)

    box(ax, (0.24, 0.035), (0.56, 0.15),
        "joint Bayesian inference\nrobot trajectory  $s_{1:T}$   +   camera covariance  $R_i$",
        face="#f1eefb", edge=VIOLET, fontsize=17)
    arrow(ax, (0.52, 0.275), (0.52, 0.195), color=VIOLET, lw=3.0)

    right = fig.add_axes([0.66, 0.14, 0.30, 0.68])
    clean(right)
    box(right, (0.01, 0.69), (0.98, 0.24),
        r"$s_{k+1}\sim p(s_{k+1}\mid s_k,u_k,Q)$" "\n\n"
        r"$z_{i,k}\sim\mathcal{N}(h_i(s_k)+b_i,\,R_i)$",
        face=PALE_GREY, edge="#bdbcb6", fontsize=17, weight="normal")
    box(right, (0.01, 0.47), (0.98, 0.12),
        r"$p(s_{1:T},R_i\mid z_{1:T},u_{1:T})$",
        face="#f1eefb", edge=VIOLET, fontsize=19)
    right.text(0.5, 0.37, "The innovation contains both:", ha="center",
               fontsize=15, fontweight="bold")
    right.text(0.5, 0.30, r"belief uncertainty  $H P^- H^T$", ha="center",
               fontsize=15, color=BLUE)
    right.text(0.5, 0.235, r"camera uncertainty  $R_i$", ha="center",
               fontsize=15, color=GREEN)
    box(right, (0.04, 0.035), (0.92, 0.13),
        f"Actual fixed-state evidence\n{total_hits} Camera-B hits\nrepeat sigma = 0.00 cm",
        face=PALE_GREEN, edge=GREEN, fontsize=12.5)
    fig.text(0.50, 0.035,
             "PROPOSED METHOD — no joint Bayesian posterior over R exists in the repository yet",
             ha="center", fontsize=12, color=MUTED, fontweight="bold")
    save(fig, "32_joint_bayesian_commissioning_flow.png")


def validation_design():
    _manifest, groups = repeat_groups()
    if not DRIVE_ESTIMATE.is_file():
        raise RuntimeError(f"Missing driven disagreement estimate: {DRIVE_ESTIMATE}")
    driven = json.loads(DRIVE_ESTIMATE.read_text(encoding="utf-8"))
    driven_sigma_px = float(driven["calibration"]["sigma_px_by_camera"][CAMERA])
    equations = int(driven["equations"])
    centred_pixels = np.concatenate(
        [result["pixels"] - result["pixels"].mean(axis=0) for result in groups.values()], axis=0)
    stationary_sigma_px = float(np.std(centred_pixels, axis=0).mean())
    fig = canvas(
        "Driven camera disagreement is not stationary detector noise",
        "Camera B · same fixed-offset observation method · actual Gazebo data",
    )
    left = fig.add_axes([0.055, 0.24, 0.35, 0.56])
    right = fig.add_axes([0.44, 0.24, 0.35, 0.56])
    for ax in (left, right):
        clean(ax, (-25, 25), (-25, 25))
        ax.set_aspect("equal")
        ax.axhline(0, color="#dddcd7", lw=1)
        ax.axvline(0, color="#dddcd7", lw=1)
        ax.set_xticks([-20, -10, 0, 10, 20])
        ax.set_yticks([-20, -10, 0, 10, 20])
        ax.tick_params(labelsize=10)
        ax.set_xlabel("centred pixel residual u (px)", fontsize=11)
        ax.set_ylabel("centred pixel residual v (px)", fontsize=11)

    left.set_title("Driven disagreement estimate", fontsize=18, fontweight="bold", pad=14)
    radius95 = math.sqrt(5.991) * driven_sigma_px
    learned = Ellipse((0, 0), 2 * radius95, 2 * radius95, facecolor=PALE_BLUE,
                      edgecolor=BLUE, lw=4)
    left.add_patch(learned)
    left.text(0, 2.5, f"$\hat{{\sigma}}_B$ = {driven_sigma_px:.2f} px", ha="center",
              fontsize=18, fontweight="bold", color=BLUE)
    left.text(0, -3.0, f"{equations} multi-camera comparisons", ha="center",
              fontsize=13, color=BLUE)
    left.text(0, -23.5, "O2 overlap-rich drive · diagnostic only\nnetwork-disagreement estimate, not Bayesian",
              ha="center", fontsize=11, color=MUTED)

    right.set_title("Independent fixed-state repeats", fontsize=18, fontweight="bold", pad=14)
    right.scatter(centred_pixels[:, 0], centred_pixels[:, 1], s=105,
                  color=GREEN, alpha=0.10, edgecolor="none")
    right.scatter([0], [0], s=140, color=GREEN, edgecolor="white", linewidth=1.2)
    right.annotate("240/240 Camera-B hits\ncoincide at exactly zero",
                   xy=(0, 0), xytext=(0, 10), ha="center", fontsize=15,
                   fontweight="bold", color=GREEN,
                   arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2))
    right.text(0, -17, f"empirical $\sigma_B$ = {stationary_sigma_px:.2f} px",
               ha="center", fontsize=17, color=GREEN, fontweight="bold")
    right.text(0, -23.5, "6 states · 40 fresh captures each\npixel arrays repeat exactly within each state",
               ha="center", fontsize=11, color=MUTED)

    compare = fig.add_axes([0.82, 0.23, 0.15, 0.57])
    clean(compare)
    compare.text(0.5, 0.95, "Actual comparison", ha="center", fontsize=17, fontweight="bold")
    compare.text(0.5, 0.73, f"{driven_sigma_px:.2f} px", ha="center", fontsize=24,
                 color=BLUE, fontweight="bold")
    compare.text(0.5, 0.64, "driven disagreement", ha="center", fontsize=12)
    compare.text(0.5, 0.51, "≠", ha="center", fontsize=30, color=ORANGE, fontweight="bold")
    compare.text(0.5, 0.38, f"{stationary_sigma_px:.2f} px", ha="center", fontsize=24,
                 color=GREEN, fontweight="bold")
    compare.text(0.5, 0.29, "stationary repeat", ha="center", fontsize=12)
    compare.text(0.5, 0.10, "Not the same\nrandom process", ha="center", fontsize=15,
                 color=ORANGE, fontweight="bold")

    decision = fig.add_axes([0.08, 0.04, 0.84, 0.13])
    clean(decision)
    box(decision, (0.05, 0.05), (0.90, 0.78),
        "Motion/state/model variation is being absorbed by the driven estimate\n"
        "→ do not interpret it as stationary detector R",
        face=PALE_ORANGE, edge=ORANGE, fontsize=17)
    save(fig, "33_driven_vs_stationary_R.png")


def commissioned_model_overview():
    fig = canvas(
        "Commissioned camera model",
        "Measurement quality and measurement availability answer different questions",
    )
    ax = fig.add_axes([0.045, 0.10, 0.91, 0.76])
    clean(ax)

    box(ax, (0.34, 0.82), (0.32, 0.12), "camera observation",
        face=PALE_GREY, edge="#bdbcb6", fontsize=18)
    arrow(ax, (0.50, 0.815), (0.50, 0.72), color=MUTED)
    box(ax, (0.32, 0.59), (0.36, 0.12), "bias-corrected measurement\n$z_i-b_i$",
        face=PALE_GREEN, edge=GREEN, fontsize=18)

    arrow(ax, (0.45, 0.585), (0.25, 0.46), color=GREEN)
    arrow(ax, (0.55, 0.585), (0.75, 0.46), color=BLUE)
    box(ax, (0.07, 0.31), (0.35, 0.14),
        "$R_i$ commissioning\n$R_{\mathrm{hit},i}$  | usable observation",
        face=PALE_GREEN, edge=GREEN, fontsize=17)
    box(ax, (0.58, 0.31), (0.35, 0.14),
        "availability\n$q_i(s)=P(\mathrm{usable}\mid s)$",
        face=PALE_BLUE, edge=BLUE, fontsize=17)

    arrow(ax, (0.25, 0.30), (0.40, 0.19), color=GREEN)
    arrow(ax, (0.75, 0.30), (0.60, 0.19), color=BLUE)
    box(ax, (0.34, 0.06), (0.32, 0.13), "camera sensor model\n$\{b_i,R_{\mathrm{hit},i},q_i\}$",
        face="#f1eefb", edge=VIOLET, fontsize=17)

    # Two consumers, intentionally separated by a vertical rule.
    ax.plot([0.70, 0.70], [0.02, 0.23], color="#d2d1cc", lw=1.5)
    box(ax, (0.73, 0.08), (0.24, 0.12), "actual hit\nEKF update uses $R_{\mathrm{hit},i}$",
        face=PALE_GREEN, edge=GREEN, fontsize=14)
    arrow(ax, (0.66, 0.125), (0.72, 0.14), color=GREEN)
    box(ax, (0.03, 0.08), (0.24, 0.12), "future observation\nplanner uses $q_i$ and $R_{\mathrm{hit},i}$",
        face=PALE_BLUE, edge=BLUE, fontsize=14)
    arrow(ax, (0.34, 0.125), (0.28, 0.14), color=BLUE)

    fig.text(0.50, 0.035,
             "A missed detection creates no filter update; it is not represented by an enormous R.",
             ha="center", fontsize=14, color=ORANGE, fontweight="bold")
    save(fig, "34_commissioned_camera_model_overview.png")


def main():
    fixed_state_repeat()
    geometry_repeat_panel()
    bayesian_commissioning_flow()
    validation_design()
    commissioned_model_overview()
    manifest, groups = repeat_groups()
    driven = json.loads(DRIVE_ESTIMATE.read_text(encoding="utf-8"))
    summary = {
        "status": "complete_actual_data",
        "schema": "camera_R_presentation_visuals.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stationary_capture": str(REPEAT_CAPTURE),
        "stationary_capture_manifest_sha256": sha256(REPEAT_MANIFEST),
        "stationary_interpretations_sha256": sha256(REPEAT_TABLE),
        "camera": CAMERA,
        "observation_method": "fixed 30.9 cm radial correction",
        "states": {
            name: {
                "attempts": len(result["attempts"]),
                "usable_hits": len(result["hits"]),
                "unique_pixel_arrays": result["unique_images"],
                "range_m": result["range_m"],
                "yaw_rad": result["yaw"],
                "bias_cm": result["bias_cm"],
                "repeat_sigma_cm": result["repeat_sigma_cm"],
                "covariance_m2": result["cov"].tolist(),
            }
            for name, result in groups.items()
        },
        "driven_comparison": {
            "source": "schema-5 O2 overlap-rich diagnostic drive",
            "estimator": "truth-free multi-camera disagreement; not a Bayesian posterior",
            "artifact": str(DRIVE_ESTIMATE),
            "artifact_sha256": sha256(DRIVE_ESTIMATE),
            "equations": int(driven["equations"]),
            "camera_B_sigma_px": float(
                driven["calibration"]["sigma_px_by_camera"][CAMERA]),
            "evidence_status": "diagnostic_only",
        },
        "verdict": (
            "All 40 Camera-B readings coincide at every stationary state. Gazebo contributes "
            "zero within-state pixel and ground-plane repeat variance under this protocol; "
            "the non-zero driven disagreement estimate therefore includes state, geometry, "
            "model, or inter-camera variation and is not stationary detector noise."
        ),
        "figures": [
            "30_fixed_state_repeat_design.png",
            "31_geometry_repeat_panel_design.png",
            "32_joint_bayesian_commissioning_flow.png",
            "33_driven_vs_stationary_R.png",
            "34_commissioned_camera_model_overview.png",
        ],
        "capture_plan": manifest["plan"],
    }
    (OUT / "actual_data_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
