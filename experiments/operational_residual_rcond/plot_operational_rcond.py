#!/usr/bin/env python3
"""Figures for EXP-RCOND — the per-camera-vs-pooled conditional covariance NULL.

This script PLOTS ONLY. It re-runs no estimation, refits nothing and recomputes no
metric: every number drawn comes verbatim out of the three locked evidence JSONs

    logs/studies/operational_residual_rcond/exp1_timing_and_coverage/timing_and_coverage.json
    logs/studies/operational_residual_rcond/exp2_operational_rcond/operational_rcond.json
    logs/studies/operational_residual_rcond/exp3_two_dof_rcond/operational_rcond.json

The only arithmetic performed here is presentational: differences between two
printed numbers, eigen-decompositions of the printed 2x2 covariances so they can
be drawn as ellipses, and a binomial standard error used purely as a *scale
reference* on the coverage axis (labelled as such on the figure).

The result being drawn is a NULL. The registry's own `next_action` reads
"No promotion; per-camera conditional covariance tied or lost to pooled
covariance." The figures are laid out so that verdict is the first thing read:

  fig_r1  head-to-head predictive scores, both calibration arms, all four metrics
  fig_r2  what the per-camera R_cond actually is, and how unstable it is
  fig_r3  (exp3 dir) the 2-DOF projection fix repairs the belief — and the
          per-camera-vs-pooled verdict is still a tie

Two arms appear throughout, and they differ ONLY in which projection calibration
was in force (see exp3 RESULTS.md):

  v2  the deployed along-bearing calibration      -> exp2_operational_rcond/
  v3  the gated 2-DOF calibration                 -> exp3_two_dof_rcond/

Outputs -> logs/studies/operational_residual_rcond/exp{2,3}_*/
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, Rectangle  # noqa: E402

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
STUDY = REPO / "logs" / "studies" / "operational_residual_rcond"
OUT_EXP2 = STUDY / "exp2_operational_rcond"
OUT_EXP3 = STUDY / "exp3_two_dof_rcond"

CAMERAS = ("camera_A", "camera_B", "camera_C", "camera_D")

# Okabe-Ito, colourblind-safe and legible on white paper.
C_POOLED = "#0072B2"     # blue   — constant pooled R_cond (3 free parameters)
C_PERCAM = "#D55E00"     # orange — per-camera R_cond (12 free parameters)
C_CAM = {
    "camera_A": "#0072B2",
    "camera_B": "#E69F00",
    "camera_C": "#009E73",
    "camera_D": "#CC79A7",
}
C_CAPTURE = ("#332288", "#88CCEE", "#DDCC77")
C_REF = "#444444"
C_BAND = "#BBBBBB"

#: Median NEES of a calibrated 2-D belief (chi2_2 median). Quoted by both RESULTS.md.
NEES_CALIBRATED = 1.39

ARMS = (
    ("v2 — deployed along-bearing calibration", OUT_EXP2),
    ("v3 — gated 2-DOF calibration", OUT_EXP3),
)


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.grid": True, "grid.color": "#CCCCCC",
        "grid.alpha": 0.35, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 10.5, "font.size": 9,
    })


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, out_dir: Path, stem: str) -> list[Path]:
    written = []
    for ext in ("pdf", "png"):
        target = out_dir / f"{stem}.{ext}"
        fig.savefig(target, bbox_inches="tight")
        written.append(target)
    plt.close(fig)
    return written


def ellipse_params(cov) -> tuple[float, float, float]:
    """(major 1-sigma axis length, minor 1-sigma axis length, angle deg) of a 2x2."""
    matrix = np.asarray(cov, dtype=float)
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))
    return math.sqrt(max(values[0], 0.0)), math.sqrt(max(values[1], 0.0)), angle


def label_pair(ax, left_value: float, right_value: float, y: float,
               left_color: str, right_color: str, dy: int = -19) -> None:
    """Print two dumbbell endpoint values on the OUTSIDE of the pair.

    The v3 arm's two methods differ by 0.039 nats, so centred labels collide;
    pushing each label away from the pair keeps them readable without moving,
    rescaling or otherwise flattering either point.
    """
    ax.annotate(f"{left_value:.3f}", xy=(left_value, y), xytext=(-7, dy),
                textcoords="offset points", ha="right", fontsize=7.6, color=left_color)
    ax.annotate(f"{right_value:.3f}", xy=(right_value, y), xytext=(7, dy),
                textcoords="offset points", ha="left", fontsize=7.6, color=right_color)


def axis_sigmas(cov) -> tuple[float, float, float]:
    """(sigma_x, sigma_y, correlation) as printed straight off the matrix."""
    matrix = np.asarray(cov, dtype=float)
    sx, sy = math.sqrt(matrix[0, 0]), math.sqrt(matrix[1, 1])
    rho = matrix[0, 1] / (sx * sy) if sx > 0 and sy > 0 else float("nan")
    return sx, sy, rho


# --------------------------------------------------------------------------- #
# fig_r1 — the head-to-head. This is the null.
# --------------------------------------------------------------------------- #

def fig_r1(exp2: dict, exp3: dict) -> tuple[list[Path], dict]:
    scores = {
        "v2": exp2["predictive_scores"],
        "v3": exp3["predictive_scores"],
    }
    n_scored = scores["v2"]["n_scored"]
    assert scores["v3"]["n_scored"] == n_scored, "arms must be scored on the same set"

    arm_labels = {
        "v2": "v2\ndeployed calibration\n(exp2)",
        "v3": "v3\ngated 2-DOF calibration\n(exp3)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.0),
                             gridspec_kw={"width_ratios": [1.15, 1.25, 1.0]})
    ax_nll, ax_cov, ax_sharp = axes

    # ---- panel 1: MNLL dumbbell -------------------------------------------- #
    ys = {"v2": 1.0, "v3": 0.0}
    for arm, y in ys.items():
        pooled = scores[arm]["constant_pooled_R_cond"]["mnll"]
        percam = scores[arm]["per_camera_R_cond"]["mnll"]
        ax_nll.plot([pooled, percam], [y, y], color="#999999", lw=2.4, zorder=1,
                    solid_capstyle="round")
        ax_nll.scatter([pooled], [y], s=125, color=C_POOLED, zorder=3,
                       edgecolor="white", linewidth=1.1)
        ax_nll.scatter([percam], [y], s=125, color=C_PERCAM, zorder=3,
                       marker="D", edgecolor="white", linewidth=1.1)
        delta = percam - pooled
        verdict = "per-camera WORSE" if delta > 0 else "per-camera better"
        ax_nll.annotate(
            f"$\\Delta$ = {delta:+.3f} nats   ({verdict})",
            xy=((pooled + percam) / 2.0, y), xytext=(0, 20),
            textcoords="offset points", ha="center", fontsize=8.6,
            fontweight="bold", color="#222222")
        ax_nll.annotate(f"{pooled:.3f}", xy=(pooled, y), xytext=(0, -17),
                        textcoords="offset points", ha="center", fontsize=7.6,
                        color=C_POOLED)
        ax_nll.annotate(f"{percam:.3f}", xy=(percam, y), xytext=(0, -17),
                        textcoords="offset points", ha="center", fontsize=7.6,
                        color=C_PERCAM)
    ax_nll.set_yticks(list(ys.values()))
    ax_nll.set_yticklabels([arm_labels[a] for a in ys], fontsize=8.2)
    ax_nll.set_ylim(-0.55, 1.6)
    ax_nll.set_xlim(-4.7, 0.5)
    ax_nll.set_xlabel("mean negative log-likelihood of the residuals  [nats]\n"
                      r"under $C_t = H P^s H^\top + R_{\rm cond}$   " "← lower is better")
    ax_nll.set_title("Predictive likelihood\n"
                     "the method gap is dwarfed by the calibration gap",
                     fontweight="bold")
    ax_nll.grid(axis="y", visible=False)
    ax_nll.scatter([], [], s=110, color=C_POOLED, label="constant pooled $R_{\\rm cond}$ (3 params)")
    ax_nll.scatter([], [], s=110, color=C_PERCAM, marker="D",
                   label="per-camera $R_{\\rm cond}$ (12 params)")
    ax_nll.legend(fontsize=7.8, loc="lower right", framealpha=0.95)

    # ---- panel 2: coverage error vs nominal -------------------------------- #
    clusters = [
        ("v2", 0.50, "coverage_50"),
        ("v2", 0.95, "coverage_95"),
        ("v3", 0.50, "coverage_50"),
        ("v3", 0.95, "coverage_95"),
    ]
    width = 0.40
    cov_rows = []
    for index, (arm, nominal, key) in enumerate(clusters):
        pooled = scores[arm]["constant_pooled_R_cond"][key]
        percam = scores[arm]["per_camera_R_cond"][key]
        se = math.sqrt(nominal * (1.0 - nominal) / n_scored)
        ax_cov.add_patch(Rectangle((index - 0.5, -se), 1.0, 2 * se,
                                   facecolor=C_BAND, edgecolor="none",
                                   alpha=0.55, zorder=0))
        ax_cov.bar(index - width / 2, pooled - nominal, width=width,
                   color=C_POOLED, zorder=2)
        ax_cov.bar(index + width / 2, percam - nominal, width=width,
                   color=C_PERCAM, zorder=2, hatch="///", edgecolor="white",
                   linewidth=0.0)
        for offset, value in ((-width / 2, pooled), (+width / 2, percam)):
            below = (value - nominal) < 0
            ax_cov.annotate(f"{value:.3f}", xy=(index + offset, value - nominal),
                            xytext=(0, -12 if below else 4), textcoords="offset points",
                            ha="center", fontsize=7.2, color="#222222")
        cov_rows.append({
            "arm": arm, "nominal": nominal, "pooled": pooled, "per_camera": percam,
            "binomial_se": round(se, 5),
        })
    ax_cov.axhline(0.0, color="#111111", lw=1.3, zorder=3)
    ax_cov.set_xticks(range(len(clusters)))
    ax_cov.set_xticklabels([f"{arm}\nnominal {int(nom * 100)}%" for arm, nom, _ in clusters],
                           fontsize=8.2)
    ax_cov.set_ylabel("empirical coverage − nominal\n(0 = calibrated; below 0 = overconfident)")
    ax_cov.set_ylim(-0.47, 0.23)
    ax_cov.set_title("Coverage calibration\n"
                     "v2: both models badly overconfident — per-camera more so",
                     fontweight="bold")
    ax_cov.annotate("grey band = $\\pm$1 binomial SE at $n$=1425 — an i.i.d. SCALE REFERENCE only.\n"
                    "Detections are serially correlated and the two arms are paired on the same\n"
                    "residuals, so this is NOT a significance test of the pooled/per-camera gap.",
                    xy=(0.5, 0.995), xycoords="axes fraction", fontsize=6.6,
                    color="#333333", ha="center", va="top")

    # ---- panel 3: sharpness ------------------------------------------------ #
    for arm, y in ys.items():
        pooled = scores[arm]["constant_pooled_R_cond"]["sharpness_log_det"]
        percam = scores[arm]["per_camera_R_cond"]["sharpness_log_det"]
        ax_sharp.plot([pooled, percam], [y, y], color="#999999", lw=2.4, zorder=1,
                      solid_capstyle="round")
        ax_sharp.scatter([pooled], [y], s=125, color=C_POOLED, zorder=3,
                         edgecolor="white", linewidth=1.1)
        ax_sharp.scatter([percam], [y], s=125, color=C_PERCAM, marker="D", zorder=3,
                         edgecolor="white", linewidth=1.1)
        ax_sharp.annotate(f"{pooled:.3f}", xy=(pooled, y), xytext=(0, 12),
                          textcoords="offset points", ha="center", fontsize=7.6,
                          color=C_POOLED)
        ax_sharp.annotate(f"{percam:.3f}", xy=(percam, y), xytext=(0, -18),
                          textcoords="offset points", ha="center", fontsize=7.6,
                          color=C_PERCAM)
    ax_sharp.set_yticks(list(ys.values()))
    ax_sharp.set_yticklabels(["v2", "v3"], fontsize=9)
    ax_sharp.set_ylim(-0.55, 1.6)
    ax_sharp.set_xlabel(r"sharpness — mean $\log|C_t|$")
    ax_sharp.set_title("Sharpness is NOT a score\n"
                       "per-camera is sharper in both arms — that is the mechanism\n"
                       "of its v2 loss, not a win",
                       fontweight="bold")
    ax_sharp.grid(axis="y", visible=False)

    fig.suptitle(
        "EXP-RCOND — NULL: a per-camera conditional covariance does not beat one constant "
        "pooled matrix\n"
        f"$n$ = {n_scored} associated detections, 4 cameras, 3 captures  ·  "
        "no ground truth enters either arm",
        fontsize=12.5, fontweight="bold")
    fig.text(
        0.5, -0.075,
        "Read with the caveat, not around it: both arms are scored on the SAME 1425 residuals "
        "that fitted them. “Held-out” in this study names the leave-one-camera-out "
        "reference trajectory, not held-out scoring data.\n"
        "The comparison is therefore biased TOWARD per-camera — 12 free parameters against "
        "3, scored in-sample — and per-camera still loses by 1.221 nats under v2 and gains "
        "0.039 nats under v3 while losing 95% coverage. That is a tie, not a promotion.",
        ha="center", va="top", fontsize=8.0, color="#222222",
        bbox={"facecolor": "#F4F4F4", "edgecolor": "#BBBBBB", "boxstyle": "round,pad=0.55"})
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    written = save(fig, OUT_EXP2, "fig_r1_pooled_vs_per_camera")
    summary = {
        "n_scored": n_scored,
        "mnll": {
            arm: {
                "pooled": scores[arm]["constant_pooled_R_cond"]["mnll"],
                "per_camera": scores[arm]["per_camera_R_cond"]["mnll"],
                "delta_per_minus_pooled": round(
                    scores[arm]["per_camera_R_cond"]["mnll"]
                    - scores[arm]["constant_pooled_R_cond"]["mnll"], 4),
            } for arm in ("v2", "v3")
        },
        "coverage": cov_rows,
        "sharpness": {
            arm: {
                "pooled": scores[arm]["constant_pooled_R_cond"]["sharpness_log_det"],
                "per_camera": scores[arm]["per_camera_R_cond"]["sharpness_log_det"],
            } for arm in ("v2", "v3")
        },
        "arm_gap_mnll_v2_to_v3": {
            "pooled": round(scores["v3"]["constant_pooled_R_cond"]["mnll"]
                            - scores["v2"]["constant_pooled_R_cond"]["mnll"], 4),
            "per_camera": round(scores["v3"]["per_camera_R_cond"]["mnll"]
                                - scores["v2"]["per_camera_R_cond"]["mnll"], 4),
        },
    }
    return written, summary


# --------------------------------------------------------------------------- #
# fig_r2 — what the per-camera estimate actually is
# --------------------------------------------------------------------------- #

def fig_r2(exp1: dict, exp2: dict, exp3: dict) -> tuple[list[Path], dict]:
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 10.2))
    (ax_v2, ax_v3), (ax_n, ax_anchor) = axes

    detail = {}
    for ax, payload, arm in ((ax_v2, exp2, "v2 — deployed calibration (exp2)"),
                             (ax_v3, exp3, "v3 — gated 2-DOF calibration (exp3)")):
        pooled = payload["pooled_shrinkage_target"]
        rows = {row["camera"]: row for row in payload["per_camera"]}
        major, minor, angle = ellipse_params(pooled)
        ax.add_patch(Ellipse((0, 0), 2 * major, 2 * minor, angle=angle,
                             facecolor="none", edgecolor="#111111", lw=2.6, ls="--",
                             zorder=5, label="constant pooled target"))
        arm_detail = {}
        for camera in CAMERAS:
            cov = payload["per_camera_R_cond"][camera]
            major, minor, angle = ellipse_params(cov)
            sx, sy, rho = axis_sigmas(cov)
            row = rows[camera]
            floored = bool(row.get("held_out_psd_projected"))
            ax.add_patch(Ellipse((0, 0), 2 * major, 2 * minor, angle=angle,
                                 facecolor=C_CAM[camera], alpha=0.13, zorder=2))
            ax.add_patch(Ellipse((0, 0), 2 * major, 2 * minor, angle=angle,
                                 facecolor="none", edgecolor=C_CAM[camera], lw=2.0,
                                 ls=":" if floored else "-", zorder=4,
                                 label=(f"{camera.replace('camera_', 'cam ')}  "
                                        f"$n$={row['held_out_n']}"
                                        f"{'  (PSD-floored)' if floored else ''}")))
            arm_detail[camera] = {
                "sigma_x_m": round(sx, 4), "sigma_y_m": round(sy, 4),
                "correlation": round(rho, 3),
                "major_sigma_m": round(major, 4), "minor_sigma_m": round(minor, 4),
                "n": row["held_out_n"], "psd_floored": floored,
                "shrinkage_lambda": row["shrinkage_lambda"],
            }
        sx, sy, rho = axis_sigmas(pooled)
        arm_detail["pooled"] = {"sigma_x_m": round(sx, 4), "sigma_y_m": round(sy, 4),
                                "correlation": round(rho, 3)}
        detail[arm.split(" ")[0]] = arm_detail

        ax.set_aspect("equal")
        limit = 0.042
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.axhline(0.0, color="#DDDDDD", lw=0.8, zorder=0)
        ax.axvline(0.0, color="#DDDDDD", lw=0.8, zorder=0)
        ax.set_xlabel("world $x$ residual  [m]")
        ax.set_ylabel("world $y$ residual  [m]")
        ax.set_title(f"1-$\\sigma$ $R_{{\\rm cond}}$ ellipses — {arm}", fontweight="bold")
        ax.legend(fontsize=7.4, loc="upper left", framealpha=0.95)

    ax_v2.annotate("camera B collapses to $\\sigma\\approx$4 mm on 295 of the\n"
                   "1425 scored residuals: the $S-\\overline{HP^sH^\\top}$ subtraction\n"
                   "went indefinite and was floored. A near-zero $R_{\\rm cond}$ on a\n"
                   "fifth of the data is what an overconfident model looks like.",
                   xy=(0.985, 0.02), xycoords="axes fraction", ha="right", va="bottom",
                   fontsize=7.0, color="#222222",
                   bbox={"facecolor": "#FFF6EC", "edgecolor": "#D55E00", "lw": 0.8,
                         "boxstyle": "round,pad=0.4"})
    ax_v3.annotate("Under v3 three of four per-camera matrices are nearly\n"
                   "rank-deficient ($|\\rho|>0.97$, 9:1 to 11:1 axis ratios) while the\n"
                   "pooled target sits at $\\rho=-0.60$. The per-camera shapes are\n"
                   "very different from the pool — and buy 0.039 nats.",
                   xy=(0.985, 0.02), xycoords="axes fraction", ha="right", va="bottom",
                   fontsize=7.0, color="#222222",
                   bbox={"facecolor": "#EEF6FF", "edgecolor": "#0072B2", "lw": 0.8,
                         "boxstyle": "round,pad=0.4"})

    # ---- panel: where the 1425 samples come from --------------------------- #
    captures = [entry["capture"] for entry in exp1["captures"]]
    bottoms = np.zeros(len(CAMERAS))
    for index, entry in enumerate(exp1["captures"]):
        counts = np.array([entry["per_camera"][cam]["associated"] for cam in CAMERAS],
                          dtype=float)
        ax_n.bar(range(len(CAMERAS)), counts, bottom=bottoms, width=0.62,
                 color=C_CAPTURE[index], edgecolor="white", linewidth=0.8,
                 label=entry["capture"])
        for x, (count, base) in enumerate(zip(counts, bottoms)):
            if count >= 40:
                ax_n.text(x, base + count / 2.0, f"{int(count)}", ha="center",
                          va="center", fontsize=7.4, color="white", fontweight="bold")
        bottoms += counts
    for x, total in enumerate(bottoms):
        ax_n.text(x, total + 12, f"{int(total)}", ha="center", fontsize=8.4,
                  fontweight="bold", color="#222222")
    ax_n.set_xticks(range(len(CAMERAS)))
    ax_n.set_xticklabels([c.replace("camera_", "cam ") for c in CAMERAS])
    ax_n.set_ylabel("associated detections")
    ax_n.set_ylim(0, max(bottoms) * 1.18)
    ax_n.set_title("Every per-camera matrix rests on one thin, uneven sample\n"
                   f"{exp1['total_associated']}/{exp1['total_detections']} detections "
                   "associate within 0.15 s (gate R0 PASS)", fontweight="bold")
    ax_n.legend(fontsize=7.6, loc="upper left")

    # ---- panel: anchor-sigma sensitivity ----------------------------------- #
    anchor_rows = {"v2": exp2["anchor_sensitivity"], "v3": exp3["anchor_sensitivity"]}
    for arm, rows in anchor_rows.items():
        style = "-" if arm == "v2" else "--"
        xs = [row["anchor_std_m"] for row in rows]
        for camera in CAMERAS:
            ys = [row[camera] for row in rows]
            ax_anchor.plot(xs, ys, color=C_CAM[camera], ls=style, lw=1.7,
                           marker="o", ms=4.5, zorder=3)
            for x, y, row in zip(xs, ys, rows):
                if row[f"{camera}_floored"]:
                    ax_anchor.scatter([x], [y], s=58, facecolor="white",
                                      edgecolor=C_CAM[camera], lw=1.5, zorder=4)
    ax_anchor.axvline(exp2["anchor_std_m"], color=C_REF, lw=1.1, ls=":", zorder=1)
    ax_anchor.annotate(f"frozen operating point\n$\\sigma_{{\\rm anchor}}$="
                       f"{exp2['anchor_std_m']} m",
                       xy=(exp2["anchor_std_m"], 0.0335), xytext=(4, 0),
                       textcoords="offset points", fontsize=7.2, color=C_REF)
    ax_anchor.set_xscale("log")
    ax_anchor.set_xticks([0.02, 0.05, 0.10, 0.20])
    ax_anchor.set_xticklabels(["0.02", "0.05", "0.10", "0.20"])
    ax_anchor.set_xlabel(r"assumed anchor $\sigma$ [m]  (a frozen input, swept — never fitted)")
    ax_anchor.set_ylabel(r"per-camera $\sigma$ of $R_{\rm cond}$  [m]")
    ax_anchor.set_ylim(-0.002, 0.038)
    ax_anchor.set_title("The per-camera values are not identified\n"
                        "every camera slides to the 0 floor as the assumed anchor loosens",
                        fontweight="bold")
    handles = [plt.Line2D([], [], color=C_CAM[c], lw=1.9,
                          label=c.replace("camera_", "cam ")) for c in CAMERAS]
    handles += [plt.Line2D([], [], color="#555555", lw=1.7, ls="-", label="v2 arm"),
                plt.Line2D([], [], color="#555555", lw=1.7, ls="--", label="v3 arm"),
                plt.Line2D([], [], color="#555555", marker="o", ls="none",
                           markerfacecolor="white", label="PSD-floored")]
    ax_anchor.legend(handles=handles, fontsize=7.2, ncol=2, loc="upper right",
                     framealpha=0.95)

    fig.suptitle("EXP-RCOND — what the per-camera $R_{\\rm cond}$ is, and why 12 parameters "
                 "buy nothing\n"
                 "ground truth is EVALUATION ONLY and enters none of these estimates",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    written = save(fig, OUT_EXP2, "fig_r2_rcond_vs_pooled")
    return written, detail


# --------------------------------------------------------------------------- #
# fig_r3 — exp3 context: the belief fix is real, the R5 verdict is not
# --------------------------------------------------------------------------- #

def fig_r3(exp2: dict, exp3: dict) -> tuple[list[Path], dict]:
    fig, (ax_nees, ax_q) = plt.subplots(1, 2, figsize=(13.0, 5.0),
                                        gridspec_kw={"width_ratios": [1.0, 1.15]})

    captures = [row["capture"] for row in exp2["gate_r1_calibration"]]
    v2 = [row["nees_median_smoothed_at_detections"] for row in exp2["gate_r1_calibration"]]
    v3 = [row["nees_median_smoothed_at_detections"] for row in exp3["gate_r1_calibration"]]
    width = 0.36
    xs = np.arange(len(captures))
    ax_nees.bar(xs - width / 2, v2, width=width, color="#999999",
                label="v2 — deployed calibration")
    ax_nees.bar(xs + width / 2, v3, width=width, color="#009E73",
                label="v3 — gated 2-DOF calibration")
    for x, (a, b) in enumerate(zip(v2, v3)):
        ax_nees.text(x - width / 2, a * 1.05, f"{a:.2f}", ha="center", fontsize=7.8)
        ax_nees.text(x + width / 2, b * 1.05, f"{b:.2f}", ha="center", fontsize=7.8)
    ax_nees.axhline(NEES_CALIBRATED, color="#111111", lw=1.4, ls="--")
    ax_nees.annotate(f"calibrated 2-D belief = {NEES_CALIBRATED}",
                     xy=(len(captures) - 0.55, NEES_CALIBRATED), xytext=(0, 5),
                     textcoords="offset points", ha="right", fontsize=7.8)
    ax_nees.set_yscale("log")
    ax_nees.set_xticks(xs)
    ax_nees.set_xticklabels(
        [c + ("\n(held out from the\ncalibration fit)" if c.startswith("fusion") else
              "\n(in-sample)") for c in captures], fontsize=7.8)
    ax_nees.set_ylabel("median NEES at detection instants   (evaluation-only metric)")
    ax_nees.set_title("Gate R1 — the 2-DOF projection fix repairs the belief\n"
                      "this is a real PASS, and it is a DIFFERENT gate from R5",
                      fontweight="bold")
    ax_nees.legend(fontsize=8, loc="upper right")

    # ---- Q sweep: the tension that bounds the whole estimate ---------------- #
    for arm, payload, style in (("v2", exp2, "-"), ("v3", exp3, "--")):
        rows = payload["process_sensitivity"]
        sigmas = [row["sigma_per_sqrt_m"] for row in rows]
        for index, capture in enumerate(captures):
            ys = [row["nees_median_at_detections"][index] for row in rows]
            ax_q.plot(sigmas, ys, color=C_CAPTURE[index], ls=style, lw=1.7,
                      marker="s" if arm == "v2" else "o", ms=4.5)
    ax_q.axhline(NEES_CALIBRATED, color="#111111", lw=1.3, ls="--", zorder=1)
    ax_q.set_xscale("log")
    ax_q.set_yscale("log")
    sigmas = [row["sigma_per_sqrt_m"] for row in exp2["process_sensitivity"]]
    ax_q.set_xticks(sigmas)
    ax_q.set_xticklabels([f"{s:g}" for s in sigmas])
    floored = {}
    for arm, payload in (("v2", exp2), ("v3", exp3)):
        floored[arm] = [sum(1 for c in CAMERAS if row[f"{c}_floored"])
                        for row in payload["process_sensitivity"]]
    top = ax_q.get_ylim()[1]
    for index, sigma in enumerate(sigmas):
        ax_q.annotate(f"{floored['v2'][index]} / {floored['v3'][index]}",
                      xy=(sigma, top * 0.72), ha="center", fontsize=7.6,
                      color="#8C4A00", fontweight="bold")
    ax_q.annotate("cameras PSD-floored (v2 / v3) — 4 floored means no per-camera\n"
                  "$R_{\\rm cond}$ is resolvable at that odometry noise",
                  xy=(0.5, 0.955), xycoords="axes fraction", ha="center", va="top",
                  fontsize=7.2, color="#8C4A00")
    ax_q.set_xlabel(r"odometry process noise  $\sigma/\sqrt{m}$   (frozen input, swept)")
    ax_q.set_ylabel("median NEES at detection instants")
    ax_q.set_title("The tension that bounds $R_{\\rm cond}$\n"
                   "the $Q$ that calibrates the belief floors the covariance estimate",
                   fontweight="bold")
    handles = [plt.Line2D([], [], color=C_CAPTURE[i], lw=1.8, label=c)
               for i, c in enumerate(captures)]
    handles += [plt.Line2D([], [], color="#555555", lw=1.7, ls="-", marker="s",
                           ms=4.5, label="v2 arm"),
                plt.Line2D([], [], color="#555555", lw=1.7, ls="--", marker="o",
                           ms=4.5, label="v3 arm")]
    ax_q.legend(handles=handles, fontsize=7.2, loc="lower left", ncol=1, framealpha=0.95)

    s2 = exp2["predictive_scores"]
    s3 = exp3["predictive_scores"]
    d2 = s2["per_camera_R_cond"]["mnll"] - s2["constant_pooled_R_cond"]["mnll"]
    d3 = s3["per_camera_R_cond"]["mnll"] - s3["constant_pooled_R_cond"]["mnll"]
    fig.suptitle("EXP-RCOND exp3 — the blocker is removed; the promotion gate is still not met",
                 fontsize=12.5, fontweight="bold")
    fig.text(
        0.5, -0.055,
        "Do not read the left panel as a win for per-camera $R_{\\rm cond}$. It is not that gate. "
        f"With the belief repaired, per-camera moves MNLL by {d3:+.3f} nats against the pooled "
        f"constant (v2: {d2:+.3f}), \n"
        "and pooled remains closer to nominal at 95% coverage (0.917 vs 0.900). "
        "Gate R5 — “beat pooled covariance on held-out evidence” — is a TIE, and "
        f"the registry records no promotion.   $n$ = {s3['n_scored']} detections.",
        ha="center", va="top", fontsize=8.0, color="#222222",
        bbox={"facecolor": "#F4F4F4", "edgecolor": "#BBBBBB", "boxstyle": "round,pad=0.55"})
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    written = save(fig, OUT_EXP3, "fig_r3_two_dof_context")
    return written, {
        "nees_at_detections": {"captures": captures, "v2": v2, "v3": v3},
        "cameras_floored_by_q": {"sigma_per_sqrt_m": sigmas, **floored},
        "mnll_delta_per_minus_pooled": {"v2": round(d2, 4), "v3": round(d3, 4)},
    }


def main() -> int:
    _style()
    exp1 = load(STUDY / "exp1_timing_and_coverage" / "timing_and_coverage.json")
    exp2 = load(OUT_EXP2 / "operational_rcond.json")
    exp3 = load(OUT_EXP3 / "operational_rcond.json")

    written = []
    paths, head_to_head = fig_r1(exp2, exp3)
    written += paths
    paths, rcond_detail = fig_r2(exp1, exp2, exp3)
    written += paths
    paths, context = fig_r3(exp2, exp3)
    written += paths

    print(json.dumps({"head_to_head": head_to_head,
                      "rcond_detail": rcond_detail,
                      "exp3_context": context}, indent=2))
    for path in written:
        print("wrote ->", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
