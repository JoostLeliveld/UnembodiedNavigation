#!/usr/bin/env python3
"""exp1: historical-v2 sensitivity of coverage versus achievable precision.

Coverage-aware planning asks one question of each point on the floor: will a camera
see me here? The correlation-floor result
(`logs/studies/bayesian_filter_showcase/exp1_graceful_vs_trusting`) says that is
only half the question, because the belief may never become sharper than the
residual systematic of whichever camera is watching. A spot covered only by a
camera with a repeated residual floor cannot be known better than that floor, however long the
robot loiters there and however reliably it is seen.

So achievable precision is a FIELD, determined jointly by availability and by which
camera supplies it, and the two fields are not the same picture. This study builds
both and measures where they disagree.

Model. At a point where camera c is usable with probability ``p_c`` per detection
opportunity (rate ``f``), and odometry accumulates variance at ``q_rate`` between
updates, the steady-state belief variance that camera alone can sustain is

    sigma_c^2(x, y)  =  floor_c^2  +  q_rate / (f * p_c(x, y))
                        ^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^
                        what the      drift accumulated while
                        camera's      waiting for the next
                        lean forbids  usable detection

and the network's achievable precision is the best any single camera can offer,

    sigma*(x, y) = min_c sigma_c(x, y).

The minimum, not a fusion: this network's overlap is 13 % and uniform fusion is
already known to lose to the best single camera. Selection is the operation that
matches the evidence.

Everything on the right-hand side was measured for the original study: ``p_c`` from the
frozen four-camera coverage artifact, ``floor_c`` from the retired-v2 per-camera residual
study, and ``f``/``q_rate`` from its recorded runtime. The floor values are now historical
and route/yaw-confounded, so this is mechanism sensitivity rather than a current selector.

Outputs -> logs/studies/achievable_precision_map/exp1_precision_vs_coverage/
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)

COVERAGE_NPZ = (
    REPO / "paper_artifacts/gp/warehouse_full_4cam_fused_v1/fused_planner_four_camera.npz"
)
OUT = REPO / "logs/studies/achievable_precision_map/exp1_precision_vs_coverage"

CAMERAS = ("camera_A", "camera_B", "camera_C", "camera_D")

#: Residual systematic per camera, metres — the floor the belief may not beat.
#: Measured (logs/studies/operational_residual_rcond/exp2_operational_rcond, oracle
#: bias norms). See exp2_does_it_generalize: stable for C, and a generous bound
#: rather than a precise constant for the others.
HISTORICAL_V2_FLOOR_M = {
    "camera_A": 0.0071,
    "camera_B": 0.0123,
    "camera_C": 0.0768,
    "camera_D": 0.0328,
}

DETECTION_RATE_HZ = 3.0          # measured runtime throughput, inference-bound
NOMINAL_SPEED_MPS = 0.3
ODOM_SIGMA_PER_SQRT_M = 0.04     # same constant the filter study uses
#: Variance accumulated per second of dead reckoning at nominal speed.
Q_RATE_M2_PER_S = ODOM_SIGMA_PER_SQRT_M**2 * NOMINAL_SPEED_MPS
#: Below this availability a camera is treated as not supplying updates at all;
#: the drift term would otherwise diverge and dominate the picture.
MIN_USABLE_P = 0.02

CAMERA_COLORS = {"camera_A": "#0072B2", "camera_B": "#009E73",
                 "camera_C": "#D55E00", "camera_D": "#CC79A7"}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "font.size": 9,
    })


def per_camera_sigma(availability: np.ndarray, floor_m: float) -> np.ndarray:
    """Steady-state belief sigma this camera alone can sustain, metres."""

    usable = np.maximum(availability, MIN_USABLE_P)
    drift_variance = Q_RATE_M2_PER_S / (DETECTION_RATE_HZ * usable)
    sigma = np.sqrt(floor_m**2 + drift_variance)
    # Where the camera effectively never fires, it sustains nothing.
    return np.where(availability < MIN_USABLE_P, np.inf, sigma)


def main() -> int:
    if not COVERAGE_NPZ.is_file():
        print(f"missing coverage artifact: {COVERAGE_NPZ}")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    _style()

    data = np.load(COVERAGE_NPZ)
    xs, ys = data["xs"], data["ys"]
    availability = {c: np.asarray(data[f"P_camera_{c[-1]}_map"], float) for c in CAMERAS}

    sigma_stack = np.stack([
        per_camera_sigma(availability[c], HISTORICAL_V2_FLOOR_M[c]) for c in CAMERAS
    ])
    availability_stack = np.stack([availability[c] for c in CAMERAS])

    # The two competing views of the same floor.
    best_coverage_index = np.argmax(availability_stack, axis=0)
    best_precision_index = np.argmin(sigma_stack, axis=0)
    achievable_sigma = np.min(sigma_stack, axis=0)
    coverage_best_p = np.max(availability_stack, axis=0)
    # What a coverage-only planner would actually get: it selects the most-available
    # camera, and then inherits whatever floor that camera happens to impose.
    sigma_if_following_coverage = np.take_along_axis(
        sigma_stack, best_coverage_index[None, ...], axis=0
    )[0]

    reachable = np.isfinite(achievable_sigma) & (coverage_best_p >= MIN_USABLE_P)
    disagree = reachable & (best_coverage_index != best_precision_index)
    # Subtract only on the mask: both sides are +inf outside it, and inf - inf is nan.
    penalty = np.zeros_like(achievable_sigma)
    penalty[disagree] = (sigma_if_following_coverage[disagree]
                         - achievable_sigma[disagree])

    stats = {
        "status": "historical_v2_sensitivity_only",
        "comparison_context": "MC-DRIVE-V2 residual floors composed with frozen coverage",
        "prohibited_use": "current camera ranking or camera-management input",
        "reachable_cells": int(reachable.sum()),
        "fraction_where_best_coverage_is_not_best_precision": float(
            disagree.sum() / max(reachable.sum(), 1)
        ),
        "achievable_sigma_m": {
            "median": float(np.median(achievable_sigma[reachable])),
            "p90": float(np.percentile(achievable_sigma[reachable], 90)),
            "max": float(np.max(achievable_sigma[reachable])),
        },
        "coverage_following_sigma_m": {
            "median": float(np.median(sigma_if_following_coverage[reachable])),
            "p90": float(np.percentile(sigma_if_following_coverage[reachable], 90)),
        },
        "penalty_where_they_disagree_m": {
            "median": float(np.median(penalty[disagree])) if disagree.any() else 0.0,
            "p90": float(np.percentile(penalty[disagree], 90)) if disagree.any() else 0.0,
            "max": float(np.max(penalty)) if disagree.any() else 0.0,
        },
        "share_of_floor_by_best_precision_camera": {
            c: float(np.mean(best_precision_index[reachable] == i))
            for i, c in enumerate(CAMERAS)
        },
        "share_of_floor_by_best_coverage_camera": {
            c: float(np.mean(best_coverage_index[reachable] == i))
            for i, c in enumerate(CAMERAS)
        },
        "config": {
            "detection_rate_hz": DETECTION_RATE_HZ,
            "nominal_speed_mps": NOMINAL_SPEED_MPS,
            "odom_sigma_per_sqrt_m": ODOM_SIGMA_PER_SQRT_M,
            "historical_v2_floor_m": HISTORICAL_V2_FLOOR_M,
            "coverage_artifact": str(COVERAGE_NPZ.relative_to(REPO)),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ figures
    masked_sigma = np.where(reachable, achievable_sigma, np.nan)
    masked_coverage = np.where(coverage_best_p >= MIN_USABLE_P, coverage_best_p, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.6))
    im0 = axes[0].pcolormesh(xs, ys, masked_coverage, cmap="viridis", vmin=0, vmax=1,
                             shading="auto")
    axes[0].set_title("What coverage-aware planning sees\n"
                      "best availability $\\max_c p_c$", fontweight="bold", fontsize=10)
    fig.colorbar(im0, ax=axes[0], shrink=0.85).set_label("probability of a usable detection")

    im1 = axes[1].pcolormesh(xs, ys, masked_sigma * 100.0, cmap="magma_r",
                             vmin=0, vmax=12, shading="auto")
    axes[1].set_title("What the robot can actually KNOW\n"
                      "achievable belief $\\sigma^*$", fontweight="bold", fontsize=10)
    fig.colorbar(im1, ax=axes[1], shrink=0.85).set_label("achievable position sigma [cm]")

    palette = ListedColormap([CAMERA_COLORS[c] for c in CAMERAS])
    shown = np.where(reachable, best_precision_index, np.nan)
    im2 = axes[2].pcolormesh(xs, ys, shown, cmap=palette,
                             norm=BoundaryNorm(np.arange(-0.5, 4.5), palette.N),
                             shading="auto")
    # Hatch the disagreement region.
    axes[2].contourf(xs, ys, disagree.astype(float), levels=[0.5, 1.5],
                     colors="none", hatches=["////"])
    axes[2].set_title("Which camera you should USE\n"
                      "(hatched: differs from the most-available one)",
                      fontweight="bold", fontsize=10)
    cbar = fig.colorbar(im2, ax=axes[2], shrink=0.85, ticks=range(4))
    cbar.ax.set_yticklabels([c.replace("camera_", "") for c in CAMERAS])

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    fig.suptitle(
        "Historical-v2 sensitivity: being seen is not the same as being known — "
        f"on {100 * stats['fraction_where_best_coverage_is_not_best_precision']:.0f} % of "
        "the reachable floor the most-available camera is not the most informative one",
        fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_m1_precision_vs_coverage.{ext}", bbox_inches="tight")
    plt.close(fig)

    print(f"reachable cells: {stats['reachable_cells']}")
    print(f"best-coverage camera != best-precision camera on "
          f"{100 * stats['fraction_where_best_coverage_is_not_best_precision']:.1f}% "
          "of the reachable floor")
    print(f"achievable sigma  median {100 * stats['achievable_sigma_m']['median']:.1f} cm  "
          f"p90 {100 * stats['achievable_sigma_m']['p90']:.1f} cm")
    print(f"following coverage median "
          f"{100 * stats['coverage_following_sigma_m']['median']:.1f} cm  "
          f"p90 {100 * stats['coverage_following_sigma_m']['p90']:.1f} cm")
    print(f"penalty where they disagree: median "
          f"{100 * stats['penalty_where_they_disagree_m']['median']:.1f} cm  "
          f"max {100 * stats['penalty_where_they_disagree_m']['max']:.1f} cm")
    print("share of floor, best-precision camera:",
          {k: round(v, 3) for k, v in stats["share_of_floor_by_best_precision_camera"].items()})
    print("share of floor, best-coverage camera:",
          {k: round(v, 3) for k, v in stats["share_of_floor_by_best_coverage_camera"].items()})
    print("wrote ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
