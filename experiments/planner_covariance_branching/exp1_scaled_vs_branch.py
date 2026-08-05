#!/usr/bin/env python3
"""exp1: where does folding availability into R agree with the honest branch model?

A fully controlled ablation — no captures, no ground truth, no fitted model. Every
number here is the *same algebra the planner runs*, evaluated over a grid of
(prior covariance, availability, conditional noise). The question is narrow and
answerable exactly:

    A usable detection arrives with probability p and, when it does, carries
    covariance R_det. The honest expected posterior is

        E[P+] = p · P_hit(R_det) + (1 - p) · P-

    The single-R planner path instead takes ONE deterministic Kalman update with
    an availability-inflated covariance, either

        R_det / p                                (scaled baseline; the R_miss -> inf
                                                 limit of the deployed blend)
        1/R = p/R_visible + (1-p)/R_miss         (the DEPLOYED precision blend)

    How far apart are they, and where?

Three readouts:

    1. relative error in the position-trace and log-det of the posterior,
    2. the equivalent isotropic covariance sigma_eff^2 that WOULD reproduce
       E[P+] through a single update — which turns out to depend on P- and so
       cannot be cached per position,
    3. the region where the single-R shortcut is within 10 %, i.e. where the old
       mapping is actually fine.

Outputs -> logs/studies/planner_covariance_branching/exp1_scaled_vs_branch/
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
sys.path.insert(0, str(REPO / "src" / "reliability"))

from reliability.observation_model import (  # noqa: E402
    equivalent_isotropic_covariance,
    expected_posterior_branch,
    kalman_posterior,
    posterior_criterion,
    scaled_covariance_baseline,
)
from reliability.single_camera_adapter import precision_blend_covariance  # noqa: E402

OUT = REPO / "logs" / "studies" / "planner_covariance_branching" / "exp1_scaled_vs_branch"

# Pixel-space priors, matching the R_plan semantics (2x2, px^2) and the sigma
# range already used by experiments/efe_hit_miss_mixture.
PRIORS = {
    "tight (5 px)": 5.0,
    "medium (30 px)": 30.0,
    "loose (80 px)": 80.0,
}
SIGMA_DET_PX = (2.0, 5.0, 10.0, 20.0)
# The runtime miss endpoint is unreconciled (40 px offline vs 120 px runtime; see
# reliability.covariance_mapping.MissEndpointPolicy). Both are swept so that no
# conclusion here depends on which one is right — and the scaled baseline needs
# no endpoint at all.
R_MISS_PX = (40.0, 120.0)
P_GRID = np.linspace(0.02, 1.0, 99)
XY_CORRELATION = 0.25
THETA_STD_DEG = 5.0
ACCEPTABLE_REL_ERROR = 0.10

H = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))

C_SCALED = "#D55E00"
C_BLEND40 = "#CC79A7"
C_BLEND120 = "#8C564B"
C_BRANCH = "#0072B2"
PRIOR_STYLES = {"tight (5 px)": ":", "medium (30 px)": "-", "loose (80 px)": "--"}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "axes.grid": True, "grid.color": "#CCCCCC",
        "grid.alpha": 0.3, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "font.size": 9,
    })


def prior_matrix(sigma_px: float):
    """3-state (x, y, theta) prior with a mild xy correlation."""
    var = sigma_px ** 2
    return (
        (var, XY_CORRELATION * var, 0.0),
        (XY_CORRELATION * var, var, 0.0),
        (0.0, 0.0, math.radians(THETA_STD_DEG) ** 2),
    )


def isotropic(sigma_px: float):
    return ((sigma_px ** 2, 0.0), (0.0, sigma_px ** 2))


def relative_error(old, reference, criterion: str) -> float:
    a = posterior_criterion(old, criterion)
    b = posterior_criterion(reference, criterion)
    if criterion == "logdet":
        # A log-det difference is already a ratio of volumes; report it as such.
        return a - b
    return (a - b) / b


def build_grid() -> list[dict]:
    rows: list[dict] = []
    for prior_name, sigma_prior in PRIORS.items():
        prior = prior_matrix(sigma_prior)
        for sigma_det in SIGMA_DET_PX:
            r_det = isotropic(sigma_det)
            for p_use in P_GRID:
                branch = expected_posterior_branch(prior, H, r_det, float(p_use))
                reference = branch.cov_expected
                row = {
                    "prior": prior_name,
                    "sigma_prior_px": sigma_prior,
                    "sigma_det_px": sigma_det,
                    "p_use": float(p_use),
                    "branch_pos_trace_px2": posterior_criterion(reference, "position_trace"),
                    "branch_logdet": posterior_criterion(reference, "logdet"),
                    "hit_pos_trace_px2": posterior_criterion(branch.cov_hit, "position_trace"),
                    "prior_pos_trace_px2": posterior_criterion(branch.cov_miss, "position_trace"),
                }

                scaled_post = kalman_posterior(
                    prior, H, scaled_covariance_baseline(r_det, float(p_use))
                )
                row["scaled_rel_trace_error"] = relative_error(
                    scaled_post, reference, "position_trace"
                )
                row["scaled_logdet_gap"] = relative_error(scaled_post, reference, "logdet")

                for r_miss in R_MISS_PX:
                    blend = precision_blend_covariance(
                        float(p_use), r_visible_uv=sigma_det, r_miss_uv=r_miss
                    )
                    blend_post = kalman_posterior(prior, H, blend)
                    tag = int(r_miss)
                    row[f"blend{tag}_rel_trace_error"] = relative_error(
                        blend_post, reference, "position_trace"
                    )
                    row[f"blend{tag}_logdet_gap"] = relative_error(
                        blend_post, reference, "logdet"
                    )

                equivalent = equivalent_isotropic_covariance(
                    prior, H, r_det, float(p_use), criterion="position_trace"
                )
                row["sigma_eff_px"] = math.sqrt(equivalent.sigma2)
                row["sigma_eff_reached"] = equivalent.reached
                row["sigma_scaled_px"] = sigma_det / math.sqrt(max(float(p_use), 1e-12))
                row["sigma_eff_over_scaled"] = row["sigma_eff_px"] / row["sigma_scaled_px"]
                rows.append(row)
    return rows


def fig_p1(rows: list[dict]) -> dict:
    """Relative position-trace error of each single-R shortcut versus p_use."""

    fig, axes = plt.subplots(1, len(SIGMA_DET_PX), figsize=(4.0 * len(SIGMA_DET_PX), 4.0),
                             sharey=True)
    for ax, sigma_det in zip(axes, SIGMA_DET_PX):
        for prior_name, style in PRIOR_STYLES.items():
            subset = [r for r in rows
                      if r["sigma_det_px"] == sigma_det and r["prior"] == prior_name]
            ps = [r["p_use"] for r in subset]
            ax.plot(ps, [r["scaled_rel_trace_error"] for r in subset],
                    color=C_SCALED, ls=style, lw=1.9,
                    label=f"$R/p$ · {prior_name}")
            ax.plot(ps, [r["blend120_rel_trace_error"] for r in subset],
                    color=C_BLEND120, ls=style, lw=1.9,
                    label=f"blend ($R_{{miss}}$=120 px) · {prior_name}")
        ax.axhspan(-ACCEPTABLE_REL_ERROR, ACCEPTABLE_REL_ERROR,
                   color="#009E73", alpha=0.12, zorder=0)
        ax.axhline(0.0, color="#666666", lw=1.0, ls=":")
        ax.set_title(rf"$\sigma_{{det}}$ = {sigma_det:.0f} px", fontweight="bold")
        ax.set_xlabel(r"$p_{\rm use}$")
    axes[0].set_ylabel("relative error in posterior position trace\n"
                       r"$(\mathrm{tr}\,P^+_{\rm single-R} - \mathrm{tr}\,E[P^+])"
                       r"\,/\,\mathrm{tr}\,E[P^+]$")
    axes[0].legend(fontsize=6.5, loc="lower right", ncol=1)
    fig.suptitle("Both single-$R$ shortcuts are optimistic everywhere except at the endpoints; "
                 "the green band is $\\pm$10 %",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_p1_relative_trace_error.{ext}", bbox_inches="tight")
    plt.close(fig)

    worst = min(rows, key=lambda r: r["scaled_rel_trace_error"])
    return {
        "worst_scaled_rel_trace_error": worst["scaled_rel_trace_error"],
        "worst_scaled_at": {k: worst[k] for k in ("prior", "sigma_det_px", "p_use")},
    }


def fig_p2(rows: list[dict]) -> dict:
    """sigma_eff versus p_use: the equivalent covariance is not a function of p alone."""

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax, ax2 = axes
    sigma_det = 5.0
    for prior_name, style in PRIOR_STYLES.items():
        subset = [r for r in rows
                  if r["sigma_det_px"] == sigma_det and r["prior"] == prior_name]
        ps = [r["p_use"] for r in subset]
        ax.plot(ps, [r["sigma_eff_px"] for r in subset], color=C_BRANCH, ls=style, lw=2.0,
                label=f"$\\sigma_{{eff}}$ · {prior_name}")
        ax2.plot(ps, [r["sigma_eff_over_scaled"] for r in subset],
                 color=C_BRANCH, ls=style, lw=2.0, label=prior_name)
    subset = [r for r in rows if r["sigma_det_px"] == sigma_det
              and r["prior"] == "medium (30 px)"]
    ax.plot([r["p_use"] for r in subset], [r["sigma_scaled_px"] for r in subset],
            color=C_SCALED, lw=2.0, label=r"$\sigma_{det}/\sqrt{p}$  (the $R/p$ law)")
    ax.set_yscale("log")
    ax.set_xlabel(r"$p_{\rm use}$")
    ax.set_ylabel(r"equivalent isotropic std  $\sigma_{\rm eff}$  [px]")
    ax.set_title(f"The single $R$ that WOULD reproduce $E[P^+]$\n"
                 f"($\\sigma_{{det}}$ = {sigma_det:.0f} px, trace-matched)",
                 fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper right")

    ax2.axhline(1.0, color="#666666", lw=1.0, ls=":")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$p_{\rm use}$")
    ax2.set_ylabel(r"$\sigma_{\rm eff}\ /\ (\sigma_{det}/\sqrt{p})$")
    ax2.set_title("The $R/p$ law is off by a factor that depends on the PRIOR,\n"
                  r"so no cached $R_{\rm plan}(s)$ can be correct", fontweight="bold")
    ax2.legend(fontsize=7.5, loc="upper right")

    fig.suptitle(r"$R_{\rm eff}$ is $R_{\rm eff}(s, P^-, H)$ — not $R_{\rm eff}(s)$",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_p2_equivalent_covariance.{ext}", bbox_inches="tight")
    plt.close(fig)

    spread = {}
    for target_p in (0.2, 0.5, 0.8):
        picked = [
            min((r for r in rows if r["sigma_det_px"] == sigma_det and r["prior"] == name),
                key=lambda r: abs(r["p_use"] - target_p))
            for name in PRIORS
        ]
        sigmas = [r["sigma_eff_px"] for r in picked]
        spread[f"p={target_p}"] = {
            "sigma_eff_px": dict(zip(PRIORS, sigmas)),
            "max_over_min": max(sigmas) / min(sigmas),
            "scaled_law_px": picked[0]["sigma_scaled_px"],
        }
    return {"sigma_det_px": sigma_det, "prior_spread": spread}


def fig_p3(rows: list[dict]) -> dict:
    """Where is the shortcut acceptable? Dense region map over (p_use, informativeness).

    Computed on its own dense grid rather than on the factorial rows: the y axis
    is the measurement-to-prior std ratio, which is what actually controls the
    error, and the factorial grid only visits 12 values of it.
    """

    sigma_prior = PRIORS["medium (30 px)"]
    prior = prior_matrix(sigma_prior)
    ratios = np.logspace(-1.5, 1.5, 61)
    ps = np.linspace(0.02, 1.0, 50)
    dense = {"scaled": np.zeros((ratios.size, ps.size)),
             "blend120": np.zeros((ratios.size, ps.size))}
    for i, ratio in enumerate(ratios):
        sigma_det = float(ratio * sigma_prior)
        r_det = isotropic(sigma_det)
        for j, p_use in enumerate(ps):
            reference = expected_posterior_branch(prior, H, r_det, float(p_use)).cov_expected
            scaled_post = kalman_posterior(
                prior, H, scaled_covariance_baseline(r_det, float(p_use))
            )
            blend_post = kalman_posterior(
                prior, H,
                precision_blend_covariance(
                    float(p_use), r_visible_uv=sigma_det, r_miss_uv=R_MISS_PX[-1]
                ),
            )
            dense["scaled"][i, j] = abs(relative_error(scaled_post, reference, "position_trace"))
            dense["blend120"][i, j] = abs(relative_error(blend_post, reference, "position_trace"))

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), sharey=True)
    im = None
    for ax, (key, label) in zip(
        axes,
        (("scaled", r"$R/p$  (endpoint-free)"),
         ("blend120", rf"deployed precision blend, $R_{{miss}}$={R_MISS_PX[-1]:.0f} px")),
    ):
        grid = dense[key]
        im = ax.pcolormesh(ps, ratios, np.log10(np.maximum(grid, 1e-4)),
                           cmap="magma_r", vmin=-3.0, vmax=0.0, shading="auto")
        cs = ax.contour(ps, ratios, grid, levels=[ACCEPTABLE_REL_ERROR],
                        colors="#00A0B0", linewidths=2.0)
        ax.clabel(cs, fmt=lambda v: f"{100 * v:.0f}%", fontsize=8)
        ax.set_yscale("log")
        ax.set_xlabel(r"$p_{\rm use}$")
        ax.set_title(label, fontweight="bold")
        ax.grid(False)
    axes[0].set_ylabel(r"$\sigma_{det}\ /\ \sigma_{\rm prior}$"
                       "\n(low = informative measurement)")
    cb = fig.colorbar(im, ax=axes.tolist())
    cb.set_label(r"$\log_{10}$ | relative position-trace error |   (dark = wrong)")
    fig.suptitle("The single-$R$ shortcut is only safe where the measurement barely informs the "
                 "belief\n(cyan contour = 10 % error; below it the shortcut is optimistic by more "
                 "than that)",
                 fontsize=12.0, fontweight="bold", y=1.06)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_p3_acceptable_region.{ext}", bbox_inches="tight")
    plt.close(fig)

    factorial = {
        key: sum(1 for r in rows if abs(r[key]) <= ACCEPTABLE_REL_ERROR) / len(rows)
        for key in ("scaled_rel_trace_error", "blend40_rel_trace_error",
                    "blend120_rel_trace_error")
    }
    return {
        "factorial_grid_fraction_within_10pct": factorial,
        "dense_grid_fraction_within_10pct": {
            key: float(np.mean(grid <= ACCEPTABLE_REL_ERROR)) for key, grid in dense.items()
        },
        "dense_grid": {
            "sigma_prior_px": sigma_prior,
            "ratio_range": [float(ratios[0]), float(ratios[-1])],
            "cells": int(ratios.size * ps.size),
        },
    }


def write_csv(rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys())
    with (OUT / "grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    """Headline numbers quoted in RESULTS.md."""

    def at(prior: str, sigma_det: float, p_use: float) -> dict:
        return min((r for r in rows if r["prior"] == prior and r["sigma_det_px"] == sigma_det),
                   key=lambda r: abs(r["p_use"] - p_use))

    spot = at("medium (30 px)", 5.0, 0.5)
    return {
        "spot_check_medium_prior_sigma5_p50": {
            "branch_pos_trace_px2": spot["branch_pos_trace_px2"],
            "scaled_rel_trace_error": spot["scaled_rel_trace_error"],
            "blend120_rel_trace_error": spot["blend120_rel_trace_error"],
            "blend40_rel_trace_error": spot["blend40_rel_trace_error"],
            "sigma_eff_px": spot["sigma_eff_px"],
            "sigma_scaled_px": spot["sigma_scaled_px"],
        },
        "median_abs_rel_trace_error": {
            key: float(np.median([abs(r[key]) for r in rows]))
            for key in ("scaled_rel_trace_error", "blend40_rel_trace_error",
                        "blend120_rel_trace_error")
        },
        "max_logdet_gap_nats": {
            key: float(max(abs(r[key]) for r in rows))
            for key in ("scaled_logdet_gap", "blend40_logdet_gap", "blend120_logdet_gap")
        },
        "grid_rows": len(rows),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    rows = build_grid()
    write_csv(rows)
    payload = {
        "config": {
            "priors_sigma_px": PRIORS,
            "sigma_det_px": list(SIGMA_DET_PX),
            "r_miss_px_swept": list(R_MISS_PX),
            "p_use_grid": [float(P_GRID[0]), float(P_GRID[-1]), len(P_GRID)],
            "xy_correlation": XY_CORRELATION,
            "theta_std_deg": THETA_STD_DEG,
            "acceptable_rel_error": ACCEPTABLE_REL_ERROR,
        },
        "summary": summarize(rows),
        "fig_p1": fig_p1(rows),
        "fig_p2": fig_p2(rows),
        "fig_p3": fig_p3(rows),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(json.dumps(payload["fig_p3"], indent=2))
    print("wrote ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
