#!/usr/bin/env python3
"""Figures for exp1: what the hit/miss mixture changes versus the precision blend.

Calls the REAL runtime expressions from ``planning.core.casadi_efe`` (no
reimplementation), evaluated numerically through CasADi DM, so the figures show
the shipped algebra rather than a paraphrase of it.

Outputs -> logs/studies/efe_hit_miss_mixture/exp1_mixture_vs_blend/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))
from shared.paths import repo_root  # noqa: E402

REPO = repo_root()
for _pkg in ("planning", "unav_common"):
    _p = REPO / "src" / _pkg
    if _p.is_dir():
        sys.path.insert(0, str(_p))

import casadi as ca  # noqa: E402
from planning.core.casadi_efe import (  # noqa: E402
    CasadiEfeParams,
    _blend_observation_covariance_ca,
    hit_miss_posterior_ca,
)

OUT = REPO / "logs" / "studies" / "efe_hit_miss_mixture" / "exp1_mixture_vs_blend"

# Runtime-representative constants. r_visible 2.5 px and r_miss 120 px are the
# planner-runtime endpoints carried by reliability.covariance_mapping.
R_VISIBLE_PX = 2.5
R_MISS_PX = 120.0

C_BLEND = "#D55E00"    # current path
C_MIX = "#0072B2"      # proposed mixture
C_HIT = "#009E73"      # hit branch
C_MISS = "#E69F00"     # miss branch


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "axes.grid": True, "grid.color": "#CCCCCC",
        "grid.alpha": 0.3, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "font.size": 9,
    })


def _params() -> CasadiEfeParams:
    return CasadiEfeParams(
        R_visible=ca.DM(np.diag([R_VISIBLE_PX ** 2, R_VISIBLE_PX ** 2])),
        R_miss=ca.DM(np.diag([R_MISS_PX ** 2, R_MISS_PX ** 2])),
        control_weight=1.0, risk_scale=1.0, ambiguity_scale=1.0,
        discount_gamma=1.0, process_noise_xy=0.01, process_noise_theta=0.01,
        visibility_sigma_kappa=1.0,
        goal_prior_u_std_start=1.0, goal_prior_v_std_start=1.0,
        goal_prior_u_std_final=1.0, goal_prior_v_std_final=1.0,
        goal_tightening_power=1.0, goal_progress_n_steps=1,
        use_belief_nogo_cost=False, time_horizon=1, dt=0.1, Du=2,
    )


def _prior(sigma_px: float) -> np.ndarray:
    """3-state prior (x, y, theta) with a mild xy correlation."""
    s2 = sigma_px ** 2
    return np.array([
        [s2, 0.25 * s2, 0.0],
        [0.25 * s2, s2, 0.0],
        [0.0, 0.0, (5.0 * np.pi / 180.0) ** 2],
    ])


H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _blend_posterior(S: np.ndarray, p: float, params) -> np.ndarray:
    """Current path: one deterministic update with the precision-blended R."""
    R = np.array(ca.DM(_blend_observation_covariance_ca(ca.DM(p), params)))
    P_hit, _, _, _ = _branches(S, R)
    return P_hit


def _branches(S: np.ndarray, R: np.ndarray):
    P_mix, P_hit, S_sym, Sigma = hit_miss_posterior_ca(
        ca.DM(S), ca.DM(H), ca.DM(R), ca.DM(1.0)
    )
    return (np.array(ca.DM(P_hit)), np.array(ca.DM(P_mix)),
            np.array(ca.DM(S_sym)), np.array(ca.DM(Sigma)))


def _mixture_posterior(S: np.ndarray, p: float, R: np.ndarray) -> np.ndarray:
    P_mix, _, _, _ = hit_miss_posterior_ca(ca.DM(S), ca.DM(H), ca.DM(R), ca.DM(p))
    return np.array(ca.DM(P_mix))


def _pos_trace(P: np.ndarray) -> float:
    return float(P[0, 0] + P[1, 1])


def fig_e1(params) -> dict:
    sigma0 = 30.0
    S = _prior(sigma0)
    R_cond = np.diag([R_VISIBLE_PX ** 2, R_VISIBLE_PX ** 2])
    ps = np.linspace(0.0, 1.0, 201)

    tr_blend, tr_mix = [], []
    for p in ps:
        tr_blend.append(_pos_trace(_blend_posterior(S, float(p), params)))
        tr_mix.append(_pos_trace(_mixture_posterior(S, float(p), R_cond)))
    tr_blend, tr_mix = np.array(tr_blend), np.array(tr_mix)
    tr_hit = _pos_trace(_branches(S, R_cond)[0])
    tr_miss = _pos_trace(S)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax.axhline(tr_miss, color=C_MISS, lw=2, ls="--",
               label=f"miss branch $P^-$ (no update) = {tr_miss:.0f}")
    ax.axhline(tr_hit, color=C_HIT, lw=2, ls="--",
               label=f"hit branch $P^+_{{hit}}$ = {tr_hit:.1f}")
    ax.plot(ps, tr_mix, color=C_MIX, lw=2.4, label="proposed mixture $E[P^+]$")
    ax.plot(ps, tr_blend, color=C_BLEND, lw=2.4, label="current precision blend")
    ax.set_yscale("log")
    ax.set_xlabel(r"$p_{\rm use}$  (probability a usable detection arrives)")
    ax.set_ylabel("posterior position trace  [px$^2$]")
    ax.set_title("The blend collapses to a near-certain measurement\n"
                 "as soon as $p_{\\rm use}$ leaves zero", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")

    i50 = int(np.argmin(np.abs(ps - 0.5)))
    ax.plot([0.5], [tr_blend[i50]], "o", color=C_BLEND, ms=7, zorder=5)
    ax.plot([0.5], [tr_mix[i50]], "o", color=C_MIX, ms=7, zorder=5)
    ax.annotate(
        f"at $p_{{\\rm use}}$=0.5 the blend reports\n"
        f"{tr_blend[i50]:.1f} px$^2$ but the true expected\n"
        f"value is {tr_mix[i50]:.0f} px$^2$ — {tr_mix[i50] / tr_blend[i50]:.0f}$\\times$ larger",
        xy=(0.5, tr_blend[i50]), xytext=(0.17, tr_blend[i50] * 6.0),
        fontsize=8, arrowprops=dict(arrowstyle="->", color="#444444", lw=1.0))

    ratio = tr_mix / np.maximum(tr_blend, 1e-12)
    ax2.plot(ps, ratio, color=C_MIX, lw=2.4)
    ax2.axhline(1.0, color="#666666", lw=1.0, ls=":")
    ax2.set_xlabel(r"$p_{\rm use}$")
    ax2.set_ylabel(r"$\mathrm{tr}\,E[P^+]\ /\ \mathrm{tr}\,P_{\rm blend}$")
    ax2.set_yscale("log")
    ax2.set_title("Understatement factor of the current model\n"
                  f"(peak {np.nanmax(ratio):.0f}$\\times$ at "
                  f"$p_{{\\rm use}}$={ps[int(np.nanargmax(ratio))]:.2f})",
                  fontweight="bold")

    fig.suptitle(
        "Precision-blending availability into R understates belief uncertainty by orders of magnitude",
        fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_e1_mixture_vs_blend.{ext}", bbox_inches="tight")
    plt.close(fig)
    return dict(tr_hit=tr_hit, tr_miss=tr_miss,
                blend_at_50=float(tr_blend[i50]), mix_at_50=float(tr_mix[i50]),
                peak_ratio=float(np.nanmax(ratio)),
                peak_ratio_p=float(ps[int(np.nanargmax(ratio))]))


def fig_e2(params) -> dict:
    ps = np.linspace(0.01, 0.99, 90)
    sigmas = np.linspace(5.0, 80.0, 80)
    R_cond = np.diag([R_VISIBLE_PX ** 2, R_VISIBLE_PX ** 2])
    Z = np.zeros((sigmas.size, ps.size))
    for i, sg in enumerate(sigmas):
        S = _prior(float(sg))
        for j, p in enumerate(ps):
            b = _pos_trace(_blend_posterior(S, float(p), params))
            m = _pos_trace(_mixture_posterior(S, float(p), R_cond))
            Z[i, j] = np.log10(max(m, 1e-12) / max(b, 1e-12))

    v = float(np.nanmax(np.abs(Z)))
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    im = ax.pcolormesh(ps, sigmas, Z, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
    cs = ax.contour(ps, sigmas, Z, levels=[1, 2, 3], colors="#333333",
                    linewidths=0.9, linestyles="--")
    ax.clabel(cs, fmt=lambda x: f"{10 ** x:.0f}x", fontsize=8)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"$\log_{10}$  ( mixture trace / blend trace )")
    ax.set_xlabel(r"$p_{\rm use}$")
    ax.set_ylabel(r"prior position std $\sigma_0$  [px]")
    ax.set_title("The current model is most wrong exactly where planning matters:\n"
                 "uncertain belief in partially-observed space (red = blend too optimistic)",
                 fontweight="bold")
    ax.grid(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_e2_discrepancy_map.{ext}", bbox_inches="tight")
    plt.close(fig)
    return dict(max_log10_ratio=v)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    params = _params()
    s1 = fig_e1(params)
    s2 = fig_e2(params)
    print("fig_e1:", s1)
    print("fig_e2:", s2)
    print("wrote ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
