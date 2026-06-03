#!/usr/bin/env python3
"""Trace the EFE objective's posterior belief covariance along the north route
for C1 (constant R) vs C2 (GP R), to test whether the GP meaningfully inflates
planner-facing covariance in the occluded gap.

Replicates the CasADi objective's per-step propagation in numpy:
   S <- predict(S, u);  mu,Sigma,Gamma = approx_observation(m,S,R_plan);
   S <- S - Gamma @ inv(Sigma) @ Gamma.T   (expected posterior, state_posterior_cov)
For C1 R_plan = R_visible (constant); for C2 R_plan = GP-blended (occlusion-aware).

Outputs per-step position-uncertainty (sqrt trace of xy cov, and 2-sigma major
axis) vs arc-length, with the GP p_vis overlaid. If C2's covariance balloons in
the low-p_vis gap while C1's stays small, the GP mechanism is real and quantified.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/visibility_comparison"))
from efe_offline_lab import load_setup  # noqa: E402


def step_posterior_S(planner, m, S, corr_method):
    """Fresh one-step expected posterior covariance used by the belief-nogo.

    Mirrors the planner EXACTLY: the carried covariance S is predict-only; the
    belief-nogo clearance at each step uses a fresh posterior of that S with the
    GP-conditioned R_plan (see base_planner._trajectory_plan_diagnostics L587 and
    casadi_efe S_drive L337). The posterior is NOT carried forward.
    """
    vis = planner.planning_visibility_diagnostics(m, S)
    _mu, Sigma, Gamma = planner.approx_observation(
        m, S, method=corr_method, R_override=vis['R_plan'])
    S_nogo = planner._expected_state_posterior_covariance(
        np.asarray(S, float), np.asarray(Sigma, float), np.asarray(Gamma, float))
    return np.asarray(S_nogo, float), float(vis['p_vis'])


def trace_route(setup, lane_xy):
    planner = setup.planner
    method = planner.approx_method
    m = np.array([*setup.start_xy_yaw], float)
    S = np.asarray(setup.S0, float).copy()
    wps = [np.asarray(p, float) for p in lane_xy]
    u = np.asarray(planner._controls_for_waypoints(m[:3], wps), float).reshape(-1, 2)
    xs, ys, sig_pred, sig_nogo, pvis, arc = [], [], [], [], [], []
    s_acc = 0.0
    prev = m[:2].copy()
    for k in range(u.shape[0]):
        m, S = planner.predict(m, S, u[k])  # predict-only carry (as the planner does)
        try:
            S_nogo, pv = step_posterior_S(planner, m, S, method)
        except Exception:
            S_nogo, pv = S, float("nan")
        s_acc += float(np.hypot(*(m[:2] - prev))); prev = m[:2].copy()
        xs.append(m[0]); ys.append(m[1])
        # predict-only carried covariance (same for C1/C2) vs the per-step
        # belief-nogo posterior (GP-conditioned, differs between C1/C2)
        ev_pred = np.linalg.eigvalsh(0.5 * (S[:2, :2] + S[:2, :2].T))
        ev_nogo = np.linalg.eigvalsh(0.5 * (S_nogo[:2, :2] + S_nogo[:2, :2].T))
        sig_pred.append(float(np.sqrt(max(ev_pred[-1], 0.0))))
        sig_nogo.append(float(np.sqrt(max(ev_nogo[-1], 0.0))))
        pvis.append(pv); arc.append(s_acc)
    return dict(x=np.array(xs), y=np.array(ys), sig_pred=np.array(sig_pred),
               sig_major=np.array(sig_nogo), pvis=np.array(pvis), arc=np.array(arc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", default="F31_b1_apron_a3_mid")
    ap.add_argument("--lane", default="mid_cross_lane", help="route name to trace")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    raw = yaml.safe_load(cfg_path.read_text())["optimizer_initial_routes_json"]
    routes = json.loads(raw) if isinstance(raw, str) else raw
    lane = {r["name"]: r["waypoints"] for r in routes}[args.lane]

    out = {}
    for cond in ("C1", "C2"):
        setup = load_setup(cfg_path, condition=cond, task_override=args.task)
        out[cond] = trace_route(setup, lane)
        t = out[cond]
        occ = t["pvis"] < 0.2
        print(f"{cond}: along '{args.lane}'  sig_major: max={t['sig_major'].max():.3f}m "
              f"end={t['sig_major'][-1]:.3f}m | mean sig_major where p_vis<0.2: "
              f"{(t['sig_major'][occ].mean() if occ.any() else float('nan')):.3f}m "
              f"(n_occluded_steps={int(occ.sum())})")

    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for cond, c in (("C1", "#2563eb"), ("C2", "#dc2626")):
        t = out[cond]
        ax[0].plot(t["arc"], t["sig_major"], color=c, lw=2, label=f"{cond} 2σ-major")
        ax[1].plot(t["arc"], t["pvis"], color=c, lw=2, label=f"{cond} p_vis")
    ax[0].set_ylabel("posterior position σ (major axis) [m]")
    ax[0].set_title(f"EFE-objective posterior covariance along '{args.lane}' (occluded route)")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].axhline(0.2, color="#888", ls="--", label="low-vis 0.2")
    ax[1].set_ylabel("GP p_vis"); ax[1].set_xlabel("arc length [m]")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    outdir = REPO / "logs/diagnostics/rollout_cov"
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"cov_trace_{args.lane}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Figure: {png}")


if __name__ == "__main__":
    main()
