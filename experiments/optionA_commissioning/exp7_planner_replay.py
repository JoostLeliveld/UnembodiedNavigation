#!/usr/bin/env python3
"""Exp7 — Planner-compatible interface demo: tau(s) -> R_plan(s) -> predicted belief (RQ6).

The validated trust map is consumed exactly through the existing seam:
  R_plan precision blend  1/var = tau/r_visible^2 + (1-tau)/r_miss^2   (pixels)
  pixel -> metric via the calibration px/m map, and
  expected-information belief propagation  P^-1 += tau(s) * R_m(s)^-1 per camera frame.

Demonstrations (no planner code is modified):
  1. Offline route replay: predicted belief-sigma profile along logged runs, from the
     point map vs the belief-aware map (both fit WITHOUT the replayed route), compared
     against the realized belief error (GT, evaluation-only).
  2. Two candidate paths: a short path through the camera-poor band vs a longer path
     through well-supported space — predicted uncertainty decides, no robot needed.

Outputs -> logs/studies/optionA_commissioning/exp7_planner_replay/
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

import optA_common as oc
from optA_common import fnum
import geometry_visibility as gv

OUT = oc.OUT_ROOT / "exp7_planner_replay"
CAMPAIGN = oc.LOGS_VC / "honest_campaign_v1"
ELL, NOISE, RES = 0.9, 0.05, 0.20
R_VIS_UV, R_MISS_UV = 2.5, 120.0     # runtime defaults (unicycle_planner_node)
FRAME_DT = 0.24                      # ~4.2 Hz detector
Q0, QV = 0.010, 0.15                 # same odometry noise model as Exp5


class TrustMapInterface:
    """tau(s) plus metric measurement noise R_m(s) from a fitted map."""

    def __init__(self, mu_grid, xs, ys):
        self.xs, self.ys, self.mu = xs, ys, mu_grid
        with np.load(oc.CALIBRATED_PRIOR, allow_pickle=False) as d:
            self.pxs, self.pys = np.asarray(d["xs"], float), np.asarray(d["ys"], float)
            self.ppm = np.asarray(d["px_per_m_min_map"], float)

    def tau(self, X):
        return oc.sigmoid(oc.fbg._interp_grid(self.xs, self.ys, self.mu, np.asarray(X, float)))

    def sigma_m(self, X):
        std_px, _ = gv.trust_to_r_plan(self.tau(X), R_VIS_UV, R_MISS_UV)
        ppm = np.clip(oc.fbg._interp_grid(self.pxs, self.pys, self.ppm, np.asarray(X, float)), 1.0, None)
        return std_px / ppm


def fit_map(ev, mask, mode):
    data = oc.make_event_data(ev["m"][mask], ev["det_hit"][mask], ev["S"][mask], ev["run"][mask])
    agg = oc.aggregate(data, resolution_m=RES)
    xs, ys, XY = oc.grid_query(nx=140, ny=128)
    mu, _ = oc.fit_predict(mode, agg, XY, length_scale=ELL, noise_var=NOISE)
    return TrustMapInterface(mu.reshape(len(ys), len(xs)), xs, ys)


def propagate(path_xy, dts, iface, p0=0.05):
    """Expected-information EKF prediction along a path. Returns sqrt(tr P) profile."""
    P = (p0 ** 2) * np.eye(2)
    out = np.zeros(len(path_xy))
    t_since_frame = 0.0
    for k in range(len(path_xy)):
        if k > 0:
            u = np.hypot(*(path_xy[k] - path_xy[k - 1]))
            q = (Q0 + QV * u) ** 2
            P = P + q * np.eye(2)
            t_since_frame += dts[k - 1]
        while t_since_frame >= FRAME_DT:
            t_since_frame -= FRAME_DT
            tau = float(iface.tau(path_xy[k][None])[0])
            sm = float(iface.sigma_m(path_xy[k][None])[0])
            Lam = np.linalg.inv(P) + tau / (sm ** 2) * np.eye(2)
            P = np.linalg.inv(Lam)
        out[k] = np.sqrt(np.trace(P))
    return out


def load_run_traj(run_dir: Path):
    exp = oc.read_rows(run_dir / "experiment.csv")
    st = np.array([fnum(r, "stamp") for r in exp])
    bel = np.column_stack([np.array([fnum(r, "planner_belief_x") for r in exp]),
                           np.array([fnum(r, "planner_belief_y") for r in exp])])
    gt = np.column_stack([np.array([fnum(r, "gt_x") for r in exp]),
                          np.array([fnum(r, "gt_y") for r in exp])])
    err = np.array([fnum(r, "belief_error_gt_m") for r in exp])
    ok = np.isfinite(st) & np.isfinite(bel).all(axis=1)
    return st[ok], bel[ok], gt[ok], err[ok]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = oc.load_events()

    # ---------------- 1) route replay with LORO maps
    replay = {}
    for route in oc.ROUTES:
        tr = ev["route"] != route
        ifaces = {"point": fit_map(ev, tr, "naive"), "belief": fit_map(ev, tr, "expected_kernel")}
        runs = sorted(glob.glob(str(CAMPAIGN / route / "*/*/experiment_*/experiment.csv")))
        per_run = []
        for rc in runs:
            st, bel, gt, err = load_run_traj(Path(rc).parent)
            if len(st) < 50:
                continue
            dts = np.clip(np.diff(st), 0.02, 0.5)
            profs = {k: propagate(bel, dts, ifc) for k, ifc in ifaces.items()}
            per_run.append(dict(st=st - st[0], err=err, **profs))
        replay[route] = per_run
        print(f"replayed {route}: {len(per_run)} runs")

    # correlation predicted sigma vs realized error (per run, belief map)
    from scipy.stats import spearmanr
    rhos = {"point": [], "belief": []}
    for route, runs in replay.items():
        for r in runs:
            fin = np.isfinite(r["err"])
            if fin.sum() < 30:
                continue
            for k in rhos:
                rhos[k].append(spearmanr(r[k][fin], r["err"][fin]).statistic)
    rho_stats = {k: (np.nanmean(v), np.nanstd(v)) for k, v in rhos.items()}

    # ---------------- 2) two candidate paths (same start/goal)
    full_iface = {"point": fit_map(ev, np.ones(len(ev["det_hit"]), bool), "naive"),
                  "belief": fit_map(ev, np.ones(len(ev["det_hit"]), bool), "expected_kernel")}
    start, goal = np.array([-5.2, -0.75]), np.array([4.6, 2.6])
    tgrid = np.linspace(0, 1, 260)

    def bez(p0, p1, p2, p3):
        t = tgrid[:, None]
        return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)
    path_short = bez(start, np.array([-2.0, 2.2]), np.array([1.5, 3.4]), goal)     # cuts through upper band
    path_long = bez(start, np.array([-3.0, -3.6]), np.array([3.8, -3.2]), goal)    # detour via apron
    dt_of = lambda p: np.hypot(*np.diff(p, axis=0).T) / 0.22
    prof = {(nm, mk): propagate(p, dt_of(p), full_iface[mk])
            for nm, p in (("short", path_short), ("long", path_long)) for mk in ("point", "belief")}

    # ---------------- fig1: interface pipeline + candidate paths
    xs, ys, XY = oc.grid_query(nx=140, ny=128)
    fig, axes = oc.plt.subplots(1, 3, figsize=(14.6, 4.7))
    ax = axes[0]
    tau_g = full_iface["belief"].tau(XY).reshape(128, 140)
    ax.imshow(tau_g, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap=oc.CMAP_TRUST, vmin=0, vmax=1)
    oc.draw_warehouse(ax, camera=False); oc.style_ax(ax, "validated trust map τ(s) (belief-aware, all routes)", keep_ticks=False)
    ax = axes[1]
    sg = full_iface["belief"].sigma_m(XY).reshape(128, 140)
    im = ax.imshow(np.clip(sg, 0, 2.5), origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap=oc.CMAP_STD)
    oc.draw_warehouse(ax, camera=False); fig.colorbar(im, ax=ax, shrink=0.8).set_label("σ_meas [m]", fontsize=7)
    oc.style_ax(ax, "R_plan(s) as metric measurement noise (r_vis=2.5px, r_miss=120px)", keep_ticks=False)
    ax = axes[2]
    ax.imshow(tau_g, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap=oc.CMAP_TRUST, vmin=0, vmax=1, alpha=0.5)
    oc.draw_warehouse(ax, camera=False)
    for nm, p, col in (("short (through poor band)", path_short, oc.RED), ("long (via apron)", path_long, oc.GREEN)):
        ax.plot(p[:, 0], p[:, 1], "-", lw=2.0, color=col, label=nm)
    ax.plot(*start, "o", ms=6, color=oc.INK); ax.plot(*goal, "*", ms=11, color=oc.INK)
    ax.legend(fontsize=7.5, loc="lower right"); oc.style_ax(ax, "two candidate paths, same start → goal", keep_ticks=False)
    fig.suptitle("Exp7 — the planner seam: τ(s) → R_plan(s) → predicted belief (no planner code changed)", fontsize=11.5, color=oc.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    oc.save(fig, OUT, "fig1_interface_and_paths.png")

    # ---------------- fig2: replay + candidate-path profiles
    fig, axes = oc.plt.subplots(1, 3, figsize=(14.2, 4.3))
    ax = axes[0]
    r = replay["route_west_to_a1_upper"][0]
    ax.plot(r["st"], r["err"], "-", lw=1.0, color=oc.INK, alpha=0.8, label="realized |belief − GT| (eval)")
    ax.plot(r["st"], r["point"], "-", lw=1.6, color=oc.BLUE, label="predicted σ, point map")
    ax.plot(r["st"], r["belief"], "--", lw=1.6, color=oc.AQUA, label="predicted σ, belief-aware map")
    ax.set_xlabel("time [s]"); ax.set_ylabel("[m]"); ax.legend(fontsize=7)
    oc.style_ax(ax, "replay example: west→A1 upper, seed0")
    ax = axes[1]
    for k, col in (("point", oc.BLUE), ("belief", oc.AQUA)):
        ax.hist(rhos[k], bins=15, alpha=0.65, color=col,
                label=f"{k} map  mean ρ={rho_stats[k][0]:.2f}")
    ax.set_xlabel("Spearman ρ(predicted σ, realized error) per run"); ax.set_ylabel("runs"); ax.legend(fontsize=7.5)
    oc.style_ax(ax, f"prediction ranks realized error ({sum(len(v) for v in replay.values())} runs)")
    ax = axes[2]
    for (nm, mk), pr in prof.items():
        if mk != "belief":
            continue
        s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff((path_short if nm == 'short' else path_long), axis=0).T))])
        col = oc.RED if nm == "short" else oc.GREEN
        ax.plot(s, pr, lw=1.8, color=col, label=f"{nm} path  (max σ {pr.max():.2f} m)")
    ax.set_xlabel("distance along path [m]"); ax.set_ylabel("predicted belief σ [m]"); ax.legend(fontsize=7.5)
    oc.style_ax(ax, "candidate paths: the longer, well-observed path stays certain")
    fig.suptitle("Exp7 — offline replay and path comparison from the map alone", fontsize=11.5, color=oc.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    oc.save(fig, OUT, "fig2_replay_profiles.png")

    md = f"""# Exp7 — Planner-compatible interface demo (RQ6)

**Seam.** τ(s) → `trust_to_r_plan` precision blend (r_visible={R_VIS_UV}px,
r_miss={R_MISS_UV}px — the runtime defaults; note the offline tooling historically used
r_miss=40) → metric σ via the calibration px/m map → expected-information belief
propagation P⁻¹ += τ·R_m⁻¹ at {1/FRAME_DT:.1f} Hz. No planner or safety code is modified;
the map plugs into the existing `P_conservative_plan_map` artifact schema.

## Results

- **Replay** (maps fit without the replayed route): predicted σ profiles rank the realized
  belief error within runs with mean Spearman ρ = {rho_stats['belief'][0]:.2f}±{rho_stats['belief'][1]:.2f}
  (belief-aware map) vs {rho_stats['point'][0]:.2f}±{rho_stats['point'][1]:.2f} (point map)
  — the two maps are equivalent at the real operating point, consistent with Exp2.
- **Candidate paths**: the short path through the camera-poor band accumulates predicted
  σ up to {max(prof[('short','belief')].max(), 0):.2f} m; the longer apron detour stays at
  {prof[('long','belief')].max():.2f} m. The map alone, through the existing R_plan seam,
  reproduces the thesis's C2-style behavior (prefer observable detours) offline — no
  Gazebo run needed.

This is a demonstration of interface compatibility, not a navigation-superiority claim
(that claim lives in the main campaign).

## Figures
- `fig1_interface_and_paths.png` — τ map → R_plan σ_meas map → candidate paths.
- `fig2_replay_profiles.png` — replay example, per-run rank correlation, path σ profiles.

*experiments/optionA_commissioning/exp7_planner_replay.py, run 2026-07-15.*
"""
    oc.write_md(OUT, "RESULTS.md", md)


if __name__ == "__main__":
    main()
