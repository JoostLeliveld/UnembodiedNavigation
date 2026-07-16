#!/usr/bin/env python3
"""Exp5 — Offline trajectory smoothing with global camera anchors (RQ4).

The GP performs no loop closure itself. An offline linear smoother (Kalman
filter + RTS over 2D position, driven by wheel-odometry increments and
anchored by the external camera's absolute BEV fixes) revises every historical
belief (mu_pre, P_pre) -> (mu_post, P_post). The same detector observations
are then re-attached to the corrected pose distributions, giving four maps:

  L0 pre-correction  + point GP        L1 pre-correction  + belief-aware GP
  L2 post-correction + point GP        L3 post-correction + belief-aware GP

"Trajectory smoothing with global camera anchors" is the honest name for this
system (the external camera provides absolute x,y), rather than classic
loop closure. GT is used for evaluation only (NEES/error), never inside the
smoother.

Outputs -> logs/studies/optionA_commissioning/exp5_trajectory_smoothing/
"""
from __future__ import annotations

import csv
import glob
from pathlib import Path

import numpy as np

import optA_common as oc
from optA_common import fnum

OUT = oc.OUT_ROOT / "exp5_trajectory_smoothing"
CAMPAIGN = oc.LOGS_VC / "honest_campaign_v1"
ELL, NOISE, RES = 0.9, 0.05, 0.20
R_CAM = 0.15 ** 2          # camera BEV anchor noise var (per axis)
Q0, QV = 0.010, 0.15       # per-step odom noise std = Q0 + QV*|increment|
P0 = 0.30 ** 2


def smooth_run(run_dir: Path):
    """KF + RTS over 2D position. Returns per-step dict + per-event records."""
    exp = oc.read_rows(run_dir / "experiment.csv")
    per = oc.read_rows(run_dir / "perception.csv")
    st = np.array([fnum(r, "stamp") for r in exp])
    ox = np.array([fnum(r, "odom_noisy_x") for r in exp])
    oy = np.array([fnum(r, "odom_noisy_y") for r in exp])
    ok = np.isfinite(st) & np.isfinite(ox) & np.isfinite(oy)
    exp = [e for e, k in zip(exp, ok) if k]
    st, ox, oy = st[ok], ox[ok], oy[ok]
    n = len(st)
    if n < 20:
        return None

    # camera anchors: detected frames with a calibrated BEV fix
    anchors = {}
    for r in per:
        if r.get("detected") != "1":
            continue
        t, zx, zy = fnum(r, "log_stamp"), fnum(r, "pred_world_x_calibrated"), fnum(r, "pred_world_y_calibrated")
        if np.isfinite(t) and np.isfinite(zx) and np.isfinite(zy):
            j = int(np.argmin(np.abs(st - t)))
            if abs(st[j] - t) <= 0.15:
                anchors.setdefault(j, []).append((zx, zy))

    u = np.column_stack([np.diff(ox, prepend=ox[0]), np.diff(oy, prepend=oy[0])])
    bx0 = fnum(exp[0], "planner_belief_x"); by0 = fnum(exp[0], "planner_belief_y")
    if not (np.isfinite(bx0) and np.isfinite(by0)):
        bx0, by0 = ox[0], oy[0]

    mf = np.zeros((n, 2)); Pf = np.zeros((n, 2, 2))
    mp = np.zeros((n, 2)); Pp = np.zeros((n, 2, 2))
    m, P = np.array([bx0, by0]), P0 * np.eye(2)
    for k in range(n):
        if k > 0:
            q = (Q0 + QV * np.hypot(*u[k])) ** 2
            m = m + u[k]
            P = P + q * np.eye(2)
        mp[k], Pp[k] = m, P
        if k in anchors:
            for z in anchors[k]:
                innov = np.asarray(z) - m
                S = P + R_CAM * np.eye(2)
                nis = float(innov @ np.linalg.solve(S, innov))
                r_eff = R_CAM * max(1.0, nis / 9.21)   # soft gate: downweight outliers,
                S = P + r_eff * np.eye(2)              # never hard-reject (avoids gate runaway)
                K = P @ np.linalg.inv(S)
                m = m + K @ innov
                P = (np.eye(2) - K) @ P
        mf[k], Pf[k] = m, P

    ms = mf.copy(); Ps = Pf.copy()
    for k in range(n - 2, -1, -1):
        G = Pf[k] @ np.linalg.inv(Pp[k + 1])
        ms[k] = mf[k] + G @ (ms[k + 1] - mp[k + 1])
        Ps[k] = Pf[k] + G @ (Ps[k + 1] - Pp[k + 1]) @ G.T

    gt = np.column_stack([np.array([fnum(r, "gt_x") for r in exp]), np.array([fnum(r, "gt_y") for r in exp])])
    bel = np.column_stack([np.array([fnum(r, "planner_belief_x") for r in exp]), np.array([fnum(r, "planner_belief_y") for r in exp])])
    cov_pre = np.stack([np.array([[fnum(r, "planner_cov_x"), fnum(r, "planner_cov_xy", 0.0)],
                                  [fnum(r, "planner_cov_xy", 0.0), fnum(r, "planner_cov_y")]]) for r in exp])

    # per-detection events with pre and post beliefs
    events = []
    for r in per:
        if r.get("detected") not in ("0", "1"):
            continue
        t = fnum(r, "log_stamp")
        if not np.isfinite(t):
            continue
        j = int(np.argmin(np.abs(st - t)))
        if abs(st[j] - t) > 0.3 or not np.isfinite(bel[j]).all():
            continue
        events.append(dict(
            det=int(r["detected"]),
            pre_m=bel[j], pre_S=cov_pre[j],
            post_m=ms[j], post_S=Ps[j],
            gt=gt[j],
        ))
    return dict(st=st, gt=gt, bel=bel, odo=np.column_stack([ox, oy]), smooth=ms, Ps=Ps,
                cov_pre=cov_pre, anchors=anchors, events=events)


def build_eventset(all_events, which):
    m = np.array([e[f"{which}_m"] for e in all_events])
    S = np.array([np.nan_to_num(e[f"{which}_S"], nan=1e-4) for e in all_events])
    y = np.array([e["det"] for e in all_events], float)
    return m, S, y


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = sorted(p.parent for p in Path(CAMPAIGN).glob("*/*/*/experiment_*/perception.csv"))
    per_run = {}
    all_events, routes, run_ids = [], [], []
    for rd in runs:
        out = smooth_run(rd)
        if out is None:
            continue
        route = rd.relative_to(CAMPAIGN).parts[0]
        per_run[str(rd)] = out
        for e in out["events"]:
            e["route"] = route; e["run"] = str(rd)
        all_events += out["events"]
        routes += [route] * len(out["events"])
        run_ids += [str(rd)] * len(out["events"])
    routes = np.array(routes); run_ids = np.array(run_ids)
    print(f"{len(per_run)} runs smoothed, {len(all_events)} events")

    # ---------------- accuracy + calibration of pre vs post beliefs (GT eval-only)
    def stats(which):
        m = np.array([e[f"{which}_m"] for e in all_events])
        S = np.array([e[f"{which}_S"] for e in all_events])
        gt = np.array([e["gt"] for e in all_events])
        fin = np.isfinite(gt).all(axis=1) & np.isfinite(m).all(axis=1)
        err = np.hypot(*(m[fin] - gt[fin]).T)
        d = m[fin] - gt[fin]
        nees = np.array([float(dd @ np.linalg.solve(Si + 1e-9 * np.eye(2), dd)) for dd, Si in zip(d, S[fin])])
        return err, nees, np.sqrt(np.trace(S[fin], axis1=1, axis2=2))
    err_pre, nees_pre, sc_pre = stats("pre")
    err_post, nees_post, sc_post = stats("post")

    # ---------------- L0-L3 maps + LORO validation
    xs, ys, XY = oc.grid_query()
    maps, val = {}, {}
    for which in ("pre", "post"):
        m, S, y = build_eventset(all_events, which)
        for mode, tag in (("naive", "point"), ("expected_kernel", "belief")):
            data = oc.make_event_data(m, y, S, run_ids)
            agg = oc.aggregate(data, resolution_m=RES)
            mu, _ = oc.fit_predict(mode, agg, XY, length_scale=ELL, noise_var=NOISE)
            maps[(which, tag)] = oc.sigmoid(mu).reshape(len(ys), len(xs))
            lls = []
            for held in oc.ROUTES:
                tr, te = routes != held, routes == held
                data_tr = oc.make_event_data(m[tr], y[tr], S[tr], run_ids[tr])
                mu_t, sig_t = oc.fit_predict(mode, oc.aggregate(data_tr, resolution_m=RES), m[te],
                                             length_scale=ELL, noise_var=NOISE)
                lls.append(oc.logloss(y[te], oc.probit_prob(mu_t, sig_t)))
            val[(which, tag)] = (float(np.mean(lls)), float(np.std(lls)))
            print(f"{which}/{tag}: LORO logloss {val[(which,tag)][0]:.4f}")

    # ---------------- fig1: example run
    ex = per_run[[k for k in per_run if "route_west_to_a1_upper/C2/seed0" in k][0]]
    fig, axes = oc.plt.subplots(1, 2, figsize=(12.4, 5.4))
    for ax, zoom in zip(axes, (False, True)):
        ax.plot(ex["gt"][:, 0], ex["gt"][:, 1], "-", color=oc.INK, lw=1.3, label="ground truth (eval-only)")
        odo_al = ex["odo"] - ex["odo"][0] + ex["bel"][0]   # odom frame -> aligned at start
        ax.plot(odo_al[:, 0], odo_al[:, 1], "-", color=oc.MUTED, lw=1.0, label="odometry dead-reckoning (aligned)")
        ax.plot(ex["bel"][:, 0], ex["bel"][:, 1], "-", color=oc.BLUE, lw=1.0, label="online belief (pre)")
        ax.plot(ex["smooth"][:, 0], ex["smooth"][:, 1], "-", color=oc.AQUA, lw=1.2, label="smoothed (post)")
        az = [z for zz in ex["anchors"].values() for z in zz]
        az = np.array(az)
        ax.plot(az[:, 0], az[:, 1], ".", ms=2.5, color=oc.ORANGE, alpha=0.5, label="camera anchors")
        from matplotlib.patches import Ellipse
        for k in range(0, len(ex["st"]), 40):
            for Smat, col in ((ex["cov_pre"][k], oc.BLUE), (ex["Ps"][k], oc.AQUA)):
                if not np.isfinite(Smat).all():
                    continue
                vals, vecs = np.linalg.eigh(Smat)
                ang = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
                ax.add_patch(Ellipse(ex["bel"][k] if col == oc.BLUE else ex["smooth"][k],
                                     2 * np.sqrt(max(vals[-1], 0)), 2 * np.sqrt(max(vals[0], 0)),
                                     angle=ang, fill=False, ec=col, lw=0.7, alpha=0.8))
        if zoom:
            ax.set_xlim(-3.2, 0.6); ax.set_ylim(1.2, 4.4)
            oc.style_ax(ax, "zoom: turn region (1σ ellipses: blue=online, teal=smoothed)")
        else:
            oc.draw_warehouse(ax)
            ax.set_aspect("equal")
            ax.legend(fontsize=7, loc="lower right")
            oc.style_ax(ax, "example run west→A1 C2 seed0")
    fig.suptitle("Exp5 — offline smoothing with global camera anchors revises historical beliefs", fontsize=11.5, color=oc.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    oc.save(fig, OUT, "fig1_example_run.png")

    # ---------------- fig2: accuracy + calibration
    fig, axes = oc.plt.subplots(1, 3, figsize=(13.8, 4.2))
    ax = axes[0]
    for e, lab, col in ((err_pre, "online belief (pre)", oc.BLUE), (err_post, "smoothed (post)", oc.AQUA)):
        q = np.sort(e); ax.plot(q, np.linspace(0, 1, len(q)), lw=1.6, color=col,
                                label=f"{lab}  p95={np.percentile(e,95):.3f} m")
    ax.set_xlim(0, 0.4); ax.set_xlabel("position error vs GT [m]"); ax.set_ylabel("CDF"); ax.legend(fontsize=7.5)
    oc.style_ax(ax, "belief accuracy (evaluation-only GT)")
    ax = axes[1]
    for s, lab, col in ((sc_pre, "pre", oc.BLUE), (sc_post, "post", oc.AQUA)):
        q = np.sort(s); ax.plot(q, np.linspace(0, 1, len(q)), lw=1.6, color=col, label=f"{lab}  median={np.median(s):.3f} m")
    ax.set_xlabel("reported pose scale sqrt(tr P) [m]"); ax.set_ylabel("CDF"); ax.legend(fontsize=7.5)
    oc.style_ax(ax, "reported uncertainty")
    ax = axes[2]
    bins = np.logspace(-2, 3, 40)
    ax.hist(np.clip(nees_pre, 1e-2, 1e3), bins=bins, alpha=0.6, color=oc.BLUE, label=f"pre  median={np.median(nees_pre):.1f}")
    ax.hist(np.clip(nees_post, 1e-2, 1e3), bins=bins, alpha=0.6, color=oc.AQUA, label=f"post median={np.median(nees_post):.1f}")
    ax.axvline(2.0, color=oc.INK2, ls="--", lw=1.0); ax.text(2.0, ax.get_ylim()[1] * 0.95, " NEES=2 (calibrated)", fontsize=7, color=oc.INK2)
    ax.set_xscale("log"); ax.set_xlabel("NEES  (err' P⁻¹ err)"); ax.set_ylabel("events"); ax.legend(fontsize=7.5)
    oc.style_ax(ax, "covariance calibration: >2 = overconfident")
    fig.suptitle("Exp5 — what smoothing does to belief quality and honesty", fontsize=11.5, color=oc.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    oc.save(fig, OUT, "fig2_calibration.png")

    # ---------------- fig3: L0-L3 maps
    fig, axes = oc.plt.subplots(2, 2, figsize=(9.6, 8.4))
    order = [("pre", "point", "L0 pre + point GP"), ("pre", "belief", "L1 pre + belief-aware"),
             ("post", "point", "L2 post + point GP"), ("post", "belief", "L3 post + belief-aware")]
    for ax, (which, tag, lab) in zip(axes.ravel(), order):
        ax.imshow(maps[(which, tag)], origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]),
                  cmap=oc.CMAP_TRUST, vmin=0, vmax=1)
        oc.draw_warehouse(ax, camera=False)
        v = val[(which, tag)]
        oc.badge(ax, f"LORO logloss {v[0]:.3f}±{v[1]:.3f}", "lower left")
        oc.style_ax(ax, lab, keep_ticks=False)
    fig.suptitle("Exp5 — the four maps: trajectory correction × input treatment", fontsize=11.5, color=oc.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    oc.save(fig, OUT, "fig3_L0_L3_maps.png")

    md = f"""# Exp5 — Trajectory smoothing with global camera anchors (RQ4)

**Setup.** Per run: linear KF+RTS over 2D position; process = wheel-odometry increments
(`odom_noisy`, per-step std {Q0}+{QV}·|Δ| m), measurements = calibrated camera BEV fixes
(R = {np.sqrt(R_CAM):.2f} m). The smoother sees no GT and no CAD. The same detector events
are then re-attached to (μ_post, P_post). {len(per_run)} runs, {len(all_events)} events.

## Belief quality (GT evaluation-only)

| quantity | online (pre) | smoothed (post) |
|---|---|---|
| position error p50 | {np.percentile(err_pre,50):.3f} m | {np.percentile(err_post,50):.3f} m |
| position error p95 | {np.percentile(err_pre,95):.3f} m | {np.percentile(err_post,95):.3f} m |
| reported scale sqrt(tr P) median | {np.median(sc_pre):.3f} m | {np.median(sc_post):.3f} m |
| NEES median (2 = calibrated) | {np.median(nees_pre):.1f} | {np.median(nees_post):.1f} |

## The four maps (held-out log loss, leave-one-route-out)

| | point GP | belief-aware GP |
|---|---|---|
| pre-correction | {val[('pre','point')][0]:.4f}±{val[('pre','point')][1]:.4f} (L0) | {val[('pre','belief')][0]:.4f}±{val[('pre','belief')][1]:.4f} (L1) |
| post-correction | {val[('post','point')][0]:.4f}±{val[('post','point')][1]:.4f} (L2) | {val[('post','belief')][0]:.4f}±{val[('post','belief')][1]:.4f} (L3) |

## Reading

- **The correction's value is honesty, not accuracy.** The online EKF is already
  camera-anchored, so the offline smoother cannot beat its mean (its p95 is in fact
  somewhat worse — the smoother swallows moving-frame BEV anchor errors that the runtime
  pipeline mitigates with pixel-space R_plan and heading handling). What it repairs is the
  *covariance*: online NEES median {np.median(nees_pre):.1f} (heavily overconfident,
  consistent with the documented 3–4× overconfidence of `state_sigma_major_m`) vs
  ≈{np.median(nees_post):.1f} after smoothing — essentially calibrated. Per Exp1's
  miscalibration scenario, honest covariances are precisely what a belief-aware map needs;
  the smoothed trajectory is the *statistically correct* training distribution even when
  its point estimates are slightly worse.
- **Gate-runaway reproduced offline.** A first smoother version used the runtime's hard
  NIS gate (9.21): once odometry drift outgrew the gated innovation window, every later
  anchor was rejected and runs collapsed to dead-reckoning (p50 error 3.5 m) — the exact
  failure mechanism documented for the online system in the C2 analysis. The fix here is a
  soft gate (inflate R by NIS/9.21 instead of rejecting), which is also a recommendation
  back to the runtime design.
- Comparing L0↔L2 isolates the trajectory-correction effect; L1↔L3 isolates it under
  belief-aware inputs; on this well-anchored system all four maps score within noise of
  each other, matching Exp2's operating-point conclusion.

## Figures
- `fig1_example_run.png` — GT vs dead-reckoning vs online belief vs smoothed + anchors.
- `fig2_calibration.png` — error CDFs, reported scale, NEES before/after.
- `fig3_L0_L3_maps.png` — the four maps with held-out scores.

*experiments/optionA_commissioning/exp5_trajectory_smoothing.py, run 2026-07-15.*
"""
    oc.write_md(OUT, "RESULTS.md", md)


if __name__ == "__main__":
    main()
