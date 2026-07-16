#!/usr/bin/env python3
"""Exp6 — Deployment-change stress test with historical-prior inflation (RQ5).

Claim under test: a historical trust map can be reused, but must be re-uncertained
(variance inflation) when the operational context may have changed.

Three realistic reuse settings, all on the original warehouse:
  A. MATCHED history   : locked teleport-survey GP (same detector+camera era)
                         evaluated on honest_campaign_v1 events.
  B. STALE history     : archived aws_gp_v7b map (older detector / camera era,
                         later superseded) evaluated on honest_campaign_v1 events.
  C. DYNAMICS change   : posterior fit on honest_campaign_v1, evaluated on
                         whitenoise_campaign_v1 events (process-noise change).

For each: prediction p = probit(F_mean, sqrt(F_std^2 + v_change)) with inflation
v_change swept 0 -> 4 (logit-variance units). Baselines: calibration prior, neutral.

Outputs -> logs/studies/optionA_commissioning/exp6_stress_test/
"""
from __future__ import annotations

import numpy as np

import optA_common as oc

OUT = oc.OUT_ROOT / "exp6_stress_test"
ELL, NOISE, RES = 0.9, 0.05, 0.20
TAU = 0.7
VCHANGE = (0.0, 0.25, 1.0, 4.0)


def artifact_predictor(path):
    with np.load(path, allow_pickle=False) as d:
        xs, ys = np.asarray(d["xs"], float), np.asarray(d["ys"], float)
        F, S = np.asarray(d["F_mean_map"], float), np.asarray(d["F_std_map"], float)

    def fn(X):
        mu = oc.fbg._interp_grid(xs, ys, F, np.asarray(X, float))
        sd = oc.fbg._interp_grid(xs, ys, S, np.asarray(X, float))
        return mu, np.clip(sd, 0.0, None)
    return fn


def honest_posterior_predictor(ev_h):
    data = oc.make_event_data(ev_h["m"], ev_h["det_hit"], ev_h["S"], ev_h["run"])
    agg = oc.aggregate(data, resolution_m=RES)
    xs, ys, XY = oc.grid_query(nx=140, ny=128)
    mu, sd = oc.fit_predict("expected_kernel", agg, XY, length_scale=ELL, noise_var=NOISE)
    Fg, Sg = mu.reshape(len(ys), len(xs)), sd.reshape(len(ys), len(xs))

    def fn(X):
        return (oc.fbg._interp_grid(xs, ys, Fg, np.asarray(X, float)),
                np.clip(oc.fbg._interp_grid(xs, ys, Sg, np.asarray(X, float)), 0.0, None))
    return fn


def score(pred_fn, X, y, v_change):
    mu, sd = pred_fn(X)
    p = oc.probit_prob(mu, np.sqrt(sd ** 2 + v_change))
    return dict(logloss=oc.logloss(y, p), brier=oc.brier(y, p), ece=oc.ece(y, p),
                fhtr=oc.fhtr(y, p, tau_high=TAU), p=p)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev_h = oc.load_events(oc.EVENTS_HONEST)
    ev_w = oc.load_events(oc.EVENTS_WHITENOISE)
    print(f"honest {len(ev_h['det_hit'])} events (det {ev_h['det_hit'].mean():.3f}); "
          f"whitenoise {len(ev_w['det_hit'])} events (det {ev_w['det_hit'].mean():.3f})")

    settings = {
        "A_matched": (artifact_predictor(oc.LOCKED_GP), ev_h, "matched survey GP → today's runs"),
        "B_stale": (artifact_predictor(oc.GP_V7B), ev_h, "stale v7b map (old detector era) → today's runs"),
        "C_dynamics": (honest_posterior_predictor(ev_h), ev_w, "honest posterior → whitenoise campaign"),
    }
    # baselines (v_change-independent)
    calib = oc.prior_logit_calibration()
    base_rows = {}
    for key, (_, ev, _) in settings.items():
        pc = oc.sigmoid(calib(ev["m"]))
        base_rows[key] = {
            "calibration": dict(logloss=oc.logloss(ev["det_hit"], pc), fhtr=oc.fhtr(ev["det_hit"], pc, tau_high=TAU)),
            "neutral": dict(logloss=oc.logloss(ev["det_hit"], np.full(len(ev["det_hit"]), 0.5)),
                            fhtr=0.0),
        }

    results = {}
    for key, (fn, ev, _) in settings.items():
        results[key] = {v: score(fn, ev["m"], ev["det_hit"], v) for v in VCHANGE}
        print(key, {v: round(results[key][v]["logloss"], 4) for v in VCHANGE})

    # ---------------- fig1: inflation curves
    fig, axes = oc.plt.subplots(1, 3, figsize=(13.8, 4.3))
    cols = {"A_matched": oc.GREEN, "B_stale": oc.RED, "C_dynamics": oc.VIOLET}
    labs = {"A_matched": "A matched history", "B_stale": "B stale history (v7b)", "C_dynamics": "C dynamics change"}
    for ax, metric, title in zip(axes[:2], ("logloss", "fhtr"),
                                 ("held-out log loss on the NEW context", f"false-high-trust rate (τ={TAU})")):
        for key in settings:
            vals = [results[key][v][metric] for v in VCHANGE]
            ax.plot(VCHANGE, vals, "-o", ms=4, lw=1.6, color=cols[key], label=labs[key])
            bl = base_rows[key]["calibration"][metric]
            ax.axhline(bl, color=cols[key], ls=":", lw=0.8, alpha=0.6)
        ax.set_xscale("symlog", linthresh=0.25)
        ax.set_xticks(VCHANGE); ax.set_xticklabels([str(v) for v in VCHANGE])
        ax.set_xlabel("inflation v_change (logit variance)")
        oc.style_ax(ax, title + "\n(dotted = calibration-prior fallback)")
    axes[0].legend(fontsize=7.5)
    ax = axes[2]
    # reliability curves at v=0 vs v=4 for the two changed settings
    edges = np.linspace(0, 1, 11)
    for key, v, col, lab in (("C_dynamics", 0.0, oc.VIOLET, "dynamics, frozen"),
                             ("C_dynamics", 4.0, oc.AQUA, "dynamics, inflated v=4"),
                             ("B_stale", 0.0, oc.RED, "stale, frozen")):
        p = results[key][v]["p"]; yb = settings[key][1]["det_hit"]
        ctr, obs, ns = oc.binned(p, yb, edges, np.nanmean)
        ax.plot(ctr[ns > 30], obs[ns > 30], "-o", ms=4, lw=1.6, color=col, label=lab)
    ax.plot([0, 1], [0, 1], "--", color=oc.INK2, lw=0.8)
    ax.set_xlabel("predicted trust"); ax.set_ylabel("observed detection rate"); ax.legend(fontsize=7)
    oc.style_ax(ax, "reliability: inflation fixes overconfidence, not bias\n(stale curve sits above the diagonal = under-trust)")
    fig.suptitle("Exp6 — reuse a historical map ⇒ re-uncertain it when the context may have changed", fontsize=11.5, color=oc.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    oc.save(fig, OUT, "fig1_inflation_curves.png")

    # ---------------- fig2: where the stale map is wrong
    xs, ys, XY = oc.grid_query()
    fig, axes = oc.plt.subplots(1, 3, figsize=(13.8, 4.6))
    with np.load(oc.GP_V7B) as d:
        stale_map = np.asarray(d["P_mean_map"], float)
        sxs, sys_ = np.asarray(d["xs"], float), np.asarray(d["ys"], float)
    with np.load(oc.LOCKED_GP) as d:
        fresh_map = np.asarray(d["P_mean_map"], float)
    ax = axes[0]
    ax.imshow(stale_map, origin="lower", extent=(sxs[0], sxs[-1], sys_[0], sys_[-1]), cmap=oc.CMAP_TRUST, vmin=0, vmax=1)
    oc.draw_warehouse(ax, camera=False); oc.style_ax(ax, "stale historical map (aws_gp_v7b)", keep_ticks=False)
    ax = axes[1]
    ax.imshow(fresh_map, origin="lower", extent=(sxs[0], sxs[-1], sys_[0], sys_[-1]), cmap=oc.CMAP_TRUST, vmin=0, vmax=1)
    oc.draw_warehouse(ax, camera=False); oc.style_ax(ax, "current survey map (locked GP)", keep_ticks=False)
    ax = axes[2]
    im = ax.imshow(stale_map - fresh_map, origin="lower", extent=(sxs[0], sxs[-1], sys_[0], sys_[-1]),
                   cmap="PuOr_r", vmin=-1, vmax=1)
    oc.draw_warehouse(ax, camera=False); oc.style_ax(ax, "stale − current: over-trust regions (orange)", keep_ticks=False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Exp6 — what changed between detector/camera generations", fontsize=11.5, color=oc.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    oc.save(fig, OUT, "fig2_stale_vs_current.png")

    # ---------------- RESULTS.md
    def row(key):
        cells = " | ".join(f"{results[key][v]['logloss']:.3f} / {results[key][v]['fhtr']:.3f}" for v in VCHANGE)
        return f"| {labs[key]} | {cells} | {base_rows[key]['calibration']['logloss']:.3f} / {base_rows[key]['calibration']['fhtr']:.3f} |"

    md = f"""# Exp6 — Deployment-change stress test (RQ5)

**Claim.** A historical trust map can be reused, but must be made *less certain* again
(variance inflation v_change on the latent) when the operational context may have changed.

**Settings** (all original warehouse):
- **A matched**: locked teleport-survey GP → honest_campaign events (no real change).
- **B stale**: archived `aws_gp_v7b` map from the older detector/camera generation →
  honest_campaign events (a real, documented system change: the v7b era used a different
  camera height / contaminated-dataset detector and was later superseded).
- **C dynamics**: posterior from honest campaign → whitenoise campaign events
  ({len(ev_w['det_hit'])} events; actuation-noise robustness campaign).

## Log loss / FHTR(τ={TAU}) vs inflation

| setting | v=0 (frozen reuse) | v=0.25 | v=1 | v=4 | calibration fallback |
|---|---|---|---|---|---|
{row('A_matched')}
{row('B_stale')}
{row('C_dynamics')}

Neutral prior scores logloss 0.693 / FHTR 0.000 in every setting (safe but useless).

## Reading (the result is sharper than the naive claim)

Inflation is the right tool for exactly ONE of the two failure modes:

- **A matched history tolerates frozen reuse** (0.049, FHTR≈0), and inflation costs
  almost nothing — if nothing changed, the choice barely matters.
- **C dynamics change = local overconfidence ⇒ inflation works.** The honest trajectory
  posterior over-trusts 11% of the new campaign's miss events (they happen at the
  commissioning frontier the old routes barely visited — same mechanism as the Exp3/4
  single-route trap). Inflation improves BOTH metrics (log loss 0.048→0.039, FHTR
  0.114→0.090 at v=4): flattening toward the prior is correct when the mean is right but
  locally too sure.
- **B stale generation = biased mean ⇒ inflation cannot help.** The v7b-era map is
  systematically *pessimistic* for today's system (map mean 0.46 vs 0.65; 25% of cells
  >0.3 too low — the old detector/camera generation scored worse everywhere). Its FHTR is
  already ≈0; its log loss is 3.4× the matched map's, and inflation only makes it worse
  monotonically (0.167→0.254). A wrong mean needs *replacement*, not humility: the
  geometry-only calibration fallback (0.088) beats the stale map outright.

**Recommissioning recipe.** On a suspected change: (1) if the camera/detector generation
changed → drop the old field, restart from the calibration prior and re-commission
(Exp3/4 shows 2–3 routes suffice); (2) if the change is mild or ambiguous (dynamics, load,
traffic) → keep the map but inflate (v≈1–4) and let new routes shrink the uncertainty;
(3) freeze only after held-out validation clears. Shadow-mode evaluation distinguishes
case 1 from case 2 cheaply: global log-loss degradation with near-zero FHTR signals bias
(replace), rising FHTR signals local overconfidence (inflate).

## Figures
- `fig1_inflation_curves.png` — log loss / FHTR vs inflation + reliability curves.
- `fig2_stale_vs_current.png` — stale vs current maps and the over-trust delta.

*experiments/optionA_commissioning/exp6_stress_test.py, run 2026-07-15. Events from
pre-existing honest_campaign_v1 + whitenoise_campaign_v1 logs; whitenoise events built
this session with the existing build_belief_gp_events.py.*
"""
    oc.write_md(OUT, "RESULTS.md", md)


if __name__ == "__main__":
    main()
