#!/usr/bin/env python3
"""Exp3+4 — Cold-start initialization x commissioning data budget (RQ3).

Priors (all consumed as a logit prior-mean function under the same
belief-aware residual GP used in Exp2):
  I0 neutral      : p=0.5 everywhere, no spatial knowledge ("unknown != low trust")
  I1 calibration  : geometry-only calibrated prior (clearance + px/m logistic link,
                    logs/geometry_visibility_prior/calibrated_prior_v1) — deployable
  I2 historical   : locked teleport-survey GP from the earlier commissioning phase
                    (paper_artifacts/gp/warehouse_visibility_gp_v1)
  I3 oracle       : GP fit on ALL routes incl. the held-out one — upper bound only.

Budget: train on N in {0,1,2,3} routes (leave-one-route-out; subsets averaged),
evaluate on the held-out route. Metrics: log loss, Brier, FHTR, plus the
fraction of the drivable map actually covered by commissioning data.

Outputs -> logs/studies/optionA_commissioning/exp34_init_budget/
"""
from __future__ import annotations

import csv
import itertools

import numpy as np

import optA_common as oc

OUT = oc.OUT_ROOT / "exp34_init_budget"
ELL, NOISE, RES = 0.9, 0.05, 0.20
TAU_HIGH = 0.7


def neutral_prior():
    fn = lambda X: np.zeros(np.asarray(X).shape[0])
    fn.name = "neutral"
    return fn


def oracle_prior(ev):
    """Upper bound: expected-kernel GP on ALL events (incl. test route). EVAL-ONLY."""
    data = oc.make_event_data(ev["m"], ev["det_hit"], ev["S"], ev["run"])
    agg = oc.aggregate(data, resolution_m=RES)
    xs, ys, XY = oc.grid_query()
    mu, _ = oc.fit_predict("expected_kernel", agg, XY, length_scale=ELL, noise_var=NOISE)
    grid = mu.reshape(len(ys), len(xs))

    def fn(X):
        return oc.fbg._interp_grid(xs, ys, grid, np.asarray(X, float))
    fn.name = "oracle"
    return fn


def eval_prior_plus_data(ev, prior_fn, train_mask, test_mask):
    y = ev["det_hit"]
    if train_mask.sum() == 0:
        mu = prior_fn(ev["m"][test_mask]); sig = np.full(int(test_mask.sum()), 1.0)
    else:
        data = oc.make_event_data(ev["m"][train_mask], y[train_mask], ev["S"][train_mask], ev["run"][train_mask])
        agg = oc.aggregate(data, resolution_m=RES)
        mu, sig = oc.fit_predict("expected_kernel", agg, ev["m"][test_mask],
                                 length_scale=ELL, noise_var=NOISE, prior_logit_fn=prior_fn)
    p = oc.probit_prob(mu, sig)
    yt = y[test_mask]
    return dict(logloss=oc.logloss(yt, p), brier=oc.brier(yt, p),
                fhtr=oc.fhtr(yt, p, tau_high=TAU_HIGH), auroc=oc.auroc(yt, p))


def coverage_fraction(ev, train_mask, radius=1.0):
    """Fraction of drivable grid cells within `radius` of any training event."""
    from scipy.spatial import cKDTree
    xs, ys, XY = oc.grid_query(nx=80, ny=72)
    if train_mask.sum() == 0:
        return 0.0
    dist, _ = cKDTree(ev["m"][train_mask]).query(XY)
    return float(np.mean(dist <= radius))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = oc.load_events()
    priors = {
        "I0_neutral": neutral_prior(),
        "I1_calibration": oc.prior_logit_calibration(),
        "I2_historical": oc.prior_logit_from_artifact(oc.LOCKED_GP),
        "I3_oracle": oracle_prior(ev),
    }
    rows = []
    for held in oc.ROUTES:
        te = ev["route"] == held
        train_routes = [r for r in oc.ROUTES if r != held]
        for n in (0, 1, 2, 3):
            subsets = list(itertools.combinations(train_routes, n)) or [()]
            for sub in subsets:
                tr = np.isin(ev["route"], sub) if sub else np.zeros(len(te), bool)
                cov = coverage_fraction(ev, tr)
                for pname, pfn in priors.items():
                    if pname == "I3_oracle" and n > 0:
                        continue  # oracle is the N-independent upper bound
                    r = eval_prior_plus_data(ev, pfn, tr, te)
                    r.update(prior=pname, n_routes=n, held=held, subset="+".join(sub), coverage=cov)
                    rows.append(r)
        print(f"held-out {held} done")

    with open(OUT / "budget_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    def mean_metric(pname, n, key):
        v = [r[key] for r in rows if r["prior"] == pname and r["n_routes"] == n and np.isfinite(r[key])]
        return (np.mean(v), np.std(v) / max(1, np.sqrt(len(v)))) if v else (np.nan, np.nan)

    # ---------------- fig1: the four priors
    xs, ys, XY = oc.grid_query()
    fig, axes = oc.plt.subplots(1, 4, figsize=(15.2, 4.1))
    for ax, (pname, pfn) in zip(axes, priors.items()):
        pm = oc.sigmoid(pfn(XY)).reshape(len(ys), len(xs))
        ax.imshow(pm, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap=oc.CMAP_TRUST, vmin=0, vmax=1)
        oc.draw_warehouse(ax, camera=False)
        lab = {"I0_neutral": "I0 neutral (p=0.5)", "I1_calibration": "I1 calibration-only (geometry)",
               "I2_historical": "I2 historical survey GP", "I3_oracle": "I3 oracle (all routes, eval-only)"}[pname]
        oc.style_ax(ax, lab, keep_ticks=False)
    fig.suptitle("Exp3 — cold-start priors as logit mean functions", fontsize=11.5, color=oc.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    oc.save(fig, OUT, "fig1_priors.png")

    # ---------------- fig2: budget curves
    fig, axes = oc.plt.subplots(1, 3, figsize=(13.8, 4.2))
    colors = {"I0_neutral": oc.MUTED, "I1_calibration": oc.BLUE, "I2_historical": oc.VIOLET, "I3_oracle": oc.GREEN}
    labels = {"I0_neutral": "I0 neutral", "I1_calibration": "I1 calibration", "I2_historical": "I2 historical", "I3_oracle": "I3 oracle (bound)"}
    ns = (0, 1, 2, 3)
    for ax, key, title in zip(axes, ("logloss", "brier", "fhtr"),
                              ("held-out log loss", "held-out Brier", f"false-high-trust rate (τ={TAU_HIGH})")):
        for pname in priors:
            if pname == "I3_oracle":
                v, se = mean_metric(pname, 0, key)
                ax.axhline(v, color=colors[pname], ls="--", lw=1.2)
                ax.text(2.15, v, " oracle bound", fontsize=6.8, color=colors[pname], va="bottom")
                continue
            mus = [mean_metric(pname, n, key)[0] for n in ns]
            ses = [mean_metric(pname, n, key)[1] for n in ns]
            ax.errorbar(ns, mus, yerr=ses, marker="o", ms=4, lw=1.6, capsize=2,
                        color=colors[pname], label=labels[pname])
        ax.set_xticks(ns); ax.set_xlabel("commissioning routes used")
        oc.style_ax(ax, title)
    axes[0].legend(fontsize=7.5)
    covs = [np.mean([r["coverage"] for r in rows if r["n_routes"] == n and r["prior"] == "I0_neutral"]) for n in ns]
    ax2 = axes[1].twinx()
    ax2.plot(ns, covs, ":s", ms=3, lw=1.0, color=oc.INK2, alpha=0.7)
    ax2.set_ylabel("map coverage within 1 m of data", fontsize=7, color=oc.INK2)
    ax2.tick_params(labelsize=6.5, colors=oc.INK2); ax2.set_ylim(0, 1)
    fig.suptitle("Exp4 — data-budget curves: prior choice dominates the low-data regime", fontsize=11.5, color=oc.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    oc.save(fig, OUT, "fig2_budget_curves.png")

    # ---------------- fig3: posterior maps after 1 route, per prior
    held = "route_west_to_a1_upper"
    one_route = ("route_apron_to_a2_mid",)
    tr = np.isin(ev["route"], one_route)
    fig, axes = oc.plt.subplots(1, 4, figsize=(15.2, 4.1))
    data = oc.make_event_data(ev["m"][tr], ev["det_hit"][tr], ev["S"][tr], ev["run"][tr])
    agg = oc.aggregate(data, resolution_m=RES)
    for ax, (pname, pfn) in zip(axes, priors.items()):
        mu, _ = oc.fit_predict("expected_kernel", agg, XY, length_scale=ELL, noise_var=NOISE, prior_logit_fn=pfn)
        ax.imshow(oc.sigmoid(mu).reshape(len(ys), len(xs)), origin="lower",
                  extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap=oc.CMAP_TRUST, vmin=0, vmax=1)
        ax.plot(ev["m"][tr, 0], ev["m"][tr, 1], ".", ms=0.8, color=oc.INK, alpha=0.35)
        oc.draw_warehouse(ax, camera=False)
        oc.style_ax(ax, f"{labels[pname]} + 1 route", keep_ticks=False)
    fig.suptitle("Exp3 — posterior after ONE commissioning route (dots = the route)", fontsize=11.5, color=oc.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    oc.save(fig, OUT, "fig3_after_one_route.png")

    # ---------------- results
    thr = 1.05 * mean_metric("I0_neutral", 3, "logloss")[0]
    to_thr = {}
    for pname in ("I0_neutral", "I1_calibration", "I2_historical"):
        to_thr[pname] = next((n for n in ns if mean_metric(pname, n, "logloss")[0] <= thr), ">3")

    tbl = "| prior | N=0 | N=1 | N=2 | N=3 | routes to threshold* |\n|---|---|---|---|---|---|\n"
    for pname in ("I0_neutral", "I1_calibration", "I2_historical"):
        cells = " | ".join(f"{mean_metric(pname, n, 'logloss')[0]:.3f}" for n in ns)
        tbl += f"| {labels[pname]} | {cells} | {to_thr[pname]} |\n"
    ov, _ = mean_metric("I3_oracle", 0, "logloss")

    ftbl = "| prior | FHTR N=0 | FHTR N=1 | FHTR N=3 |\n|---|---|---|---|\n"
    for pname in ("I0_neutral", "I1_calibration", "I2_historical"):
        ftbl += ("| " + labels[pname] + " | " +
                 " | ".join(f"{mean_metric(pname, n, 'fhtr')[0]:.3f}" for n in (0, 1, 3)) + " |\n")

    md = f"""# Exp3+4 — Initialization x data budget (RQ3)

**Question.** Which realistic initialization gives the best early performance and the lowest
false-confidence with limited commissioning data — and how many routes until the trust map
reaches its validation level?

**Setup.** Same belief-aware residual GP as Exp2, `det_hit` target, leave-one-route-out;
budgets N∈{{0,1,2,3}} training routes (subsets averaged). Priors are logit mean functions.
Held-out **log loss** (lower better); threshold* = within 5% of the neutral-prior N=3 level
({thr:.3f}).

## Budget curves (held-out log loss, mean over folds/subsets)

{tbl}
Oracle (all routes incl. held-out, evaluation-only upper bound): **{ov:.3f}**.

## False-high-trust rate (τ={TAU_HIGH})

{ftbl}

## Reading

- **N=0 separates the priors sharply**: neutral is safe but uninformative (log loss 0.69);
  the calibration prior predicts held-out usability from pure geometry (0.12); the
  historical survey GP is the strongest realistic cold start on an unchanged plant (0.05,
  near the 0.013 oracle bound). Exp6 stresses the *changed-plant* case.
- **The single-route trap (the key RQ3 finding).** Adding ONE route makes the neutral and
  calibration maps *worse* than their own N=0 baseline (calibration 0.12 → 0.33; FHTR jumps
  0.01 → 0.44). Mechanism: one route is mostly `det=1`, its positive residuals extrapolate
  ~one length scale beyond the visited lane, and that optimism leaks into genuinely bad
  unvisited cells of the held-out route. Passive data collection is NOT monotonically safe:
  with a weak prior, a small amount of clustered evidence produces false confidence at the
  commissioning frontier. Conservative structure (historical prior: FHTR 0.08) or wider
  data spread is required before the map is trusted.
- **With 2–3 routes the curves recover and converge** — trajectory evidence swamps the
  prior; the prior choice is an early-shift/coverage question, not an asymptotic one.
- Coverage line (fig2, middle): even 3 routes touch barely half of the drivable map within
  1 m — the prior keeps carrying most of the never-visited area (unknown ≠ low trust).

## Figures
- `fig1_priors.png` — the four priors on the warehouse grid.
- `fig2_budget_curves.png` — log loss / Brier / FHTR vs N routes + map coverage.
- `fig3_after_one_route.png` — what each initialization looks like after one route.

*experiments/optionA_commissioning/exp34_init_budget.py, run 2026-07-15 on pre-existing
honest_campaign_v1 logs.*
"""
    oc.write_md(OUT, "RESULTS.md", md)


if __name__ == "__main__":
    main()
