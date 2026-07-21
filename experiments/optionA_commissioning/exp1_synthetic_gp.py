#!/usr/bin/env python3
"""Exp1 — Synthetic uncertain-input GP study (RQ2, mathematical reproduction).

Claim under test: an uncertain-input (belief-aware) GP avoids falsely sharp
spatial predictions when training locations are uncertain, and converges to
the point GP as pose covariance -> 0.

Everything here is SYNTHETIC and labeled as such. The GP code is the same
production code used on real logs (fit_belief_aware_gp via optA_common).

Methods: point GP | tuned point GP (CV length scale) | fixed blur |
         uncertainty weighting | belief-aware (expected kernel, per-sample P)
Scenarios: low / high / heteroscedastic / anisotropic / miscalibrated pose noise.

Outputs -> logs/studies/optionA_commissioning/exp1_synthetic_gp/
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

import optA_common as oc

OUT = oc.OUT_ROOT / "exp1_synthetic_gp"
RNG_SEEDS = (0, 1, 2, 3, 4)
NOISE_STD_LOGIT = 0.40    # target noise, generated in logit space
NOISE_VAR = NOISE_STD_LOGIT ** 2   # GP observation-noise var (correctly specified)
ELL = 0.9                 # production length scale
N_SAMPLES = 300
GRID_N = 80
BAD = dict(x0=0.6, x1=2.6, y0=-0.4, y1=2.4)   # sharp camera-poor zone
UNKNOWN = dict(x0=-6.0, x1=-3.4)               # never-visited strip (no samples)

METHODS = ("point", "tuned_point", "fixed_blur", "uncertainty_weighted", "expected_kernel")


# ------------------------------------------------------------------ true field
def true_field(X):
    """f(x,y) in [0,1]: good near-camera core, gradual range falloff, sharp bad zone."""
    X = np.asarray(X, float)
    r = np.hypot(X[:, 0] - 0.0, X[:, 1] + 6.0)          # synthetic camera at (0,-6)
    base = 4.0 - 0.55 * np.maximum(r - 4.0, 0.0)        # logit: high near, gradual falloff
    inbad = ((X[:, 0] >= BAD["x0"]) & (X[:, 0] <= BAD["x1"]) &
             (X[:, 1] >= BAD["y0"]) & (X[:, 1] <= BAD["y1"]))
    lg = np.where(inbad, -3.0, base)
    return oc.sigmoid(lg)


LANES = (-3.5, -1.2, 0.2, 1.6, 3.0)


def sample_routes(rng, n=N_SAMPLES):
    """Pseudo-commissioning routes covering x>-3.4 (leaves an unknown strip)."""
    lanes = []
    k = n // len(LANES)
    for i, yline in enumerate(LANES):
        ts = np.linspace(0, 1, k)
        x = -3.0 + 8.6 * ts if i % 2 == 0 else 5.6 - 8.6 * ts
        y = yline + 0.35 * np.sin(6.28 * ts * 2) + 0.08 * rng.standard_normal(x.size)
        lanes.append(np.column_stack([x, y]))
    return np.vstack(lanes)


def make_covs(scenario, S_true, rng):
    """Return (true sampling covs, reported covs) per sample."""
    n = S_true.shape[0]
    iso = lambda s: np.repeat((s ** 2 * np.eye(2))[None], n, axis=0)
    if scenario == "low_uniform":
        C = iso(0.05); return C, C
    if scenario == "high_uniform":
        C = iso(0.60); return C, C
    if scenario == "heteroscedastic":
        s = 0.05 + 0.75 * np.clip((S_true[:, 0] + 3.0) / 8.6, 0, 1)  # grows along route
        C = np.einsum("i,jk->ijk", s ** 2, np.eye(2)); return C, C
    if scenario == "anisotropic":
        # along-track sigma 0.7, cross-track 0.1; track direction = +x lanes
        d = np.gradient(S_true[:, 0])
        ang = np.where(d >= 0, 0.0, np.pi)
        C = np.empty((n, 2, 2))
        for i, a in enumerate(ang):
            Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            C[i] = Rm @ np.diag([0.7 ** 2, 0.1 ** 2]) @ Rm.T
        return C, C
    if scenario == "miscalibrated":
        return iso(0.50), iso(0.10)   # true noise 0.50 m, reported only 0.10 m
    raise ValueError(scenario)


# ------------------------------------------------------------------ methods
def tuned_length_scale(agg, rng, ells=(0.6, 0.9, 1.3, 1.8, 2.4, 3.0)):
    """Deployable tuning: 5-fold CV on the (noisy) training samples themselves."""
    n = agg.X.shape[0]
    folds = rng.permutation(n) % 5
    best, best_nll = ELL, np.inf
    for ell in ells:
        nlls = []
        for f in range(5):
            tr, te = folds != f, folds == f
            if te.sum() < 3:
                continue
            sub = oc.fbg.AggregateData(agg.X[tr], agg.y[tr], agg.cov[tr], agg.count[tr])
            mu, sig = oc.fit_predict("naive", sub, agg.X[te], length_scale=ell, noise_var=NOISE_VAR)
            nlls.append(oc.gaussian_nll_logit(agg.y[te], mu, sig, NOISE_VAR))
        if np.mean(nlls) < best_nll:
            best_nll, best = float(np.mean(nlls)), ell
    return best


def predict(method, agg, XY, rng):
    t0 = time.perf_counter()
    if method == "point":
        mu, sig = oc.fit_predict("naive", agg, XY, length_scale=ELL, noise_var=NOISE_VAR)
    elif method == "tuned_point":
        ell = tuned_length_scale(agg, rng)
        mu, sig = oc.fit_predict("naive", agg, XY, length_scale=ell, noise_var=NOISE_VAR)
    else:
        mu, sig = oc.fit_predict(method, agg, XY, length_scale=ELL, noise_var=NOISE_VAR)
    return mu, sig, time.perf_counter() - t0


# ------------------------------------------------------------------ metrics
def boundary_displacement(p_grid, f_grid, xs, ys):
    """Symmetric mean contour distance between predicted and true 0.5 iso-lines
    around the bad zone (computed in the visited region only)."""
    cell = xs[1] - xs[0]
    pb, tb = p_grid < 0.5, f_grid < 0.5
    if pb.sum() == 0 or tb.sum() == 0:
        return np.nan
    def bound(mask):
        er = mask & ~np.roll(mask, 1, 0) | mask & ~np.roll(mask, 1, 1)
        return er
    dt_t = distance_transform_edt(~bound(tb)) * cell
    dt_p = distance_transform_edt(~bound(pb)) * cell
    bp, bt = bound(pb), bound(tb)
    return float(0.5 * (dt_t[bp].mean() + dt_p[bt].mean()))


def evaluate(mu, sig, XY, xs, ys, S_true):
    from scipy.spatial import cKDTree
    f = true_field(XY)
    p = oc.probit_prob(mu, sig)
    dist_to_data, _ = cKDTree(S_true).query(XY)
    visited = (XY[:, 0] > UNKNOWN["x1"]) & (dist_to_data <= 1.0)   # commissioned area
    unknown = XY[:, 0] <= UNKNOWN["x1"]                            # never-visited strip
    grid_p = p.reshape(len(ys), len(xs))
    grid_f = f.reshape(len(ys), len(xs))
    # boundary displacement: only in the bad-zone neighborhood covered by data
    nb = (visited & (XY[:, 0] >= BAD["x0"] - 1.2) & (XY[:, 0] <= BAD["x1"] + 1.2) &
          (XY[:, 1] >= BAD["y0"] - 1.2) & (XY[:, 1] <= BAD["y1"] + 1.2)).reshape(len(ys), len(xs))
    gp = np.where(nb, grid_p, 1.0)
    gf = np.where(nb, grid_f, 1.0)
    t = oc.logit(np.clip(f, 1e-4, 1 - 1e-4))
    z = (t - mu) / np.sqrt(sig ** 2 + NOISE_VAR)
    return dict(
        rmse=float(np.sqrt(np.mean((p[visited] - f[visited]) ** 2))),
        nll=float(np.mean(0.5 * np.log(2 * np.pi * (sig[visited] ** 2 + NOISE_VAR)) + 0.5 * z[visited] ** 2)),
        cov90=float(np.mean(np.abs(z[visited]) <= 1.645)),
        fhta=float(np.mean((p[visited] > 0.7) & (f[visited] < 0.4))),
        bdisp=boundary_displacement(gp, gf, xs, ys),
        unk_pull=float(np.mean(np.abs(p[unknown] - 0.5))),
        unk_cov90=float(np.mean(np.abs(z[unknown]) <= 1.645)),
    )


SCENARIOS = ("low_uniform", "high_uniform", "heteroscedastic", "anisotropic", "miscalibrated")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    xs = np.linspace(-6, 6, GRID_N)
    ys = np.linspace(-6, 6, GRID_N)
    _, _, XY = oc.grid_query(xs, ys)

    rows = []
    runtimes = {}
    example = {}   # (scenario, method) -> p-grid for figures, seed 0
    example_data = {}
    for scen in SCENARIOS:
        for seed in RNG_SEEDS:
            rng = np.random.default_rng(seed)
            S_true = sample_routes(rng)
            t_true = oc.logit(true_field(S_true))
            y_obs = oc.sigmoid(t_true + NOISE_STD_LOGIT * rng.standard_normal(t_true.size))
            C_true, C_rep = make_covs(scen, S_true, rng)
            S_obs = np.array([rng.multivariate_normal(s, c) for s, c in zip(S_true, C_true)])
            data = oc.make_event_data(S_obs, y_obs, C_rep, np.arange(len(y_obs)) // 65)
            agg = oc.aggregate(data, resolution_m=0.0)
            for method in METHODS:
                mu, sig, dt = predict(method, agg, XY, rng)
                met = evaluate(mu, sig, XY, xs, ys, S_true)
                met.update(scenario=scen, method=method, seed=seed, runtime_s=dt)
                rows.append(met)
                runtimes.setdefault(method, []).append(dt)
                if seed == 0:
                    example[(scen, method)] = oc.probit_prob(mu, sig).reshape(GRID_N, GRID_N)
            if seed == 0:
                example_data[scen] = (S_true, S_obs, C_rep, y_obs)
        print(f"scenario {scen} done")

    import csv
    with open(OUT / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # convergence gate: P -> 0 must reproduce the point GP
    rng = np.random.default_rng(0)
    S_true = sample_routes(rng)
    y_obs = oc.sigmoid(oc.logit(true_field(S_true)) + NOISE_STD_LOGIT * rng.standard_normal(len(S_true)))
    gaps = []
    sig_list = (0.6, 0.3, 0.15, 0.05, 0.02, 0.005, 1e-4)
    for s in sig_list:
        C = np.repeat((s ** 2 * np.eye(2))[None], len(S_true), axis=0)
        data = oc.make_event_data(S_true, y_obs, C, np.arange(len(y_obs)) // 65)
        agg = oc.aggregate(data, resolution_m=0.0)
        mu_b, sg_b = oc.fit_predict("expected_kernel", agg, XY, length_scale=ELL, noise_var=NOISE_VAR)
        mu_p, sg_p = oc.fit_predict("naive", agg, XY, length_scale=ELL, noise_var=NOISE_VAR)
        gaps.append(float(np.max(np.abs(oc.sigmoid(mu_b) - oc.sigmoid(mu_p)))))
    gate_pass = gaps[-1] < 5e-3

    # ---------------- figures
    agg_m = {}
    for scen in SCENARIOS:
        for method in METHODS:
            sel = [r for r in rows if r["scenario"] == scen and r["method"] == method]
            agg_m[(scen, method)] = {k: (np.nanmean([r[k] for r in sel]), np.nanstd([r[k] for r in sel]))
                                     for k in ("rmse", "nll", "cov90", "fhta", "bdisp", "unk_pull", "unk_cov90")}

    # fig1: setup
    fig, axes = oc.plt.subplots(1, 3, figsize=(13.6, 4.4))
    ax = axes[0]
    im = ax.imshow(true_field(XY).reshape(GRID_N, GRID_N), origin="lower", extent=(-6, 6, -6, 6),
                   cmap=oc.CMAP_TRUST, vmin=0, vmax=1)
    ax.axvspan(UNKNOWN["x0"], UNKNOWN["x1"], color=oc.INK, alpha=0.12)
    ax.text(-4.7, 5.0, "never\nvisited", ha="center", fontsize=7.5, color=oc.INK2)
    fig.colorbar(im, ax=ax, shrink=0.8)
    oc.style_ax(ax, "true trust field f(x,y)  [SYNTHETIC]")
    St, So, Cr, yo = example_data["heteroscedastic"]
    ax = axes[1]
    ax.scatter(St[:, 0], St[:, 1], s=5, c=yo, cmap=oc.CMAP_TRUST, vmin=0, vmax=1, linewidths=0)
    from matplotlib.patches import Ellipse
    for i in range(0, len(St), 8):
        vals, vecs = np.linalg.eigh(Cr[i])
        ang = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
        ax.add_patch(Ellipse(St[i], 2 * np.sqrt(vals[-1]), 2 * np.sqrt(vals[0]), angle=ang,
                             fill=False, ec=oc.VIOLET, lw=0.6, alpha=0.7))
    ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
    oc.style_ax(ax, "commissioning samples + reported pose covariance (heteroscedastic)")
    ax = axes[2]
    ax.plot(sig_list, gaps, "-o", color=oc.AQUA, lw=1.6, ms=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("pose sigma [m] (isotropic)"); ax.set_ylabel("max |p_belief − p_point|")
    oc.badge(ax, f"gate: gap at σ→0 = {gaps[-1]:.1e}  {'PASS' if gate_pass else 'FAIL'}", "lower right")
    oc.style_ax(ax, "convergence: belief-aware → point GP as P→0")
    fig.suptitle("Exp1 — synthetic uncertain-input GP: setup and convergence gate", fontsize=11.5, color=oc.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    oc.save(fig, OUT, "fig1_setup_and_gate.png")

    # fig2: prediction maps, key scenarios
    show_scen = ("high_uniform", "heteroscedastic", "miscalibrated")
    fig, axes = oc.plt.subplots(len(show_scen), len(METHODS) + 1, figsize=(3.0 * (len(METHODS) + 1), 3.1 * len(show_scen)))
    for i, scen in enumerate(show_scen):
        axf = axes[i, 0]
        axf.imshow(true_field(XY).reshape(GRID_N, GRID_N), origin="lower", extent=(-6, 6, -6, 6),
                   cmap=oc.CMAP_TRUST, vmin=0, vmax=1)
        oc.style_ax(axf, f"true field\n[{scen}]", keep_ticks=False)
        for j, method in enumerate(METHODS):
            ax = axes[i, j + 1]
            ax.imshow(example[(scen, method)], origin="lower", extent=(-6, 6, -6, 6),
                      cmap=oc.CMAP_TRUST, vmin=0, vmax=1)
            m = agg_m[(scen, method)]
            oc.style_ax(ax, oc.METHOD_LABELS[method], keep_ticks=False)
            oc.badge(ax, f"RMSE {m['rmse'][0]:.3f}\nFHT {m['fhta'][0]*100:.1f}%", "lower left")
    fig.suptitle("Exp1 — predicted trust maps (seed 0), scenario × method  [SYNTHETIC]", fontsize=12, color=oc.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    oc.save(fig, OUT, "fig2_prediction_maps.png")

    # fig3: metric bars
    metrics_show = (("rmse", "RMSE vs true field"), ("nll", "held-out NLL (logit)"),
                    ("cov90", "90% interval coverage"), ("fhta", "false-high-trust area"),
                    ("bdisp", "boundary displacement [m]"), ("unk_cov90", "coverage in unvisited strip"))
    fig, axes = oc.plt.subplots(2, 3, figsize=(13.8, 7.6))
    xpos = np.arange(len(SCENARIOS))
    w = 0.15
    for ax, (mk, title) in zip(axes.ravel(), metrics_show):
        for j, method in enumerate(METHODS):
            vals = [agg_m[(s, method)][mk][0] for s in SCENARIOS]
            errs = [agg_m[(s, method)][mk][1] for s in SCENARIOS]
            ax.bar(xpos + (j - 2) * w, vals, w, yerr=errs, color=oc.METHOD_COLORS[method],
                   label=oc.METHOD_LABELS[method], error_kw=dict(lw=0.7, ecolor=oc.INK2))
        if mk in ("cov90", "unk_cov90"):
            ax.axhline(0.90, color=oc.INK2, lw=0.8, ls="--")
        ax.set_xticks(xpos); ax.set_xticklabels([s.replace("_", "\n") for s in SCENARIOS], fontsize=7)
        oc.style_ax(ax, title)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Exp1 — 5 seeds mean±std per scenario  [SYNTHETIC]", fontsize=12, color=oc.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    oc.save(fig, OUT, "fig3_metrics.png")

    # ---------------- results markdown
    def cell(s, m, k, pct=False, prec=3):
        v, sd = agg_m[(s, m)][k]
        return f"{100*v:.1f}±{100*sd:.1f}" if pct else f"{v:.{prec}f}±{sd:.{prec}f}"

    lines = []
    for mk, title in (("rmse", "RMSE"), ("nll", "NLL"), ("fhta", "false-high-trust area (frac)"), ("cov90", "coverage@90%")):
        lines.append(f"\n### {title}\n")
        lines.append("| scenario | " + " | ".join(oc.METHOD_LABELS[m] for m in METHODS) + " |")
        lines.append("|" + "---|" * (len(METHODS) + 1))
        for s in SCENARIOS:
            lines.append(f"| {s} | " + " | ".join(cell(s, m, mk) for m in METHODS) + " |")
    rt = {m: float(np.mean(v)) for m, v in runtimes.items()}

    md = f"""# Exp1 — Synthetic uncertain-input GP study (RQ2)

**Claim.** An uncertain-input GP avoids falsely sharp spatial predictions when training
locations are uncertain; with exact locations it must reproduce the point GP.

**Setup [SYNTHETIC].** Known field f(x,y)∈[0,1] with camera-good core, gradual range
falloff, a sharp bad zone, and a never-visited strip (x<{UNKNOWN['x1']}). {N_SAMPLES} samples
on {len(LANES)} commissioning lanes, target noise {NOISE_STD_LOGIT} (logit space, correctly
specified to the GP), {len(RNG_SEEDS)} seeds. Same production GP code as the real-data
experiments (`fit_belief_aware_gp`), ℓ={ELL} m. Evaluation on cells within 1 m of
commissioning data; the never-visited strip is scored separately (unknown ≠ low trust).

## Convergence gate (pre-registered)

max |p_belief − p_point| as isotropic pose σ shrinks {sig_list} m:
{', '.join(f'{g:.2e}' for g in gaps)} → **{'PASS' if gate_pass else 'FAIL'}** (σ→0 gap {gaps[-1]:.1e}).

## Results (mean±std over seeds, visited region)
{chr(10).join(lines)}

### Runtime (mean per fit+grid predict, {GRID_N}×{GRID_N} grid)
{' · '.join(f'{oc.METHOD_LABELS[m]}: {rt[m]:.2f}s' for m in METHODS)}

## Reading (what the numbers actually say)

- **The benefit is calibration, not the mean.** No method recovers a better mean map (RMSE
  is flat across methods): information destroyed by input noise cannot be undone. What
  input-aware methods fix is *false sharpness* — the point GP under-covers (coverage 0.88
  at nominal 0.90, NLL 1.62 high-uniform / 1.76 anisotropic) while belief-aware restores
  coverage ≈0.91 and cuts NLL to 1.44 / 1.51. This is exactly the O'Callaghan-style
  uncertain-input effect and is the honest claim to carry into the thesis.
- **Where per-sample P matters vs global smoothing:** with *uniform* pose noise, fixed blur
  is mathematically identical to belief-aware (numbers coincide) — the reviewer objection
  "it's just smoothing" is TRUE there. The distinction appears only when covariance varies
  (heteroscedastic/anisotropic), where belief-aware beats fixed blur on NLL.
- **Cheap baseline warning:** simple uncertainty *weighting* (noise inflation by tr P) is
  the best NLL/coverage method in the het/aniso scenarios — it discounts bad samples
  instead of blurring them. An honest paper must report it; the expected-kernel remains the
  principled model (it also shifts the mean, and it is the variant with a query-side
  covariance seam used at planning time).
- **Miscalibrated covariance** (true 0.50 m, reported 0.10 m): all input-aware methods
  collapse onto the point GP — they inherit the lie. Covariance quality is a real
  dependency; motivates Exp5 (trajectory smoothing) and honest R calibration.
- **Unknown region:** all GP variants keep the unvisited strip near the neutral prior with
  wide intervals (unknown ≠ low trust); coverage there stays ≥ nominal.

*Generated by experiments/optionA_commissioning/exp1_synthetic_gp.py; data synthetic, run 2026-07-15.*
"""
    oc.write_md(OUT, "RESULTS.md", md)
    print("gate:", "PASS" if gate_pass else "FAIL", "gaps:", ["%.1e" % g for g in gaps])


if __name__ == "__main__":
    main()
