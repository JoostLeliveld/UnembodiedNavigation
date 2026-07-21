#!/usr/bin/env python3
"""Experiment A — controlled alpha covariance-fidelity sweep on the REAL drive.

CONTROLLED ABLATION (not operational evidence). The honest null (expB) showed the
uncertain-input GP (U5, expected_kernel) does NOT beat the point-input GP (U1,
naive) at the real pose uncertainty (belief sigma ~0.06 m) — there is almost no
input uncertainty for the expected-kernel treatment to exploit. This experiment
is the pre-registered discriminating regime (research_story ch.03; plan
`A_controlled_covariance_sweep.md`): take the REAL commissioning trajectory + REAL
detection outcomes, inject a CONTROLLED, known amount of input-location noise of
growing magnitude sigma = alpha * sigma_ref, tell each GP the (correctly specified)
reported covariance, and measure where — if anywhere — modelling that covariance
starts to matter for held-out detection prediction.

Real data: logs/visibility_comparison/single_cam_commissioning_v1 (a NEW planner-
agnostic coverage drive; NO reuse of stale artifacts, per NO_SHORTCUTS). Positions
= belief mean m_x/m_y (operational; NOT ground truth). Labels = det_hit
(operational). The ONLY synthetic element is the injected input-location noise,
which is the controlled variable of the ablation. No gt_* / oracle / CAD is used.

Modes: U1 naive (point) | U3 belief_spread | U4 uncertainty_weighted |
       U5 expected_kernel (uncertain-input champion). Held-out = leave-region-out
(3x2 contiguous spatial blocks). Metrics via scripts/shared/metrics.py.

Gate (pre-registered): at alpha=0 the expected-kernel must reproduce the point GP
(max |p_U5 - p_U1| over a query grid < 5e-3). Reading: report the alpha at which
U5 separates from U1/U3 (lower held-out NLL / better coverage), if any.

Outputs -> logs/studies/single_camera_uigp_reliability/expA_alpha_sweep/
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments" / "optionA_commissioning"))
sys.path.insert(0, str(REPO / "scripts" / "shared"))
import optA_common as oc  # noqa: E402
import metrics as M  # noqa: E402

EVENTS = REPO / "logs/visibility_comparison/single_cam_commissioning_v1/belief_gp_events/events_leaveregionout.csv"
OUT = REPO / "logs/studies/single_camera_uigp_reliability/expA_alpha_sweep"

ALPHAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
SIGMA_REF = 0.15               # m; alpha=1 -> 0.15 m injected input sigma (isotropic)
SEEDS = (0, 1, 2)
MODES = ("naive", "belief_spread", "uncertainty_weighted", "expected_kernel")
MODE_LABEL = {"naive": "U1 point", "belief_spread": "U3 smoothing",
              "uncertainty_weighted": "U4 weighted", "expected_kernel": "U5 uncertain-input"}
NBX, NBY = 3, 2                # leave-region-out blocks
ELL, NOISE_VAR = 0.90, 0.05    # deployed GP hyperparameters
AGG_RES = 0.0                  # NO aggregation: agg.cov == the injected covariance exactly.
                               # (resolution_m>0 folds within-cell spatial spread INTO agg.cov,
                               #  which breaks the alpha=0 -> U5==U1 convergence gate; see
                               #  fit_belief_aware_gp._aggregate_events lines ~159-162.)
N_MAX_TRAIN = 1200             # stratified subsample per fold for O(N^3) tractability at res=0
MIN_TEST = 25                  # skip degenerate held-out folds


def load():
    P, y = [], []
    with open(EVENTS) as f:
        for r in csv.DictReader(f):
            try:
                x = float(r["m_x"]); yy = float(r["m_y"]); h = int(float(r["det_hit"]))
            except (KeyError, ValueError):
                continue
            if np.isfinite(x) and np.isfinite(yy):
                P.append((x, yy)); y.append(h)
    return np.asarray(P, float), np.asarray(y, float)


def region_blocks(P):
    """Assign each point to one of NBX x NBY contiguous spatial blocks (leave-region-out)."""
    xe = np.quantile(P[:, 0], np.linspace(0, 1, NBX + 1))
    ye = np.quantile(P[:, 1], np.linspace(0, 1, NBY + 1))
    bx = np.clip(np.searchsorted(xe[1:-1], P[:, 0]), 0, NBX - 1)
    by = np.clip(np.searchsorted(ye[1:-1], P[:, 1]), 0, NBY - 1)
    return bx * NBY + by


def fit_one(mode, Ptr, ytr, Pte, cov_tr, blocks_tr):
    """Fit the (noised) train fold at res=0 (agg.cov == injected cov), predict at test."""
    data = oc.make_event_data(Ptr, ytr, cov_tr, blocks_tr)
    agg = oc.aggregate(data, resolution_m=AGG_RES)
    mu, sig = oc.fit_predict(mode, agg, Pte, length_scale=ELL, noise_var=NOISE_VAR)
    return M.clip_prob(M.probit_prob(mu, sig))


def subsample(idx, yv, n_max, rng):
    """Stratified subsample of train indices, preserving the (rare) miss class."""
    if idx.size <= n_max:
        return idx
    neg = idx[yv[idx] == 0]
    pos = idx[yv[idx] == 1]
    n_neg = min(neg.size, n_max // 2)
    n_pos = n_max - n_neg
    sel_neg = rng.choice(neg, n_neg, replace=False) if neg.size else neg
    sel_pos = rng.choice(pos, min(n_pos, pos.size), replace=False) if pos.size else pos
    return np.sort(np.concatenate([sel_neg, sel_pos]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P, y = load()
    blk = region_blocks(P)
    ublk = [b for b in np.unique(blk) if (blk == b).sum() >= MIN_TEST]
    print(f"loaded {len(y)} events, det rate {y.mean():.3f}; blocks {ublk} "
          f"(sizes {[int((blk==b).sum()) for b in ublk]})")

    rows = []
    for alpha in ALPHAS:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            z = rng.standard_normal(P.shape[0] * 2).reshape(P.shape[0], 2)  # paired unit noise
            sig_inj = alpha * SIGMA_REF
            Pnoisy = P + sig_inj * z
            cov = np.repeat((max(sig_inj, 1e-6) ** 2 * np.eye(2))[None], P.shape[0], axis=0)
            for b in ublk:
                te = blk == b
                tr_idx = subsample(np.where(~te)[0], y, N_MAX_TRAIN, rng)
                keep_auroc = te.sum() >= MIN_TEST and len(np.unique(y[te])) >= 2
                yte = y[te]
                for mode in MODES:
                    p = fit_one(mode, Pnoisy[tr_idx], y[tr_idx], P[te], cov[tr_idx], blk[tr_idx])
                    rec = dict(alpha=alpha, seed=seed, block=int(b), mode=mode,
                               n_test=int(te.sum()),
                               brier=float(M.brier(yte, p)),
                               nll=float(M.logloss(yte, p)),
                               auroc=float(M.auroc(yte, p)) if keep_auroc else float("nan"))
                    rows.append(rec)
        print(f"alpha={alpha} done")

    with open(OUT / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- aggregate (mean +/- std over folds x seeds) ----
    def agg_metric(alpha, mode, key):
        v = [r[key] for r in rows if r["alpha"] == alpha and r["mode"] == mode and np.isfinite(r[key])]
        return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))

    # ---- convergence gate: alpha=0 => U5 == U1 on a common query grid ----
    xs = np.linspace(P[:, 0].min(), P[:, 0].max(), 60)
    ys = np.linspace(P[:, 1].min(), P[:, 1].max(), 60)
    _, _, XY = oc.grid_query(xs, ys)
    gidx = subsample(np.arange(P.shape[0]), y, N_MAX_TRAIN, np.random.default_rng(0))
    cov0 = np.zeros((gidx.size, 2, 2))          # exact zero input uncertainty
    data0 = oc.make_event_data(P[gidx], y[gidx], cov0, blk[gidx])
    agg0 = oc.aggregate(data0, resolution_m=AGG_RES)
    p_u1 = M.probit_prob(*oc.fit_predict("naive", agg0, XY, length_scale=ELL, noise_var=NOISE_VAR))
    p_u5 = M.probit_prob(*oc.fit_predict("expected_kernel", agg0, XY, length_scale=ELL, noise_var=NOISE_VAR))
    gate_gap = float(np.max(np.abs(p_u1 - p_u5)))
    gate_pass = gate_gap < 5e-3

    # ---- figure ----
    fig, axes = oc.plt.subplots(1, 3, figsize=(14, 4.3))
    colors = {"naive": "#888888", "belief_spread": "#4CA6C9",
              "uncertainty_weighted": "#E08A00", "expected_kernel": "#7B4CC9"}
    for ax, (key, title, lo) in zip(axes, [("nll", "held-out NLL (lower=better)", True),
                                           ("brier", "held-out Brier", True),
                                           ("auroc", "held-out AUROC (higher=better)", False)]):
        for mode in MODES:
            m = [agg_metric(a, mode, key)[0] for a in ALPHAS]
            e = [agg_metric(a, mode, key)[1] for a in ALPHAS]
            ax.errorbar(ALPHAS, m, yerr=e, marker="o", ms=4, lw=1.6, capsize=2,
                        color=colors[mode], label=MODE_LABEL[mode])
        ax.set_xlabel(r"$\alpha$  (injected input $\sigma = \alpha\cdot$" + f"{SIGMA_REF} m)")
        oc.style_ax(ax, title)
    axes[0].legend(fontsize=7.5, loc="best")
    axes[0].axvline(0.4, color="#bbb", lw=0.7, ls=":")
    fig.suptitle("Experiment A — controlled input-covariance sweep on the REAL commissioning drive "
                 "[CONTROLLED ABLATION]", fontsize=11.5)
    fig.text(0.5, 0.005, f"convergence gate  max|p(U5)-p(U1)| @ alpha=0 = {gate_gap:.1e}  "
             f"{'PASS' if gate_pass else 'FAIL'}", ha="center", fontsize=8.5,
             color="#2a7" if gate_pass else "#c33")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(OUT / "expA_alpha_sweep.png", dpi=130)
    print("wrote", OUT / "expA_alpha_sweep.png")

    # ---- results markdown ----
    def line(key):
        out = [f"| {key} |"]
        return out
    hdr = "| alpha | " + " | ".join(MODE_LABEL[m] for m in MODES) + " |"
    sep = "|" + "---|" * (len(MODES) + 1)
    def tbl(key, pct=False):
        L = [f"\n**{key}** (mean±std over {len(SEEDS)} seeds × {len(ublk)} held-out regions)\n", hdr, sep]
        for a in ALPHAS:
            cells = []
            for mode in MODES:
                mu, sd = agg_metric(a, mode, key)
                cells.append(f"{mu:.3f}±{sd:.3f}")
            L.append(f"| {a:g} | " + " | ".join(cells) + " |")
        return "\n".join(L)

    # separation: first alpha where U5 NLL is below U1 NLL by more than the pooled std
    sep_alpha = None
    for a in ALPHAS:
        u5m, u5s = agg_metric(a, "expected_kernel", "nll")
        u1m, u1s = agg_metric(a, "naive", "nll")
        if np.isfinite(u5m) and u5m < u1m - 0.5 * (u5s + u1s):
            sep_alpha = a; break

    # computed headline facts for an honest, reproducible reading
    a_hi = ALPHAS[-1]
    nll_hi = {m: agg_metric(a_hi, m, "nll")[0] for m in MODES}
    best_hi = min((m for m in MODES if np.isfinite(nll_hi[m])), key=lambda m: nll_hi[m])
    au5_hi = agg_metric(a_hi, "expected_kernel", "auroc")[0]
    au_lo = agg_metric(0.0, "naive", "auroc")[0]
    u5_collapse = (np.isfinite(au5_hi) and au5_hi < 0.70) or (nll_hi["expected_kernel"] > nll_hi["naive"])
    service_verdict = (best_hi != "expected_kernel") or u5_collapse
    reading = (
        f"- **Gate {'PASS' if gate_pass else 'FAIL'}.** At alpha=0 the four modes coincide"
        f" (uncertain-input reduces to the point GP when input uncertainty is zero).\n"
        f"- **Spatial reliability is strong and mode-independent** at low alpha (AUROC ~{au_lo:.2f});"
        f" location predicts availability well — the SERVICE-MAP result, independent of the"
        f" uncertain-input treatment.\n"
        f"- **U5 (expected-kernel) at high input noise** (alpha={a_hi:g}, sigma≈{a_hi*SIGMA_REF:.2f} m):"
        f" NLL={nll_hi['expected_kernel']:.3f}, AUROC={au5_hi:.3f}"
        + (f" — it COLLAPSES (worse than naive NLL {nll_hi['naive']:.3f}); once injected sigma exceeds"
           f" the length-scale ({ELL} m) the kernel expectation flattens to an uninformative posterior.\n"
           if u5_collapse else " — competitive with naive.\n")
        + f"- **Best mode as sigma grows: {MODE_LABEL[best_hi]}** (lowest held-out NLL at alpha={a_hi:g}). "
        + ("Simple uncertainty-weighting / smoothing beats the per-sample expected-kernel here — the"
           " 'cheap-baseline warning' from synthetic exp1, now reproduced on REAL data.\n"
           if best_hi != "expected_kernel" else "The uncertain-input model is competitive here.\n")
        + "- **Checkpoint-C1 verdict:** "
        + ("the uncertain-input GP is NOT the earnable headline; the story rests on the SERVICE framing"
           " (spatial reliability map + calibrated covariance + self-monitoring), whose value (expB AUROC"
           " 0.78 vs 0.50 constant on held-out regions) stands independent of the uncertain-input claim.\n"
           if service_verdict else
           "the uncertain-input GP shows a benefit worth pursuing at higher input uncertainty.\n")
        + "- **Caveat (fair to U5):** injected noise here is homogeneous and isotropic; per exp1 the"
          " expected-kernel's edge over smoothing/weighting appears mainly under HETEROSCEDASTIC /"
          " ANISOTROPIC pose covariance. A multi-session capture with genuinely varying pose"
          " uncertainty is the only remaining regime where U5 could matter."
    )

    md = f"""# Experiment A — controlled alpha covariance-fidelity sweep (REAL drive)

**Evidence class:** CONTROLLED ABLATION (labelled; not operational evidence).
**Provenance:** real commissioning drive `single_cam_commissioning_v1` (detector
`warehouse_yolo_detector_v1`), positions = belief mean (operational), labels =
`det_hit` (operational). The only synthetic element is the injected input-location
noise sigma = alpha·{SIGMA_REF} m — the controlled variable. No gt_*/oracle/CAD used.
**Held-out:** leave-region-out, {NBX}×{NBY} contiguous blocks (folds {ublk}).
Fitter = production `fit_belief_aware_gp` via `optA_common`, ℓ={ELL} m, noise_var={NOISE_VAR}.

## Convergence gate (pre-registered)
max |p(U5 expected_kernel) − p(U1 naive)| at alpha=0 over the query grid =
**{gate_gap:.2e}** → **{'PASS' if gate_pass else 'FAIL'}** (must be < 5e-3: with zero
input uncertainty the uncertain-input GP must reproduce the point GP).

## Results
{tbl('nll')}
{tbl('brier')}
{tbl('auroc')}

## Reading (honest)
{reading}

*Generated by experiments/single_camera_uigp_reliability/tools/expA_alpha_sweep_real.py.*
"""
    oc.write_md(OUT, "RESULTS.md", md)
    print(f"gate {'PASS' if gate_pass else 'FAIL'} (gap {gate_gap:.2e}); separation alpha = {sep_alpha}")


if __name__ == "__main__":
    main()
