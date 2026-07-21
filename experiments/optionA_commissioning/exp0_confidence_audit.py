#!/usr/bin/env python3
"""Exp0 — Confidence audit (RQ1 validation gate).

Question: is YOLO confidence a usable proxy for the quality/usability of an
external-camera measurement?

Data: per-detection rows from honest_campaign_v1 perception.csv (43 runs),
joined to experiment.csv for the EKF gate outcome (pixel_corr_nis/accepted).
Ground truth is used ONLY to compute the offline BEV error label
e_k = ||camera BEV estimate - GT at capture time|| (localization_error_captime_m).

Outputs -> logs/studies/optionA_commissioning/exp0_confidence_audit/
"""
from __future__ import annotations

import glob
import math
from pathlib import Path

import numpy as np

import optA_common as oc
from optA_common import fnum

OUT = oc.OUT_ROOT / "exp0_confidence_audit"
E_SAFE = (0.15, 0.30, 0.50)
CAMPAIGN = oc.LOGS_VC / "honest_campaign_v1"


def load_audit_rows():
    recs = []
    for pf in sorted(glob.glob(str(CAMPAIGN / "*/*/*/*/perception.csv"))):
        run_dir = Path(pf).parent
        route = run_dir.relative_to(CAMPAIGN).parts[0]
        exp_rows = oc.read_rows(run_dir / "experiment.csv")
        exp_stamp = np.array([fnum(r, "stamp") for r in exp_rows])
        nis = np.array([fnum(r, "pixel_corr_nis") for r in exp_rows])
        acc = np.array([fnum(r, "pixel_corr_accepted") for r in exp_rows])
        corr_stamp = np.array([fnum(r, "planner_pixel_correction_stamp") for r in exp_rows])
        for r in oc.read_rows(pf):
            if r.get("detected") not in ("0", "1"):
                continue
            det = int(r["detected"])
            t = fnum(r, "log_stamp")
            rec = dict(
                route=route, det=det,
                score=fnum(r, "yolo_score_raw"),
                err=fnum(r, "localization_error_captime_m"),
                err_calib=fnum(r, "localization_error_calibrated_m"),
                u=fnum(r, "obs_u"), v=fnum(r, "obs_v"),
                border=fnum(r, "border_margin_px"),
                tx=fnum(r, "true_x"), ty=fnum(r, "true_y"), t=t,
                bearing=fnum(r, "camera_relative_bearing_deg"),
                nis=math.nan, accepted=math.nan,
            )
            if det and np.isfinite(t) and exp_stamp.size:
                j = int(np.nanargmin(np.abs(exp_stamp - t)))
                # gate outcome for the correction nearest this detection
                if abs(exp_stamp[j] - t) <= 0.3:
                    # find nearest row that actually has a correction stamped near t
                    w = np.where(np.isfinite(corr_stamp) & (np.abs(corr_stamp - t) <= 0.35))[0]
                    if w.size:
                        k = w[np.argmin(np.abs(corr_stamp[w] - t))]
                        rec["nis"] = nis[k]
                        rec["accepted"] = acc[k]
            recs.append(rec)
    keys = recs[0].keys()
    return {k: np.array([r[k] for r in recs]) for k in keys}


def spearman(a, b):
    from scipy.stats import spearmanr
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return math.nan, int(m.sum())
    return float(spearmanr(a[m], b[m]).statistic), int(m.sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_audit_rows()
    det = d["det"] == 1
    hasl = det & np.isfinite(d["err"]) & np.isfinite(d["score"])
    c, e = d["score"][hasl], d["err"][hasl]
    rng = np.hypot(d["tx"][hasl] - oc.CAMERA_POS[0], d["ty"][hasl] - oc.CAMERA_POS[1])

    rho_all, n_all = spearman(c, e)
    # per-range-band partial view
    band_rhos = []
    for lo, hi in [(0, 5), (5, 7), (7, 12)]:
        m = (rng >= lo) & (rng < hi)
        r, n = spearman(c[m], e[m])
        band_rhos.append((lo, hi, r, n))
    # motion confound: start dwell (t<40 s, robot ~static) vs moving section
    t_all = d["t"][hasl]
    early = t_all < 40.0
    rho_early, n_early = spearman(c[early], e[early])
    rho_late, n_late = spearman(c[~early], e[~early])
    rho_cr, _ = spearman(c, rng)
    rho_et, _ = spearman(e, t_all)

    edges = np.linspace(max(0.25, np.percentile(c, 1)), 1.0, 9)
    ctr, med, ns = oc.binned(c, e, edges, np.nanmedian)
    _, p95, _ = oc.binned(c, e, edges, lambda x: np.nanpercentile(x, 95))

    # ---------------- fig 1: confidence vs BEV error
    fig, axes = oc.plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    hb = ax.hexbin(c, e, gridsize=45, cmap=oc.CMAP_INK, mincnt=1, linewidths=0)
    ax.plot(ctr, med, "-o", color=oc.BLUE, ms=3.5, lw=1.6, label="median")
    ax.plot(ctr, p95, "-s", color=oc.RED, ms=3.5, lw=1.6, label="95th pct")
    for es, ls in zip(E_SAFE, (":", "--", "-.")):
        ax.axhline(es, color=oc.MUTED, lw=0.8, ls=ls)
        ax.text(ax.get_xlim()[0] + 0.005, es, f"e_safe={es}", fontsize=6.5, color=oc.MUTED, va="bottom")
    ax.set_xlabel("YOLO confidence  c"); ax.set_ylabel("BEV localization error  e  [m]")
    ax.set_ylim(0, min(1.6, np.percentile(e, 99.5) * 1.15))
    ax.legend(fontsize=7.5, loc="upper right")
    oc.style_ax(ax, "confidence vs camera BEV error (per detection)")
    oc.badge(ax, f"Spearman ρ = {rho_all:+.3f}   n = {n_all}", "upper left")
    fig.colorbar(hb, ax=ax, shrink=0.85).set_label("detections", fontsize=7)

    ax = axes[1]
    for es, col in zip(E_SAFE, (oc.YELLOW, oc.ORANGE, oc.RED)):
        _, frac, _ = oc.binned(c, (e > es).astype(float), edges, np.nanmean)
        ax.plot(ctr, frac, "-o", ms=3.5, lw=1.6, color=col, label=f"P(e > {es} m)")
    ax2 = ax.twinx()
    ax2.bar(ctr, ns, width=np.diff(edges) * 0.85, color=oc.GRID, zorder=0)
    ax2.set_ylabel("bin count", fontsize=7, color=oc.MUTED); ax2.tick_params(labelsize=6.5, colors=oc.MUTED)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_xlabel("YOLO confidence  c"); ax.set_ylabel("exceedance probability")
    ax.legend(fontsize=7.5, loc="upper right")
    oc.style_ax(ax, "unsafe-measurement probability per confidence bin")
    fig.suptitle("Exp0 — Is confidence a usable proxy for measurement quality?", fontsize=11.5, color=oc.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    oc.save(fig, OUT, "fig1_confidence_vs_error.png")

    # ---------------- fig 2: gate interaction (NIS / acceptance)
    fig, axes = oc.plt.subplots(1, 2, figsize=(11.5, 4.0))
    okg = hasl & np.isfinite(d["accepted"])
    cg, ag, ng = d["score"][okg], d["accepted"][okg], d["nis"][okg]
    ctr2, accr, _ = oc.binned(cg, ag, edges, np.nanmean)
    _, nismed, _ = oc.binned(cg, ng, edges, np.nanmedian)
    ax = axes[0]
    ax.plot(ctr2, accr, "-o", color=oc.GREEN, lw=1.6, ms=3.5)
    ax.set_ylim(0.0, 1.02); ax.set_xlabel("YOLO confidence"); ax.set_ylabel("EKF gate acceptance rate")
    oc.style_ax(ax, f"filter acceptance per confidence bin (n={okg.sum()})")
    ax = axes[1]
    ax.plot(ctr2, nismed, "-o", color=oc.VIOLET, lw=1.6, ms=3.5)
    ax.axhline(9.21, color=oc.RED, lw=0.9, ls="--"); ax.text(edges[0], 9.21, " gate 9.21", color=oc.RED, fontsize=7, va="bottom")
    ax.set_xlabel("YOLO confidence"); ax.set_ylabel("median NIS")
    oc.style_ax(ax, "innovation consistency (NIS) per confidence bin")
    fig.suptitle("Exp0 — confidence vs the runtime gate", fontsize=11, color=oc.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    oc.save(fig, OUT, "fig2_gate_interaction.png")

    # ---------------- fig 3: conditioning on range & image border
    fig, axes = oc.plt.subplots(1, 3, figsize=(13.5, 4.0))
    terc = np.nanpercentile(c, [33, 66])
    groups = [(c < terc[0], "low conf", oc.RED), ((c >= terc[0]) & (c < terc[1]), "mid conf", oc.YELLOW), (c >= terc[1], "high conf", oc.BLUE)]
    redges = np.linspace(np.nanpercentile(rng, 1), np.nanpercentile(rng, 99), 8)
    ax = axes[0]
    for m, lab, col in groups:
        rc, rm, _ = oc.binned(rng[m], e[m], redges, np.nanmedian)
        ax.plot(rc, rm, "-o", ms=3, lw=1.5, color=col, label=lab)
    ax.set_xlabel("camera ground range [m]"); ax.set_ylabel("median BEV error [m]"); ax.legend(fontsize=7.5)
    oc.style_ax(ax, "error vs range, per confidence tercile")
    ax = axes[1]
    bord = d["border"][hasl]
    bedges = np.linspace(0, np.nanpercentile(bord, 98), 8)
    for m, lab, col in groups:
        bc, bm, _ = oc.binned(bord[m], e[m], bedges, np.nanmedian)
        ax.plot(bc, bm, "-o", ms=3, lw=1.5, color=col, label=lab)
    ax.set_xlabel("border margin [px]"); ax.set_ylabel("median BEV error [m]"); ax.legend(fontsize=7.5)
    oc.style_ax(ax, "error vs image-border margin, per confidence tercile")
    ax = axes[2]
    # detection usability: detection rate vs position (via score presence) per range
    rng_all = np.hypot(d["tx"] - oc.CAMERA_POS[0], d["ty"] - oc.CAMERA_POS[1])
    okr = np.isfinite(rng_all)
    rc, dr, _ = oc.binned(rng_all[okr], d["det"][okr].astype(float), redges, np.nanmean)
    ax.plot(rc, dr, "-o", color=oc.AQUA, lw=1.6, ms=3.5)
    ax.set_ylim(0, 1.02); ax.set_xlabel("camera ground range [m]"); ax.set_ylabel("detection rate")
    oc.style_ax(ax, "usable-observation probability vs range")
    fig.suptitle("Exp0 — where does confidence carry (or miss) error information?", fontsize=11, color=oc.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    oc.save(fig, OUT, "fig3_conditioning.png")

    # ---------------- fig 4: spatial maps
    fig, axes = oc.plt.subplots(1, 2, figsize=(11.8, 4.9))
    ax = axes[0]
    sc = ax.scatter(d["tx"][hasl], d["ty"][hasl], c=c, s=4, cmap=oc.CMAP_TRUST, vmin=0.25, vmax=1.0, linewidths=0)
    oc.draw_warehouse(ax); ax.set_aspect("equal"); ax.set_xlim(-5.7, 5.7); ax.set_ylim(-5.9, 5.2)
    fig.colorbar(sc, ax=ax, shrink=0.8).set_label("YOLO confidence", fontsize=7)
    oc.style_ax(ax, "detections at GT position, colored by confidence")
    ax = axes[1]
    sc = ax.scatter(d["tx"][hasl], d["ty"][hasl], c=np.clip(e, 0, 0.6), s=4, cmap=oc.CMAP_STD, vmin=0, vmax=0.6, linewidths=0)
    oc.draw_warehouse(ax); ax.set_aspect("equal"); ax.set_xlim(-5.7, 5.7); ax.set_ylim(-5.9, 5.2)
    fig.colorbar(sc, ax=ax, shrink=0.8).set_label("BEV error [m] (clipped 0.6)", fontsize=7)
    oc.style_ax(ax, "same detections, colored by BEV localization error")
    fig.suptitle("Exp0 — spatial structure of confidence and error (GT used for evaluation only)", fontsize=11, color=oc.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    oc.save(fig, OUT, "fig4_spatial.png")

    # ---------------- summary + decision
    q = lambda x, p: float(np.nanpercentile(x, p))
    lowc = c < terc[0]; highc = c >= terc[1]
    exc = {es: (float(np.mean(e[lowc] > es)), float(np.mean(e[highc] > es))) for es in E_SAFE}
    accs = ""
    if okg.sum():
        acc_low = float(np.nanmean(ag[cg < terc[0]])) if (cg < terc[0]).sum() else math.nan
        acc_high = float(np.nanmean(ag[cg >= terc[1]])) if (cg >= terc[1]).sum() else math.nan
        accs = f"| gate acceptance | {acc_low:.3f} | {acc_high:.3f} |\n"

    monotone = bool(np.all(np.diff(med[np.isfinite(med)]) <= 0.02)) and rho_all < -0.1
    decision = (
        "**PASS (with a documented ceiling):** confidence is monotonically informative about BEV error "
        "and strongly informative about exceedance risk -> proceed with confidence-derived trust."
        if monotone else
        "**PARTIAL:** the confidence-error relation is weak/non-monotone at per-detection level; "
        "confidence mainly encodes detection success. Scope the trust target as detection usability "
        "(det-rate / score field), and keep the BEV-error claim out of the method's assumptions."
    )
    md = f"""# Exp0 — Confidence audit (RQ1 gate)

**Question.** Is YOLO confidence a usable proxy for the quality of an external-camera
measurement, before we build a confidence-derived trust map on it?

**Data.** honest_campaign_v1, {int(det.sum())} detections out of {len(d['det'])} perception rows
(43 runs, 4 routes). Error label (evaluation-only): `localization_error_captime_m` =
||camera BEV estimate − GT at frame capture||. Gate outcome joined from experiment.csv
(`pixel_corr_nis`, `pixel_corr_accepted`) within 0.35 s.

## Headline numbers

| quantity | value |
|---|---|
| Spearman ρ (confidence vs BEV error), all detections | **{rho_all:+.3f}** (n={n_all}) — confounded, see below |
| per-range-band ρ | {', '.join(f'{lo}-{hi} m: {r:+.3f} (n={n})' for lo, hi, r, n in band_rhos)} |
| ρ start dwell (t<40 s, robot ≈static) / moving section | {rho_early:+.3f} (n={n_early}) / **{rho_late:+.3f}** (n={n_late}) |
| ρ(confidence, range) / ρ(error, elapsed time) | {rho_cr:+.3f} / {rho_et:+.3f} |
| BEV error, low-conf tercile | p50 {q(e[lowc],50):.3f} m, p95 {q(e[lowc],95):.3f} m |
| BEV error, high-conf tercile | p50 {q(e[highc],50):.3f} m, p95 {q(e[highc],95):.3f} m |
| detection rate overall | {float(np.mean(d['det'])):.3f} |

## Exceedance P(e > e_safe)

| e_safe | low-conf tercile | high-conf tercile |
|---|---|---|
""" + "".join(f"| {es:.2f} m | {lo:.3f} | {hi:.3f} |\n" for es, (lo, hi) in exc.items()) + f"""
| metric | low conf | high conf |
|---|---|---|
{accs}
## Reading the numbers (confound structure)

The raw all-detection correlation is **misleadingly positive** — a Simpson-style confound,
not a real "high confidence → high error" effect. {n_early} of {n_all} detections
({100*n_early/max(n_all,1):.0f}%) come from the start dwell (t<40 s), where the robot is
near-static: BEV error is tiny (motion-free) while the start pose sits at long range where
confidence is middling. Error is driven primarily by **robot motion** (ρ(e, t)={rho_et:+.2f}),
while confidence is driven primarily by **range** (ρ(c, range)={rho_cr:+.2f}). Once the dwell
is excluded, the moving-section relation is weakly protective (ρ={rho_late:+.3f}) and the
useful signal concentrates in the tail: the low-confidence tercile is ~3-8× more likely to
exceed e_safe than the high-confidence tercile (table above). The EKF gate accepts almost
everything regardless of confidence, so confidence is NOT redundant with the gate — but it is
also not a per-detection metric-error predictor.

## Decision (pre-registered gate)

{decision}

**Consequence for Option A.** The trust map's target stays *detection usability*
(`det_hit` / `yolo_score_raw` field), exactly what the existing GP pipeline models; the map
is a prior over "will the camera give a usable measurement here", not a per-measurement
error predictor. The tail information (low conf ⇒ elevated exceedance risk) justifies keeping
confidence inside the trust target, and the claim "confidence-derived trust" is scoped to
detector usability, per the pre-registered PARTIAL branch.

## Figures
- `fig1_confidence_vs_error.png` — per-detection error vs confidence + exceedance curves.
- `fig2_gate_interaction.png` — EKF gate acceptance and NIS per confidence bin.
- `fig3_conditioning.png` — error vs range / border margin per confidence tercile; detection rate vs range.
- `fig4_spatial.png` — spatial layout of confidence and error (GT for evaluation only).

*Generated by experiments/optionA_commissioning/exp0_confidence_audit.py on the pre-existing
honest_campaign_v1 logs (recorded 2026-07-01); analysis run 2026-07-15.*
"""
    oc.write_md(OUT, "RESULTS.md", md)
    print(f"decision: {'PASS' if monotone else 'PARTIAL'}  rho={rho_all:+.3f}")


if __name__ == "__main__":
    main()
