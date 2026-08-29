#!/usr/bin/env python3
"""Is the fused correction overconfident because of bias, or because of shared error?

    python3 experiments/estimator_consistency/decompose.py

Offline, on the schema-4 drives. Writes logs/studies/estimator_consistency/numbers.json.

A consistent planar estimator has mean NEES 2. Ours is well above that. This separates
the two things that can cause it, because they need opposite fixes:

  bias                 a constant offset the covariance never claimed to cover.
                       Removing the mean residual removes it. Commissioning fixes it.

  understated Sigma    the covariance is too small for the scatter that is actually there.
                       Survives debiasing. If it comes from fusing correlated cameras as
                       independent, it gets worse with more cameras, and commissioning
                       each camera separately can never find it.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "fusion_on_fixed_routes"))
import aligned as A  # noqa: E402

OUT = REPO / "logs/studies/estimator_consistency"
CAMPAIGN = REPO / "logs/studies/fusion_on_fixed_routes/drives_realistic_speed_n1_20260829"
#: planar NEES targets for a consistent estimator
NEES_MEAN_TARGET = 2.0
NEES_MEDIAN_TARGET = 2.0 * math.log(2.0)


def _nees(residuals: np.ndarray, covs: np.ndarray) -> np.ndarray:
    out = []
    for r, C in zip(residuals, covs):
        try:
            out.append(float(r @ np.linalg.inv(C) @ r))
        except np.linalg.LinAlgError:
            out.append(math.nan)
    return np.array(out)


def _summary(name, residuals, covs):
    """NEES as measured, and again after the mean residual is removed."""
    n = _nees(residuals, covs)
    debiased = residuals - residuals.mean(axis=0)
    nd = _nees(debiased, covs)
    ok = np.isfinite(n) & np.isfinite(nd)
    n, nd = n[ok], nd[ok]
    bias = residuals.mean(axis=0)
    # how much the stated covariance would have to grow for mean NEES to reach 2
    inflate = float(np.mean(nd) / NEES_MEAN_TARGET) if len(nd) else math.nan
    # the empirical spread the covariance is failing to describe
    emp = np.cov(residuals.T) if len(residuals) > 2 else np.full((2, 2), math.nan)
    stated = covs.mean(axis=0)
    return {
        "name": name,
        "n": int(len(n)),
        "bias_cm": [round(float(bias[0]) * 100, 3), round(float(bias[1]) * 100, 3)],
        "bias_norm_cm": round(float(np.linalg.norm(bias)) * 100, 3),
        "rms_residual_cm": round(float(np.sqrt((residuals ** 2).sum(axis=1).mean())) * 100, 3),
        "nees_mean": round(float(np.mean(n)), 2),
        "nees_median": round(float(np.median(n)), 2),
        "nees_mean_after_debias": round(float(np.mean(nd)), 2),
        "nees_median_after_debias": round(float(np.median(nd)), 2),
        "bias_share_of_overconfidence_pct": (
            round(float((np.mean(n) - np.mean(nd)) / max(np.mean(n) - NEES_MEAN_TARGET, 1e-9) * 100), 1)
            if np.mean(n) > NEES_MEAN_TARGET else None),
        "covariance_inflation_needed": round(inflate, 2),
        "stated_sigma_cm": [round(math.sqrt(max(stated[0, 0], 0)) * 100, 2),
                            round(math.sqrt(max(stated[1, 1], 0)) * 100, 2)],
        "empirical_sigma_cm": [round(math.sqrt(max(emp[0, 0], 0)) * 100, 2),
                               round(math.sqrt(max(emp[1, 1], 0)) * 100, 2)],
    }


#: The arms whose observation model is the frozen method. O1 and O2 deliberately misread
#: what a detector's box means, so their residuals describe those models, not the estimator.
HULL_ARMS = ("F1", "F2", "F3", "F4")


def collect(runs, *, admitted_only=True):
    """Per-camera readings and fused answers, each already scored at its own instant.

    Everything comes through ``aligned``: it deduplicates the four log rows the manager
    writes per detection and pairs each quantity with the truth at the stamp that quantity
    describes. Reading the CSV directly would silently undo both.
    """
    per_cam_r, per_cam_c, per_cam_id = [], [], []
    fused_r, fused_c, fused_n = [], [], []
    by_batch = defaultdict(dict)
    for run in runs:
        for row in A.readings(run, admitted_only=admitted_only):
            xy = np.array([row["obs_x"], row["obs_y"]], dtype=float)
            truth = np.asarray(row["truth"], dtype=float)
            C = np.asarray(row["cov"], dtype=float)
            if not (np.isfinite(xy).all() and np.isfinite(truth).all() and np.isfinite(C).all()):
                continue
            if C[0, 0] <= 0 or C[1, 1] <= 0:
                continue
            r = xy - truth
            per_cam_r.append(r); per_cam_c.append(C); per_cam_id.append(row["camera"])
            by_batch[(str(run), row["source_batch_id"])][row["camera"]] = r
        for row in A.fused_answers(run):
            xy = np.asarray(row["fused_xy"], dtype=float)
            truth = np.asarray(row["truth"], dtype=float)
            C = np.asarray(row["fused_cov"], dtype=float)
            if not (np.isfinite(xy).all() and np.isfinite(truth).all() and np.isfinite(C).all()):
                continue
            if C[0, 0] <= 0 or C[1, 1] <= 0:
                continue
            fused_r.append(xy - truth); fused_c.append(C); fused_n.append(row["n_used"])
    return (np.array(per_cam_r), np.array(per_cam_c), per_cam_id,
            np.array(fused_r), np.array(fused_c), np.array(fused_n, dtype=float), by_batch)


def cross_camera_correlation(by_batch):
    """Do two cameras err the same way at the same instant?

    This is the quantity independent fusion assumes is zero. It is measured on the RAW
    residuals, so a common bias counts -- which is the point: a shared offset is exactly
    what precision-addition cannot represent.
    """
    pairs = defaultdict(lambda: ([], []))
    for cams in by_batch.values():
        ids = sorted(cams)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                pairs[(a, b)][0].append(cams[a])
                pairs[(a, b)][1].append(cams[b])
    out = {}
    allx, ally = [], []
    for (a, b), (ra, rb) in sorted(pairs.items()):
        ra, rb = np.array(ra), np.array(rb)
        if len(ra) < 20:
            continue
        cx = float(np.corrcoef(ra[:, 0], rb[:, 0])[0, 1])
        cy = float(np.corrcoef(ra[:, 1], rb[:, 1])[0, 1])
        out[f"{a}|{b}"] = {"n": int(len(ra)), "corr_x": round(cx, 3), "corr_y": round(cy, 3)}
        allx.extend(ra[:, 0] * rb[:, 0]); ally.extend(ra[:, 1] * rb[:, 1])
    # the shared component: mean product of two cameras' residuals is the covariance
    # they have in common, which independent fusion sets to zero by assumption
    shared_xx = float(np.mean(allx)) if allx else math.nan
    shared_yy = float(np.mean(ally)) if ally else math.nan
    out["_shared"] = {
        "shared_cov_xx_m2": shared_xx, "shared_cov_yy_m2": shared_yy,
        "implied_shared_sigma_cm": round(
            math.sqrt(max((shared_xx + shared_yy) / 2.0, 0.0)) * 100, 2),
        "note": ("mean product of two cameras' residuals at the same detector batch. "
                 "Independent fusion assumes this is zero; whatever it is, is the part "
                 "that does not average away."),
    }
    return out


def main() -> int:
    runs = sorted(p for p in CAMPAIGN.glob("*/*/*/experiment_*")
                  if (p / "fusion_observations.csv").is_file()
                  and p.parts[-3] in HULL_ARMS)
    if not runs:
        raise SystemExit(f"no hull-arm drives under {CAMPAIGN}")
    # What the filter actually consumed. The unconditional set is reported beside it,
    # because admission runs partly through agreeing with the other cameras and so is
    # not a clean sample of the sensor.
    pcr, pcc, pcid, fr, fc, fn, by_batch = collect(runs, admitted_only=True)
    upcr, upcc, _, _, _, _, _ = collect(runs, admitted_only=False)
    OUT.mkdir(parents=True, exist_ok=True)

    result = {
        "drives": len(runs),
        "campaign": str(CAMPAIGN.relative_to(REPO)),
        "nees_targets": {"mean": NEES_MEAN_TARGET, "median": round(NEES_MEDIAN_TARGET, 3)},
        "single_camera_admitted": _summary("one ADMITTED camera reading vs its stated covariance", pcr, pcc),
        "single_camera_unconditional": _summary("every camera reading, admitted or not", upcr, upcc),
        "fused": _summary("the fused correction against its stated covariance", fr, fc),
        "fused_by_camera_count": {},
        "cross_camera": cross_camera_correlation(by_batch),
    }
    for k in (1, 2, 3, 4, 5):
        m = fn == k
        if m.sum() > 30:
            result["fused_by_camera_count"][str(k)] = _summary(f"{k} camera(s)", fr[m], fc[m])
    (OUT / "numbers.json").write_text(json.dumps(result, indent=2) + "\n")

    def show(s):
        print(f"  {s['name']:52} n={s['n']:5}")
        print(f"    residual RMS {s['rms_residual_cm']:6.2f} cm   bias {s['bias_norm_cm']:5.2f} cm "
              f"{s['bias_cm']}")
        print(f"    stated sigma {s['stated_sigma_cm']}  vs empirical {s['empirical_sigma_cm']} cm")
        print(f"    NEES mean {s['nees_mean']:7.2f} -> after removing the bias {s['nees_mean_after_debias']:7.2f}"
              f"   (target {NEES_MEAN_TARGET})")
        if s["bias_share_of_overconfidence_pct"] is not None:
            print(f"    bias explains {s['bias_share_of_overconfidence_pct']:5.1f}% of the excess; "
                  f"covariance would need x{s['covariance_inflation_needed']}")

    print(f"{len(runs)} hull-arm drives (F1-F4 only; O1 and O2 misread the box by design)\n")
    show(result["single_camera_admitted"]); print()
    show(result["single_camera_unconditional"]); print()
    show(result["fused"]); print()
    print("  fused, split by how many cameras contributed:")
    for k, s in result["fused_by_camera_count"].items():
        print(f"    {k} camera(s): NEES {s['nees_mean']:7.2f} -> debiased {s['nees_mean_after_debias']:7.2f}"
              f"   inflation x{s['covariance_inflation_needed']:<6} n={s['n']}")
    sh = result["cross_camera"]["_shared"]
    print(f"\n  shared error between two cameras at the same instant: "
          f"{sh['implied_shared_sigma_cm']} cm")
    for k, v in sorted(result["cross_camera"].items()):
        if k != "_shared":
            print(f"    {k:24} n={v['n']:5}  corr x {v['corr_x']:+.3f}  y {v['corr_y']:+.3f}")
    print(f"\nwrote {OUT / 'numbers.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
