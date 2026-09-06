"""What the NEES tail is made of: range, camera, box mismatch, and how much is shared.

Regenerates every number `docs/ANSWERS.md` quotes for Q6 (what owns the tail) and Q2 (how
much of the cameras' error is shared), so neither is a scratchpad throwaway.

The finding, in one line: the stated covariance is `J^-1 sigma_px^2 J^-T` -- pure pixel noise
-- so it shrinks without limit as a camera gets closer, while the real error bottoms out at a
range-independent floor of about a centimetre. Close in, that floor is larger than the whole
claimed ellipse, so NEES explodes. It is a NEAR-range defect, the opposite end of the range
axis from the far-range bias that has been chased instead.

Ground truth forms residuals and does nothing else. Every quantity is scored against the truth
at the instant that quantity describes, through `aligned.readings`, which also drops the ~4
duplicate rows the manager writes per detection.

    python3 experiments/estimator_consistency/tail_anatomy.py [campaign_dir ...]

With no argument it reads the 1 m/s campaign, which is what ANSWERS.md's Q6 tables were built
from. Pass the 0.22 m/s campaign for the numbers at the speed the paper actually uses -- they
are much milder, and ANSWERS.md says so.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/fusion_on_fixed_routes"))
import aligned as A  # noqa: E402

DEFAULT = REPO / "logs/studies/fusion_on_fixed_routes/drives_realistic_speed_n1_20260829"
HULL_ARMS = ("F1", "F2", "F3", "F4")     # O1/O2 misread the box by design
OUT = REPO / "logs/studies/estimator_consistency/tail_anatomy.json"
RANGE_BINS = ((0, 4), (4, 6), (6, 8), (8, 10), (10, 13), (13, 16), (16, 20), (20, 99))
TAIL_NEES = 20.0


def load(campaigns):
    rows, batches = [], defaultdict(dict)
    for campaign in campaigns:
        for manifest in sorted(Path(campaign).rglob("run_manifest.json")):
            run = manifest.parent
            parts = run.relative_to(campaign).parts
            if len(parts) < 2 or parts[1] not in HULL_ARMS:
                continue
            for e in A.readings(run, admitted_only=True, dedupe=True):
                C, err = np.asarray(e["cov"], float), np.asarray(e["error"], float)
                if not (np.isfinite(C).all() and np.isfinite(err).all()):
                    continue
                if C[0, 0] <= 0 or C[1, 1] <= 0:
                    continue
                try:
                    nees = float(err @ np.linalg.inv(C) @ err)
                except np.linalg.LinAlgError:
                    continue
                rows.append(dict(
                    camera=e["camera"], nees=nees,
                    err_cm=float(np.linalg.norm(err) * 100),
                    sigma_cm=float(math.sqrt(0.5 * (C[0, 0] + C[1, 1])) * 100),
                    range_m=float(e["range_m"]), conf=float(e["conf"]),
                    bbox_h_px=float(e["bbox_h_px"]),
                    pred_h_px=float(e.get("pred_h_px", math.nan))))
                batches[(str(run), e["source_batch_id"])][e["camera"]] = (err, float(e["range_m"]))
    return rows, {k: v for k, v in batches.items() if len(v) >= 2}


def by_range(rows):
    out = {}
    for lo, hi in RANGE_BINS:
        sub = [r for r in rows if lo <= r["range_m"] < hi]
        if len(sub) < 20:
            continue
        sig = float(np.median([r["sigma_cm"] for r in sub]))
        err = float(np.median([r["err_cm"] for r in sub]))
        out[f"{lo}-{hi} m"] = dict(
            n=len(sub), claimed_sigma_cm=sig, real_error_cm=err,
            real_over_claim=err / sig if sig else math.nan,
            tail_rate=float(np.mean([r["nees"] > TAIL_NEES for r in sub])),
            median_nees=float(np.median([r["nees"] for r in sub])))
    return out


def by_camera(rows):
    out = {}
    for c in sorted({r["camera"] for r in rows}):
        sub = [r for r in rows if r["camera"] == c]
        out[c] = dict(n=len(sub),
                      tail_rate=float(np.mean([r["nees"] > TAIL_NEES for r in sub])),
                      median_error_cm=float(np.median([r["err_cm"] for r in sub])),
                      median_nees=float(np.median([r["nees"] for r in sub])),
                      median_range_m=float(np.median([r["range_m"] for r in sub])))
    return out


def floor_sweep(rows, floors=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5)):
    """A range-independent floor added in quadrature: what it does to the 4-sigma tail."""
    sig = np.array([r["sigma_cm"] for r in rows])
    err = np.array([r["err_cm"] for r in rows])
    return {f"{f:.1f} cm": dict(
        beyond_4_sigma=float(np.mean(err / np.sqrt(sig ** 2 + f ** 2) > 4.0)),
        median_real_over_claim=float(np.median(err / np.sqrt(sig ** 2 + f ** 2))))
        for f in floors}


def _spearman(a, b):
    ra, rb = np.empty(len(a)), np.empty(len(b))
    ra[np.argsort(a)] = np.arange(len(a))
    rb[np.argsort(b)] = np.arange(len(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def box_mismatch(rows):
    """|detected/predicted - 1| as a truth-free tail flag. Needs schema 5."""
    sub = [r for r in rows if math.isfinite(r["pred_h_px"]) and r["pred_h_px"] > 0
           and math.isfinite(r["bbox_h_px"])]
    if len(sub) < 50:
        return {"note": "no pred_h_px on these drives (schema <= 4)"}
    mis = np.array([abs(r["bbox_h_px"] / r["pred_h_px"] - 1.0) for r in sub])
    nees = np.array([r["nees"] for r in sub])
    rng = np.array([r["range_m"] for r in sub])
    out = {}
    for lo, hi in ((0, .01), (.01, .02), (.02, .04), (.04, .08), (.08, 9)):
        m = (mis >= lo) & (mis < hi)
        if m.sum() < 10:
            continue
        idx = np.where(m)[0]
        out[f"{lo:.2f}-{hi:.2f}"] = dict(
            n=int(m.sum()), median_nees=float(np.median(nees[m])),
            tail_rate=float(np.mean(nees[m] > TAIL_NEES)),
            median_error_cm=float(np.median([sub[i]["err_cm"] for i in idx])),
            median_range_m=float(np.median(rng[m])))
    out["spearman_mismatch_vs_nees"] = _spearman(mis, nees)
    out["spearman_range_vs_nees"] = _spearman(rng, nees)
    return out


def shared_error(batches):
    """`E[r_i . r_j]` over distinct camera pairs estimates the shared variance; and what
    fusing actually delivers against what independent errors would give."""
    def closest(v):
        vals = [x[1] for x in v.values() if math.isfinite(x[1])]
        return min(vals) if vals else math.nan

    def pairs(sel):
        cross, self_ = [], []
        for v in sel:
            rs = [v[c][0] for c in v]
            for i in range(len(rs)):
                self_.append(float(rs[i] @ rs[i]))
                for j in range(i + 1, len(rs)):
                    cross.append(float(rs[i] @ rs[j]))
        if len(cross) < 30:
            return None
        c, s = float(np.mean(cross)), float(np.mean(self_))
        return dict(n_pairs=len(cross), rms_error_cm=math.sqrt(s) * 100,
                    shared_cm=math.sqrt(max(c, 0.0)) * 100,
                    shared_fraction_of_squared_error=(max(c, 0.0) / s) if s else math.nan)

    vals = list(batches.values())
    out = {"all_batches": pairs(vals)}
    for lo, hi in ((0, 6), (6, 10), (10, 16), (16, 99)):
        sel = [v for v in vals
               if lo <= (closest(v) if math.isfinite(closest(v)) else -1) < hi]
        got = pairs(sel)
        if got:
            out[f"closest camera {lo}-{hi} m"] = got

    gain = {}
    for n in (2, 3, 4, 5):
        sel = [v for v in vals if len(v) == n]
        if len(sel) < 20:
            continue
        mean_single = [float(np.mean([np.linalg.norm(x[0]) for x in v.values()])) * 100
                       for v in sel]
        fused = [float(np.linalg.norm(np.mean([x[0] for x in v.values()], axis=0))) * 100
                 for v in sel]
        gain[f"{n} cameras"] = dict(
            batches=len(sel),
            best_single_cm=float(np.median([min(np.linalg.norm(x[0]) for x in v.values()) * 100
                                            for v in sel])),
            mean_single_cm=float(np.median(mean_single)),
            equal_weight_fused_cm=float(np.median(fused)),
            independence_would_give_cm=float(np.median(mean_single)) / math.sqrt(n))
    out["fusion_gain"] = gain
    return out


def main() -> int:
    campaigns = [Path(a).resolve() for a in sys.argv[1:]] or [DEFAULT]
    rows, batches = load(campaigns)
    if not rows:
        raise SystemExit(f"no hull-arm readings under {campaigns}")
    nees = np.array([r["nees"] for r in rows])
    result = dict(
        campaigns=[str(c) for c in campaigns], n_readings=len(rows),
        n_multi_camera_batches=len(batches),
        overall=dict(median_nees=float(np.median(nees)), mean_nees=float(nees.mean()),
                     tail_rate=float(np.mean(nees > TAIL_NEES)),
                     tail_share_of_nees_mass=float(nees[nees > TAIL_NEES].sum() / nees.sum())),
        by_range=by_range(rows), by_camera=by_camera(rows),
        floor_sweep=floor_sweep(rows), box_mismatch=box_mismatch(rows),
        shared_error=shared_error(batches),
        note="A consistent 2-D estimator gives median NEES 1.386 and mean 2.0.")

    o = result["overall"]
    print(f"{len(rows)} admitted hull readings, {len(batches)} multi-camera batches")
    print(f"NEES median {o['median_nees']:.2f}  mean {o['mean_nees']:.2f}  "
          f"tail(>{TAIL_NEES:.0f}) {100*o['tail_rate']:.2f}%  "
          f"carrying {100*o['tail_share_of_nees_mass']:.0f}% of the NEES mass\n")
    print(f"{'range':>10s} {'n':>6s} {'claimed':>10s} {'real':>9s} {'real/claim':>11s} {'tail':>7s}")
    for k, v in result["by_range"].items():
        print(f"{k:>10s} {v['n']:6d} {v['claimed_sigma_cm']:7.2f} cm "
              f"{v['real_error_cm']:6.2f} cm {v['real_over_claim']:10.2f}x "
              f"{100*v['tail_rate']:6.1f}%")
    print(f"\n{'camera':>8s} {'n':>6s} {'tail':>7s} {'median err':>12s} {'med range':>11s}")
    for k, v in result["by_camera"].items():
        print(f"{k:>8s} {v['n']:6d} {100*v['tail_rate']:6.1f}% "
              f"{v['median_error_cm']:9.2f} cm {v['median_range_m']:8.1f} m")
    print("\nfloor added in quadrature -> readings beyond 4 sigma")
    for k, v in result["floor_sweep"].items():
        print(f"  {k:>7s}  {100*v['beyond_4_sigma']:5.2f}%   "
              f"median real/claim {v['median_real_over_claim']:.2f}x")
    fg = result["shared_error"].get("fusion_gain", {})
    if fg:
        print(f"\n{'cameras':>9s} {'batches':>8s} {'mean single':>12s} {'fused':>9s} "
              f"{'independence':>14s}")
        for k, v in fg.items():
            print(f"{k:>9s} {v['batches']:8d} {v['mean_single_cm']:9.2f} cm "
                  f"{v['equal_weight_fused_cm']:6.2f} cm "
                  f"{v['independence_would_give_cm']:11.2f} cm")
    bm = result["box_mismatch"]
    if "spearman_mismatch_vs_nees" in bm:
        print(f"\nbox mismatch vs NEES  Spearman {bm['spearman_mismatch_vs_nees']:+.3f}   "
              f"(range vs NEES {bm['spearman_range_vs_nees']:+.3f})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
