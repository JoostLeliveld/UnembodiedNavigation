#!/usr/bin/env python3
"""Opportunity-gated availability: separate "robot out of view" from real detector misses.

The expB honest-null caveat #4: det_hit=0 conflates two very different things —
(a) the robot was geometrically OUT of the camera's field of view / occluded (no
opportunity to detect), and (b) the robot WAS geometrically visible but the
detector still failed (a real camera-service failure). A deployed reliability map
must not blame the camera for (a).

We use the shipped geometry-visibility field F (paper_artifacts/gp/
warehouse_visibility_gp_v1, logit space) as the OPPORTUNITY signal: F(x,y) high =
geometry says the robot at (x,y) is in view. Then:
  - opportunity = F(belief_xy) > F_OPP  (geometrically could be seen)
  - availability a(s) = P(det_hit | opportunity, s)  -- the deployment-relevant field
Compare the ungated field P(det_hit | s) (what expB fit) with the opportunity-gated
field, on region-disjoint held-out folds. Report how the 492 misses split into
out-of-FOV vs in-FOV (the genuinely interesting detector failures).

CONTROLLED/OPERATIONAL: positions = belief mean (operational), det_hit
(operational), F geometry field (MODEL, first-principles geometry — allowed
operational input, it is NOT gt/oracle). No gt_*/CAD-as-model-input.

Outputs -> logs/studies/single_camera_uigp_reliability/expC_opportunity_gated/
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
GP = REPO / "paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz"
OUT = REPO / "logs/studies/single_camera_uigp_reliability/expC_opportunity_gated"

F_OPP = -3.0        # logit; F > F_OPP  ==  geometry gives >~5% chance the robot is in view
NBX, NBY = 3, 2
ELL, NOISE_VAR = 0.90, 0.05
AGG_RES = 0.20      # naive GP ignores input covariance, so aggregation is fine/fast here


def load_events():
    P, y = [], []
    for r in csv.DictReader(open(EVENTS)):
        try:
            x = float(r["m_x"]); yy = float(r["m_y"]); h = int(float(r["det_hit"]))
        except (KeyError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(yy):
            P.append((x, yy)); y.append(h)
    return np.asarray(P, float), np.asarray(y, float)


def query(grid, xs, ys, XY):
    ix = np.clip(np.searchsorted(xs, XY[:, 0]), 0, len(xs) - 1)
    iy = np.clip(np.searchsorted(ys, XY[:, 1]), 0, len(ys) - 1)
    return grid[iy, ix]


def region_blocks(P):
    xe = np.quantile(P[:, 0], np.linspace(0, 1, NBX + 1))
    ye = np.quantile(P[:, 1], np.linspace(0, 1, NBY + 1))
    bx = np.clip(np.searchsorted(xe[1:-1], P[:, 0]), 0, NBX - 1)
    by = np.clip(np.searchsorted(ye[1:-1], P[:, 1]), 0, NBY - 1)
    return bx * NBY + by


def heldout_naive(P, y, blk, keep):
    """Region-disjoint held-out P(y|s) with the point GP, over the subset `keep`."""
    briers, nlls, aurocs = [], [], []
    for b in np.unique(blk[keep]):
        te = keep & (blk == b)
        tr = keep & (blk != b)
        if te.sum() < 20 or tr.sum() < 50:
            continue
        data = oc.make_event_data(P[tr], y[tr], np.zeros((int(tr.sum()), 2, 2)), blk[tr])
        agg = oc.aggregate(data, resolution_m=AGG_RES)
        mu, sig = oc.fit_predict("naive", agg, P[te], length_scale=ELL, noise_var=NOISE_VAR)
        p = M.clip_prob(M.probit_prob(mu, sig))
        briers.append(M.brier(y[te], p)); nlls.append(M.logloss(y[te], p))
        if len(np.unique(y[te])) == 2:
            aurocs.append(M.auroc(y[te], p))
    def ms(v):
        return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))
    return ms(briers), ms(nlls), ms(aurocs), len(briers)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P, y = load_events()
    blk = region_blocks(P)
    d = np.load(GP, allow_pickle=True)
    F = query(d["F_mean_map"], d["xs"], d["ys"], P)
    opp = F > F_OPP

    n = len(y)
    miss = y == 0
    miss_oofov = miss & ~opp        # geometrically not-in-view misses (NOT camera failures)
    miss_infov = miss & opp         # in-view but still missed (REAL detector failures)
    print(f"events {n}  det rate {y.mean():.3f}  misses {int(miss.sum())}")
    print(f"opportunity (F>{F_OPP}) rate = {opp.mean():.3f}  (n_opp={int(opp.sum())})")
    print(f"  misses out-of-FOV (not camera's fault) : {int(miss_oofov.sum())} "
          f"({miss_oofov.sum()/max(miss.sum(),1):.0%} of misses)")
    print(f"  misses IN-FOV (real detector failures) : {int(miss_infov.sum())} "
          f"({miss_infov.sum()/max(miss.sum(),1):.0%} of misses)")

    all_mask = np.ones(n, dtype=bool)
    (bU, nU, aU, kU) = heldout_naive(P, y, blk, all_mask)                 # ungated (expB-style)
    # gated: fit availability a(s)=P(det|opportunity) on opportunity frames only
    (bG, nG, aG, kG) = heldout_naive(P, y, blk, opp)

    det_all = y.mean()
    det_opp = y[opp].mean() if opp.any() else float("nan")

    lines = [
        "| field | target | held-out Brier | held-out NLL | held-out AUROC | folds | base rate |",
        "|---|---|---|---|---|---|---|",
        f"| ungated (expB) | P(det \\| s), all frames | {bU[0]:.3f}±{bU[1]:.3f} | {nU[0]:.3f}±{nU[1]:.3f} | {aU[0]:.3f}±{aU[1]:.3f} | {kU} | {det_all:.3f} |",
        f"| opportunity-gated | a(s)=P(det \\| in-view, s) | {bG[0]:.3f}±{bG[1]:.3f} | {nG[0]:.3f}±{nG[1]:.3f} | {aG[0]:.3f}±{aG[1]:.3f} | {kG} | {det_opp:.3f} |",
    ]

    md = f"""# Experiment C — opportunity-gated availability (single camera, real drive)

**Evidence class:** REAL EXPERIMENT (operational inputs) + MODEL (geometry field as
the opportunity gate). Provenance: `single_cam_commissioning_v1` (detector
`warehouse_yolo_detector_v1`); opportunity from the shipped geometry-visibility
field `warehouse_visibility_gp_v1` (logit F, F>{F_OPP} = geometrically in view).
No gt_*/oracle/CAD-as-model-input.

## The point
A deployed reliability map must not blame the camera when the robot was simply not
in view. Splitting the {int(miss.sum())} detection misses by geometric visibility:

- **out-of-FOV misses** (robot not geometrically visible; NOT a camera failure):
  **{int(miss_oofov.sum())}** ({miss_oofov.sum()/max(miss.sum(),1):.0%} of misses)
- **in-FOV misses** (robot visible but detector still failed; the REAL service
  failures the map should learn): **{int(miss_infov.sum())}**
  ({miss_infov.sum()/max(miss.sum(),1):.0%} of misses)

Opportunity rate over the drive = {opp.mean():.2f}. Detection rate rises from
{det_all:.3f} (all frames) to {det_opp:.3f} once we condition on the robot being
geometrically in view — i.e. most misses are the robot leaving the camera's
footprint, exactly as a warehouse operator would expect.

## Held-out prediction (region-disjoint, point GP)
{chr(10).join(lines)}

## Reading
- The ungated field mostly learns the geometric footprint (where the camera can
  see at all); the opportunity-gated field a(s) isolates the deployment-relevant
  question: *given the robot is in view, will the detector produce a usable
  observation here?*
- This is the availability layer a(s) of the factorised service model (ch.04); the
  next layer is quality q(s) = P(usable localization | detection), which needs a
  localization-error label and is best measured on a multi-session capture.

*Generated by experiments/single_camera_uigp_reliability/tools/expC_opportunity_gated_availability.py.*
"""
    oc.write_md(OUT, "RESULTS.md", md)
    print("wrote", OUT / "RESULTS.md")


if __name__ == "__main__":
    main()
