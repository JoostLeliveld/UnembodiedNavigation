"""The half-centimetre lean left once the observation model is right.

**This is the only thing here that deserves the word bias.**  The box-versus-centre problem
is 30 cm and belongs to ``observation``; it is not fitted and not corrected.  What is left
after that is 0.53 cm, and it goes to 0.29 cm.

Small, but not ignorable for the reason that always applies to a lean: random scatter shrinks
as the square root of the number of looks, and a lean does not shrink at all.  A robot that
sees itself a thousand times still carries it.

**Why it drifts with distance, and why that is not the detector.**  Splitting the residual by
how much of the robot is visible separates three things that look alike when pooled: on an
unoccluded robot the hull matches the rendered outline to 0.006 px and the detector
reproduces its own label convention to about a tenth of a pixel.  The leftover comes from
mildly occluded sightings that ``admission`` still admits, and those get commoner with
distance -- a quarter of sightings under 8 m, nearly all beyond 20 m.  So the correction is
indexed on distance because *occlusion prevalence* is, which makes it a property of **this
stock arrangement** rather than of the cameras.  Restock the warehouse and it must be
re-measured.

**How little data.**  It converges by about ten floor positions; from there to a hundred the
held-out residual barely moves, so more parking buys confidence in the number rather than a
better number.  Below five it is a lottery -- the median still looks fine but individual
draws land worse than no correction at all.

A calibration detail and a backup slide.  Nothing downstream rests on it.
"""
from __future__ import annotations

import collections
import math

import numpy as np

DESIGN = "b(range) = c0 + c1*range_m + c2*range_m^2, in pixels, per axis"


def design_row(range_m):
    """The three terms the correction is built from, for one sighting."""
    return np.array([1.0, range_m, range_m ** 2])


def fit(sightings):
    """Least squares: six numbers, three per axis.  Returns a (3, 2) coefficient matrix."""
    A = np.array([design_row(s["range_m"]) for s in sightings])
    B = np.array([[s["du_px"], s["dv_px"]] for s in sightings])
    return np.linalg.lstsq(A, B, rcond=None)[0]


def apply(coeffs, range_m):
    """The pixel offset to subtract from a sighting taken at this distance."""
    return design_row(range_m) @ coeffs


def ground_error(sighting, coeffs=None):
    """A sighting's error on the floor, in centimetres, with or without the correction."""
    residual = np.array([sighting["du_px"], sighting["dv_px"]])
    if coeffs is not None:
        residual = residual - apply(coeffs, sighting["range_m"])
    return (sighting["Jinv"] @ residual) * 100.0


GROUPS = {
    "camera": lambda s: s["camera"],
    "range": lambda s: int(s["range_m"] // 4),
    "heading": lambda s: int((math.degrees(s["rel_heading_rad"]) + 180) // 30),
    "area": lambda s: (int(s["x"] // 5), int(s["y"] // 5)),
}


def score(sightings, coeffs, stated_spread, min_bin=25):
    """What is left, pooled and in the worst group of any kind.

    ``worst group`` is a maximum over dozens of noisy group means, so it is inflated even
    when nothing is really there.  Always read it against ``worst_group_null`` rather than as
    a finding on its own.
    """
    e = np.array([ground_error(s, coeffs) for s in sightings])
    pooled = float(np.linalg.norm(e.mean(axis=0)))
    worst, where, ngroups = 0.0, "", 0
    for gname, gf in GROUPS.items():
        buckets = collections.defaultdict(list)
        for s, ev in zip(sightings, e):
            buckets[gf(s)].append(ev)
        for k, sub in buckets.items():
            if len(sub) < min_bin:
                continue
            ngroups += 1
            v = float(np.linalg.norm(np.mean(sub, axis=0)))
            if v > worst:
                worst, where = v, f"{gname} {k}"
    return {"pooled_mean_error_cm": pooled, "worst_conditional_cm": worst,
            "worst_conditional_where": where, "n_groups": ngroups,
            "random_spread_cm": stated_spread, "r_b": pooled / stated_spread,
            "rmse_cm": float(np.sqrt((np.linalg.norm(e, axis=1) ** 2).mean())),
            "median_cm": float(np.median(np.linalg.norm(e, axis=1)))}


def worst_group_null(sightings, coeffs, observed, draws=300, seed=0, min_bin=25):
    """Is the worst group distinguishable from a maximum over many noisy bins?

    Shuffle the residuals between sightings, destroying every conditional structure, and
    recompute the same statistic.  On this capture the observed value landed on the 90th
    percentile of that null -- so it was not a finding.  Heading was the one grouping that
    survived the same test.
    """
    rng = np.random.default_rng(seed)
    resid = [np.array([s["du_px"], s["dv_px"]]) - apply(coeffs, s["range_m"]) for s in sightings]
    null = []
    for _ in range(draws):
        perm = rng.permutation(len(resid))
        e = [(sightings[i]["Jinv"] @ resid[perm[i]]) * 100.0 for i in range(len(sightings))]
        w = 0.0
        for gf in GROUPS.values():
            buckets = collections.defaultdict(list)
            for s, ev in zip(sightings, e):
                buckets[gf(s)].append(ev)
            for sub in buckets.values():
                if len(sub) >= min_bin:
                    w = max(w, float(np.linalg.norm(np.mean(sub, axis=0))))
        null.append(w)
    null = np.array(null)
    return {"median_cm": float(np.median(null)), "p90_cm": float(np.percentile(null, 90)),
            "fraction_of_shuffles_at_or_above_observed": float((null >= observed).mean()),
            "note": ("worst-conditional is a maximum over many noisy group means, so it is "
                     "inflated even with no real structure; compare against this null "
                     "before calling it a finding")}


def mechanism(sightings, min_bin=25):
    """Where the residual comes from: the hull, the detector, or admitted occlusion."""
    usable = [s for s in sightings if not math.isnan(s["visible_height"])]
    by_visible, prevalence = {}, {}
    for lo, hi in ((0.85, 0.90), (0.90, 0.95), (0.95, 0.98), (0.98, 0.995), (0.995, 2.0)):
        sub = [s for s in usable if lo <= s["visible_height"] < hi]
        if len(sub) < min_bin:
            continue
        by_visible[f"{lo:.3f}-{hi:.3f}"] = {
            "n": len(sub),
            "total_residual_px": float(np.mean([s["det_bottom_v"] - s["pred_hull_bottom_v"] for s in sub])),
            "geometry_px": float(np.mean([s["mask_bottom_v"] + 1.0 - s["pred_hull_bottom_v"] for s in sub])),
            "detector_px": float(np.mean([s["det_bottom_v"] - s["mask_bottom_v"] - 1.0 for s in sub])),
        }
    for lo, hi in ((0, 8), (8, 14), (14, 20), (20, 25)):
        sub = [s for s in usable if lo <= s["range_m"] < hi]
        if len(sub) < min_bin:
            continue
        prevalence[f"{lo}-{hi}m"] = {
            "n": len(sub),
            "median_visible_height": float(np.median([s["visible_height"] for s in sub])),
            "fraction_partly_hidden": float(np.mean([s["visible_height"] < 0.98 for s in sub])),
            "total_residual_px": float(np.mean([s["det_bottom_v"] - s["pred_hull_bottom_v"] for s in sub])),
        }
    return {"by_visible_height": by_visible, "occlusion_prevalence_by_range": prevalence,
            "note": ("The hull term vanishes on unoccluded robots, so the observation model "
                     "is correct and the detector reproduces its label convention to about a "
                     "tenth of a pixel. What the correction removes is driven by mildly "
                     "occluded sightings that admission admits, and those get commoner with "
                     "range -- which is why a correction indexed on range works. It is "
                     "therefore a property of THIS stock arrangement, not of the cameras, "
                     "and must be re-measured if the warehouse is restocked.")}
