"""Three checks on the observability result.

1. Prior sensitivity. A posterior that tracks the prior was never learned from the
   data. Scaling the prior separates the two.
2. Count-matched crossing angle. Wide-separation pairs outnumber narrow ones 12:1
   in this capture, so the wide arm must be subsampled to the narrow arm's site
   count before the comparison means anything.
3. How many co-observed sites a commissioning pass actually needs.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calib_geometry as cg  # noqa: E402
import observability as ob  # noqa: E402

RNG = np.random.default_rng(0)
DRAWS = 8


def posterior(info, free, prior_scale=1.0):
    prior = np.tile(ob.PRIOR_SD * prior_scale, len(free))
    return np.sqrt(np.diag(np.linalg.inv(info + np.diag(1.0 / prior ** 2))))


def summarise(sd, free):
    p = cg.N_PARAM
    return {name: {pn: float(sd[i * p + k]) for k, pn in enumerate(cg.PARAM_NAMES)}
            for i, name in enumerate(free)}


def main() -> None:
    cameras, marker_z, offsets = cg.load_capture(ob.DATASET)
    cov_px, _ = ob.pixel_noise()
    rows = ob.sightings()
    names = tuple(sorted(cameras))
    multi = rows.groupby("site").camera.nunique()
    rows["n_cams"] = rows.site.map(multi)

    out = {"stated_prior_sd": dict(zip(cg.PARAM_NAMES, [float(v) for v in ob.PRIOR_SD]))}

    # ---- 1. prior sensitivity, on the full capture -------------------------
    info_all, _ = ob.information(rows, cameras, marker_z, offsets, cov_px, names)
    out["prior_sensitivity"] = {}
    for scale in (0.25, 1.0, 4.0, 20.0):
        sd = posterior(info_all, names, scale)
        p = cg.N_PARAM
        per_param = {pn: float(np.mean([sd[i * p + k] for i in range(len(names))]))
                     for k, pn in enumerate(cg.PARAM_NAMES)}
        out["prior_sensitivity"][f"prior_x{scale}"] = {
            "prior_sd": {pn: float(ob.PRIOR_SD[k] * scale)
                         for k, pn in enumerate(cg.PARAM_NAMES)},
            "posterior_sd_mean_over_cameras": {k: round(v, 5) for k, v in per_param.items()},
        }

    # ---- 2. count-matched crossing angle ----------------------------------
    pair_rows = []
    for site, grp in rows[rows.n_cams >= 2].groupby("site", sort=False):
        pose = (float(grp.x.iloc[0]), float(grp.y.iloc[0]), float(grp.yaw_rad.iloc[0]))
        for a, b in combinations(sorted(set(grp.camera)), 2):
            sep = abs(ob.bearing_deg(cameras[a], pose) - ob.bearing_deg(cameras[b], pose))
            pair_rows.append({"site": site, "separation_deg": min(sep, 360.0 - sep)})
    pairs = pd.DataFrame(pair_rows)
    wide = set(pairs.loc[pairs.separation_deg > 100.0, "site"])
    narrow = set(pairs.loc[pairs.separation_deg <= 100.0, "site"])
    wide_only, narrow_only = sorted(wide - narrow), sorted(narrow - wide)
    n = len(narrow_only)

    def arm(sites):
        info, _ = ob.information(rows[rows.site.isin(set(sites))], cameras, marker_z,
                                offsets, cov_px, names)
        return posterior(info, names)

    narrow_sd = arm(narrow_only)
    draws = [arm([wide_only[i] for i in RNG.choice(len(wide_only), n, replace=False)])
             for _ in range(DRAWS)]
    p = cg.N_PARAM
    out["count_matched_crossing_angle"] = {
        "sites_per_arm": n,
        "wide_draws": DRAWS,
        "narrow_pairs_median_separation_deg": round(
            float(pairs[pairs.site.isin(narrow_only)].separation_deg.median()), 1),
        "wide_pairs_median_separation_deg": round(
            float(pairs[pairs.site.isin(wide_only)].separation_deg.median()), 1),
        "posterior_sd_mean_over_cameras": {
            pn: {
                "narrow": round(float(np.mean([narrow_sd[i * p + k] for i in range(4)])), 5),
                "wide_median_of_draws": round(float(np.median(
                    [np.mean([d[i * p + k] for i in range(4)]) for d in draws])), 5),
            } for k, pn in enumerate(cg.PARAM_NAMES)
        },
    }

    # ---- 3. how many co-observed sites are needed --------------------------
    overlap_sites = sorted(set(rows.loc[rows.n_cams >= 2, "site"]))
    ladder = []
    for count in (25, 50, 100, 200, 400, 921):
        count = min(count, len(overlap_sites))
        reps = 4 if count < len(overlap_sites) else 1
        got = []
        for _ in range(reps):
            pick = ([overlap_sites[i] for i in RNG.choice(len(overlap_sites), count, replace=False)]
                    if count < len(overlap_sites) else overlap_sites)
            got.append(arm(pick))
        ladder.append({
            "overlap_sites": count,
            "posterior_sd_mean_over_cameras": {
                pn: round(float(np.median([np.mean([g[i * p + k] for i in range(4)])
                                           for g in got])), 5)
                for k, pn in enumerate(cg.PARAM_NAMES)
            },
        })
    out["sites_needed"] = ladder

    dest = ob.OUT / "sensitivity.json"
    dest.write_text(json.dumps(out, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
