"""What can the four cameras learn about their own mounting error, without truth?

Reads the real four-camera keypoint capture, takes the pixel noise from the real
held-out predictions of `yolo_pose_4cam_v1`, and computes the information the
capture carries about all 24 mounting-error parameters (6 per camera) after the
3176 unknown robot poses have been marginalised out. No pose is ever assumed
known; the truth columns are used only to place the sightings on the floor, which
in service is the robot's own belief.

The question is not "what is the drift" (there is none injected here) but "how
much of a drift COULD this data resolve" -- a property of the geometry and the
noise, not of any particular fault. That is why it needs no injected fault and no
new Gazebo run.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib_geometry as cg  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "logs/perception_datasets/projected_keypoint_dataset_4cam_v1"
PREDICTIONS = REPO / "logs/studies/keypoint_measurement/v1_4cam_retrained/per_sample.csv"
OUT = REPO / "logs/studies/overlap_self_calibration"

# The scale a resolvable fault is measured against: one unit is the smallest drift
# the existing GT-free change detector fires on (calibration_drift_lifecycle:
# 0.1 deg of rotation, 0.025 m of translation).
LADDER = np.array([0.1, 0.1, 0.1, 0.025, 0.025, 0.025])

# Stated prior on a drifted wall bracket. Reported alongside every posterior so
# the reader can see which numbers the data bought and which the prior did.
PRIOR_SD = np.array([1.0, 1.0, 1.0, 0.05, 0.05, 0.05])

# Readings whose worst pixel residual exceeds this are excluded from the noise
# estimate: 1 of 820 held-out both-markers-rendered readings.
RESIDUAL_TRIM_PX = 5.0


def pixel_noise() -> tuple[np.ndarray, dict]:
    """The measured 4x4 pixel residual covariance of this reading, in px^2."""
    d = pd.read_csv(PREDICTIONS)
    d = d[(d.detected == 1) & (d.front_labelled_visible == 1) & (d.rear_labelled_visible == 1)]
    cols = ["res_front_u", "res_front_v", "res_rear_u", "res_rear_v"]
    res = d[cols].to_numpy(float)
    keep = np.abs(res).max(axis=1) <= RESIDUAL_TRIM_PX
    res = res[keep]
    cov = np.cov(res - res.mean(axis=0), rowvar=False, ddof=1)
    meta = {
        "readings_used": int(len(res)),
        "readings_trimmed": int((~keep).sum()),
        "mean_px": [round(float(v), 4) for v in res.mean(axis=0)],
        "sd_px": [round(float(v), 4) for v in np.sqrt(np.diag(cov))],
    }
    return cov, meta


def sightings() -> pd.DataFrame:
    """Every accepted sighting where BOTH marker disks actually rendered."""
    d = pd.read_csv(DATASET / "capture_diagnostics.csv")
    d = d[(d.accepted == 1) & (d.front_visible == 1) & (d.rear_visible == 1)].copy()
    d["site"] = list(zip(d.x.round(6), d.y.round(6), d.yaw_rad.round(6)))
    return d


def bearing_deg(camera, pose) -> float:
    """Compass bearing from the robot to the camera, degrees."""
    c = np.asarray(camera.center_m, float)
    return float(np.degrees(np.arctan2(c[1] - pose[1], c[0] - pose[0])))


def information(rows: pd.DataFrame, cameras, marker_z, offsets, cov_px,
                free: tuple[str, ...]) -> tuple[np.ndarray, dict]:
    """Information on the mounting errors of `free`, poses marginalised out.

    Per site: stack every sighting's Jacobian, then Schur-complement the site's
    own three pose unknowns away. What is left is what the sighting says about
    the cameras rather than about where the robot was.
    """
    p = cg.N_PARAM
    slot = {name: i for i, name in enumerate(free)}
    total = np.zeros((p * len(free), p * len(free)))
    w = np.linalg.inv(cov_px)
    conds, sites_used, sightings_used = [], 0, 0

    for _, grp in rows.groupby("site", sort=False):
        pose = (float(grp.x.iloc[0]), float(grp.y.iloc[0]), float(grp.yaw_rad.iloc[0]))
        a = np.zeros((3, 3))
        b = np.zeros((3, p * len(free)))
        d = np.zeros((p * len(free), p * len(free)))
        for cam_name in grp.camera:
            j_pose, j_theta = cg.jacobians(cameras[cam_name], pose, marker_z, offsets)
            a += j_pose.T @ w @ j_pose
            sightings_used += 1
            if cam_name not in slot:
                continue                      # this camera is treated as known
            k = slot[cam_name] * p
            b[:, k:k + p] += j_pose.T @ w @ j_theta
            d[k:k + p, k:k + p] += j_theta.T @ w @ j_theta
        conds.append(float(np.linalg.cond(a)))
        total += d - b.T @ np.linalg.solve(a, b)
        sites_used += 1

    total = 0.5 * (total + total.T)
    return total, {
        "sites": sites_used,
        "sightings": sightings_used,
        "pose_information_condition_median": round(float(np.median(conds)), 1),
        "pose_information_condition_p95": round(float(np.percentile(conds, 95)), 1),
    }


def report(info: np.ndarray, free: tuple[str, ...]) -> dict:
    """Turn an information matrix into what a reader needs: what is learned, what is not."""
    p = cg.N_PARAM
    ladder = np.tile(LADDER, len(free))
    prior = np.tile(PRIOR_SD, len(free))

    # Nullspace / spectrum in ladder units: a direction with sd 1 can just resolve
    # a fault the size of the existing detector's threshold.
    scaled = info * np.outer(ladder, ladder)
    vals, vecs = np.linalg.eigh(scaled)
    vals = np.maximum(vals, 0.0)
    sd_dir = np.where(vals > 0, 1.0 / np.sqrt(np.maximum(vals, 1e-300)), np.inf)

    # Posterior with the stated prior. This is the honest per-parameter answer.
    post = np.linalg.inv(info + np.diag(1.0 / prior ** 2))
    sd_post = np.sqrt(np.diag(post))

    per_cam = {}
    for i, name in enumerate(free):
        s = sd_post[i * p:(i + 1) * p]
        per_cam[name] = {
            pname: {"posterior_sd": round(float(v), 5),
                    "prior_sd": float(PRIOR_SD[k]),
                    "shrinkage": round(float(v / PRIOR_SD[k]), 4)}
            for k, (pname, v) in enumerate(zip(cg.PARAM_NAMES, s))
        }

    order = np.argsort(sd_dir)[::-1]
    worst = []
    for idx in order[:4]:
        vec = vecs[:, idx]
        heavy = np.argsort(np.abs(vec))[::-1][:5]
        worst.append({
            "sd_in_detector_units": (None if not np.isfinite(sd_dir[idx])
                                     else round(float(sd_dir[idx]), 3)),
            "dominant_terms": [
                f"{free[j // p]}.{cg.PARAM_NAMES[j % p]} {vec[j]:+.2f}" for j in heavy
            ],
        })

    return {
        "directions_total": int(len(vals)),
        "directions_resolved_below_detector_threshold": int((sd_dir < 1.0).sum()),
        "directions_worse_than_10x_threshold": int((sd_dir > 10.0).sum()),
        "direction_sd_detector_units": {
            "best": round(float(sd_dir.min()), 4),
            "median": round(float(np.median(sd_dir)), 4),
            "worst": (None if not np.isfinite(sd_dir.max()) else round(float(sd_dir.max()), 3)),
        },
        "least_identified_directions": worst,
        "per_camera": per_cam,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "observability.json"))
    args = ap.parse_args()

    cameras, marker_z, offsets = cg.load_capture(DATASET)
    cov_px, noise_meta = pixel_noise()
    rows = sightings()
    names = tuple(sorted(cameras))

    multi = rows.groupby("site").camera.nunique()
    rows["n_cams"] = rows.site.map(multi)

    # Which pairs share a site, and how far apart their bearings are.
    pair_rows, seps = [], {}
    for site, grp in rows[rows.n_cams >= 2].groupby("site", sort=False):
        pose = (float(grp.x.iloc[0]), float(grp.y.iloc[0]), float(grp.yaw_rad.iloc[0]))
        cams = sorted(set(grp.camera))
        for a, b in combinations(cams, 2):
            sep = abs(bearing_deg(cameras[a], pose) - bearing_deg(cameras[b], pose))
            sep = min(sep, 360.0 - sep)
            seps.setdefault((a, b), []).append(sep)
            pair_rows.append({"site": site, "a": a, "b": b, "separation_deg": sep})
    pairs = pd.DataFrame(pair_rows)

    arms = {
        "all_sightings": rows,
        "overlap_only": rows[rows.n_cams >= 2],
        "single_camera_sites_only": rows[rows.n_cams == 1],
    }
    wide = set(pairs.loc[pairs.separation_deg > 100.0, "site"]) if len(pairs) else set()
    narrow = set(pairs.loc[pairs.separation_deg <= 100.0, "site"]) if len(pairs) else set()
    arms["overlap_wide_pairs_only"] = rows[rows.site.isin(wide - narrow)]
    arms["overlap_narrow_pairs_only"] = rows[rows.site.isin(narrow - wide)]

    out = {
        "capture": DATASET.name,
        "predictions": str(PREDICTIONS.relative_to(REPO)),
        "reading": "two marker keypoints -> four pixels per sighting",
        "pixel_noise": noise_meta,
        "pixel_covariance_px2": [[round(float(v), 5) for v in row] for row in cov_px],
        "detector_threshold_units": {"rotation_deg": 0.1, "translation_m": 0.025},
        "stated_prior_sd": dict(zip(cg.PARAM_NAMES, [float(v) for v in PRIOR_SD])),
        "overlap_census": {
            "sites_total": int(len(multi)),
            "sites_by_camera_count": {str(k): int(v) for k, v in
                                      multi.value_counts().sort_index().items()},
            "pair_bearing_separation_deg": {
                f"{a}|{b}": {"sites": len(v), "median_separation": round(float(np.median(v)), 1)}
                for (a, b), v in sorted(seps.items(), key=lambda kv: -len(kv[1]))
            },
        },
        "arms": {},
        "one_camera_at_a_time": {},
    }

    for arm, subset in arms.items():
        if not len(subset):
            continue
        info, meta = information(subset, cameras, marker_z, offsets, cov_px, names)
        out["arms"][arm] = {**meta, **report(info, names)}
        print(f"[{arm}] sites={meta['sites']} sightings={meta['sightings']}")

    # Can one camera identify its own mounting error if the other three are trusted?
    for cam in names:
        subset = rows[rows.camera == cam]
        info, meta = information(subset, cameras, marker_z, offsets, cov_px, (cam,))
        out["one_camera_at_a_time"][cam] = {**meta, **report(info, (cam,))}
        print(f"[alone: {cam}] sites={meta['sites']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
