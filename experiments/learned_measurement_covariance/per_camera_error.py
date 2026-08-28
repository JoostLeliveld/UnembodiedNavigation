"""What each camera's hull reading is actually worth: its bias, its spread, and whether
the covariance it states matches either of them.

Every reading is scored against the truth at the instant the CAMERA saw it (``obs_stamp``),
never at logging time -- scoring late charges the pipeline delay to the sensor. Ground truth
is used to score and for nothing else.

**One row per reading.** The camera manager decides at 20 Hz while the detector produces
5 Hz, so each detection is republished on about four consecutive decisions and written to
``fusion_observations.csv`` four times: 25656 rows for 6418 distinct readings across these
drives. Counting rows made every ``n`` here 4x too large, made the likelihood comparison
between the one-term and two-term models about 4x more decisive than the data supports, and
weighted each camera by how long the manager kept re-fusing it. Loading goes through
``aligned.readings``, which keeps the first row per (camera, capture time).

**Readings are also reported unconditionally.** ``used == 1`` means the arm's fusion rule
accepted the reading, and acceptance runs through the disagreement gate -- through agreeing
with the other cameras, which is close to agreeing with the truth. So the admitted set is
what the runtime consumed but is not a clean per-camera error; both are printed.

Three questions, in order:

1. **Bias.** Is the mean residual zero, per camera and in the camera's own frame (along its
   line of sight, and across it)?
2. **Spread.** Is the spread of the centred residual the size the stated covariance claims?
   The runtime states ``R = J^-1 sigma_px^2 J^-T`` with ONE commissioned pixel number for
   every camera, so the only way it can differ per camera is through geometry.
3. **Where the spread comes from.** Push each residual back into pixels. If the network's
   error really is a pixel-space detector error, the implied pixel noise is the same number
   at every camera and every range. Where it is not, something that lives in METRES -- shape,
   ground plane, the pose the prediction was made from -- is being charged to pixels.

The candidate fix is one extra parameter, stated before it is fitted:

    R = (sigma_px / sigma_commissioned)^2 * C_stated  +  sigma_m^2 * I

**Decision rule, written before the fit.** The two-term model replaces the one-term model
only if, on the HELD-OUT drive, it brings every camera's normalised squared error inside
[0.5, 2.0] where the one-term model does not, AND improves the pooled log-likelihood. A null
is a result: if one pixel number already does this, the extra parameter is not earned.

    python3 experiments/learned_measurement_covariance/per_camera_error.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/measurement_commissioning"))
sys.path.insert(0, str(REPO / "experiments/fusion_on_fixed_routes"))
from camera import camera_models  # noqa: E402
import aligned as A  # noqa: E402

DRIVES = REPO / "logs/studies/fusion_on_fixed_routes"
OUT = REPO / "logs/studies/learned_measurement_covariance"
DATASET = REPO / "logs/perception_datasets/warehouse_v2_yolo_shared_20260822"
CAMS = camera_models(DATASET)

#: the pixel noise every camera's stated covariance was built from
SIGMA_PX = json.loads((REPO / "logs/studies/measurement_commissioning/calibration.json")
                      .read_text())["calibration"]["sigma_px"]
#: drives that log the capture time. The last one is held out from every fit.
FIT = (("drives_commissioning_v2", "K0"), ("drives_commissioning_v2", "K1"),
       ("drives_commissioning_v3", "K0"), ("drives_commissioning_v3", "K2"))
HELD_OUT = (("drives_commissioning_v4", "K3"),)
#: median of a 2-D chi-square, for turning a median NEES into a scale factor
CHI2_2_MEDIAN = 2.0 * math.log(2.0)


def _run_dir(campaign: str, arm: str) -> Path:
    """The one drive for this arm, chosen deterministically.

    This took ``hits[0]`` of an unordered ``rglob``, so which drive a fit used depended
    on filesystem order. Sorted, and it refuses rather than guesses when a campaign
    holds more than one run for the arm.
    """
    hits = sorted({p.parent for p in (DRIVES / campaign).rglob("fusion_observations.csv")
                   if f"/{arm}/" in str(p)})
    if not hits:
        raise SystemExit(f"no drive for {campaign}/{arm}")
    if len(hits) > 1:
        listed = "\n  ".join(str(h.relative_to(REPO)) for h in hits)
        raise SystemExit(
            f"{campaign}/{arm} holds {len(hits)} drives; name one explicitly rather "
            f"than letting the loader pick:\n  {listed}")
    return hits[0]


def readings(campaign: str, arm: str, *, admitted_only: bool = True) -> list[dict]:
    """Every reading, once, scored at its own capture time, in the camera's own frame."""

    run = _run_dir(campaign, arm)
    out = []
    for entry in A.readings(run, admitted_only=admitted_only, dedupe=True):
        cov = entry["cov"]
        if not np.isfinite(cov).all() or np.linalg.det(cov) <= 0.0:
            continue
        truth = entry["truth"]
        camera = CAMS[f"camera_{entry['camera']}"]
        ray = truth - np.asarray(camera.cam_pos[:2])
        norm = float(np.linalg.norm(ray))
        if norm <= 0.0:
            continue
        ray = ray / norm                                     # away from the camera
        across = np.array([-ray[1], ray[0]])
        error = entry["error"]
        out.append(dict(camera=entry["camera"], drive=f"{campaign}/{arm}", cov=cov,
                        error=error, along=float(error @ ray), across=float(error @ across),
                        ray=ray, range_m=entry["range_m"],
                        stamp=entry["decision_stamp"], cap=entry["obs_stamp"],
                        truth=truth, truth_yaw=entry["truth_yaw"],
                        bbox_h_px=entry["bbox_h_px"], conf=entry["conf"],
                        used=entry["used"]))
    return out


def _stack(rows):
    """(covariances, residuals) as arrays -- every 2x2 solve below is done in closed form."""

    return (np.array([r["cov"] for r in rows]), np.array([r["error"] for r in rows]))


def _quadratic(rows, scale: float = 1.0, floor_m: float = 0.0):
    """``e' R^-1 e`` and ``det R`` under ``R = scale^2 C + floor^2 I``, for every reading."""

    cov, err = rows if isinstance(rows, tuple) else _stack(rows)
    a = scale ** 2 * cov[:, 0, 0] + floor_m ** 2
    b = scale ** 2 * cov[:, 0, 1]
    d = scale ** 2 * cov[:, 1, 1] + floor_m ** 2
    det = a * d - b * b
    ex, ey = err[:, 0], err[:, 1]
    quad = (d * ex * ex - 2.0 * b * ex * ey + a * ey * ey) / det
    return quad, det


def _nees(rows, scale: float = 1.0, floor_m: float = 0.0):
    """Normalised squared error under ``R = scale^2 * C + floor^2 I``. Honest is 2.0."""

    return _quadratic(rows, scale, floor_m)[0]


def _nll(rows, scale: float, floor_m: float) -> float:
    """Mean negative log-likelihood of the residuals under the same model."""

    quad, det = _quadratic(rows, scale, floor_m)
    return float(np.mean(0.5 * (np.log(det) + quad)))


def fit_two_term(rows, floors=np.arange(0.0, 0.061, 0.0005), scales=np.arange(0.5, 3.01, 0.05)):
    """Grid search over (pixel-noise scale, metric floor). Two parameters, no more."""

    stacked = _stack(rows)
    best = (math.inf, 1.0, 0.0)
    for floor in floors:
        for scale in scales:
            value = _nll(stacked, float(scale), float(floor))
            if value < best[0]:
                best = (value, float(scale), float(floor))
    return dict(nll=best[0], sigma_px=best[1] * SIGMA_PX, floor_cm=best[2] * 100,
                scale=best[1], floor_m=best[2])


def per_camera_table(rows, title, scale=1.0, floor_m=0.0):
    print(f"\n{title}")
    print(f"  {'cam':4} {'n':>5} {'bias along':>11} {'bias across':>12} "
          f"{'spread along':>13} {'spread across':>14} {'it claims':>10} {'NSE':>6}")
    table = {}
    for camera in sorted({r["camera"] for r in rows}):
        sub = [r for r in rows if r["camera"] == camera]
        along = np.array([r["along"] for r in sub]) * 100
        across = np.array([r["across"] for r in sub]) * 100
        claim = np.median([math.sqrt(np.trace(scale ** 2 * r["cov"]
                                              + floor_m ** 2 * np.eye(2)) / 2)
                           for r in sub]) * 100
        nse = float(np.median(_nees(sub, scale, floor_m)) / CHI2_2_MEDIAN * 2.0)
        print(f"  {camera:4} {len(sub):5d} {along.mean():+8.2f} cm {across.mean():+9.2f} cm "
              f"{along.std():10.2f} cm {across.std():11.2f} cm {claim:7.2f} cm {nse:6.2f}")
        table[camera] = dict(n=len(sub), bias_along_cm=float(along.mean()),
                             bias_across_cm=float(across.mean()),
                             spread_along_cm=float(along.std()),
                             spread_across_cm=float(across.std()),
                             claimed_cm=float(claim), nse=nse,
                             median_error_cm=float(np.median(
                                 [np.linalg.norm(r["error"]) for r in sub]) * 100),
                             median_range_m=float(np.median([r["range_m"] for r in sub])))
    return table


def implied_pixel_noise(rows, bins=(0, 4, 8, 12, 16, 26)):
    """The residual pushed back into pixels: sigma_px * sqrt(NIS), no Jacobian needed.

    ``C = J^-1 sigma^2 J^-T`` exactly, so ``e' C^-1 e = |J e|^2 / sigma^2``. If the error is
    a pixel-space detector error, this comes out the same at every range. If a term that
    lives in metres is present, it blows up where the geometry is sharp -- close in.
    """

    out = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        sub = [r for r in rows if lo <= r["range_m"] < hi]
        if len(sub) < 30:
            continue
        nis = _nees(sub)
        out[f"{lo}-{hi} m"] = dict(
            n=len(sub),
            implied_px=float(SIGMA_PX * math.sqrt(np.median(nis) / CHI2_2_MEDIAN)),
            median_error_cm=float(np.median([np.linalg.norm(r["error"]) for r in sub]) * 100),
            claimed_cm=float(np.median([math.sqrt(np.trace(r["cov"]) / 2) for r in sub]) * 100))
    return out


def main():
    fit_rows = [r for c, a in FIT for r in readings(c, a)]
    test_rows = [r for c, a in HELD_OUT for r in readings(c, a)]
    print(f"fit set {len(fit_rows)} readings over {len(FIT)} drives; "
          f"held out {len(test_rows)} readings ({HELD_OUT[0][0]}/{HELD_OUT[0][1]})")
    print(f"every stated covariance is J^-1 ({SIGMA_PX:.3f} px)^2 J^-T -- one pixel number, "
          f"all five cameras")

    one = per_camera_table(fit_rows, "FIT SET, as deployed: one pixel number through geometry")
    fitted = fit_two_term(fit_rows)
    print(f"\n  two-term fit on the fit set: pixel noise {fitted['sigma_px']:.2f} px "
          f"(deployed {SIGMA_PX:.2f}), metric floor {fitted['floor_cm']:.2f} cm")
    px_only = fit_two_term(fit_rows, floors=np.array([0.0]))
    print(f"  pixel noise alone, refitted : {px_only['sigma_px']:.2f} px, "
          f"mean NLL {px_only['nll']:+.3f} against {fitted['nll']:+.3f} for the two-term model")

    # What the admission gate is worth to these numbers. `used == 1` means the fusion
    # rule accepted the reading, and acceptance runs through the disagreement gate, so
    # the admitted set is conditioned on agreeing with the other cameras. Printed rather
    # than argued about.
    all_rows = [r for c, a in FIT + HELD_OUT
                for r in readings(c, a, admitted_only=False)]
    admitted = [r for r in all_rows if r["used"]]
    if all_rows and len(admitted) != len(all_rows):
        print(f"\n  every reading vs the ones the rule used: {len(all_rows)} -> "
              f"{len(admitted)} ({100.0 * len(admitted) / len(all_rows):.0f}% admitted)")
        for label, subset in (("every reading", all_rows), ("admitted only", admitted)):
            err = np.array([np.linalg.norm(r["error"]) for r in subset]) * 100
            print(f"    {label:16} median error {np.median(err):5.2f} cm, "
                  f"p95 {np.percentile(err, 95):6.2f} cm, "
                  f"NSE {np.median(_nees(subset)) / CHI2_2_MEDIAN * 2.0:5.2f}")

    held_one = per_camera_table(test_rows, "HELD OUT, as deployed")
    held_px = per_camera_table(test_rows, "HELD OUT, pixel noise refitted, no floor",
                               scale=px_only["scale"])
    held_two = per_camera_table(test_rows, "HELD OUT, two terms: pixels through geometry "
                                           "PLUS a metric floor",
                                scale=fitted["scale"], floor_m=fitted["floor_m"])

    print("\nwhere the spread comes from: the residual pushed back into pixels")
    print(f"  {'range':10} {'n':>6} {'implied pixel noise':>21} {'error':>9} {'it claims':>11}")
    bands = implied_pixel_noise(fit_rows + test_rows)
    for band, v in bands.items():
        print(f"  {band:10} {v['n']:6d} {v['implied_px']:16.2f} px "
              f"{v['median_error_cm']:7.2f} cm {v['claimed_cm']:8.2f} cm")
    per_cam_px = {}
    for camera in sorted({r["camera"] for r in fit_rows + test_rows}):
        sub = [r for r in fit_rows + test_rows if r["camera"] == camera]
        per_cam_px[camera] = dict(
            implied_px=float(SIGMA_PX * math.sqrt(np.median(_nees(sub)) / CHI2_2_MEDIAN)),
            median_range_m=float(np.median([r["range_m"] for r in sub])), n=len(sub))
    print("\n  per camera:")
    for camera, v in per_cam_px.items():
        print(f"  {camera:4} {v['n']:6d} {v['implied_px']:16.2f} px   "
              f"at a median {v['median_range_m']:.1f} m")

    verdict_one = all(0.5 <= v["nse"] <= 2.0 for v in held_one.values())
    verdict_two = all(0.5 <= v["nse"] <= 2.0 for v in held_two.values())
    print(f"\nDECISION RULE: every camera inside [0.5, 2.0] on the held-out drive?")
    print(f"  as deployed        : {verdict_one}   "
          f"(worst {max(v['nse'] for v in held_one.values()):.2f})")
    print(f"  pixel noise alone  : "
          f"{all(0.5 <= v['nse'] <= 2.0 for v in held_px.values())}   "
          f"(worst {max(v['nse'] for v in held_px.values()):.2f})")
    print(f"  two terms          : {verdict_two}   "
          f"(worst {max(v['nse'] for v in held_two.values()):.2f})")

    everything = _height_ratios(fit_rows + test_rows)
    mechanism = why_the_bias(everything)
    print("\nthe bias, against how much of the robot the detector's box actually covered")
    print(f"  {'height ratio':14} {'n':>6} {'bias along the ray':>20} {'spread':>10} {'median range':>14}")
    for band, v in mechanism["by_height_ratio"].items():
        print(f"  {band:14} {v['n']:6d} {v['bias_along_cm']:+16.2f} cm "
              f"{v['spread_along_cm']:7.2f} cm {v['median_range_m']:11.1f} m")
    print(f"\n  {'range':10} {'n':>6} {'bias along':>13} {'share of boxes under 95% of predicted':>40}")
    for band, v in mechanism["by_range"].items():
        print(f"  {band:10} {v['n']:6d} {v['bias_along_cm']:+9.2f} cm "
              f"{v['fraction_under_0_95'] * 100:38.1f}%")

    gates = gate_options(everything)
    print("\nwhat tightening the admission check buys, and what it costs")
    print(f"  {'rule':48} {'kept':>6} {'bias':>9} {'worst camera':>14} {'error':>9} {'worst NSE':>10}")
    for label, v in gates.items():
        print(f"  {label:48} {v['kept_fraction'] * 100:5.0f}% {v['bias_along_cm']:+6.2f} cm "
              f"{v['worst_camera_bias_cm']:11.2f} cm {v['median_error_cm']:6.2f} cm "
              f"{v['worst_camera_nse']:10.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_camera_error.json").write_text(json.dumps(dict(
        sigma_px_deployed=SIGMA_PX, fit_drives=[f"{c}/{a}" for c, a in FIT],
        held_out_drives=[f"{c}/{a}" for c, a in HELD_OUT],
        n_fit=len(fit_rows), n_held_out=len(test_rows),
        fit_set_as_deployed=one, held_out_as_deployed=held_one,
        held_out_pixel_refit=held_px, held_out_two_term=held_two,
        two_term_fit=fitted, pixel_only_fit=px_only,
        implied_pixel_noise_by_range=bands, implied_pixel_noise_by_camera=per_cam_px,
        decision=dict(rule="every camera NSE in [0.5, 2.0] on the held-out drive",
                      as_deployed=verdict_one, two_term=verdict_two),
        bias_mechanism=mechanism, gate_options=gates,
    ), indent=1))
    print(f"\nwrote {OUT / 'per_camera_error.json'}")




def why_the_bias(rows):
    """Is the radial bias the gate letting truncated boxes through?

    The gate keeps a detection whose box is at least 85% of the predicted height. A box
    shortened by an occluder keeps its top and loses its bottom, so its bottom-centre
    back-projects FURTHER from the camera -- outward along the ray, which is exactly where
    the bias sits. The height ratio is available online; here it is recomputed against the
    truth so the two can be compared without the prior in the way.
    """

    from unav_common.robot_hull import VISUAL_HULL, silhouette_box  # noqa: PLC0415

    out = {"by_height_ratio": {}, "by_range": {}}
    for r in rows:
        box = silhouette_box(CAMS[f"camera_{r['camera']}"], r["truth"][0], r["truth"][1],
                             r["truth_yaw"], VISUAL_HULL)
        r["height_ratio"] = ((r["bbox_h_px"] / (box[3] - box[1]))
                             if box is not None and box[3] > box[1] else float("nan"))
    good = [r for r in rows if math.isfinite(r.get("height_ratio", float("nan")))]
    for lo, hi in ((0.85, 0.92), (0.92, 0.96), (0.96, 0.99), (0.99, 1.02), (1.02, 3.0)):
        sub = [r for r in good if lo <= r["height_ratio"] < hi]
        if len(sub) < 30:
            continue
        out["by_height_ratio"][f"{lo:.2f}-{hi:.2f}"] = dict(
            n=len(sub), bias_along_cm=float(np.mean([r["along"] for r in sub]) * 100),
            spread_along_cm=float(np.std([r["along"] for r in sub]) * 100),
            median_range_m=float(np.median([r["range_m"] for r in sub])))
    for lo, hi in ((0, 4), (4, 8), (8, 12), (12, 16), (16, 26)):
        sub = [r for r in good if lo <= r["range_m"] < hi]
        if len(sub) < 30:
            continue
        out["by_range"][f"{lo}-{hi} m"] = dict(
            n=len(sub), bias_along_cm=float(np.mean([r["along"] for r in sub]) * 100),
            median_height_ratio=float(np.median([r["height_ratio"] for r in sub])),
            fraction_under_0_95=float(np.mean([r["height_ratio"] < 0.95 for r in sub])))
    return out


def _height_ratios(rows):
    """How much of the robot the detector's box actually covered, against the hull."""

    from unav_common.robot_hull import VISUAL_HULL, silhouette_box  # noqa: PLC0415

    for r in rows:
        box = silhouette_box(CAMS[f"camera_{r['camera']}"], r["truth"][0], r["truth"][1],
                             r["truth_yaw"], VISUAL_HULL)
        r["height_ratio"] = ((r["bbox_h_px"] / (box[3] - box[1]))
                             if box is not None and box[3] > box[1] else float("nan"))
    return [r for r in rows if math.isfinite(r["height_ratio"])]


def gate_options(rows):
    """What tightening the admission check buys, and what it costs in readings.

    The paper's currency is availability, so a bias removed by refusing readings is not
    free -- it is paid for in corrections that never arrive. Both numbers, side by side.
    """

    options = {
        "as deployed (85% of predicted height, any range)": lambda r: True,
        "at least 96% of the predicted height": lambda r: r["height_ratio"] >= 0.96,
        "at least 99% of the predicted height": lambda r: r["height_ratio"] >= 0.99,
        "closer than 16 m": lambda r: r["range_m"] < 16.0,
        "closer than 16 m AND at least 96%": lambda r: r["range_m"] < 16.0 and r["height_ratio"] >= 0.96,
    }
    out = {}
    for label, keep in options.items():
        sub = [r for r in rows if keep(r)]
        if not sub:
            continue
        per_cam = {}
        for camera in sorted({r["camera"] for r in sub}):
            cam_rows = [r for r in sub if r["camera"] == camera]
            per_cam[camera] = dict(
                kept=len(cam_rows) / max(1, len([r for r in rows if r["camera"] == camera])),
                bias_along_cm=float(np.mean([r["along"] for r in cam_rows]) * 100),
                nse=float(np.median(_nees(cam_rows)) / CHI2_2_MEDIAN * 2.0))
        out[label] = dict(
            kept_fraction=len(sub) / len(rows),
            bias_along_cm=float(np.mean([r["along"] for r in sub]) * 100),
            worst_camera_bias_cm=float(max(abs(v["bias_along_cm"]) for v in per_cam.values())),
            median_error_cm=float(np.median([np.linalg.norm(r["error"]) for r in sub]) * 100),
            worst_camera_nse=float(max(v["nse"] for v in per_cam.values())),
            per_camera=per_cam)
    return out

if __name__ == "__main__":
    main()
