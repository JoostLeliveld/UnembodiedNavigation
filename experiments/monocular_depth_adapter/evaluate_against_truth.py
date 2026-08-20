#!/usr/bin/env python3
"""Score the depth models against real Gazebo depth.

    python3 experiments/monocular_depth_adapter/evaluate_against_truth.py

Two sides, kept strictly apart.

**Truth (evaluation only).** Real Gazebo depth from the camera's own co-located
depth sensor, captured by ``capture_depth_truth.py``. Nothing under
``monodepth/`` can see it.

**The model's side (deployment-legal only).** The one assumption allowed here is
the one a real warehouse can also make: *the floor is a plane at z = 0 and the
camera calibration is known*. That is enough to compute, analytically, the true
depth of any pixel whose ray lands on open floor — no depth sensor, no CAD
heights, no ground truth. Fitting scale and shift on those pixels is the whole
correction, and it is the same trick the repo's occlusion prior already uses.

Verified on this capture: analytic floor depth matches the Gazebo sensor to a
**median 2.4 mm / MAE 3.7 mm** on 470 595 free-floor pixels, which is what makes
the anchor trustworthy — and confirms the sensor returns optical-axis depth
rather than range along the ray (that alternative is off by 0.91 m MAE).

Three arms per model:

``raw``          what the network said, believed as metres.
``floor``        scale+shift fitted on open-floor pixels only, then applied
                 everywhere. Deployment-legal.
``oracle_affine``scale+shift fitted against the truth itself, on the scored
                 pixels. NOT deployable — it is the ceiling the floor anchor is
                 trying to reach, and the gap between the two is the price of
                 having to infer the anchor from the floor.

**Scored on pixels the anchor never saw**: everything that is not open floor —
racks, crates, walls, the robot. Scoring an anchor on its own fitting set would
report the fit, not the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import frozen_set as fs  # noqa: E402

sys.path.insert(0, str(fs.REPO / "src" / "unav_common"))
sys.path.insert(0, str(fs.REPO / "scripts" / "shared"))
from metrics import spearman as _spearman  # noqa: E402  THE shared one; never re-derive

#: The shared implementation returns (rho, n_finite).
def spearman(a, b) -> float:  # noqa: E302
    return float(_spearman(a, b)[0])


#: Rank correlations run on a strided subsample; full-frame costs minutes and
#: moves the third decimal.
RANK_STRIDE = 8

from monodepth import MonocularDepthAdapter, available_models  # noqa: E402
from monodepth.types import CameraIntrinsics, DepthRequest  # noqa: E402

TRUTH_DIR = fs.REPO / "logs/studies/monocular_depth_adapter/depth_truth_warehouse_aws"
OUT_DIR = fs.REPO / "logs/studies/monocular_depth_adapter/truth_evaluation"

#: Open-floor band for the anchor: in front of the rack rows, inside the walls.
#: Deployment-legal — it is a 2-D map of where the floor is, which any warehouse
#: has, not a height map and not CAD.
FREE_FLOOR_Y = (-4.9, -1.6)
FREE_FLOOR_ABS_X = 5.2

#: Camera A of warehouse_aws, from the capture manifest.
CAM_POS = (0.0, -5.5, 4.8)
CAM_LOOK_AT = (-1.3425623890456766e-05, -1.8449838730392667, 0.0)
IMG_W, IMG_H, FOV_H = 1280, 720, 1.5708


def _camera():
    from unav_common.camera_model import ObliqueCameraModel

    return ObliqueCameraModel(cam_pos=list(CAM_POS), look_at=list(CAM_LOOK_AT),
                              img_width=IMG_W, img_height=IMG_H, fov_h_rad=FOV_H)


def analytic_floor_depth(cam) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optical-axis depth to the plane z=0 for every pixel, plus where it lands.

    Pure calibration: no sensor, no CAD, no truth. This is the whole "floor
    assumption" — and the only outside knowledge the correction is allowed.
    """
    K, R, C = cam.K, cam.R, np.asarray(cam.cam_pos, dtype=float)
    u, v = np.meshgrid(np.arange(IMG_W, dtype=float), np.arange(IMG_H, dtype=float))
    rays_cam = np.linalg.inv(K) @ np.vstack([u.ravel(), v.ravel(), np.ones(u.size)])
    rays_world = R.T @ rays_cam
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -C[2] / rays_world[2]
    hit = C[:, None] + rays_world * t
    depth_z = t.reshape(IMG_H, IMG_W)
    hx = hit[0].reshape(IMG_H, IMG_W)
    hy = hit[1].reshape(IMG_H, IMG_W)
    # Rays pointing at or above the horizon never meet the floor; blank them so
    # nothing downstream mistakes a negative or infinite intersection for a hit.
    above_horizon = ~(np.isfinite(depth_z) & (depth_z > 0))
    depth_z[above_horizon] = np.nan
    hx[above_horizon] = np.nan
    hy[above_horizon] = np.nan
    return depth_z, hx, hy


def free_floor_mask(hx: np.ndarray, hy: np.ndarray, floor_depth: np.ndarray) -> np.ndarray:
    """Pixels whose floor ray lands on open floor, from the 2-D free-space map."""
    ok = np.isfinite(floor_depth) & (floor_depth > 0)
    return (ok & np.isfinite(hy) & (hy > FREE_FLOOR_Y[0]) & (hy < FREE_FLOOR_Y[1])
            & (np.abs(hx) < FREE_FLOOR_ABS_X))


def fit_scale_shift(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                    *, iters: int = 3, clip_sigma: float = 2.5) -> tuple[float, float]:
    """Robust least squares for ``target ~ a*pred + b`` over ``mask``.

    Iteratively trims outliers: some pixels the 2-D map calls floor are in fact
    a crate or the robot standing on it, and a plain fit would chase them.
    """
    p = pred[mask].astype(np.float64)
    t = target[mask].astype(np.float64)
    keep = np.isfinite(p) & np.isfinite(t)
    a, b = 1.0, 0.0
    for _ in range(iters):
        if keep.sum() < 64:
            break
        A = np.stack([p[keep], np.ones(keep.sum())], axis=1)
        (a, b), *_ = np.linalg.lstsq(A, t[keep], rcond=None)
        resid = np.abs(a * p + b - t)
        scale = resid[keep].std() or 1.0
        keep = np.isfinite(p) & np.isfinite(t) & (resid < clip_sigma * scale)
    return float(a), float(b)


def anchor_target(reference_depth: np.ndarray, convention) -> np.ndarray:
    """What the affine fit should regress onto, in the model's own space.

    An inverse-depth network is linear in 1/depth, not in depth. Fitting
    ``a*disparity + b ~ depth`` fits the wrong curve entirely — the tell is a
    NEGATIVE slope, because disparity falls where depth rises. Measured on this
    capture before the fix: scale -0.527, and 0.615 m MAE against a 0.300 m
    oracle. Regressing onto 1/depth instead is the standard alignment for this
    family and costs nothing.
    """
    from monodepth.types import DepthConvention

    if convention is DepthConvention.INVERSE_DEPTH:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(reference_depth > 0, 1.0 / reference_depth, np.nan)
    return reference_depth


def apply_anchor(pred: np.ndarray, convention, a: float, b: float) -> np.ndarray:
    """Push a fitted (a, b) back out as metres, in the model's own space."""
    from monodepth.types import DepthConvention

    fitted = a * pred + b
    if convention is DepthConvention.INVERSE_DEPTH:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(fitted > 1e-6, 1.0 / fitted, np.nan)
    return fitted


def depth_errors(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict:
    p, t = pred[mask].astype(np.float64), truth[mask].astype(np.float64)
    ok = np.isfinite(p) & np.isfinite(t) & (t > 0)
    p, t = p[ok], t[ok]
    if p.size == 0:
        return {"n": 0}
    err = p - t
    rel = np.abs(err) / t
    ratio = np.maximum(p / t, t / np.maximum(p, 1e-9))
    return {
        "n": int(p.size),
        "mae_m": float(np.abs(err).mean()),
        "rmse_m": float(np.sqrt((err ** 2).mean())),
        "median_abs_err_m": float(np.median(np.abs(err))),
        "bias_m": float(err.mean()),
        "abs_rel": float(rel.mean()),
        "delta1": float((ratio < 1.25).mean()),
        "p95_abs_err_m": float(np.percentile(np.abs(err), 95)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=None,
                        help="default: every model that ran in the reference inference run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--uncertainty", default="native+flip",
                        help="also score whether the uncertainty signal tracks the real error")
    args = parser.parse_args()

    manifest = json.loads((TRUTH_DIR / "manifest.json").read_text(encoding="utf-8"))
    frames = manifest["frames"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cam = _camera()
    intr = CameraIntrinsics.from_matrix(cam.K, IMG_W, IMG_H)
    floor_depth, hx, hy = analytic_floor_depth(cam)
    anchor_mask = free_floor_mask(hx, hy, floor_depth)

    from PIL import Image

    requests, truths = [], {}
    for fr in frames:
        rgb = np.asarray(Image.open(TRUTH_DIR / fr["rgb_file"]).convert("RGB"), dtype=np.uint8)
        requests.append(DepthRequest(fr["frame_id"], rgb, intr,
                                     source_path=str((TRUTH_DIR / fr["rgb_file"]).relative_to(fs.REPO))))
        truths[fr["frame_id"]] = np.load(TRUTH_DIR / fr["depth_file"])

    # The anchor's own pixels are excluded from scoring; what is left is the
    # structure the model actually had to infer.
    truth0 = truths[frames[0]["frame_id"]]
    scored_mask = np.isfinite(truth0) & (truth0 > 0) & ~anchor_mask
    print(f"anchor pixels (open floor): {anchor_mask.sum():,}   "
          f"scored pixels (everything else): {scored_mask.sum():,}")

    sanity = depth_errors(floor_depth, truth0, anchor_mask)
    print(f"floor-assumption sanity check vs Gazebo depth on the anchor pixels: "
          f"MAE {sanity['mae_m']*1000:.1f} mm, median {sanity['median_abs_err_m']*1000:.1f} mm")

    models = args.models
    if models is None:
        index_path = fs.REPO / "logs/studies/monocular_depth_adapter/bs1_native_flip/index.json"
        models = (sorted(json.loads(index_path.read_text())["models"])
                  if index_path.is_file() else available_models())

    results: dict = {
        "truth": {
            "source": "Gazebo external_camera/depth_camera (evaluation only)",
            "capture_dir": str(TRUTH_DIR.relative_to(fs.REPO)),
            "n_frames": len(frames),
            "convention": "metric_z (optical-axis), confirmed against analytic floor depth",
        },
        "anchor": {
            "assumption": "floor is a plane at z=0 and the camera calibration is known",
            "legal_inputs": ["camera calibration", "2-D map of open floor"],
            "illegal_inputs_not_used": ["ground truth depth", "CAD heights", "detector output"],
            "n_anchor_pixels": int(anchor_mask.sum()),
            "floor_assumption_vs_gazebo": sanity,
        },
        "scored_region": "every pixel that is NOT open floor (racks, crates, walls, robot)",
        "models": {},
    }

    for model_name in models:
        print(f"\n=== {model_name} ===", flush=True)
        try:
            with MonocularDepthAdapter(model_name, device=args.device, batch_size=1,
                                       uncertainty=args.uncertainty) as adapter:
                preds = adapter.predict(requests)
                convention = adapter.info.convention
        except Exception as exc:  # noqa: BLE001 - a model that cannot run is a result
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            results["models"][model_name] = {"failed": f"{type(exc).__name__}: {exc}"}
            continue

        per_frame = {"raw": [], "floor": [], "oracle_affine": []}
        scales = []
        signal_rho: dict[str, list] = {"native_confidence": [], "flip_spread": []}
        for req, pred in zip(requests, preds):
            truth = truths[req.image_id]
            valid = pred.valid & scored_mask
            anchor = pred.valid & anchor_mask

            if convention.is_metric:
                per_frame["raw"].append(depth_errors(pred.depth, truth, valid))

            # Deployment-legal: fitted on open floor, from calibration alone.
            a, b = fit_scale_shift(pred.depth, anchor_target(floor_depth, convention), anchor)
            per_frame["floor"].append(
                depth_errors(apply_anchor(pred.depth, convention, a, b), truth, valid))
            scales.append(a)

            # Not deployable: fitted against the truth, on the scored pixels. The
            # ceiling the floor anchor is chasing.
            ao, bo = fit_scale_shift(pred.depth, anchor_target(truth, convention), valid)
            per_frame["oracle_affine"].append(
                depth_errors(apply_anchor(pred.depth, convention, ao, bo), truth, valid))

            # Does either uncertainty signal know where the model is wrong? With
            # truth in hand this is answerable directly, instead of by proxy
            # through cross-model disagreement.
            anchored = apply_anchor(pred.depth, convention, a, b)
            abs_err = np.abs(anchored - truth)
            for name, signal in (("native_confidence", pred.native_confidence),
                                 ("flip_spread", pred.uncertainty
                                  if pred.uncertainty_kind == "flip_consistency" else None)):
                if signal is None:
                    continue
                m = valid & np.isfinite(signal) & np.isfinite(abs_err)
                if m.sum() > 1000:
                    signal_rho[name].append(
                        spearman(signal[m][::RANK_STRIDE], abs_err[m][::RANK_STRIDE]))

        def summarise(rows):
            rows = [r for r in rows if r.get("n")]
            if not rows:
                return None
            return {k: float(np.median([r[k] for r in rows]))
                    for k in ("mae_m", "rmse_m", "median_abs_err_m", "bias_m",
                              "abs_rel", "delta1", "p95_abs_err_m")}

        entry = {
            "convention": convention.value,
            "n_frames": len(preds),
            "median_floor_anchor_scale": float(np.median(scales)),
            "arms": {k: summarise(v) for k, v in per_frame.items()},
            "uncertainty_vs_true_error": {
                name: {
                    "median_spearman": float(np.median(vals)),
                    "n_frames": len(vals),
                    "reading": ("a signal that knows where the model is wrong correlates "
                                "POSITIVELY with |error| if it is an uncertainty, and "
                                "NEGATIVELY if it is a confidence"),
                }
                for name, vals in signal_rho.items() if vals
            },
        }
        results["models"][model_name] = entry

        for arm in ("raw", "floor", "oracle_affine"):
            s = entry["arms"][arm]
            if s is None:
                print(f"  {arm:<14} n/a (not a metric model)")
            else:
                print(f"  {arm:<14} MAE {s['mae_m']:.3f} m   RMSE {s['rmse_m']:.3f} m   "
                      f"AbsRel {s['abs_rel']:.3f}   within 25% {100*s['delta1']:.1f}%   "
                      f"bias {s['bias_m']:+.3f} m")
        print(f"  floor anchor scale x{entry['median_floor_anchor_scale']:.3f}")
        for name, blob in entry["uncertainty_vs_true_error"].items():
            print(f"  {name:<14} vs true |error|: spearman {blob['median_spearman']:+.3f}")

    out_path = out_dir / "truth_evaluation.json"
    scored = [m for m, e in results["models"].items() if "failed" not in e]

    # Do not let a run that measured nothing destroy a run that measured
    # something. This happened for real on 2026-08-11: another session's Gazebo
    # took the whole 4 GB card, every model OOM'd, and the all-failed result
    # overwrote a complete evaluation.
    if not scored and out_path.is_file():
        salvage = out_path.with_name("truth_evaluation.FAILED.json")
        salvage.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nevery model failed; kept the previous {out_path.name} and wrote "
              f"{salvage.name} instead")
        return 1

    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path.relative_to(fs.REPO)} ({len(scored)} model(s) scored)")
    return 0 if scored else 1


if __name__ == "__main__":
    raise SystemExit(main())
